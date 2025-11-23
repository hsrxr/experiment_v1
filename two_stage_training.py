import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import time

# === 1. 配置与全局常数 (Global Constants) ===

@dataclass
class ExperimentConfig:
    # 实验设置
    run_id: str = "test_run"
    k: float = 2.4            # 扩展倍数
    x: float = 0.2            # 第一阶段预算占比
    c_total: float = 1.0e18   # 总算力预算 (FLOPs)
    
    # 模型架构 (Base N2)
    vocab_size: int = 50257   # GPT-2 style
    hidden_dim: int = 512
    n_layers_n2: int = 12
    n_heads: int = 8
    seq_len: int = 1024
    
    # 训练超参
    global_batch_size: int = 524288 # 524k tokens
    max_lr: float = 6.0e-4
    min_lr: float = 6.0e-5    # Decay target
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 42
    
    # 系统
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === 2. 模型架构 (Llama-style) ===

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight

class RoPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        # (max_len, dim/2) -> (max_len, dim) via complex
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, q, k):
        # q, k: (B, Seq, Heads, HeadDim)
        seq_len = q.shape[1]
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # SwiGLU
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Attention(nn.Module):
    def __init__(self, args: ExperimentConfig):
        super().__init__()
        self.head_dim = args.hidden_dim // args.n_heads
        self.n_heads = args.n_heads
        self.wq = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.wk = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.wv = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.wo = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.rope = RoPE(self.head_dim)

    def forward(self, x):
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim)

        q, k = self.rope(q, k)

        # Flash Attention (Simplified implementation)
        # Transpose for attention: (B, Heads, S, HeadDim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(out)

class TransformerBlock(nn.Module):
    def __init__(self, args: ExperimentConfig):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MLP(args.hidden_dim, 4 * args.hidden_dim) # 4x standard
        self.attention_norm = RMSNorm(args.hidden_dim)
        self.ffn_norm = RMSNorm(args.hidden_dim)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class LlamaModel(nn.Module):
    def __init__(self, args: ExperimentConfig, n_layers: int):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.hidden_dim = args.hidden_dim
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(n_layers)])
        self.norm = RMSNorm(args.hidden_dim)
        self.output = nn.Linear(args.hidden_dim, args.vocab_size, bias=False)
        
        # Tie weights
        self.output.weight = self.tok_embeddings.weight

    def forward(self, x):
        h = self.tok_embeddings(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.output(h)
        return logits

    def count_non_emb_params(self):
        """计算非Embedding参数量 (6ND近似)"""
        total_params = sum(p.numel() for p in self.parameters())
        emb_params = self.tok_embeddings.weight.numel() 
        # Output weight is tied, so only subtract one instance
        return total_params - emb_params

# === 3. 预算计算 (Implementation of Section 9.1) ===

def calculate_stage_tokens(c_total: float, x: float, n1_params_non_emb: int, n2_params_non_emb: int) -> Tuple[int, int]:
    """
    基于 FLOPs 比例计算两阶段 Token 数
    Budget1 = C_total * x
    Budget2 = C_total * (1-x)
    Tokens = Budget / (6 * NonEmbParams)
    """
    if x <= 0: return 0, int(c_total / (6 * n2_params_non_emb))
    if x >= 1.0: return int(c_total / (6 * n1_params_non_emb)), 0

    tokens_s1 = (c_total * x) / (6 * n1_params_non_emb)
    tokens_s2 = (c_total * (1 - x)) / (6 * n2_params_non_emb)
    
    return int(tokens_s1), int(tokens_s2)

# === 4. 模型生长与权重迁移 (Implementation of Section 6 & 9.2) ===

def create_and_transfer_model(old_model: LlamaModel, args: ExperimentConfig, target_layers: int) -> LlamaModel:
    """
    Top Stacking + Zero Output Init
    """
    logger.info(f"正在进行模型生长: {len(old_model.layers)} Layers -> {target_layers} Layers")
    
    new_model = LlamaModel(args, n_layers=target_layers).to(args.device).to(args.dtype)
    L1 = len(old_model.layers)
    
    # 1. 复制 Embedding 和 Final Norm
    new_model.tok_embeddings.weight.data.copy_(old_model.tok_embeddings.weight.data)
    new_model.norm.weight.data.copy_(old_model.norm.weight.data)
    # Output weight is tied, automatically handled
    
    # 2. Layer 复制逻辑
    for i in range(target_layers):
        source_idx = i if i < L1 else i % L1  # Top Stacking: 循环复制
        
        # 深拷贝旧层权重
        new_model.layers[i].load_state_dict(old_model.layers[source_idx].state_dict())
        
        # 3. Zero Init (Section 6.1 Point 3)
        # 仅置零 o_proj 和 down_proj
        if i >= L1: # 仅对新生成的层做 Zero Init
            nn.init.zeros_(new_model.layers[i].attention.wo.weight)
            nn.init.zeros_(new_model.layers[i].feed_forward.down_proj.weight)
            logger.info(f"  Layer {i}: Initialized from Layer {source_idx}, Zero-init applied to o_proj/down_proj.")
        else:
            # logger.info(f"  Layer {i}: Copied exactly from Layer {source_idx}.")
            pass
            
    return new_model

def get_layer_index(name: str) -> Optional[int]:
    """辅助函数: 从参数名 'layers.5.attention...' 中提取 5"""
    parts = name.split('.')
    for i, part in enumerate(parts):
        if part == 'layers' and i + 1 < len(parts):
            try:
                return int(parts[i+1])
            except ValueError:
                pass
    return None

def map_optimizer_states(opt_n1: torch.optim.Optimizer, model_n2: nn.Module, L1: int):
    """
    Section 9.2: 优化器状态迁移
    """
    logger.info("正在迁移优化器状态...")
    
    # 获取旧状态
    # 注意: PyTorch optimizer.state_dict()['state'] 是以 param_id (int) 为 key 的
    # 我们需要构建 map: param_name -> state
    
    # 1. 构建旧模型的 Name -> State 映射
    # 问题: opt_n1.state 里的 key 是 id, 不是 name。
    # 需要通过 opt_n1.param_groups[0]['params'] (ids) 和 model_n1.named_parameters() 对应起来
    # 但这里 model_n1 已经由 old_model 变量持有，参数对象没变
    
    # 更简单的方法：我们假设 old_model 的参数对象在 opt_n1 中是可索引的
    # 但是我们传入的是 opt_n1 对象。
    
    # 实际上，我们需要 model_n1 来辅助映射，但函数签名里没传 model_n1。
    # 在本脚本中，我们可以在调用前保持引用。
    # 为简化，我们在调用处直接处理，这里假设我们能获取旧参数的 map。
    pass # 逻辑移至主循环中处理

# === 5. 学习率调度器 (WSD Schedule) ===

class WSDScheduler:
    def __init__(self, optimizer, total_steps, warmup_ratio, decay_ratio, start_lr, max_lr, end_lr):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.end_lr = end_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        # Warmup
        if self.current_step < self.warmup_steps:
            return self.max_lr * (self.current_step / max(1, self.warmup_steps))
        
        # Stable
        if self.current_step < self.warmup_steps + self.stable_steps:
            return self.max_lr
            
        # Decay (Linear or generic, PDF implies simple decay 20%)
        decay_progress = (self.current_step - (self.warmup_steps + self.stable_steps)) / max(1, self.decay_steps)
        # Assuming linear decay to end_lr (or 0)
        return self.max_lr - (self.max_lr - self.end_lr) * min(1.0, decay_progress)

# === 6. 数据模拟 (Mock Data) ===

class SyntheticDataset(torch.utils.data.Dataset):
    """
    模拟 OpenWebText 数据流，用于冒烟测试
    """
    def __init__(self, vocab_size, seq_len, length=1000000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 返回随机 token
        return torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)

def get_dataloader(args, batch_size):
    # 实际应用中请替换为真实的 Dataset
    dataset = SyntheticDataset(args.vocab_size, args.seq_len)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size // args.seq_len, num_workers=0)
    # 注意: 这里 batch_size 是 token 数，这只是一个模拟。
    # 真实实现中 batch_size 应该是 sequences 数 = global_batch_size / seq_len

# === 7. 主流程 ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="A-3")
    parser.add_argument("--k", type=float, default=2.4)
    parser.add_argument("--x", type=float, default=0.2)
    parser.add_argument("--baseline", action="store_true", help="是否为 Baseline (B-1)")
    cmd_args = parser.parse_args()

    # 1. 初始化配置
    cfg = ExperimentConfig(
        run_id=cmd_args.run_id,
        k=cmd_args.k,
        x=cmd_args.x
    )
    
    # 设置种子
    torch.manual_seed(cfg.seed)
    
    # 计算层数
    L2 = cfg.n_layers_n2
    L1 = int(L2 / cfg.k) if not cmd_args.baseline else L2 # 如果是 Baseline，全程 L2
    
    logger.info(f"=== Experiment: {cfg.run_id} ===")
    logger.info(f"Device: {cfg.device}, Precision: {cfg.dtype}")
    logger.info(f"Settings: k={cfg.k}, x={cfg.x}, Baseline={cmd_args.baseline}")
    logger.info(f"Arch: Stage1 Layers={L1}, Stage2 Layers={L2}")

    # 2. 实例化 Stage 1 模型
    model = LlamaModel(cfg, n_layers=L1).to(cfg.device).to(cfg.dtype)
    n1_non_emb = model.count_non_emb_params()
    logger.info(f"Stage 1 Non-Emb Params: {n1_non_emb / 1e6:.2f}M")
    
    # 虚拟 N2 仅用于计算 budget
    temp_n2 = LlamaModel(cfg, n_layers=L2)
    n2_non_emb = temp_n2.count_non_emb_params()
    del temp_n2
    
    # 3. 计算预算 (Tokens)
    if cmd_args.baseline:
        # Baseline: x=1.0 for N2 essentially (Original doc says Run B-1 k=1.0 x=1.0)
        # 但 B-1 是单阶段，所以直接全程算 Tokens
        tokens_s1 = 0
        tokens_s2 = int(cfg.c_total / (6 * n2_non_emb))
    else:
        tokens_s1, tokens_s2 = calculate_stage_tokens(cfg.c_total, cfg.x, n1_non_emb, n2_non_emb)
        
    logger.info(f"Budget Tokens: Stage1={tokens_s1/1e9:.3f}B, Stage2={tokens_s2/1e9:.3f}B")

    # 转换 Tokens 为 Steps
    bs_tokens = cfg.global_batch_size
    steps_s1 = max(1, tokens_s1 // bs_tokens)
    steps_s2 = max(1, tokens_s2 // bs_tokens)
    
    logger.info(f"Training Steps: Stage1={steps_s1}, Stage2={steps_s2}")

    # 4. 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.max_lr, 
        betas=(cfg.beta1, cfg.beta2),
        eps=1e-8,
        weight_decay=cfg.weight_decay
    )
    
    # 5. Stage 1 训练
    if not cmd_args.baseline and steps_s1 > 0:
        logger.info(">>> Starting Stage 1 Training")
        
        # Stage 1 Schedule: Warmup 10% -> Stable 90% (No decay specified in doc for Stage 1 of A-3?)
        # Doc says: "Stage 1: Warmup (10%) -> Stable (90%, No Decay)"
        scheduler = WSDScheduler(
            optimizer, steps_s1, 
            warmup_ratio=0.1, decay_ratio=0.0, # No decay
            start_lr=0, max_lr=cfg.max_lr, end_lr=cfg.max_lr
        )
        
        model.train()
        # 模拟训练循环
        for step in range(steps_s1):
            optimizer.zero_grad()
            # Mock Input
            inputs = torch.randint(0, cfg.vocab_size, (cfg.global_batch_size // cfg.seq_len, cfg.seq_len)).to(cfg.device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), inputs.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            
            if step % 10 == 0:
                logger.info(f"Stage 1 | Step {step}/{steps_s1} | Loss: {loss.item():.4f} | LR: {scheduler.get_lr():.2e}")

        logger.info("Stage 1 Completed.")

    # 6. 过渡阶段 (Transition) - 仅针对实验组
    if not cmd_args.baseline:
        logger.info(">>> Initiating Transition (Scaling Up)")
        
        # 6.1 保存旧模型引用用于映射
        old_model = model
        # 保存旧优化器状态，并映射参数名 -> 状态
        old_opt_state = optimizer.state_dict()['state']
        old_param_map = {n: p for n, p in old_model.named_parameters()}
        # 这里的 key 是 parameter tensor 的 id
        # 我们需要一个 map: name -> state
        name_to_state = {}
        # 必须通过 optimizer.param_groups 来找 id
        id_to_idx = {id(p): i for i, p in enumerate(optimizer.param_groups[0]['params'])} # id(p) works if object persistent
        # 注意: torch optim state dict use param_id usually corresponding to insertion order or explicit id
        # 为简化，假设 param_groups 顺序与 named_parameters 顺序一致 (Standard PyTorch behavior)
        
        param_names = list(old_param_map.keys())
        # optimizer.param_groups[0]['params'] 是 tensor 对象列表
        # optimizer.state 是 dict {tensor_id: state} 或 {tensor_obj: state}
        # PyTorch 版本差异。通常 state_dict 使用 id。
        
        # Robust mapping:
        # Index in named_parameters -> State
        idx_to_state = {}
        for idx, p in enumerate(optimizer.param_groups[0]['params']):
            if p in optimizer.state:
                idx_to_state[idx] = optimizer.state[p]
        
        # 6.2 创建新模型
        new_model = create_and_transfer_model(old_model, cfg, target_layers=L2)
        model = new_model # Replace
        
        # 6.3 优化器状态迁移 (Section 9.2 Implementation Logic)
        logger.info("Mapping Optimizer States...")
        
        # 新优化器 (必须重新初始化，因为参数对象变了)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.max_lr, 
            betas=(cfg.beta1, cfg.beta2),
            eps=1e-8,
            weight_decay=cfg.weight_decay
        )
        
        # 构建新状态
        new_opt_state_dict = optimizer.state_dict()
        new_param_groups = new_opt_state_dict['param_groups']
        # 同样假设顺序一致
        
        count_inherited = 0
        count_zeroed = 0
        
        for i, (name, param) in enumerate(model.named_parameters()):
            if not param.requires_grad: continue
            
            # 这里的 param 是新模型的参数
            # 找到对应的 param_id (在新 optimizer 中的 key)
            # 在刚初始化的 optimizer 中，state 还是空的，我们需要填充它
            
            # 1. 判断属于旧层还是新层
            layer_idx = get_layer_index(name)
            
            # 这里的 L1 是旧模型的层数
            is_old_layer = False
            
            # 特殊处理: Embedding 和 Norm 总是继承
            if "tok_embeddings" in name or "norm" in name or "output" in name:
                is_old_layer = True
            elif layer_idx is not None and layer_idx < L1:
                is_old_layer = True
            
            target_state = {}
            
            if is_old_layer:
                # 尝试从旧状态中找
                # 假设名字完全匹配 (因为我们是复制的结构)
                # Find index of this name in old model
                if name in param_names:
                    old_idx = param_names.index(name)
                    if old_idx in idx_to_state:
                        old_s = idx_to_state[old_idx]
                        target_state = copy.deepcopy(old_s)
                        count_inherited += 1
            
            if not target_state: # 新层 or 没找到状态 -> Zero Init
                target_state = {
                    'step': 0, # 重置 step ? PDF 里的伪代码写 step: 0
                    'exp_avg': torch.zeros_like(param),
                    'exp_avg_sq': torch.zeros_like(param)
                }
                count_zeroed += 1
                
            # 将构造好的 state 塞回 optimizer
            # 需要找到当前 param 在 optimizer 中的 key
            # optimizer.state[param] = target_state (如果直接操作对象)
            optimizer.state[param] = target_state

        logger.info(f"Optimizer State Mapped: {count_inherited} Inherited, {count_zeroed} Zero-Initialized.")
        
        # 释放旧模型显存
        del old_model
        torch.cuda.empty_cache()

    # 7. Stage 2 训练
    logger.info(">>> Starting Stage 2 Training")
    
    # Stage 2 Schedule: Re-Warmup (200 steps) -> Stable -> Decay (20%)
    if not cmd_args.baseline:
        # A-3 Schedule
        # "Re-Warmup (200 steps) -> Stable (80%) -> Decay (20%)"
        # Note: Stable ratio is remaining part.
        # Steps breakdown:
        rw_steps = 200
        decay_steps = int(steps_s2 * 0.2)
        # Remaining is stable
        
        # Custom Scheduler for Stage 2
        # Need a simplified lambda or class. Re-using WSD with manual overrides.
        class Stage2Scheduler(WSDScheduler):
            def get_lr(self):
                # Re-Warmup
                if self.current_step < 200:
                    return self.max_lr * (self.current_step / 200)
                # Stable
                if self.current_step < (self.total_steps - int(self.total_steps * 0.2)):
                    return self.max_lr
                # Decay
                decay_len = int(self.total_steps * 0.2)
                progress = (self.current_step - (self.total_steps - decay_len)) / decay_len
                return self.max_lr - (self.max_lr - self.end_lr) * min(1.0, progress)
                
        scheduler = Stage2Scheduler(optimizer, steps_s2, 0, 0, 0, cfg.max_lr, cfg.min_lr)

    else:
        # Baseline B-1 Schedule: Warmup(10%) -> Stable(70%) -> Decay(20%)
        # Here steps_s2 is total steps
        scheduler = WSDScheduler(
            optimizer, steps_s2,
            warmup_ratio=0.1, decay_ratio=0.2,
            start_lr=0, max_lr=cfg.max_lr, end_lr=cfg.min_lr
        )

    model.train()
    for step in range(steps_s2):
        optimizer.zero_grad()
        # Mock Input
        inputs = torch.randint(0, cfg.vocab_size, (cfg.global_batch_size // cfg.seq_len, cfg.seq_len)).to(cfg.device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), inputs.view(-1))
        loss.backward()
        
        # Transition 检查: 第1个step grad_norm
        if step == 0 and not cmd_args.baseline:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0) # Just calculate
            logger.info(f"Transition Check | Step 0 Grad Norm: {total_norm.item():.4f} (Expected < 2.0)")
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip) # Re-clip
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
        optimizer.step()
        scheduler.step()
        
        if step % 10 == 0:
            logger.info(f"Stage 2 | Step {step}/{steps_s2} | Loss: {loss.item():.4f} | LR: {scheduler.get_lr():.2e}")

    logger.info(">>> Experiment Completed Successfully.")

if __name__ == "__main__":
    main()