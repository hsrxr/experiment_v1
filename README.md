# Scaling Law 两阶段训练调度验证 (v3.4) 代码库

本代码库实现了《实验规格说明书 v3.4》中描述的所有逻辑，用于验证两阶段训练在固定 FLOPs 预算下的优势。

## 环境依赖

* Python 3.8+
* PyTorch 2.0+ (支持 `torch.bfloat16` 和 `RMSNorm`)
* CUDA 环境 (推荐，否则需在代码中修改 device 为 cpu)

安装:

```
pip install torch numpy
```

## 代码结构说明

### `two_stage_training.py`

这是单一文件的完整实现，包含了：

1. **Llama 架构**: 含 RoPE, SwiGLU, RMSNorm。
2. **Budget Calculator**: 严格按照文档公式 $D_1 = \frac{C_{total} \times x}{6 \times N1_{non\\_emb}}$ 计算。
3. **Top Stacking**: 实现了层复制和 `o_proj`, `down_proj` 的 Zero Init。
4. **Optimizer State Mapping**: 实现了从 $N_1$ 到 $N_2$ 的 AdamW 状态精确迁移。

### 运行实验

请直接使用提供的 shell 脚本：

```
chmod +x run_experiments.sh
./run_experiments.sh
```

或者手动运行：

**1. 运行 Baseline (B-1):**

```
python two_stage_training.py --run_id "B-1" --baseline
```

**2. 运行 实验组 (A-3):**

```
python two_stage_training.py --run_id "A-3" --k 2.4 --x 0.2
```

## 注意事项

1. **显存占用**: 默认 Global Batch Size 为 524k tokens。在单卡调试时，脚本内部逻辑会将这视为积累的 tokens 总量。实际 `inputs` tensor 的大小是 `global_batch_size // seq_len`。如果显存不足，请在 Python 脚本的 `ExperimentConfig` 中调小 `global_batch_size`。
2. **数据**: 目前使用 `SyntheticDataset` 生成随机 token 进行冒烟测试。正式训练请替换 `get_dataloader` 中的 Dataset。
