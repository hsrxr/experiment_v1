#!/bin/bash

# 确保脚本有执行权限: chmod +x run_experiments.sh

echo "=== Scaling Law Two-Stage Verification (v3.4) ==="
echo "Note: Default Global Batch Size is 524,288. For local debugging, modify 'global_batch_size' in python script."

# 1. 冒烟测试 (Smoke Test)
# 使用极小的 Budget 和 steps 快速验证流程
echo "[Phase 1] Running Smoke Test (A-3 Small)..."
python two_stage_training.py --run_id "A-3-Smoke" --k 2.4 --x 0.2 
if [ $? -ne 0 ]; then
    echo "Smoke Test Failed!"
    exit 1
fi
echo "Smoke Test Passed."

# 2. 核心实验 (Phase 2)
# Baseline B-1
echo "[Phase 2] Running Baseline B-1 (k=1.0, x=1.0)..."
# 注意: Baseline 逻辑中 x 实际被忽略，全程用 N2，但为了参数一致性传参
python two_stage_training.py --run_id "B-1" --baseline --k 1.0 --x 1.0

# Experiment A-3
echo "[Phase 2] Running Experiment A-3 (k=2.4, x=0.2)..."
python two_stage_training.py --run_id "A-3" --k 2.4 --x 0.2

echo "All Runs Initiated."