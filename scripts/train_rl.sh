#!/bin/bash
# ============================================
# LightningGrep RL 训练脚本
# REINFORCE + SWE-bench
# ============================================

set -e

echo "============================================"
echo "LightningGrep RL Training"
echo "============================================"

# 配置
SFT_MODEL="${SFT_MODEL:-outputs/sft_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl_v1}"
NUM_STEPS="${NUM_STEPS:-500}"

# 检查 SFT 模型
echo "[1/5] 检查 SFT 模型..."
if [ ! -d "$SFT_MODEL" ]; then
    echo "❌ SFT 模型不存在: $SFT_MODEL"
    echo "请先完成 SFT 训练，或设置 SFT_MODEL 环境变量"
    exit 1
fi
echo "✓ SFT 模型: $SFT_MODEL"
echo ""

# 下载 SWE-bench 数据
echo "[2/5] 准备 SWE-bench 数据..."
if [ ! -f "data/swebench/rl_train.json" ]; then
    echo "下载并处理 SWE-bench 数据..."
    pip install -q datasets
    python src/data/download_swebench.py --output_dir data/swebench
else
    echo "✓ 数据已存在"
fi

TRAIN_COUNT=$(python -c "import json; print(len(json.load(open('data/swebench/rl_train.json'))))")
VAL_COUNT=$(python -c "import json; print(len(json.load(open('data/swebench/rl_val.json'))))")
echo "✓ 训练集: $TRAIN_COUNT 条"
echo "✓ 验证集: $VAL_COUNT 条"
echo ""

# 下载仓库
echo "[3/5] 下载 SWE-bench 仓库..."
if [ ! -d "data/swebench/repos" ] || [ -z "$(ls -A data/swebench/repos 2>/dev/null)" ]; then
    echo "开始下载仓库（这可能需要一些时间）..."
    python src/data/download_repos.py \
        --data data/swebench/verified_all.json \
        --output data/swebench/repos \
        --workers 4
else
    REPO_COUNT=$(ls -d data/swebench/repos/*/ 2>/dev/null | wc -l)
    echo "✓ 已有 $REPO_COUNT 个仓库"
fi
echo ""

# 检查 GPU
echo "[4/5] 检查 GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 开始训练
echo "[5/5] 开始 RL 训练..."
echo "============================================"
echo ""

python src/training/rl_reinforce.py \
    --sft_model "$SFT_MODEL" \
    --base_model Qwen/Qwen3-1.7B \
    --train_data data/swebench/rl_train.json \
    --val_data data/swebench/rl_val.json \
    --repo_dir data/swebench/repos \
    --output_dir "$OUTPUT_DIR" \
    --num_steps "$NUM_STEPS" \
    --batch_size 2 \
    --num_rollouts 4 \
    --learning_rate 1e-5 \
    --max_turns 4 \
    --temperature 0.7

echo ""
echo "============================================"
echo "✓ RL 训练完成！"
echo "  模型保存在: $OUTPUT_DIR"
echo "============================================"

# 评测
echo ""
echo "开始评测..."
python src/evaluation/eval_retrieval.py \
    --model "$OUTPUT_DIR/best" \
    --test_data data/swebench/rl_test.json \
    --repo_dir data/swebench/repos \
    --output results/rl_eval.json \
    --max_turns 4

echo ""
echo "✓ 评测完成！结果保存在: results/rl_eval.json"
