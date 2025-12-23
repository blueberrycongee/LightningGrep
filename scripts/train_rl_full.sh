#!/bin/bash
# ============================================
# LightningGrep RL 训练 - 第二步：大规模正式训练
# ============================================
# 前提：train_rl_test.sh 跑通了
# 预期时间：5-10 小时
# ============================================

echo "============================================"
echo "LightningGrep RL 训练 - 大规模正式训练"
echo "============================================"

# 检查测试是否完成
if [ ! -d "outputs/rl_test" ]; then
    echo "⚠️ 请先运行 train_rl_test.sh 验证 pipeline"
    exit 1
fi

python scripts/run_rl.py \
    --sft_model outputs/sft_v2 \
    --output_dir outputs/rl_v1 \
    --split lite \
    --max_samples 100 \
    --num_steps 500 \
    --batch_size 4 \
    --num_rollouts 4 \
    --max_turns 4 \
    --temperature 0.7 \
    --learning_rate 5e-6 \
    --quantization 4bit \
    --dtype bf16 \
    --lora_r 16 \
    --lora_alpha 32 \
    --save_every 50

echo ""
echo "============================================"
echo "训练完成！"
echo "模型保存在: outputs/rl_v1/"
echo "日志文件: outputs/rl_v1/training_log.jsonl"
echo "============================================"
