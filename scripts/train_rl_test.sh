#!/bin/bash
# ============================================
# LightningGrep RL 训练 - 第一步：小规模测试
# ============================================
# 目的：验证 pipeline 能跑通，观察 reward 是否上升
# 预期时间：30-60 分钟
# 成功标志：没报错 + reward 有上升趋势
# ============================================

echo "============================================"
echo "LightningGrep RL 训练 - 小规模测试"
echo "============================================"

python scripts/run_rl.py \
    --sft_model outputs/sft_v2 \
    --output_dir outputs/rl_test \
    --split lite \
    --max_samples 30 \
    --num_steps 50 \
    --batch_size 2 \
    --num_rollouts 4 \
    --max_turns 4 \
    --max_parallel_calls 8 \
    --temperature 0.7 \
    --learning_rate 5e-6 \
    --quantization none \
    --dtype bf16 \
    --lora_r 16 \
    --lora_alpha 32 \
    --grad_clip 1.0 \
    --max_traj_length 4096 \
    --scale_by_tool_calls \
    --debug \
    --save_every 20

echo ""
echo "============================================"
echo "测试完成！检查结果："
echo "1. 查看日志: cat outputs/rl_test/training_log_*.jsonl"
echo "2. 如果 reward 有上升且 loss 不为 0，运行: bash scripts/train_rl_full.sh"
echo "============================================"
