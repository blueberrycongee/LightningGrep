# LightningGrep SFT 训练指南

## 环境要求

- **GPU**: NVIDIA A100 (40GB/80GB)
- **CUDA**: 11.8+
- **Python**: 3.10+

## 快速开始

### 方法一：一键脚本（推荐）

```bash
# 1. Clone 仓库
git clone https://github.com/blueberrycongee/LightningGrep.git
cd LightningGrep

# 2. 运行训练
chmod +x scripts/train.sh
./scripts/train.sh
```

### 方法二：手动运行

```bash
# 1. 安装依赖
pip install torch transformers peft accelerate datasets bitsandbytes tqdm

# 2. 启动训练
python src/training/sft_qlora.py \
    --train_data data/code_search/sft_all_train.json \
    --val_data data/code_search/sft_all_val.json \
    --output_dir outputs/sft_v2
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | Qwen/Qwen3-1.7B | 基座模型 |
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 4 | 每 GPU batch size |
| `--gradient_accumulation` | 4 | 梯度累积（有效 batch=16） |
| `--max_length` | 4096 | 最大序列长度 |
| `--lora_r` | 64 | LoRA rank |
| `--lora_alpha` | 128 | LoRA alpha |
| `--learning_rate` | 2e-4 | 学习率 |
| `--early_stopping_patience` | 3 | Early stopping patience |

## 显存不足？

如果 40GB A100 OOM，降低参数：

```bash
python src/training/sft_qlora.py \
    --train_data data/code_search/sft_all_train.json \
    --val_data data/code_search/sft_all_val.json \
    --output_dir outputs/sft_v2 \
    --batch_size 2 \
    --max_length 2048
```

## 断点续训

训练中断后，添加 `--resume` 继续：

```bash
python src/training/sft_qlora.py \
    --train_data data/code_search/sft_all_train.json \
    --val_data data/code_search/sft_all_val.json \
    --output_dir outputs/sft_v2 \
    --resume
```

## 训练输出

训练完成后，模型保存在 `outputs/sft_v2/`：

```
outputs/sft_v2/
├── adapter_model.safetensors  # LoRA 权重
├── adapter_config.json        # LoRA 配置
├── tokenizer.json             # Tokenizer
└── ...
```

## 预计时间

- **数据量**: 2176 训练 / 544 验证
- **A100 40GB**: ~30-60 分钟
- **A100 80GB**: ~20-40 分钟
