"""划分训练/验证集"""
import json
import random
import argparse
from pathlib import Path


def split_data(input_path: str, train_ratio: float = 0.9, seed: int = 42):
    """将数据划分为训练集和验证集"""
    random.seed(seed)
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # 输出路径
    input_path = Path(input_path)
    train_path = input_path.parent / f"{input_path.stem}_train.json"
    val_path = input_path.parent / f"{input_path.stem}_val.json"
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"原始数据: {len(data)} 条")
    print(f"训练集: {len(train_data)} 条 -> {train_path}")
    print(f"验证集: {len(val_data)} 条 -> {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="输入数据路径")
    parser.add_argument("--ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    split_data(args.input, args.ratio, args.seed)
