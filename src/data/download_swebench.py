"""
下载并处理 SWE-bench 数据集
用于 RL 训练和评测
"""
import os
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm


def download_swebench(output_dir: str = "data/swebench"):
    """下载 SWE-bench 数据集"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("下载 SWE-bench 数据集")
    print("=" * 50)
    
    # 下载 SWE-bench Verified (高质量子集)
    print("\n[1/3] 下载 SWE-bench Verified...")
    verified = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    print(f"  Verified: {len(verified)} 条")
    
    # 下载 SWE-bench Lite
    print("\n[2/3] 下载 SWE-bench Lite...")
    lite = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"  Lite: {len(lite)} 条")
    
    # 处理数据
    print("\n[3/3] 处理数据...")
    
    verified_data = process_swebench(verified)
    lite_data = process_swebench(lite)
    
    # 保存原始数据
    with open(output_dir / "verified_all.json", "w", encoding="utf-8") as f:
        json.dump(verified_data, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "lite_all.json", "w", encoding="utf-8") as f:
        json.dump(lite_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 数据保存到: {output_dir}")
    return verified_data, lite_data


def process_swebench(dataset) -> List[Dict]:
    """处理 SWE-bench 数据"""
    data = []
    for item in dataset:
        # 从 patch 中提取文件列表
        patch_files = extract_files_from_patch(item.get("patch", ""))
        
        data.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "query": item["problem_statement"],  # Issue 描述作为 query
            "ground_truth_files": patch_files,   # Patch 涉及的文件
            "patch": item.get("patch", ""),
        })
    return data


def extract_files_from_patch(patch: str) -> List[str]:
    """从 git patch 中提取文件路径"""
    files = []
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            # diff --git a/path/to/file b/path/to/file
            parts = line.split()
            if len(parts) >= 4:
                file_path = parts[2].lstrip("a/")
                if file_path not in files:
                    files.append(file_path)
        elif line.startswith("---") and not line.startswith("--- /dev/null"):
            # --- a/path/to/file
            parts = line.split()
            if len(parts) >= 2:
                file_path = parts[1].lstrip("a/")
                if file_path not in files and file_path != "/dev/null":
                    files.append(file_path)
    return files


def split_by_repo(
    data: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """按 repo 划分数据，确保同一 repo 的数据在同一集合"""
    random.seed(seed)
    
    # 按 repo 分组
    repo_data = defaultdict(list)
    for item in data:
        repo_data[item["repo"]].append(item)
    
    # 随机划分 repos
    repos = list(repo_data.keys())
    random.shuffle(repos)
    
    split_idx = int(len(repos) * train_ratio)
    train_repos = set(repos[:split_idx])
    val_repos = set(repos[split_idx:])
    
    train_data = []
    val_data = []
    
    for repo, items in repo_data.items():
        if repo in train_repos:
            train_data.extend(items)
        else:
            val_data.extend(items)
    
    return train_data, val_data


def prepare_rl_data(output_dir: str = "data/swebench", seed: int = 42):
    """准备 RL 训练数据"""
    output_dir = Path(output_dir)
    
    # 加载数据
    with open(output_dir / "verified_all.json", "r", encoding="utf-8") as f:
        verified_data = json.load(f)
    
    with open(output_dir / "lite_all.json", "r", encoding="utf-8") as f:
        lite_data = json.load(f)
    
    print("=" * 50)
    print("划分数据集")
    print("=" * 50)
    
    # 划分 Verified 为 train/val
    train_data, val_data = split_by_repo(verified_data, train_ratio=0.8, seed=seed)
    
    # 找出 Lite 中不在 Verified 的作为测试集
    verified_ids = {item["instance_id"] for item in verified_data}
    test_data = [item for item in lite_data if item["instance_id"] not in verified_ids]
    
    # 如果 test 太少，用 Lite 全部
    if len(test_data) < 50:
        test_data = lite_data
    
    print(f"训练集: {len(train_data)} (from Verified)")
    print(f"验证集: {len(val_data)} (from Verified)")
    print(f"测试集: {len(test_data)} (from Lite)")
    
    # 统计 repos
    train_repos = set(item["repo"] for item in train_data)
    val_repos = set(item["repo"] for item in val_data)
    test_repos = set(item["repo"] for item in test_data)
    
    print(f"\n训练 repos: {len(train_repos)}")
    print(f"验证 repos: {len(val_repos)}")
    print(f"测试 repos: {len(test_repos)}")
    
    # 保存
    with open(output_dir / "rl_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "rl_val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "rl_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 数据保存到: {output_dir}")
    print(f"  - rl_train.json")
    print(f"  - rl_val.json")
    print(f"  - rl_test.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/swebench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_download", action="store_true", help="跳过下载，直接划分")
    args = parser.parse_args()
    
    if not args.skip_download:
        download_swebench(args.output_dir)
    
    prepare_rl_data(args.output_dir, args.seed)
