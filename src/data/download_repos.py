"""
下载 SWE-bench 仓库
克隆到指定 commit 用于 RL 训练
"""
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def clone_repo(repo: str, commit: str, output_dir: Path) -> bool:
    """克隆仓库到指定 commit"""
    repo_name = repo.replace("/", "_")
    repo_path = output_dir / repo_name
    
    if repo_path.exists():
        return True
    
    try:
        # 克隆
        url = f"https://github.com/{repo}.git"
        subprocess.run(
            ["git", "clone", "--depth", "100", url, str(repo_path)],
            capture_output=True,
            timeout=300,
            check=True
        )
        
        # Checkout 到指定 commit
        subprocess.run(
            ["git", "checkout", commit],
            cwd=str(repo_path),
            capture_output=True,
            timeout=60,
        )
        
        return True
    except Exception as e:
        print(f"  ✗ {repo}: {e}")
        return False


def download_repos(
    data_path: str,
    output_dir: str = "data/swebench/repos",
    max_workers: int = 4,
):
    """下载所有需要的仓库"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 收集需要的 repo 和 commit
    repo_commits = {}
    for item in data:
        repo = item["repo"]
        commit = item["base_commit"]
        if repo not in repo_commits:
            repo_commits[repo] = commit
    
    print(f"需要下载 {len(repo_commits)} 个仓库")
    print(f"保存到: {output_dir}")
    print()
    
    # 多线程下载
    success = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(clone_repo, repo, commit, output_dir): repo
            for repo, commit in repo_commits.items()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            repo = futures[future]
            try:
                if future.result():
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ {repo}: {e}")
                failed += 1
    
    print(f"\n✓ 完成: {success} 成功, {failed} 失败")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/swebench/verified_all.json")
    parser.add_argument("--output", type=str, default="data/swebench/repos")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    download_repos(args.data, args.output, args.workers)
