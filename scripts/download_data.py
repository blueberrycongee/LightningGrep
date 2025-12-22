"""
下载 HotpotQA 数据集
"""
import os
import urllib.request
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

URLS = {
    "hotpot_train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "hotpot_dev": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
}

def download_file(url: str, filepath: str):
    """下载文件"""
    if os.path.exists(filepath):
        print(f"[跳过] {filepath} 已存在")
        return
    
    print(f"[下载] {url}")
    print(f"[保存] {filepath}")
    urllib.request.urlretrieve(url, filepath)
    print(f"[完成] 大小: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for name, url in URLS.items():
        filepath = os.path.join(DATA_DIR, f"{name}.json")
        download_file(url, filepath)
    
    # 验证数据
    print("\n[验证数据]")
    for name in URLS.keys():
        filepath = os.path.join(DATA_DIR, f"{name}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  {name}: {len(data)} 条样本")
            
            # 打印一个样本的结构
            if data:
                sample = data[0]
                print(f"    字段: {list(sample.keys())}")

if __name__ == "__main__":
    main()
