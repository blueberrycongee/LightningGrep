"""
使用 LLM 合成训练数据
支持：OpenAI / 硅基流动 (SiliconFlow)
"""
import os
import json
import argparse
from typing import Optional
from openai import OpenAI
from tqdm import tqdm

from synthesis_prompt import create_synthesis_prompt, SYSTEM_PROMPT

# 配置
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")

# API 配置
API_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "Qwen/Qwen2.5-72B-Instruct",
        "env_key": "SILICONFLOW_API_KEY",
    },
}


def load_hotpotqa(split: str = "train", limit: Optional[int] = None) -> list:
    """加载 HotpotQA 数据"""
    filepath = os.path.join(RAW_DIR, f"hotpot_{split}.json")
    if split == "dev":
        filepath = os.path.join(RAW_DIR, "hotpot_dev.json")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
    
    return data


def filter_samples(data: list) -> list:
    """过滤样本：保留多跳问题，排除过简单或过复杂的"""
    filtered = []
    for sample in data:
        # 跳过过于简单的（只有一个支持事实）
        if len(sample["supporting_facts"]) < 2:
            continue
        
        # 跳过过于复杂的（超过 5 个支持事实）
        if len(sample["supporting_facts"]) > 5:
            continue
        
        # 跳过 context 过长的
        total_sentences = sum(len(sentences) for _, sentences in sample["context"])
        if total_sentences > 50:
            continue
        
        filtered.append(sample)
    
    return filtered


def extract_json(text: str) -> Optional[dict]:
    """从文本中提取 JSON（支持 markdown 代码块）"""
    import re
    
    # 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass
    
    # 尝试从 ```json ... ``` 中提取
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # 尝试从 ``` ... ``` 中提取
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # 尝试找到第一个 { 和最后一个 }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    
    return None


def synthesize_one(client: OpenAI, sample: dict, model: str) -> Optional[dict]:
    """合成一条数据"""
    system_prompt, user_prompt = create_synthesis_prompt(sample)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        
        result_text = response.choices[0].message.content
        result = extract_json(result_text)
        
        if result is None:
            print(f"[警告] JSON 解析失败")
            return None
        
        # 添加原始数据的元信息
        result["_id"] = sample.get("_id", "")
        result["_type"] = sample.get("type", "")
        result["_level"] = sample.get("level", "")
        
        return result
    
    except Exception as e:
        print(f"[错误] {e}")
        return None


def validate_result(result: dict) -> tuple:
    """验证合成结果，返回 (is_valid, issues)"""
    issues = []
    
    # 新格式必需字段
    required_fields = ["question", "is_parallel", "rounds", "answer"]
    
    for field in required_fields:
        if field not in result:
            issues.append(f"缺少字段: {field}")
    
    if issues:
        return False, issues
    
    # 检查 rounds 格式
    rounds = result.get("rounds", [])
    if not isinstance(rounds, list) or len(rounds) == 0:
        issues.append("rounds 必须是非空数组")
    else:
        for i, r in enumerate(rounds):
            if "think" not in r:
                issues.append(f"round {i} 缺少 think")
            if "searches" not in r:
                issues.append(f"round {i} 缺少 searches")
    
    return len(issues) == 0, issues


def load_existing_results(output_path: str) -> tuple:
    """加载已有结果，用于断点续传"""
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_ids = {r.get("_id") for r in existing if "_id" in r}
            return existing, existing_ids
        except:
            pass
    return [], set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="合成数量")
    parser.add_argument("--provider", type=str, default="siliconflow", 
                        choices=["openai", "siliconflow"], help="API 提供商")
    parser.add_argument("--model", type=str, default=None, help="使用的模型（默认根据 provider 选择）")
    parser.add_argument("--output", type=str, default="synthetic_v1.json", help="输出文件名")
    parser.add_argument("--dry-run", action="store_true", help="只打印 prompt，不调用 API")
    parser.add_argument("--resume", action="store_true", help="断点续传：跳过已完成的样本")
    args = parser.parse_args()
    
    # 获取 API 配置
    config = API_CONFIGS[args.provider]
    model = args.model or config["default_model"]
    api_key = os.environ.get(config["env_key"])
    
    if not api_key and not args.dry_run:
        print(f"[错误] 请设置 {config['env_key']} 环境变量")
        print(f"  Windows: $env:{config['env_key']} = 'your-api-key'")
        print(f"  Linux:   export {config['env_key']}='your-api-key'")
        return
    
    print(f"[配置] Provider: {args.provider}, Model: {model}")
    
    # 加载数据
    print(f"[加载] HotpotQA 训练集...")
    data = load_hotpotqa("train")
    print(f"  原始样本数: {len(data)}")
    
    # 过滤
    data = filter_samples(data)
    print(f"  过滤后样本数: {len(data)}")
    
    # 限制数量
    data = data[:args.limit]
    print(f"  本次合成: {len(data)} 条")
    
    if args.dry_run:
        # 只打印一个示例 prompt
        print("\n[示例 Prompt]")
        system_prompt, user_prompt = create_synthesis_prompt(data[0])
        print("=" * 50)
        print("[System]")
        print(system_prompt[:500] + "...")
        print("=" * 50)
        print("[User]")
        print(user_prompt[:1000] + "...")
        return
    
    # 初始化客户端
    client = OpenAI(api_key=api_key, base_url=config["base_url"])
    
    # 合成
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    output_path = os.path.join(SYNTHETIC_DIR, args.output)
    
    # 断点续传：加载已有结果
    results = []
    existing_ids = set()
    skipped_count = 0
    
    if args.resume:
        results, existing_ids = load_existing_results(output_path)
        if results:
            print(f"[续传] 已加载 {len(results)} 条已完成的结果")
    
    valid_count = len(results)
    issue_counts = {}
    
    for sample in tqdm(data, desc="合成中"):
        # 跳过已完成的样本
        sample_id = sample.get("_id", "")
        if sample_id in existing_ids:
            skipped_count += 1
            continue
        
        result = synthesize_one(client, sample, model)
        if result:
            is_valid, issues = validate_result(result)
            if is_valid:
                results.append(result)
                valid_count += 1
            else:
                for issue in issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # 每 10 条保存一次（容错）
        if len(results) % 10 == 0 and len(results) > 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 最终保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[完成]")
    print(f"  有效样本: {valid_count}/{len(data)}")
    if skipped_count > 0:
        print(f"  跳过已完成: {skipped_count}")
    print(f"  保存到: {output_path}")
    
    if issue_counts:
        print(f"\n[质量问题统计]")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {issue}: {count} 次")


if __name__ == "__main__":
    main()
