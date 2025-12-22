"""
代码检索评测脚本
评估模型的文件检索能力

使用方法:
    python src/evaluation/eval_retrieval.py \
        --model outputs/rl_v1/best \
        --test_data data/swebench/rl_test.json \
        --repo_dir data/swebench/repos \
        --output results/eval_results.json
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.environment.code_search_env import CodeSearchEnv, TOOLS_DEFINITION


@dataclass
class EvalResult:
    """单个样本的评测结果"""
    instance_id: str
    repo: str
    query: str
    ground_truth_files: List[str]
    predicted_files: List[str]
    precision: float
    recall: float
    f1: float
    f_beta: float  # β=0.5
    tool_calls: int
    turns: int
    latency: float  # 秒


@dataclass
class EvalMetrics:
    """总体评测指标"""
    num_samples: int
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_f_beta: float
    avg_tool_calls: float
    avg_turns: float
    avg_latency: float
    recall_at_1: float  # 至少找到 1 个正确文件的比例
    recall_at_all: float  # 找到所有正确文件的比例


def compute_metrics(
    predicted: List[str],
    ground_truth: List[str],
    beta: float = 0.5
) -> Tuple[float, float, float, float]:
    """计算 Precision, Recall, F1, F-beta"""
    if not ground_truth:
        return 0.0, 0.0, 0.0, 0.0
    
    gt_set = set(ground_truth)
    pred_set = set(predicted)
    
    if not pred_set:
        return 0.0, 0.0, 0.0, 0.0
    
    correct = len(gt_set & pred_set)
    
    precision = correct / len(pred_set)
    recall = correct / len(gt_set)
    
    if precision + recall == 0:
        f1 = 0.0
        f_beta = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        beta_sq = beta ** 2
        f_beta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    
    return precision, recall, f1, f_beta


def parse_tool_calls(text: str) -> List[Dict]:
    """从生成文本中解析 tool calls"""
    import re
    tool_calls = []
    
    # 尝试解析 <tool_calls> 格式
    pattern = r'<tool_calls>\s*(.*?)\s*</tool_calls>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            calls = json.loads(match)
            if isinstance(calls, list):
                tool_calls.extend(calls)
            elif isinstance(calls, dict):
                tool_calls.append(calls)
        except:
            pass
    
    return tool_calls


def run_search(
    model,
    tokenizer,
    env: CodeSearchEnv,
    query: str,
    max_turns: int = 4,
    temperature: float = 0.1,
    device: str = "cuda",
) -> Tuple[List[str], int, int, float]:
    """运行代码搜索"""
    env.reset()
    start_time = time.time()
    
    messages = [
        {"role": "system", "content": f"You are a code search agent. Use the tools to find relevant code files. Tools: {json.dumps(TOOLS_DEFINITION)}"},
        {"role": "user", "content": query}
    ]
    
    for turn in range(max_turns):
        # 构建输入
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 解析 tool calls
        tool_calls = parse_tool_calls(generated_text)
        
        if not tool_calls:
            messages.append({"role": "assistant", "content": generated_text})
            break
        
        # 执行
        results, done = env.step(tool_calls)
        
        messages.append({
            "role": "assistant",
            "content": generated_text,
            "tool_calls": tool_calls
        })
        
        for tc, result in zip(tool_calls, results):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result["content"]
            })
        
        if done:
            break
    
    latency = time.time() - start_time
    search_result = env.get_result()
    
    return search_result.files_found, search_result.tool_calls, search_result.turns, latency


def evaluate(
    model,
    tokenizer,
    test_data: List[Dict],
    repo_dir: str,
    max_turns: int = 4,
    temperature: float = 0.1,
    device: str = "cuda",
) -> Tuple[List[EvalResult], EvalMetrics]:
    """评测模型"""
    results = []
    
    for item in tqdm(test_data, desc="Evaluating"):
        repo_path = Path(repo_dir) / item["repo"].replace("/", "_")
        
        if not repo_path.exists():
            print(f"  跳过: {item['repo']} (仓库不存在)")
            continue
        
        env = CodeSearchEnv(
            repo_path=str(repo_path),
            max_turns=max_turns,
        )
        
        try:
            predicted_files, tool_calls, turns, latency = run_search(
                model, tokenizer, env,
                item["query"],
                max_turns=max_turns,
                temperature=temperature,
                device=device,
            )
        except Exception as e:
            print(f"  错误: {item['instance_id']}: {e}")
            continue
        
        precision, recall, f1, f_beta = compute_metrics(
            predicted_files,
            item["ground_truth_files"],
            beta=0.5
        )
        
        results.append(EvalResult(
            instance_id=item["instance_id"],
            repo=item["repo"],
            query=item["query"][:200],  # 截断
            ground_truth_files=item["ground_truth_files"],
            predicted_files=predicted_files,
            precision=precision,
            recall=recall,
            f1=f1,
            f_beta=f_beta,
            tool_calls=tool_calls,
            turns=turns,
            latency=latency,
        ))
    
    # 计算总体指标
    if not results:
        return results, EvalMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    n = len(results)
    metrics = EvalMetrics(
        num_samples=n,
        avg_precision=sum(r.precision for r in results) / n,
        avg_recall=sum(r.recall for r in results) / n,
        avg_f1=sum(r.f1 for r in results) / n,
        avg_f_beta=sum(r.f_beta for r in results) / n,
        avg_tool_calls=sum(r.tool_calls for r in results) / n,
        avg_turns=sum(r.turns for r in results) / n,
        avg_latency=sum(r.latency for r in results) / n,
        recall_at_1=sum(1 for r in results if r.recall > 0) / n,
        recall_at_all=sum(1 for r in results if r.recall == 1.0) / n,
    )
    
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="代码检索评测")
    
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B", help="基座模型")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据")
    parser.add_argument("--repo_dir", type=str, required=True, help="仓库目录")
    parser.add_argument("--output", type=str, default="results/eval_results.json", help="输出文件")
    parser.add_argument("--max_turns", type=int, default=4, help="最大轮数")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载数据
    print("\n[1/3] 加载测试数据...")
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"  测试样本: {len(test_data)} 条")
    
    # 加载模型
    print("\n[2/3] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 检查是否是 LoRA 模型
    adapter_config = Path(args.model) / "adapter_config.json"
    if adapter_config.exists():
        # 加载 LoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.model)
    else:
        # 直接加载
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    model.eval()
    
    # 评测
    print("\n[3/3] 开始评测...")
    results, metrics = evaluate(
        model, tokenizer, test_data, args.repo_dir,
        max_turns=args.max_turns,
        temperature=args.temperature,
        device=device,
    )
    
    # 打印结果
    print("\n" + "=" * 50)
    print("评测结果")
    print("=" * 50)
    print(f"样本数: {metrics.num_samples}")
    print(f"Precision: {metrics.avg_precision:.4f}")
    print(f"Recall: {metrics.avg_recall:.4f}")
    print(f"F1: {metrics.avg_f1:.4f}")
    print(f"F-β (β=0.5): {metrics.avg_f_beta:.4f}")
    print(f"Tool calls (avg): {metrics.avg_tool_calls:.1f}")
    print(f"Turns (avg): {metrics.avg_turns:.1f}")
    print(f"Latency (avg): {metrics.avg_latency:.2f}s")
    print(f"Recall@1: {metrics.recall_at_1:.4f}")
    print(f"Recall@All: {metrics.recall_at_all:.4f}")
    print("=" * 50)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metrics": asdict(metrics),
        "results": [asdict(r) for r in results],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 结果保存到: {output_path}")


if __name__ == "__main__":
    main()
