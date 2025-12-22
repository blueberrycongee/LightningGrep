"""
将合成的结构化数据转换为 SFT 训练格式
"""
import json
import os
import argparse
from typing import List, Dict


def convert_sample_to_sft(sample: dict) -> dict:
    """
    将一条合成数据转换为 SFT 训练格式（检索工具版本）
    
    输出格式：
    {
        "input": "Query: ...",
        "output": "<think>...</think><search>...</search><information>...</information><result>...</result>",
        "mask_tags": ["information"]  # 训练时不计算 loss 的标签
    }
    
    注意：这是检索工具，不是 QA。输出的是相关文档位置，不是答案。
    """
    question = sample.get("question", "")
    rounds = sample.get("rounds", [])
    final_think = sample.get("final_think", "找到相关文档")
    sources = sample.get("sources", [])
    
    # 构建完整的输出轨迹
    output_parts = []
    source_idx = 0
    
    for round_info in rounds:
        think = round_info.get("think", "分析查询")
        searches = round_info.get("searches", [])
        is_parallel = round_info.get("parallel", False)
        
        # 思考
        output_parts.append(f"<think>{think}</think>")
        
        # 搜索（并行用 ## 分隔）
        if is_parallel and len(searches) > 1:
            search_content = " ## ".join(searches)
        else:
            search_content = searches[0] if searches else ""
        output_parts.append(f"<search>{search_content}</search>")
        
        # 信息返回（从 sources 获取，这部分会被 mask）
        for search_query in searches:
            if source_idx < len(sources):
                source = sources[source_idx]
                doc_name = source.get("doc", "Document")
                lines = source.get("lines", [0])
                # 模拟搜索结果
                info_content = f"[{doc_name}] (lines {lines}): 相关信息..."
            else:
                info_content = f"搜索结果..."
            output_parts.append(f"<information>{info_content}</information>")
            source_idx += 1
    
    # 最终思考
    output_parts.append(f"<think>{final_think}</think>")
    
    # 输出检索结果（文档 + 行号），不是答案
    result_lines = []
    for source in sources:
        doc_name = source.get("doc", "Document")
        lines = source.get("lines", [0])
        result_lines.append(f"  - {doc_name}: lines {lines}")
    
    result_content = "\n".join(result_lines) if result_lines else "  无相关结果"
    output_parts.append(f"<result>\n{result_content}\n</result>")
    
    return {
        "input": f"Query: {question}",
        "output": "\n".join(output_parts),
        "mask_tags": ["information"],  # 训练时 mask 这个标签内的内容
        "_id": sample.get("_id", ""),
        "_is_parallel": sample.get("is_parallel", False)
    }


def convert_file(input_path: str, output_path: str = None, train_ratio: float = None) -> dict:
    """
    转换整个文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（如果指定 train_ratio 则忽略）
        train_ratio: 训练集比例（0-1），如果指定则输出 train/val 两个文件
    """
    import random
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    converted = []
    stats = {"total": len(data), "success": 0, "failed": 0}
    
    for sample in data:
        try:
            result = convert_sample_to_sft(sample)
            converted.append(result)
            stats["success"] += 1
        except Exception as e:
            print(f"[错误] 转换失败: {e}")
            stats["failed"] += 1
    
    # 划分训练集/验证集
    if train_ratio is not None and 0 < train_ratio < 1:
        random.seed(42)  # 固定种子，保证可复现
        random.shuffle(converted)
        
        split_idx = int(len(converted) * train_ratio)
        train_data = converted[:split_idx]
        val_data = converted[split_idx:]
        
        # 输出文件名
        base_path = input_path.replace(".json", "")
        train_path = f"{base_path}_train.json"
        val_path = f"{base_path}_val.json"
        
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        stats["train_count"] = len(train_data)
        stats["val_count"] = len(val_data)
        stats["train_path"] = train_path
        stats["val_path"] = val_path
    else:
        # 不划分，输出单个文件
        if output_path is None:
            output_path = input_path.replace(".json", "_sft.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        
        stats["output_path"] = output_path
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="转换合成数据为 SFT 格式")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--split", type=float, default=None, 
                        help="训练集比例（如 0.9 表示 90%% 训练，10%% 验证）")
    args = parser.parse_args()
    
    print(f"[转换] {args.input}")
    if args.split:
        print(f"[划分] 训练集 {args.split*100:.0f}% / 验证集 {(1-args.split)*100:.0f}%")
    
    stats = convert_file(args.input, args.output, args.split)
    
    print(f"\n[完成]")
    print(f"  总数: {stats['total']}")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['failed']}")
    
    if "train_count" in stats:
        print(f"\n[输出]")
        print(f"  训练集: {stats['train_path']} ({stats['train_count']} 条)")
        print(f"  验证集: {stats['val_path']} ({stats['val_count']} 条)")
    elif "output_path" in stats:
        print(f"  输出: {stats['output_path']}")


if __name__ == "__main__":
    main()
