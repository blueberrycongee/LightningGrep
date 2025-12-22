# LightningGrep ⚡

> **开源的代码检索 Agent 模型**
>
> Open-Source Code Retrieval Agent - Replicating Windsurf's Fast Context

## 🎯 项目简介

LightningGrep 是一个开源的**代码检索 Agent**，复刻 Windsurf 的 Fast Context 功能：

- **并行工具调用**：一次调用多个 grep/read/glob，减少搜索轮数
- **标准 FC 格式**：使用 Qwen 标准 Function Calling 格式训练
- **多轮搜索**：3-4 轮搜索，每轮 5-8 个并行工具调用
- **精准定位**：返回相关文件 + 行号范围

### 定位

```
用户查询 ──→ LightningGrep Agent ──→ grep/read/glob 工具 ──→ 相关代码位置
                   │
                   ├── Round 1: 并行 grep 搜索关键词
                   ├── Round 2: 并行 read 读取文件
                   └── Round 3: 确认并返回结果
```

### 示例输出

```json
Query: "Find where ParseError is raised"

Round 1 (5 parallel calls):
├── grep "raise ParseError" src/
├── grep "ParseError(" src/
├── grep "class ParseError" src/
├── glob "src/*parser*.py"
└── glob "src/*error*.py"

Round 2 (4 parallel calls):
├── read src/parser/json.py:130-150
├── read src/parser/xml.py:75-95
├── read src/error.py:10-30
└── grep "except ParseError" src/

Final Result:
├── src/parser/json.py [140-144] - high
├── src/parser/xml.py [85-87] - high
└── src/error.py [12-15] - medium
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 合成训练数据

使用 DeepSeek-V3 通过硅基流动 API 合成代码检索轨迹数据：

```bash
# 设置 API Key
export SILICONFLOW_API_KEY="your-api-key"  # Linux/Mac
# 或
set SILICONFLOW_API_KEY=your-api-key       # Windows

# 合成 1000 条数据
python src/data_synthesis/synthesize_code_search_llm.py \
    --num 1000 \
    --output data/code_search/sft_1k.json \
    --model deepseek-ai/DeepSeek-V3

# 如果中断，用 --resume 继续（每 10 条自动保存）
python src/data_synthesis/synthesize_code_search_llm.py \
    --num 1000 \
    --output data/code_search/sft_1k.json \
    --model deepseek-ai/DeepSeek-V3 \
    --resume
```

### 3. 分布式合成（可选）

多人协作，加速数据合成：

```bash
# 你：合成第一部分
python synthesize_code_search_llm.py --num 500 --output sft_part1.json

# 朋友：合成第二部分
python synthesize_code_search_llm.py --num 500 --output sft_part2.json

# 合并数据
python -c "
import json, glob
data = []
for f in glob.glob('sft_part*.json'):
    data.extend(json.load(open(f)))
json.dump(data, open('sft_merged.json', 'w'), ensure_ascii=False, indent=2)
print(f'Merged {len(data)} samples')
"
```

### 4. 训练模型（QLoRA）

```bash
python src/training/sft_qlora.py \
    --train_data data/code_search/sft_1k.json \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --output_dir outputs/code_search_v1
```

### 5. 测试模型

```bash
python scripts/test_model.py \
    --model outputs/code_search_v1 \
    --query "Find where TimeoutError is raised"
```

## 📁 项目结构

```
LightningGrep/
├── data/
│   └── code_search/              # 代码检索训练数据
├── src/
│   ├── data_synthesis/
│   │   └── synthesize_code_search_llm.py  # 数据合成脚本
│   └── training/
│       └── sft_qlora.py          # QLoRA SFT 训练
├── scripts/
│   ├── download_data.py          # 数据下载
│   └── test_model.py             # 模型测试
├── outputs/                      # 训练输出
├── research-plan.md              # 研究计划
└── requirements.txt
```

## 📋 研究计划

### 目标

复刻 Windsurf 的 Fast Context 功能，训练一个能进行**并行代码检索**的小模型。

### 阶段规划

| 阶段 | 内容 | 状态 |
|------|------|------|
| **V1** | 数据合成 + SFT | 🔄 进行中 |
| **V2** | SWE-Bench RL | ⏳ 待开始 |

### V1 进度

- [x] 数据格式设计（Qwen FC 标准格式）
- [x] 数据合成脚本（DeepSeek-V3）
- [x] 断点续传支持
- [x] 分布式合成支持
- [ ] 合成 1000+ 条 SFT 数据（⏳ 进行中...）
- [ ] QLoRA SFT 训练
- [ ] 评测

## 📊 数据格式

使用 **Qwen 标准 Function Calling 格式**：

```json
{
  "messages": [
    {"role": "user", "content": "Find where TimeoutError is raised"},
    {
      "role": "assistant",
      "content": "Search for TimeoutError in source files",
      "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "grep", "arguments": "{\"query\": \"raise TimeoutError\", \"path\": \"src/\"}"}},
        {"id": "call_2", "type": "function", "function": {"name": "grep", "arguments": "{\"query\": \"TimeoutError(\", \"path\": \"src/\"}"}}
      ]
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "src/client.py:42: raise TimeoutError('Connection timeout')"},
    {"role": "tool", "tool_call_id": "call_2", "content": "src/server.py:128: raise TimeoutError(f'Request timeout after {timeout}s')"},
    {"role": "assistant", "content": "{\"result\": {\"files\": [...], \"summary\": \"...\"}}"}
  ],
  "tools": [...]
}
```

## 🔬 方法论

### 训练流程

```
SFT（格式 + 基础策略）
  │
  │  使用合成数据，教模型：
  │  - Qwen FC 格式输出
  │  - 并行工具调用（5-8 个/轮）
  │  - 多轮搜索策略（3-4 轮）
  │
  ▼
 RL（SWE-Bench 优化）
  │
  │  真实代码库 + 真实 Issue，优化：
  │  - 搜索精准度
  │  - 效率（减少轮数）
  │
  ▼
最终模型
```

### SWE-Bench 数据

RL 阶段使用 [SWE-Bench](https://www.swebench.com/) 作为训练环境：

| 字段 | 说明 |
|------|------|
| `problem_statement` | GitHub Issue 原文描述 |
| `repo` | 完整的 Git 仓库 |
| `base_commit` | 问题发生时的 commit |
| `patch` | 正确修复的 diff（包含文件+行号） |

**训练流程**：
```
Issue 描述 → Agent 搜索 → 对比 Patch 中的文件+行号 → 计算 Reward
```

### RL 训练方法（参考 SWE-grep）

根据 [Windsurf SWE-grep 博客](https://www.cognition.ai/blog/swe-grep)：

| 技术 | 说明 |
|------|------|
| **Policy Gradient** | 基础 RL 算法 |
| **Per-Sequence Importance Sampling** | 处理多轮交互的重要性采样 |
| **Leave-One-Out Baseline** | 减少方差，用 N-1 个样本估计 baseline |
| **Weighted F1 奖励** | β=0.5，平衡 Precision 和 Recall |
| **Mask 环境 Token** | 训练时不学习工具返回内容的生成 |

**奖励函数**：
```python
# Weighted F1 (β=0.5，偏向 Recall)
precision = correct_files / predicted_files
recall = correct_files / ground_truth_files
reward = (1 + β²) * (precision * recall) / (β² * precision + recall)
```

### 工具定义

| 工具 | 参数 | 说明 |
|------|------|------|
| `grep` | query, path | 搜索文本模式 |
| `read` | file, start, end | 读取文件行 |
| `glob` | pattern | 列出匹配文件 |

## 📚 参考

- [SWE-grep Blog](https://www.cognition.ai/blog/swe-grep) - Windsurf 的并行检索方法（未开源）
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) - 开源 RL 检索模型
- [ParallelSearch](https://arxiv.org/abs/2508.09303) - 并行查询分解（代码未公开）
- [HotpotQA](https://hotpotqa.github.io/) - 多跳问答数据集

## 📝 License

MIT

