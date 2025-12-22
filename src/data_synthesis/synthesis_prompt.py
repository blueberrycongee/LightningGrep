"""
数据合成 Prompt - 用于训练并行工具调用 Agent
"""

SYSTEM_PROMPT = """你是一个数据标注专家，为训练并行检索 Agent 生成训练数据。

Agent 的能力：
1. 分析问题，决定需要搜索什么
2. 如果多个子问题相互独立，可以**并行搜索**（一次输出多个查询）
3. 如果有依赖关系，必须**串行搜索**（分多轮）

你的任务：给定问答对，生成 Agent 应该如何搜索的轨迹数据。

输出 JSON 格式，包含：
- question: 问题
- is_parallel: 是否可以并行
- rounds: 每轮的思考和搜索查询
- answer: 最终答案
- sources: 答案来源（文档标题 + 行号）
"""

USER_PROMPT_TEMPLATE = """为以下问答生成 Agent 搜索轨迹：

**问题**：{question}
**答案**：{answer}
**支持事实**：{supporting_facts}
**文档**：
{context}

输出 JSON：
```json
{{
  "question": "问题",
  "is_parallel": true/false,
  "rounds": [
    {{
      "think": "分析：为什么这样搜索",
      "searches": ["查询1", "查询2"],
      "parallel": true/false
    }}
  ],
  "final_think": "根据搜索结果的分析",
  "answer": "答案",
  "sources": [
    {{"doc": "文档标题", "lines": [0, 2]}}
  ]
}}
```

**并行判断规则**：
- ✅ 并行：比较两个独立实体（"A和B哪个早"→同时查A和B）
- ✅ 并行：查询两个不相关的事实
- ❌ 串行：需要先知道X才能查Y（"X妻子是哪国人"→先查妻子是谁，再查国籍）
- ❌ 串行：只涉及一个实体的一个属性

**示例1（并行）**：
问题：A和B哪个杂志先创办？
```json
{{
  "is_parallel": true,
  "rounds": [
    {{"think": "两个杂志独立，可以并行查询创办时间", "searches": ["A杂志创办时间", "B杂志创办时间"], "parallel": true}}
  ],
  "final_think": "A是1844年，B是1989年，A更早",
  "answer": "A"
}}
```

**示例2（串行）**：
问题：X的妻子是哪国人？
```json
{{
  "is_parallel": false,
  "rounds": [
    {{"think": "需要先找到X的妻子是谁", "searches": ["X的妻子是谁"], "parallel": false}},
    {{"think": "知道妻子是Y后，查Y的国籍", "searches": ["Y的国籍"], "parallel": false}}
  ],
  "final_think": "Y是美国人",
  "answer": "美国人"
}}
```"""


def format_context(context: list) -> str:
    """格式化 HotpotQA 的 context"""
    result = []
    for title, sentences in context:
        result.append(f"### {title}")
        for i, sent in enumerate(sentences):
            result.append(f"  [{i}] {sent}")
        result.append("")
    return "\n".join(result)


def format_supporting_facts(supporting_facts: list) -> str:
    """格式化支持事实"""
    result = []
    for title, sent_idx in supporting_facts:
        result.append(f"- {title}, 句子 {sent_idx}")
    return "\n".join(result)


def create_synthesis_prompt(sample: dict) -> tuple:
    """创建合成 prompt"""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=sample["question"],
        answer=sample["answer"],
        supporting_facts=format_supporting_facts(sample["supporting_facts"]),
        context=format_context(sample["context"]),
    )
    return SYSTEM_PROMPT, user_prompt


# 新格式示例
EXAMPLE_PARALLEL = {
    "question": "Which magazine was started first, A or B?",
    "is_parallel": True,
    "rounds": [
        {
            "think": "两个杂志独立，可以并行查询创办时间",
            "searches": ["A杂志创办时间", "B杂志创办时间"],
            "parallel": True
        }
    ],
    "final_think": "A是1844年，B是1989年，A更早",
    "answer": "A",
    "sources": [
        {"doc": "A Magazine", "lines": [0, 1]},
        {"doc": "B Magazine", "lines": [0, 1]}
    ]
}

EXAMPLE_SEQUENTIAL = {
    "question": "What nationality was X's wife?",
    "is_parallel": False,
    "rounds": [
        {
            "think": "需要先找到X的妻子是谁",
            "searches": ["X的妻子是谁"],
            "parallel": False
        },
        {
            "think": "知道妻子是Y后，查Y的国籍",
            "searches": ["Y的国籍"],
            "parallel": False
        }
    ],
    "final_think": "Y是美国人",
    "answer": "American",
    "sources": [
        {"doc": "Y", "lines": [0, 1]}
    ]
}
