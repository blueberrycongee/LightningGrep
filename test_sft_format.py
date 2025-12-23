"""测试 RL 环境中真实的 prompt"""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
sys.path.insert(0, 'src')
from environment.code_search_env import TOOLS_DEFINITION

def format_sft_prompt(messages, tools=None, add_generation_prompt=True):
    parts = []
    if tools:
        tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
        parts.append(f"<|im_start|>system\nYou are a code search agent. Available tools:\n{tools_str}<|im_end|>")
    for msg in messages:
        role = msg["role"]
        if role == "system":
            content = msg["content"]
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            content = msg["content"]
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_str = json.dumps(tool_calls, ensure_ascii=False)
                parts.append(f"<|im_start|>assistant\n{content}\n<tool_calls>\n{tc_str}\n</tool_calls><|im_end|>")
            else:
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>tool\n<tool_call_id>{tool_call_id}</tool_call_id>\n{content}<|im_end|>")
    result = "\n".join(parts)
    if add_generation_prompt:
        result += "\n<|im_start|>assistant\n"
    return result

# 加载真实数据
data = json.load(open('data/swebench/flask_test.json', encoding='utf-8'))
instance = data[0]
print("=" * 60)
print("测试 RL 环境真实 Prompt")
print("=" * 60)
print(f"Instance: {instance['instance_id']}")
print(f"Query: {instance['query'][:150]}...")
print()

# RL 环境的 prompt（更强约束）
system_prompt = """You are a code search agent. Find relevant files and code locations.

## Tool Call Format (MUST follow EXACTLY)

<tool_calls>
[{"id": "call_1", "type": "function", "function": {"name": "NAME", "arguments": "{\"key\": \"value\"}"}}]
</tool_calls>

CRITICAL RULES for arguments:
- arguments MUST be a JSON string with escaped quotes: "{\"key\": \"value\"}"
- WRONG: "arguments": "{"key": "value"}"
- RIGHT: "arguments": "{\"key\": \"value\"}"

## Available Tools

1. grep: {"query": "pattern", "path": "dir/"} - Search text in files
2. read: {"file": "path", "start": 1, "end": 50} - Read file lines  
3. glob: {"pattern": "**/*.py"} - List files by pattern
4. find: {"name": "filename"} - Find files by name

## Example

User: Find error handling code
Assistant: Let me search for error handling patterns.
<tool_calls>
[{"id": "call_1", "type": "function", "function": {"name": "grep", "arguments": "{\"query\": \"except\", \"path\": \"src/\"}"}}, {"id": "call_2", "type": "function", "function": {"name": "glob", "arguments": "{\"pattern\": \"**/*error*.py\"}"}}]
</tool_calls>

## Rules
- Maximum 4 turns, 8 tool calls per turn
- Final answer format:
<answer>
1. path/file.py:10-25
2. path/other.py:100-120
</answer>"""

user_prompt = f"""Repository: {instance['repo']}

Issue:
{instance['query'][:500]}

Find the relevant files and code locations."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

print("加载模型...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "outputs/sft_v2")
tokenizer = AutoTokenizer.from_pretrained("outputs/sft_v2", trust_remote_code=True)

prompt = format_sft_prompt(messages, tools=TOOLS_DEFINITION)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Prompt tokens: {inputs.input_ids.shape[1]}")

print("\n生成中 (temperature=0, greedy)...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=500, do_sample=False, pad_token_id=tokenizer.pad_token_id)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
print(f"生成 tokens: {len(outputs[0]) - inputs.input_ids.shape[1]}")
print("=" * 60)
print("模型输出:")
print("=" * 60)
print(response)
