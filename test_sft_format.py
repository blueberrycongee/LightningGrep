"""测试 SFT 格式的 prompt"""
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
    result = "\n".join(parts)
    if add_generation_prompt:
        result += "\n<|im_start|>assistant\n"
    return result

print("加载模型...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "outputs/sft_v2")
tokenizer = AutoTokenizer.from_pretrained("outputs/sft_v2", trust_remote_code=True)

messages = [{"role": "user", "content": "Find routing files in Flask."}]
prompt = format_sft_prompt(messages, tools=TOOLS_DEFINITION)
print("Prompt 最后 200 字符:")
print(prompt[-200:])

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"\nPrompt tokens: {inputs.input_ids.shape[1]}")

print("\n生成中 (temperature=0.7)...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
print(f"生成 tokens: {len(outputs[0]) - inputs.input_ids.shape[1]}")
print("---响应---")
print(response[:800])
