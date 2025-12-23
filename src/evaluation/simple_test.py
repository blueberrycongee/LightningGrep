"""简单测试 - 只输出模型响应"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 测试用例
TESTS = [
    {
        "tree": """flask_blog/
├── app/
│   ├── models/
│   │   ├── user.py
│   │   └── post.py
│   ├── routes/
│   │   ├── auth.py
│   │   └── blog.py
│   └── utils/
│       └── email.py
├── config.py
└── run.py""",
        "query": "Find the user authentication function"
    },
    {
        "tree": """ecommerce-api/
├── src/main/java/com/shop/
│   ├── controller/
│   │   ├── OrderController.java
│   │   └── PaymentController.java
│   ├── service/
│   │   ├── PaymentService.java
│   │   └── OrderService.java
│   └── util/
│       └── JwtUtils.java
├── pom.xml
└── README.md""",
        "query": "Find the payment processing logic"
    },
    {
        "tree": """admin-dashboard/
├── src/
│   ├── hooks/
│   │   ├── useAuth.ts
│   │   └── useFetch.ts
│   ├── services/
│   │   └── api.ts
│   └── components/
│       └── Table.tsx
├── package.json
└── tsconfig.json""",
        "query": "Find the authentication hook"
    }
]

TOOLS = [
    {"name": "grep", "description": "Search pattern in files", "parameters": {"pattern": "string", "path": "string"}},
    {"name": "read_file", "description": "Read file content", "parameters": {"path": "string"}},
    {"name": "list_dir", "description": "List directory", "parameters": {"path": "string"}}
]

def build_prompt(tree: str, query: str) -> str:
    tools_str = json.dumps(TOOLS, ensure_ascii=False)
    return f"""<|im_start|>system
You are a code search agent. Available tools:
{tools_str}

Output format:
<tool_calls>
[{{"name": "grep", "arguments": {{"pattern": "xxx", "path": "yyy"}}}}]
</tool_calls>
<|im_end|>
<|im_start|>user
File tree:
{tree}

Query: {query}
<|im_end|>
<|im_start|>assistant
"""

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()
    print("模型加载完成\n")

    for i, test in enumerate(TESTS, 1):
        print(f"{'='*60}")
        print(f"测试 {i}")
        print(f"{'='*60}")
        print(f"Query: {test['query']}")
        print(f"\nFile tree:\n{test['tree']}")
        
        prompt = build_prompt(test['tree'], test['query'])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,  # 不截断
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        
        print(f"\n模型输出（共 {len(response)} 字符）:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        print()

if __name__ == "__main__":
    main()
