"""测试容错解析器"""
import json
import re

print("测试容错解析器")
print("=" * 60)

# 模型实际输出（多个 <tool_calls> 块，未转义引号）
response = '''Let me search for blueprint
<tool_calls>
[{"id": "call_1", "type": "function", "function": {"name": "grep", "arguments": "{"query": "Blueprint", "path": "src/"}"}}]
</tool_calls>

<tool_calls>
[{"id": "call_2", "type": "function", "function": {"name": "grep", "arguments": "{"query": "raise", "path": "src/"}"}}]
</tool_calls>'''

print("原始响应:")
print(response)
print()

# 容错解析器（和 run_rl.py 一致）
def parse_tool_calls(response):
    tool_calls = []
    
    # 提取所有 <tool_calls>...</tool_calls> 内容
    pattern = r'<tool_calls>\s*(.*?)\s*</tool_calls>'
    matches = re.findall(pattern, response, re.DOTALL)
    
    print(f"找到 {len(matches)} 个 <tool_calls> 块")
    
    for i, calls_json in enumerate(matches):
        calls_json = calls_json.strip()
        print(f"\n块 {i+1}: {calls_json[:100]}...")
        
        # 方法1：尝试直接解析
        try:
            calls = json.loads(calls_json)
            print("  ✅ JSON 直接解析成功")
            if isinstance(calls, list):
                for call in calls:
                    if "function" in call:
                        func = call["function"]
                        name = func.get("name", "")
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except:
                            args = {}
                        tool_calls.append({"name": name, "arguments": args})
            continue
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON 解析失败: {e}")
        
        # 方法2：容错解析
        call_pattern = r'"name"\s*:\s*"(\w+)"[^}]*"arguments"\s*:\s*"\{([^}]*)\}"'
        call_matches = re.findall(call_pattern, calls_json)
        
        for name, args_inner in call_matches:
            args = {}
            kv_pattern = r'"(\w+)"\s*:\s*"([^"]*)"'
            for key, value in re.findall(kv_pattern, args_inner):
                args[key] = value
            
            if name and args:
                tool_calls.append({"name": name, "arguments": args})
                print(f"  ✅ 容错解析: {name} -> {args}")
    
    return tool_calls

# 测试
result = parse_tool_calls(response)
print("\n" + "=" * 60)
print(f"最终结果: {len(result)} 个工具调用")
for tc in result:
    print(f"  - {tc['name']}: {tc['arguments']}")
