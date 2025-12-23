"""
SFT 模型效果测试脚本
用模拟的项目结构测试模型是否能正确输出 tool_calls
"""

import os
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 模拟项目结构 - 用于构造真实的测试场景
# 格式与用户实际看到的文件树一致
MOCK_PROJECTS = [
    {
        "name": "flask_blog",
        "description": "Flask 博客项目",
        "structure": """flask_blog/
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── user.py
│   │   ├── post.py
│   │   └── comment.py
│   ├── routes/
│   │   ├── auth.py
│   │   ├── blog.py
│   │   └── api.py
│   ├── utils/
│   │   ├── email.py
│   │   └── validators.py
│   └── templates/
│       ├── base.html
│       └── [+12 files (12 html) & 2 dirs]
├── tests/
│   ├── test_auth.py
│   ├── test_blog.py
│   └── conftest.py
├── migrations/
│   └── versions/
├── config.py
├── requirements.txt
├── .env.example
└── run.py""",
        "queries": [
            "Find the user authentication function",
            "Where is the login route defined?",
            "Find email sending functionality",
        ]
    },
    {
        "name": "ecommerce_api",
        "description": "Spring Boot 电商后端",
        "structure": """ecommerce-api/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/
│   │   │       └── shop/
│   │   │           ├── controller/
│   │   │           │   ├── ProductController.java
│   │   │           │   ├── OrderController.java
│   │   │           │   ├── UserController.java
│   │   │           │   └── PaymentController.java
│   │   │           ├── service/
│   │   │           │   ├── PaymentService.java
│   │   │           │   ├── OrderService.java
│   │   │           │   ├── InventoryService.java
│   │   │           │   └── impl/
│   │   │           │       └── [+4 files (4 java) & 0 dirs]
│   │   │           ├── repository/
│   │   │           │   └── [+5 files (5 java) & 0 dirs]
│   │   │           ├── model/
│   │   │           │   ├── Product.java
│   │   │           │   ├── Order.java
│   │   │           │   └── User.java
│   │   │           ├── config/
│   │   │           │   ├── SecurityConfig.java
│   │   │           │   └── JwtConfig.java
│   │   │           └── util/
│   │   │               ├── JwtUtils.java
│   │   │               └── EncryptionUtils.java
│   │   └── resources/
│   │       ├── application.yml
│   │       └── application-dev.yml
│   └── test/
│       └── java/
├── pom.xml
├── Dockerfile
└── README.md""",
        "queries": [
            "Find the payment processing logic",
            "Where is JWT token generation?",
            "Find order creation endpoint",
        ]
    },
    {
        "name": "react_dashboard",
        "description": "React + TypeScript 管理后台",
        "structure": """admin-dashboard/
├── src/
│   ├── components/
│   │   ├── common/
│   │   │   ├── Button.tsx
│   │   │   ├── Modal.tsx
│   │   │   └── Table.tsx
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── Footer.tsx
│   │   └── charts/
│   │       ├── LineChart.tsx
│   │       ├── BarChart.tsx
│   │       └── PieChart.tsx
│   ├── pages/
│   │   ├── Dashboard/
│   │   │   ├── index.tsx
│   │   │   └── components/
│   │   ├── Users/
│   │   │   ├── index.tsx
│   │   │   ├── UserList.tsx
│   │   │   └── UserDetail.tsx
│   │   └── Settings/
│   │       └── index.tsx
│   ├── hooks/
│   │   ├── useAuth.ts
│   │   ├── useFetch.ts
│   │   └── useLocalStorage.ts
│   ├── services/
│   │   ├── api.ts
│   │   ├── auth.service.ts
│   │   └── user.service.ts
│   ├── store/
│   │   ├── index.ts
│   │   └── slices/
│   │       ├── authSlice.ts
│   │       └── userSlice.ts
│   ├── types/
│   │   └── index.ts
│   ├── utils/
│   │   ├── formatters.ts
│   │   └── validators.ts
│   ├── App.tsx
│   └── main.tsx
├── public/
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md""",
        "queries": [
            "Find the authentication hook",
            "Where is the data table component?",
            "Find API service configuration",
        ]
    },
    {
        "name": "python_ml_project",
        "description": "Python 机器学习项目",
        "structure": """ml-pipeline/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── transformer.py
│   │   └── cnn.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── metrics.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_experiments.ipynb
├── configs/
│   ├── model_config.yaml
│   └── train_config.yaml
├── tests/
│   └── [+5 files (5 py) & 0 dirs]
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   └── checkpoints/
├── requirements.txt
├── setup.py
└── README.md""",
        "queries": [
            "Find the model training loop",
            "Where is data preprocessing implemented?",
            "Find the learning rate scheduler",
        ]
    }
]

# 工具定义（与训练时一致）
TOOLS = [
    {
        "name": "grep",
        "description": "Search for a pattern in files",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex)"},
                "path": {"type": "string", "description": "Directory or file to search"},
                "include": {"type": "string", "description": "File pattern to include"}
            },
            "required": ["pattern", "path"]
        }
    },
    {
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "start_line": {"type": "integer", "description": "Start line number"},
                "end_line": {"type": "integer", "description": "End line number"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_dir",
        "description": "List files in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"}
            },
            "required": ["path"]
        }
    }
]


def build_prompt(project: dict, query: str, max_tree_lines: int = 50) -> str:
    """
    构造测试 prompt
    
    Args:
        project: 项目信息
        query: 用户查询
        max_tree_lines: 文件树最大行数（截断逻辑）
    """
    tools_str = json.dumps(TOOLS, ensure_ascii=False, indent=2)
    
    # 文件树截断逻辑
    tree_lines = project['structure'].strip().split('\n')
    if len(tree_lines) > max_tree_lines:
        tree = '\n'.join(tree_lines[:max_tree_lines])
        tree += f"\n... (truncated, {len(tree_lines) - max_tree_lines} more items)"
    else:
        tree = project['structure']
    
    # JSON Schema 示例 - 让模型知道精确的输出格式
    output_schema = '''
Output format - you MUST use this exact JSON structure:
<tool_calls>
[{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}]
</tool_calls>

Example:
<tool_calls>
[{"name": "grep", "arguments": {"pattern": "def authenticate", "path": "app/models/"}}]
</tool_calls>
'''
    
    prompt = f"""<|im_start|>system
You are a code search agent. You help users find code in repositories.

Available tools:
{tools_str}

{output_schema}
<|im_end|>
<|im_start|>user
Project structure:
{tree}

Query: {query}
<|im_end|>
<|im_start|>assistant
"""
    return prompt


def validate_tool_call(tool_call: dict, project: dict) -> dict:
    """
    验证工具调用的合理性
    
    Returns:
        dict: {"valid": bool, "issues": list, "score": float}
    """
    issues = []
    score = 1.0
    
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})
    
    # 处理 arguments 是字符串的情况
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            issues.append("arguments 不是有效的 JSON")
            return {"valid": False, "issues": issues, "score": 0.3}
    
    # 1. 检查工具名称
    valid_tools = ["grep", "read_file", "list_dir"]
    if name not in valid_tools:
        issues.append(f"未知工具: {name}")
        score -= 0.5
    
    # 2. 检查必需参数
    if name == "grep":
        if "pattern" not in args:
            issues.append("grep 缺少 pattern 参数")
            score -= 0.3
        if "path" not in args:
            issues.append("grep 缺少 path 参数")
            score -= 0.2
        # 检查 pattern 是否为空
        if args.get("pattern", "") == "":
            issues.append("pattern 为空")
            score -= 0.3
            
    elif name == "read_file":
        if "path" not in args:
            issues.append("read_file 缺少 path 参数")
            score -= 0.5
            
    elif name == "list_dir":
        if "path" not in args:
            issues.append("list_dir 缺少 path 参数")
            score -= 0.3
    
    # 3. 检查路径是否在项目结构中（粗略检查）
    if "path" in args:
        path = args["path"]
        structure = project.get("structure", "")
        # 检查路径的某部分是否出现在结构中
        path_parts = path.replace("\\", "/").split("/")
        found = any(part in structure for part in path_parts if part and part != ".")
        if not found and path not in [".", "./", "/"]:
            issues.append(f"路径 '{path}' 可能不存在于项目中")
            score -= 0.1  # 轻微扣分，因为可能是模糊搜索
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "score": max(0, score)
    }


def test_model(model_path: str, base_model_name: str = "Qwen/Qwen3-1.7B"):
    """测试 SFT 模型"""
    print("=" * 60)
    print("SFT 模型测试")
    print("=" * 60)
    
    # 加载模型
    print("\n[1/3] 加载模型...")
    print(f"  Base: {base_model_name}")
    print(f"  LoRA: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("  ✓ 模型加载完成")
    
    # 测试
    print("\n[2/3] 开始测试...")
    results = []
    
    for project in MOCK_PROJECTS:
        print(f"\n📁 项目: {project['name']} ({project['description']})")
        
        for query in project['queries']:
            print(f"\n  🔍 Query: {query}")
            
            prompt = build_prompt(project, query)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,  # 增加以避免截断
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
            
            # 检查输出格式
            has_tool_calls = "<tool_calls>" in response
            valid_json = False
            tool_name = None
            parsed_call = None
            validation = {"valid": False, "issues": ["未解析"], "score": 0}
            
            if has_tool_calls:
                try:
                    start = response.find("<tool_calls>") + len("<tool_calls>")
                    end = response.find("</tool_calls>")
                    if end > start:
                        json_str = response[start:end].strip()
                        parsed = json.loads(json_str)
                        valid_json = True
                        if isinstance(parsed, list) and len(parsed) > 0:
                            parsed_call = parsed[0]
                            tool_name = parsed_call.get("name", "unknown")
                            # 处理 arguments 是字符串的情况
                            args = parsed_call.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    parsed_call["arguments"] = json.loads(args)
                                except:
                                    pass  # 保持原样
                            # 验证工具调用合理性
                            validation = validate_tool_call(parsed_call, project)
                    elif end == -1 and start > 0:
                        # 没有 </tool_calls> 结束标签，可能被截断
                        validation = {"valid": False, "issues": ["输出被截断，缺少 </tool_calls>"], "score": 0.2}
                except json.JSONDecodeError as e:
                    # 尝试修复常见的 JSON 错误
                    error_msg = str(e)
                    if "Expecting ',' delimiter" in error_msg or "Unterminated string" in error_msg:
                        validation = {"valid": False, "issues": ["JSON 被截断或格式错误"], "score": 0.1}
                    else:
                        validation = {"valid": False, "issues": [f"JSON 解析错误: {error_msg[:50]}"], "score": 0}
                except Exception as e:
                    validation = {"valid": False, "issues": [f"解析异常: {str(e)[:50]}"], "score": 0}
            
            # 综合评分
            is_success = has_tool_calls and valid_json and validation["valid"]
            status = "✅" if is_success else ("⚠️" if valid_json else "❌")
            
            print(f"     {status} tool_calls: {has_tool_calls}, valid_json: {valid_json}, tool: {tool_name}")
            print(f"     验证分数: {validation['score']:.2f}, 问题: {validation['issues'] if validation['issues'] else '无'}")
            
            # 打印部分输出
            short_response = response[:200].replace('\n', ' ')
            print(f"     Response: {short_response}...")
            
            results.append({
                "project": project['name'],
                "query": query,
                "has_tool_calls": has_tool_calls,
                "valid_json": valid_json,
                "tool_name": tool_name,
                "parsed_call": parsed_call,
                "validation": validation,
                "response": response,
                "prompt": prompt  # 保存完整 prompt 用于分析
            })
    
    # 统计
    print("\n" + "=" * 60)
    print("[3/3] 测试结果统计")
    print("=" * 60)
    
    total = len(results)
    has_tool_calls_count = sum(1 for r in results if r['has_tool_calls'])
    valid_json_count = sum(1 for r in results if r['valid_json'])
    fully_valid_count = sum(1 for r in results if r['validation']['valid'])
    avg_validation_score = sum(r['validation']['score'] for r in results) / total if total > 0 else 0
    
    print(f"\n总测试: {total}")
    print(f"有 tool_calls: {has_tool_calls_count}/{total} ({100*has_tool_calls_count/total:.1f}%)")
    print(f"JSON 有效: {valid_json_count}/{total} ({100*valid_json_count/total:.1f}%)")
    print(f"完全合理: {fully_valid_count}/{total} ({100*fully_valid_count/total:.1f}%)")
    print(f"平均验证分数: {avg_validation_score:.2f}")
    
    # 工具使用分布
    tool_counts = {}
    for r in results:
        if r['tool_name']:
            tool_counts[r['tool_name']] = tool_counts.get(r['tool_name'], 0) + 1
    
    if tool_counts:
        print(f"\n工具使用分布:")
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            print(f"  - {tool}: {count}")
    
    # 常见问题统计
    all_issues = []
    for r in results:
        all_issues.extend(r['validation']['issues'])
    if all_issues:
        issue_counts = {}
        for issue in all_issues:
            # 简化问题描述
            key = issue.split(":")[0] if ":" in issue else issue
            issue_counts[key] = issue_counts.get(key, 0) + 1
        print(f"\n常见问题:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  - {issue}: {count}次")
    
    # 评级 - 基于完全合理的比例
    score = fully_valid_count / total * 100
    if score >= 80:
        grade = "优秀 🌟"
    elif score >= 60:
        grade = "良好 👍"
    elif score >= 40:
        grade = "及格 😐"
    else:
        grade = "需改进 ⚠️"
    
    print(f"\n综合评分: {score:.1f}% - {grade}")
    
    # 保存详细结果到文件
    save_results(results, model_path)
    
    return results


def save_results(results: list, model_path: str):
    """保存测试结果到文件，方便分析模型输出的逻辑性"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确定输出目录
    output_dir = os.path.join(os.path.dirname(model_path), "test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存 JSON 格式（机器可读）
    json_path = os.path.join(output_dir, f"sft_test_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 JSON 结果保存到: {json_path}")
    
    # 2. 保存可读报告（人工分析）
    report_path = os.path.join(output_dir, f"sft_test_{timestamp}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SFT 模型测试报告\n\n")
        f.write(f"- 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 模型路径: {model_path}\n")
        f.write(f"- 测试用例数: {len(results)}\n\n")
        
        # 统计
        total = len(results)
        success = sum(1 for r in results if r['valid_json'])
        f.write(f"## 📊 统计\n\n")
        f.write(f"- 成功率: {success}/{total} ({100*success/total:.1f}%)\n\n")
        
        # 详细结果
        f.write("## 📝 详细测试结果\n\n")
        f.write("每个测试用例都是 **独立的**，无历史上下文。\n\n")
        
        for i, r in enumerate(results, 1):
            status = "✅" if r['valid_json'] else "❌"
            f.write(f"---\n\n")
            f.write(f"### 测试 {i}: {status}\n\n")
            f.write(f"**项目**: {r['project']}\n\n")
            f.write(f"**Query**: {r['query']}\n\n")
            f.write(f"**工具调用**: {r['tool_name'] or '无'}\n\n")
            
            # 模型输出
            f.write(f"**模型输出**:\n\n")
            f.write("```\n")
            f.write(r['response'])
            f.write("\n```\n\n")
            
            # 逻辑分析
            f.write(f"**逻辑分析**:\n\n")
            analysis = analyze_response_logic(r)
            for point in analysis:
                f.write(f"- {point}\n")
            f.write("\n")
    
    print(f"📄 Markdown 报告保存到: {report_path}")
    print(f"\n💡 提示: 查看报告文件可分析模型输出的逻辑性")


def analyze_response_logic(result: dict) -> list:
    """分析单个响应的逻辑性"""
    analysis = []
    response = result['response']
    query = result['query'].lower()
    
    # 1. 检查是否有 tool_calls
    if result['has_tool_calls']:
        analysis.append("✅ 正确使用了 tool_calls 格式")
    else:
        analysis.append("❌ 没有使用 tool_calls 格式")
        return analysis
    
    # 2. 检查 JSON 是否有效
    if result['valid_json']:
        analysis.append("✅ JSON 格式正确")
    else:
        analysis.append("❌ JSON 格式错误")
        return analysis
    
    # 3. 检查工具选择是否合理
    tool = result['tool_name']
    if tool == 'grep':
        analysis.append("✅ 选择 grep 工具 - 适合搜索代码模式")
    elif tool == 'list_dir':
        analysis.append("⚠️ 选择 list_dir 工具 - 适合浏览目录结构")
    elif tool == 'read_file':
        analysis.append("⚠️ 选择 read_file 工具 - 通常需要先搜索再读取")
    else:
        analysis.append(f"❓ 使用了未知工具: {tool}")
    
    # 4. 检查搜索模式是否与 query 相关
    try:
        start = response.find("<tool_calls>") + len("<tool_calls>")
        end = response.find("</tool_calls>")
        json_str = response[start:end].strip()
        parsed = json.loads(json_str)
        
        if isinstance(parsed, list) and len(parsed) > 0:
            args = parsed[0].get("arguments", {})
            
            # 检查 grep pattern
            if "pattern" in args:
                pattern = args["pattern"].lower()
                # 检查 pattern 是否与 query 相关
                keywords = extract_keywords(query)
                matches = [k for k in keywords if k in pattern]
                if matches:
                    analysis.append(f"✅ 搜索模式与查询相关: '{args['pattern']}' 包含关键词 {matches}")
                else:
                    analysis.append(f"⚠️ 搜索模式可能不够精确: '{args['pattern']}'")
            
            # 检查路径是否合理
            if "path" in args:
                path = args["path"]
                if path in [".", "./", "/"]:
                    analysis.append(f"⚠️ 搜索路径较宽泛: '{path}'")
                else:
                    analysis.append(f"✅ 指定了搜索路径: '{path}'")
    except:
        pass
    
    return analysis


def extract_keywords(query: str) -> list:
    """从 query 中提取关键词"""
    # 关键词映射
    keyword_map = {
        "authentication": ["auth", "authenticate", "login", "user"],
        "login": ["login", "auth", "signin"],
        "email": ["email", "mail", "send"],
        "payment": ["payment", "pay", "transaction", "charge"],
        "jwt": ["jwt", "token", "auth"],
        "order": ["order", "create", "purchase"],
        "hook": ["hook", "use"],
        "table": ["table", "data", "grid"],
        "api": ["api", "service", "fetch", "request"],
    }
    
    keywords = []
    for key, variants in keyword_map.items():
        if key in query:
            keywords.extend(variants)
    
    return list(set(keywords))


def compare_with_base(model_path: str, base_model_name: str = "Qwen/Qwen3-1.7B"):
    """对比 SFT 模型和原始模型"""
    print("\n" + "=" * 60)
    print("对比测试：SFT vs 原始模型")
    print("=" * 60)
    
    # 简化测试
    project = MOCK_PROJECTS[0]
    query = project['queries'][0]
    prompt = build_prompt(project, query)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # 测试原始模型
    print("\n[原始模型]")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        outputs = base_model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
    base_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    print(f"Response: {base_response[:300]}...")
    
    # 清理显存
    del base_model
    torch.cuda.empty_cache()
    
    # 测试 SFT 模型
    print("\n[SFT 模型]")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    sft_model = PeftModel.from_pretrained(base_model, model_path)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(sft_model.device)
    with torch.no_grad():
        outputs = sft_model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
    sft_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    print(f"Response: {sft_response[:300]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 SFT 模型效果")
    parser.add_argument("--model_path", type=str, required=True, help="SFT 模型路径")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B", help="基座模型")
    parser.add_argument("--compare", action="store_true", help="是否与原始模型对比")
    
    args = parser.parse_args()
    
    # 运行测试
    results = test_model(args.model_path, args.base_model)
    
    if args.compare:
        compare_with_base(args.model_path, args.base_model)
