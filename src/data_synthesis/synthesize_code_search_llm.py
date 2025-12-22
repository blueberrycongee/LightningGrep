"""
使用 LLM 合成代码检索训练数据
支持：硅基流动 / OpenAI / Anthropic
"""
import os
import json
import argparse
import random
from typing import Optional, List, Dict
from openai import OpenAI
from tqdm import tqdm
import anthropic

# 配置
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "code_search")

# API 配置
API_CONFIGS = {
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "Qwen/Qwen2.5-72B-Instruct",
        "env_key": "SILICONFLOW_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "base_url": "https://anyrouter.top",  # anyrouter 代理（不带 /v1）
        "default_model": "claude-haiku-4-5-20251001",  # Claude Haiku 4.5（便宜快速）
        "env_key": "ANTHROPIC_API_KEY",
    },
}

# ==================== Prompt ====================

SYSTEM_PROMPT = """You are an expert at generating training data for code search AI agents.

Generate realistic search trajectories for code retrieval tasks.

## Output Format (JSON)
{
  "query": "user query",
  "trajectory": {
    "rounds": [
      {
        "thought": "reasoning for this search",
        "tool_calls": [
          {"name": "grep", "args": {"query": "search term", "path": "search/path/"}},
          {"name": "read", "args": {"file": "path/to/file.py", "start": 1, "end": 50}},
          {"name": "glob", "args": {"pattern": "*.py"}}
        ],
        "results": ["file.py:42: matching line content"]
      }
    ],
    "final": {
      "files": [{"path": "path/to/file.py", "lines": [40, 80], "relevance": "high"}],
      "summary": "brief summary of findings"
    }
  }
}

## Rules
1. Use 3-4 rounds for complex queries, 2-3 rounds for simple queries
2. Use 5-8 parallel tool calls per round when searching multiple targets
3. STRONGLY PREFER PARALLEL: always search multiple related keywords in ONE round
4. Generate REALISTIC file paths, function names, and search results
5. Search terms should be specific and meaningful (not generic words like "the", "is")
6. Include a mix of grep, read, and glob operations

## Parallelism Examples
GOOD (parallel - 6 tools in one round):
{
  "thought": "Search for all authentication related code",
  "tool_calls": [
    {"name": "grep", "args": {"query": "def login", "path": "src/auth/"}},
    {"name": "grep", "args": {"query": "def logout", "path": "src/auth/"}},
    {"name": "grep", "args": {"query": "def authenticate", "path": "src/"}},
    {"name": "grep", "args": {"query": "class AuthService", "path": "src/"}},
    {"name": "grep", "args": {"query": "JWT", "path": "src/auth/"}},
    {"name": "glob", "args": {"pattern": "src/auth/*.py"}}
  ]
}

BAD (too few tools):
{
  "thought": "Search for login",
  "tool_calls": [{"name": "grep", "args": {"query": "login", "path": "."}}]
}

## Examples of good search terms
- Function search: "def login", "func Marshal", "class UserService"
- Error search: "raise TimeoutError", "return err"
- Config search: "timeout =", "max_connections"

## Tool descriptions
- grep(query, path): Search for text pattern in files
- read(file, start, end): Read lines from a file
- glob(pattern): List files matching pattern"""


# 查询模板 - 更多样化
QUERY_TEMPLATES = [
    # Function/Class location
    "Find the implementation of {func} function",
    "Where is the {cls} class defined",
    "Locate the {func} method in the codebase",
    "Search for {func} implementation",
    
    # Feature understanding
    "How does {feature} work in this project",
    "Find code that handles {action}",
    "Where is {feature} implemented",
    
    # Error handling
    "Find where {error} is raised",
    "Locate {error} handling code",
    "Search for {error} exception",
    
    # Configuration
    "Find {config} configuration",
    "Where is {config} setting defined",
    
    # Multi-target
    "Compare {func1} and {func2} implementations",
    "Find all usages of {func}",
    "List all {pattern} in the codebase",
    
    # Bug fixing scenarios
    "Find the bug in {feature} handling",
    "Debug {action} issue",
    "Fix {error} in {feature}",
]

FILL_WORDS = {
    "func": ["Marshal", "Unmarshal", "Encode", "Decode", "Parse", "Serialize",
             "Connect", "Send", "Receive", "Handle", "Process", "Validate",
             "authenticate", "authorize", "login", "logout", "register",
             "cache_get", "cache_set", "query", "execute", "commit", "rollback",
             "read_file", "write_file", "delete", "update", "create", "fetch"],
    "func1": ["sync", "encode", "read", "get", "push"],
    "func2": ["async", "decode", "write", "set", "pull"],
    "cls": ["Client", "Server", "Handler", "Manager", "Controller", "Service",
            "Repository", "Factory", "Builder", "Adapter", "Proxy", "Observer",
            "Connection", "Session", "Request", "Response", "Config", "Logger"],
    "feature": ["JSON parsing", "HTTP handling", "authentication", "caching",
                "logging", "error handling", "database connection", "API routing",
                "WebSocket", "file upload", "session management", "rate limiting",
                "input validation", "output formatting", "request timeout"],
    "action": ["request timeout", "connection error", "data validation",
               "user authentication", "file upload", "cache invalidation",
               "database query", "API response", "error recovery", "retry logic"],
    "error": ["TimeoutError", "ConnectionError", "ValidationError",
              "AuthenticationError", "NotFoundError", "PermissionError",
              "DatabaseError", "NetworkError", "ParseError", "ConfigError"],
    "config": ["database", "cache", "logging", "server", "client", "timeout",
               "retry", "pool_size", "max_connections", "buffer_size"],
    "pattern": ["handlers", "middleware", "routes", "models", "services", "utils"],
}


def generate_query() -> str:
    """Generate a random query"""
    template = random.choice(QUERY_TEMPLATES)
    
    for key, values in FILL_WORDS.items():
        placeholder = "{" + key + "}"
        if placeholder in template:
            template = template.replace(placeholder, random.choice(values))
    
    return template


def create_prompt(query: str, scenario_hint: str = None) -> str:
    """Create user prompt for LLM"""
    hint = ""
    if scenario_hint:
        hint = f"\nScenario hint: {scenario_hint}"
    
    return f"""Generate a realistic code search trajectory for this query:

Query: {query}
{hint}
Requirements:
- Generate realistic file paths (e.g., src/auth/login.py, internal/encoder/json.go)
- Use meaningful search terms related to the query
- Include realistic code snippets in results
- Decide if parallel or sequential search is appropriate

Output the trajectory as JSON."""


def extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response"""
    import re
    
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    
    # Try ```json ... ```
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # Try ``` ... ```
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # Find first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    
    return None


def validate_trajectory(result: dict) -> tuple:
    """Validate generated trajectory"""
    issues = []
    
    if "trajectory" not in result:
        issues.append("Missing 'trajectory' field")
        return False, issues
    
    traj = result["trajectory"]
    
    if "rounds" not in traj or not isinstance(traj["rounds"], list):
        issues.append("Missing or invalid 'rounds'")
    elif len(traj["rounds"]) == 0:
        issues.append("Empty rounds")
    elif len(traj["rounds"]) > 4:
        issues.append("Too many rounds (max 4)")
    else:
        for i, r in enumerate(traj["rounds"]):
            if "thought" not in r:
                issues.append(f"Round {i}: missing 'thought'")
            if "tool_calls" not in r:
                issues.append(f"Round {i}: missing 'tool_calls'")
            elif len(r["tool_calls"]) > 8:
                issues.append(f"Round {i}: too many tool_calls (max 8)")
    
    if "final" not in traj:
        issues.append("Missing 'final' result")
    
    return len(issues) == 0, issues


def trajectory_to_sft(data: dict) -> dict:
    """Convert to Qwen standard FC format for SFT training"""
    messages = []
    query = data.get("query", "")
    traj = data.get("trajectory", {})
    
    # System message with tools definition
    tools = [
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search for text pattern in files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search pattern"},
                        "path": {"type": "string", "description": "Search path"}
                    },
                    "required": ["query", "path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read lines from a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "File path"},
                        "start": {"type": "integer", "description": "Start line"},
                        "end": {"type": "integer", "description": "End line"}
                    },
                    "required": ["file", "start", "end"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "List files matching pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern"}
                    },
                    "required": ["pattern"]
                }
            }
        }
    ]
    
    # User message
    messages.append({"role": "user", "content": query})
    
    call_id_counter = 0
    
    # Each round
    for round_data in traj.get("rounds", []):
        thought = round_data.get("thought", "")
        tool_calls_raw = round_data.get("tool_calls", [])
        results = round_data.get("results", [])
        
        # Convert to standard FC format
        tool_calls = []
        for tc in tool_calls_raw:
            call_id_counter += 1
            tool_calls.append({
                "id": f"call_{call_id_counter}",
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False)
                }
            })
        
        # Assistant message with tool_calls (Qwen FC format)
        messages.append({
            "role": "assistant",
            "content": thought if thought else None,  # thought as content
            "tool_calls": tool_calls
        })
        
        # Tool results (one per tool call)
        for i, tc in enumerate(tool_calls):
            result_content = results[i] if i < len(results) else ""
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_content if isinstance(result_content, str) else json.dumps(result_content, ensure_ascii=False)
            })
    
    # Final result as regular assistant message
    final = traj.get("final", {})
    messages.append({
        "role": "assistant",
        "content": json.dumps({"result": final}, ensure_ascii=False)
    })
    
    return {
        "messages": messages,
        "tools": tools,
        "_query": query,
        "_is_parallel": len(traj.get("rounds", [{}])[0].get("tool_calls", [])) > 1
    }


def synthesize_batch(
    num_samples: int,
    output_path: str,
    provider: str = "siliconflow",
    model: str = None,
    resume: bool = False,
):
    """Synthesize training data using LLM"""
    
    # API setup
    config = API_CONFIGS[provider]
    model = model or config["default_model"]
    api_key = os.environ.get(config["env_key"])
    
    if not api_key:
        print(f"[Error] Set {config['env_key']} environment variable")
        return
    
    # 根据 provider 选择客户端
    if provider == "anthropic":
        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=config["base_url"]
        )
    else:
        client = OpenAI(api_key=api_key, base_url=config["base_url"])
    
    print(f"[Config] Provider: {provider}, Model: {model}")
    print(f"[Target] {num_samples} samples")
    
    # Load existing results for resume
    results = []
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[Resume] Loaded {len(results)} existing samples")
    
    # Scenario hints for diversity
    scenarios = [
        "Go project with internal/ and pkg/ structure",
        "Python Django project with apps/",
        "TypeScript Node.js project with src/",
        "React frontend with components/",
        "Microservices with multiple services/",
        "CLI tool with cmd/ and internal/",
        None, None, None,  # No hint for variety
    ]
    
    valid_count = len(results)
    failed_count = 0
    
    pbar = tqdm(total=num_samples, initial=len(results), desc="Synthesizing")
    
    while valid_count < num_samples:
        query = generate_query()
        scenario = random.choice(scenarios)
        user_prompt = create_prompt(query, scenario)
        
        try:
            # 根据 provider 使用不同的 API 格式
            if provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=2000,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                )
                result_text = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.8,
                    max_tokens=2000,
                )
                result_text = response.choices[0].message.content
            result = extract_json(result_text)
            
            if result is None:
                failed_count += 1
                continue
            
            # Add query if not present
            if "query" not in result:
                result["query"] = query
            
            # Validate
            is_valid, issues = validate_trajectory(result)
            
            if is_valid:
                # Convert to SFT format
                sft_data = trajectory_to_sft(result)
                results.append(sft_data)
                valid_count += 1
                pbar.update(1)
                
                # Save every 10 samples
                if valid_count % 10 == 0:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
            else:
                failed_count += 1
        
        except Exception as e:
            print(f"\n[Error] {e}")
            failed_count += 1
    
    pbar.close()
    
    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[Done]")
    print(f"  Valid: {valid_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output: {output_path}")
    
    # Stats
    parallel_count = sum(1 for r in results if r.get("_is_parallel", False))
    print(f"  Parallel: {parallel_count} ({parallel_count*100//len(results)}%)")
    print(f"  Sequential: {len(results) - parallel_count} ({(len(results)-parallel_count)*100//len(results)}%)")


def main():
    parser = argparse.ArgumentParser(description="Synthesize code search training data")
    parser.add_argument("--num", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/code_search/sft_data.json")
    parser.add_argument("--provider", type=str, default="siliconflow",
                        choices=list(API_CONFIGS.keys()))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing file")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    synthesize_batch(
        num_samples=args.num,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
