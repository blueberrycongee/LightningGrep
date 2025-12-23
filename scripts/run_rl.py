"""
一键 RL 训练脚本
自动下载 SWE-Bench 数据 + 按需克隆仓库 + 开始训练

使用方法:
    python scripts/run_rl.py --sft_model outputs/sft_v2
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model

from src.environment.code_search_env import CodeSearchEnv, TOOLS_DEFINITION


# ========== 数据下载 ==========

def load_swebench_data(split: str = "lite", cache_dir: str = "data/swebench"):
    """从 HuggingFace 加载 SWE-Bench 数据"""
    cache_file = Path(cache_dir) / f"{split}.json"
    
    # 优先使用本地缓存
    if cache_file.exists():
        print(f"  从缓存加载: {cache_file}")
        with open(cache_file, "r") as f:
            return json.load(f)
    
    # 检查预处理的数据文件
    local_files = {
        "lite": Path(cache_dir) / "lite_all.json",
        "verified": Path(cache_dir) / "verified_all.json",
    }
    if split in local_files and local_files[split].exists():
        print(f"  从本地文件加载: {local_files[split]}")
        with open(local_files[split], "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # 转换字段名（兼容不同格式）
        data = []
        for item in raw_data:
            data.append({
                "instance_id": item.get("instance_id", ""),
                "repo": item.get("repo", ""),
                "base_commit": item.get("base_commit", ""),
                "problem_statement": item.get("problem_statement") or item.get("query", ""),
                "patch": item.get("patch", ""),
                "hints_text": item.get("hints_text", ""),
            })
        return data
    
    print(f"  从 HuggingFace 下载 SWE-Bench {split}...")
    
    # 尝试使用镜像（国内访问）
    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    except:
        pass
    
    try:
        if split == "lite":
            dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        else:
            dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
    except Exception as e:
        print(f"\n  ⚠️ 无法下载数据集: {e}")
        print(f"  请手动下载并放到: {cache_file}")
        print(f"  或设置环境变量: export HF_ENDPOINT=https://hf-mirror.com")
        raise
    
    data = []
    for item in dataset:
        data.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "hints_text": item.get("hints_text", ""),
        })
    
    # 缓存
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"  缓存到: {cache_file}")
    return data


def parse_patch_files(patch: str) -> List[str]:
    """从 patch 中提取修改的文件列表"""
    files = []
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            # diff --git a/path/to/file b/path/to/file
            parts = line.split()
            if len(parts) >= 4:
                file_path = parts[2][2:]  # 去掉 "a/"
                files.append(file_path)
    return files


def parse_patch_lines(patch: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    从 patch 中提取修改的文件和行范围
    
    Returns:
        {file_path: [(start_line, end_line), ...]}
    """
    import re
    
    result = {}
    current_file = None
    
    for line in patch.split("\n"):
        # 新文件
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[2][2:]  # 去掉 "a/"
                result[current_file] = []
        
        # 行范围: @@ -42,6 +42,8 @@
        elif line.startswith("@@") and current_file:
            match = re.search(r'\+(\d+)(?:,(\d+))?', line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                end = start + count - 1
                result[current_file].append((start, end))
    
    return result


# ========== 仓库管理 ==========

class RepoManager:
    """按需克隆和管理仓库"""
    
    def __init__(self, cache_dir: str = "data/swebench/repos"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cloned_repos = set()
        self._scan_existing()
    
    def _scan_existing(self):
        """扫描已克隆的仓库"""
        if not self.cache_dir.exists():
            return
        for repo_dir in self.cache_dir.iterdir():
            if repo_dir.is_dir() and (repo_dir / ".git").exists():
                self.cloned_repos.add(repo_dir.name)
    
    def get_repo_path(self, repo: str, commit: str) -> Path:
        """获取仓库路径，如果不存在则克隆"""
        # repo 格式: "owner/name"
        repo_name = repo.replace("/", "__")
        repo_path = self.cache_dir / repo_name
        
        if repo_name not in self.cloned_repos:
            self._clone_repo(repo, repo_path)
            self.cloned_repos.add(repo_name)
        
        # 切换到指定 commit
        self._checkout_commit(repo_path, commit)
        
        return repo_path
    
    def _clone_repo(self, repo: str, repo_path: Path):
        """克隆仓库（支持镜像和重试，显示进度）"""
        import shutil
        import time
        import re
        
        # GitHub 镜像列表（国内优先）
        mirrors = [
            f"https://ghproxy.com/https://github.com/{repo}.git",  # ghproxy 镜像（默认）
            f"https://gitclone.com/github.com/{repo}.git",  # gitclone 镜像
            f"https://github.com/{repo}.git",  # 原始 GitHub（最后尝试）
        ]
        
        mirror_names = ["ghproxy镜像", "gitclone镜像", "GitHub原始"]
        
        print(f"  克隆仓库: {repo}...")
        
        for i, url in enumerate(mirrors):
            mirror_name = mirror_names[i] if i < len(mirror_names) else f"镜像{i}"
            if i > 0:
                print(f"  尝试 {mirror_name}...")
            
            # 清理之前失败的目录
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            
            try:
                # 使用 Popen 实时显示进度
                process = subprocess.Popen(
                    ["git", "clone", "--depth", "100", "--progress", url, str(repo_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                
                start_time = time.time()
                last_print_time = 0
                
                # 实时读取 stderr（git 进度信息在 stderr）
                while True:
                    # 非阻塞读取
                    line = process.stderr.readline()
                    if not line and process.poll() is not None:
                        break
                    
                    if line:
                        line = line.strip()
                        # 解析进度信息
                        # 格式: "Receiving objects:  45% (1234/2745), 12.50 MiB | 1.25 MiB/s"
                        if "Receiving" in line or "Compressing" in line or "Resolving" in line:
                            current_time = time.time()
                            # 每秒最多更新一次
                            if current_time - last_print_time >= 0.5:
                                # 提取百分比和速度
                                match = re.search(r'(\d+)%.*?\|\s*([\d.]+\s*\w+/s)', line)
                                if match:
                                    percent = match.group(1)
                                    speed = match.group(2)
                                    elapsed = int(current_time - start_time)
                                    print(f"\r    进度: {percent}% | 速度: {speed} | 耗时: {elapsed}s", end="", flush=True)
                                else:
                                    # 没有速度信息，只显示百分比
                                    match = re.search(r'(\d+)%', line)
                                    if match:
                                        percent = match.group(1)
                                        elapsed = int(current_time - start_time)
                                        print(f"\r    进度: {percent}% | 耗时: {elapsed}s", end="", flush=True)
                                last_print_time = current_time
                
                # 等待进程结束
                return_code = process.wait(timeout=300)
                print()  # 换行
                
                if return_code == 0:
                    elapsed = int(time.time() - start_time)
                    print(f"  ✅ 克隆成功 (耗时 {elapsed}s)")
                    return
                else:
                    stderr = process.stderr.read()
                    print(f"  ⚠️ 失败: {stderr[:100]}...")
                    
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"\n  ⚠️ 超时 (>5分钟)，尝试下一个镜像...")
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
                continue
                
            except Exception as e:
                print(f"\n  ⚠️ 异常: {str(e)[:100]}...")
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
                continue
        
        # 所有镜像都失败
        raise RuntimeError(f"无法克隆仓库 {repo}，请检查网络或手动克隆到 {repo_path}")
    
    def _checkout_commit(self, repo_path: Path, commit: str):
        """切换到指定 commit"""
        try:
            subprocess.run(
                ["git", "checkout", commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            # 可能需要 fetch 更多历史
            print(f"    commit 不在浅克隆中，尝试获取完整历史...")
            
            # 方法1: unshallow
            result = subprocess.run(
                ["git", "fetch", "--unshallow"],
                cwd=repo_path,
                capture_output=True,
            )
            
            # 方法2: 如果 unshallow 失败，尝试 fetch 特定 commit
            if result.returncode != 0:
                subprocess.run(
                    ["git", "fetch", "origin", commit],
                    cwd=repo_path,
                    capture_output=True,
                )
            
            # 再次尝试 checkout
            try:
                subprocess.run(
                    ["git", "checkout", commit],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                # 最后尝试：完全重新克隆（不用浅克隆）
                print(f"    ⚠️ 无法 checkout {commit[:12]}，尝试完整克隆...")
                import shutil
                repo_name = repo_path.name
                repo = repo_name.replace("__", "/")
                shutil.rmtree(repo_path, ignore_errors=True)
                self.cloned_repos.discard(repo_name)
                
                # 完整克隆（不用 --depth）
                subprocess.run(
                    ["git", "clone", f"https://github.com/{repo}.git", str(repo_path)],
                    check=True,
                    capture_output=True,
                    timeout=600,
                )
                self.cloned_repos.add(repo_name)
                
                # 最后一次尝试
                subprocess.run(
                    ["git", "checkout", commit],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                )


# ========== SFT 格式 Prompt 构造（和训练时一致）==========

def format_sft_prompt(
    messages: List[Dict], 
    tools: List[Dict] = None,
    add_generation_prompt: bool = True
) -> str:
    """
    将 messages 转换为 SFT 训练时的格式
    必须和 sft_qlora.py 中的 format_fc_messages 保持一致
    """
    parts = []
    
    # 添加 tools 定义（如果有）
    if tools:
        tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
        parts.append(f"<|im_start|>system\nYou are a code search agent. Available tools:\n{tools_str}<|im_end|>")
    
    for msg in messages:
        role = msg["role"]
        
        if role == "system":
            # 自定义 system prompt
            parts.append(f"<|im_start|>system\n{msg['content']}<|im_end|>")
        
        elif role == "user":
            parts.append(f"<|im_start|>user\n{msg['content']}<|im_end|>")
        
        elif role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            
            if tool_calls:
                # 有 tool calls 的 assistant 消息
                tc_str = json.dumps(tool_calls, ensure_ascii=False)
                if content:
                    parts.append(f"<|im_start|>assistant\n{content}\n<tool_calls>\n{tc_str}\n</tool_calls><|im_end|>")
                else:
                    parts.append(f"<|im_start|>assistant\n<tool_calls>\n{tc_str}\n</tool_calls><|im_end|>")
            else:
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        elif role == "tool":
            # Tool 结果
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>tool\n<tool_call_id>{tool_call_id}</tool_call_id>\n{content}<|im_end|>")
    
    result = "\n".join(parts)
    
    # 添加生成提示
    if add_generation_prompt:
        result += "\n<|im_start|>assistant\n"
    
    return result


# ========== 奖励计算（使用 env.compute_reward，支持文件+行级 F1）==========


def get_repo_tree(repo_path: Path, max_depth: int = 2, max_items: int = 50) -> str:
    """
    获取仓库目录结构（用于 prompt）
    
    主代理会先分析仓库结构，然后传给子代理
    """
    lines = []
    count = [0]  # 用列表来在闭包中修改
    
    def walk(path: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth or count[0] >= max_items:
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        except PermissionError:
            return
        
        # 过滤掉常见的无关目录
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.tox', 'dist', 'build', '.eggs'}
        items = [x for x in items if x.name not in skip_dirs]
        
        for i, item in enumerate(items):
            if count[0] >= max_items:
                lines.append(f"{prefix}... (truncated)")
                return
            
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            
            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                count[0] += 1
                new_prefix = prefix + ("    " if is_last else "│   ")
                walk(item, new_prefix, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{item.name}")
                count[0] += 1
    
    lines.append(f"{repo_path.name}/")
    walk(repo_path)
    
    return "\n".join(lines)


def parse_answer_format(text: str) -> List[Dict]:
    """
    解析 <answer> 格式的输出
    
    格式:
    <answer>
    1. path/to/file.py:10-25
    2. path/to/other.py:100-120
    </answer>
    
    Returns:
        [{"file": "path/to/file.py", "start_line": 10, "end_line": 25}, ...]
    """
    import re
    
    results = []
    
    # 提取 <answer>...</answer> 内容
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not answer_match:
        return results
    
    answer_content = answer_match.group(1).strip()
    
    # 解析每一行
    # 格式: 1. path/to/file.py:10-25  或  path/to/file.py:10-25
    line_pattern = r'(?:\d+\.\s*)?([^:\s]+):(\d+)-(\d+)'
    
    for match in re.finditer(line_pattern, answer_content):
        file_path = match.group(1).strip()
        start_line = int(match.group(2))
        end_line = int(match.group(3))
        
        results.append({
            "file": file_path,
            "start_line": start_line,
            "end_line": end_line
        })
        
        # 最多 8 个
        if len(results) >= 8:
            break
    
    return results


def compute_answer_reward(
    answer: List[Dict],
    ground_truth_files: List[str],
    ground_truth_lines: Dict[str, List[Tuple[int, int]]] = None,
    beta: float = 0.5
) -> float:
    """
    计算基于 <answer> 的 reward
    
    博客设计: reward = (file_F1 + line_F1) / 2
    使用 β=0.5，Precision 权重是 Recall 的 2 倍
    
    Args:
        answer: [{"file": "path", "start_line": 10, "end_line": 20}, ...]
        ground_truth_files: 真实文件列表
        ground_truth_lines: 真实行范围 {file: [(start, end), ...]}
        beta: F-beta 的 beta 值
    
    Returns:
        reward: 0.0 到 1.0
    """
    if not answer:
        return 0.0
    
    # 提取提交的文件和行
    submitted_files = set()
    submitted_lines = {}
    
    for r in answer:
        file = r["file"]
        submitted_files.add(file)
        
        if file not in submitted_lines:
            submitted_lines[file] = []
        submitted_lines[file].append((r["start_line"], r["end_line"]))
    
    # 1. 文件级 F-beta
    gt_file_set = set(ground_truth_files)
    correct_files = len(gt_file_set & submitted_files)
    
    precision = correct_files / len(submitted_files) if submitted_files else 0
    recall = correct_files / len(gt_file_set) if gt_file_set else 0
    
    if precision + recall == 0:
        file_f1 = 0.0
    else:
        beta_sq = beta ** 2
        file_f1 = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    
    # 2. 行级 F-beta（如果有 ground truth）
    if ground_truth_lines:
        gt_lines = set()
        for file, ranges in ground_truth_lines.items():
            for start, end in ranges:
                for line in range(start, end + 1):
                    gt_lines.add((file, line))
        
        found_lines = set()
        for file, ranges in submitted_lines.items():
            for start, end in ranges:
                for line in range(start, end + 1):
                    found_lines.add((file, line))
        
        if not found_lines:
            line_f1 = 0.0
        else:
            correct_lines = len(gt_lines & found_lines)
            
            precision = correct_lines / len(found_lines) if found_lines else 0
            recall = correct_lines / len(gt_lines) if gt_lines else 0
            
            if precision + recall == 0:
                line_f1 = 0.0
            else:
                beta_sq = beta ** 2
                line_f1 = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
        
        # 博客: 文件和行的平均
        return (file_f1 + line_f1) / 2
    
    return file_f1


# ========== SWE-grep 风格 RL 训练 ==========
# 完整实现 Cognition/Windsurf 博客描述的技术

@dataclass
class Trajectory:
    """一条搜索轨迹"""
    instance_id: str
    query: str
    messages: List[Dict]           # 完整对话历史
    files_found: List[str]
    ground_truth: List[str]
    reward: float
    turn_data: List[Dict]          # 每轮的 input_ids 和 generated_ids（用于后续计算 log_prob）
    total_tokens: int              # 总 token 数
    tool_calls_count: int          # 工具调用总数
    turns: int                     # 实际轮数
    format_error: bool = False     # 是否有格式错误
    submitted_answer: List[Dict] = None  # 模型提交的答案


class SWEGrepTrainer:
    """
    SWE-grep 风格的 REINFORCE 训练器
    
    完整实现博客提到的技术：
    1. 4 turns × 8 parallel tool calls
    2. Weighted F1 (β=0.5) reward
    3. Per-sequence importance sampling (简化版)
    4. Leave-one-out baseline
    5. 按工具调用数 scaling advantage
    6. 格式错误/过长轨迹 mask from loss
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        repo_manager: RepoManager,
        learning_rate: float = 1e-5,
        num_rollouts: int = 4,      # 博客: g completions from same prompt
        max_turns: int = 4,          # 博客: 4 turns (3 exploration + 1 answer)
        max_parallel_calls: int = 8, # 博客: up to 8 parallel tool calls
        temperature: float = 0.7,
        max_traj_length: int = 4096, # 博客: T_max
        grad_clip: float = 1.0,
        scale_by_tool_calls: bool = True,  # 博客: scaling advantages
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.repo_manager = repo_manager
        self.num_rollouts = num_rollouts
        self.max_turns = max_turns
        self.max_parallel_calls = max_parallel_calls
        self.temperature = temperature
        self.max_traj_length = max_traj_length
        self.grad_clip = grad_clip
        self.scale_by_tool_calls = scale_by_tool_calls
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
        )
        
        # 统计
        self.total_steps = 0
        self.reward_history = []
    
    def collect_trajectory(self, instance: Dict) -> Trajectory:
        """
        收集一条轨迹（不计算梯度，只采样）
        
        博客方法：先采样轨迹，记录 token ids，之后统一计算 log_prob
        """
        repo_path = self.repo_manager.get_repo_path(
            instance["repo"],
            instance["base_commit"]
        )
        
        env = CodeSearchEnv(
            str(repo_path), 
            max_turns=self.max_turns,
            max_parallel_calls=self.max_parallel_calls
        )
        
        ground_truth_files = parse_patch_files(instance["patch"])
        ground_truth_lines = parse_patch_lines(instance["patch"])
        repo_tree = get_repo_tree(repo_path)
        
        # System prompt + User query
        system_prompt = """You are a code search agent. Your task is to find relevant files and code locations.

Available tools:
- grep: Search for text patterns in files
- read: Read specific lines from a file  
- glob: List files matching a pattern
- find: Find files by name

Rules:
1. You have 4 turns maximum: use turns 1-3 for exploration, turn 4 for your answer
2. You can make up to 8 parallel tool calls per turn
3. After exploring, provide your final answer in the <answer> format

Answer format (provide in your final response):
<answer>
1. path/to/file.py:10-25
2. path/to/other.py:100-120
3. path/to/another.py:50-80
</answer>

Rules for your answer:
- Maximum 8 locations, ordered by importance (most important first)
- Format: filepath:start_line-end_line
- Only include locations you are confident about"""
        
        user_prompt = f"""Repository: {instance["repo"]}

Directory structure:
```
{repo_tree}
```

Issue:
{instance.get("problem_statement") or instance.get("query", "")}

Find the relevant files and code locations. After exploring with tools, provide your answer in the <answer> format."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 收集每轮的信息
        turn_data = []  # [(input_ids, generated_ids), ...]
        total_tokens = 0
        tool_calls_count = 0
        format_error = False
        actual_turns = 0
        
        for turn in range(self.max_turns):
            actual_turns = turn + 1
            
            # 构建输入（使用 SFT 训练时的格式）
            prompt = format_sft_prompt(
                messages,
                tools=TOOLS_DEFINITION if turn == 0 else None,  # 只在第一轮加 tools
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_ids = inputs["input_ids"]
            
            # 生成（不计算梯度）
            # Qwen3 会先 <think> 思考，需要足够的 token 空间
            # 不限制长度，由 max_traj_length 控制总轨迹长度
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=8192,  # 足够容纳思考 + 工具调用
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_ids = outputs.sequences[0][input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            
            # 调试：打印模型输出
            if hasattr(self, 'debug') and self.debug and turn == 0:
                print(f"\n  [DEBUG] 模型输出 (前500字符):\n{response[:500]}\n")
            
            # 保存这轮的输入和生成（用于后续计算 log_prob）
            turn_data.append({
                "input_ids": input_ids.clone(),
                "generated_ids": generated_ids.clone(),
            })
            total_tokens += len(generated_ids)
            
            # 检查是否超长（博客: mask overlong trajectories）
            if total_tokens > self.max_traj_length:
                format_error = True
                break
            
            # 解析 tool calls
            tool_calls = self._parse_tool_calls(response)
            
            # 检查是否有 <answer> 格式的输出（最后一轮可以没有 tool call）
            parsed_answer = parse_answer_format(response)
            
            if not tool_calls:
                if parsed_answer:
                    # 有 <answer> 格式，这是最终答案
                    messages.append({"role": "assistant", "content": response})
                    break
                elif turn < self.max_turns - 1:
                    # 非最后一轮没有 tool call 也没有 answer = 格式错误
                    format_error = True
                break
            
            tool_calls_count += len(tool_calls)
            
            # 执行 tool calls
            results, done = env.step(tool_calls)
            
            # 更新消息（使用 SFT 格式）
            # 提取 <tool_calls> 之前的内容作为 content
            content_before_tools = response.split("<tool_calls>")[0].strip()
            
            # 构造 SFT 格式的 tool_calls
            sft_tool_calls = []
            for i, tc in enumerate(tool_calls):
                sft_tool_calls.append({
                    "id": tc.get("id", f"call_{i+1}"),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"], ensure_ascii=False)
                    }
                })
            
            messages.append({
                "role": "assistant", 
                "content": content_before_tools,
                "tool_calls": sft_tool_calls
            })
            
            # 添加 tool 结果（带 tool_call_id）
            for i, result in enumerate(results):
                messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_calls[i].get("id", f"call_{i+1}") if i < len(tool_calls) else "",
                    "content": result["content"]
                })
            
            # 检查是否有 <answer>（可以在任何轮次提交答案）
            if parsed_answer:
                break
            
            if done:
                break
        
        # 从所有消息中提取最终答案
        final_answer = []
        for msg in messages:
            if msg.get("role") == "assistant":
                answer = parse_answer_format(msg.get("content", ""))
                if answer:
                    final_answer = answer  # 取最后一个有效答案
        
        # 计算 reward（博客: weighted F1, β=0.5）
        if format_error:
            reward = 0.0
        elif final_answer:
            # 有 <answer> 格式，使用它
            reward = compute_answer_reward(final_answer, ground_truth_files, ground_truth_lines, beta=0.5)
        else:
            # 没有 <answer>，使用隐式收集的结果（兼容模式）
            # 这样模型可以逐渐学习
            reward = env.compute_reward(ground_truth_files, ground_truth_lines, beta=0.5, use_submission=False)
        
        # 注意：log_prob 在 train_step 中计算，这里只保存 turn_data
        # 这样确保在 train 模式下计算梯度
        
        problem = instance.get("problem_statement") or instance.get("query", "")
        return Trajectory(
            instance_id=instance["instance_id"],
            query=problem[:100],
            messages=messages,
            files_found=list(env.files_found),
            ground_truth=ground_truth_files,
            reward=reward,
            turn_data=turn_data,  # 保存用于后续计算 log_prob
            total_tokens=total_tokens,
            tool_calls_count=tool_calls_count,
            turns=actual_turns,
            format_error=format_error,
            submitted_answer=final_answer,
        )
    
    def _parse_tool_calls(self, response: str) -> List[Dict]:
        """
        解析工具调用（匹配 SFT 训练时的格式）
        
        SFT 格式:
        <tool_calls>
        [{"id": "call_1", "type": "function", "function": {"name": "grep", "arguments": "{...}"}}]
        </tool_calls>
        """
        import re
        
        tool_calls = []
        
        # 提取 <tool_calls>...</tool_calls> 内容
        pattern = r'<tool_calls>\s*(.*?)\s*</tool_calls>'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            try:
                # 解析 JSON 数组
                calls_json = match.group(1).strip()
                calls = json.loads(calls_json)
                
                if isinstance(calls, list):
                    for call in calls:
                        # SFT 格式: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
                        if "function" in call:
                            func = call["function"]
                            name = func.get("name", "")
                            args_str = func.get("arguments", "{}")
                            try:
                                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                            except json.JSONDecodeError:
                                args = {}
                            tool_calls.append({
                                "id": call.get("id", f"call_{len(tool_calls)+1}"),
                                "name": name,
                                "arguments": args
                            })
            except json.JSONDecodeError:
                pass
        
        return tool_calls[:8]  # 最多 8 个并行调用
    
    def train_step(self, instances: List[Dict]) -> Dict:
        """
        一步训练（博客: REINFORCE with per-sequence importance sampling）
        
        博客公式:
        L = sum_j [ A_j * sum_t log π(a_t|s_t) ]
        其中 A_j = R_j - baseline_j (leave-one-out)
        
        稳定性技巧:
        1. 格式错误轨迹 mask from loss
        2. 过长轨迹 mask from loss  
        3. 按工具调用数 scaling advantages
        4. 梯度裁剪
        """
        all_trajectories = []
        
        # 1. 收集轨迹（采样阶段，不需要梯度）
        self.model.eval()
        for instance in instances:
            for _ in range(self.num_rollouts):
                try:
                    traj = self.collect_trajectory(instance)
                    all_trajectories.append(traj)
                except Exception as e:
                    print(f"  ⚠️ Episode 失败: {e}")
                    continue
        
        if not all_trajectories:
            return {"loss": 0.0, "reward": 0.0, "num_trajectories": 0}
        
        # 2. 过滤有效轨迹（博客: mask from loss）
        valid_trajectories = [
            t for t in all_trajectories 
            if not t.format_error and t.turn_data and t.total_tokens <= self.max_traj_length
        ]
        
        if not valid_trajectories:
            avg_reward = sum(t.reward for t in all_trajectories) / len(all_trajectories)
            # 调试：打印为什么没有有效轨迹
            if hasattr(self, 'debug') and self.debug:
                for i, t in enumerate(all_trajectories):
                    print(f"  [DEBUG] 轨迹 {i}: format_error={t.format_error}, turn_data={len(t.turn_data)}, tokens={t.total_tokens}, reward={t.reward:.3f}")
            return {"loss": 0.0, "reward": avg_reward, "num_trajectories": len(all_trajectories), "valid": 0}
        
        # 3. 计算 rewards
        rewards = torch.tensor(
            [t.reward for t in valid_trajectories], 
            device=self.model.device,
            dtype=torch.float32
        )
        
        # 4. Leave-one-out baseline（博客公式）
        n = len(rewards)
        if n > 1:
            total_reward = rewards.sum()
            baselines = (total_reward - rewards) / (n - 1)
            advantages = rewards - baselines
        else:
            advantages = rewards - rewards.mean()
        
        # 5. 按 token 数 normalize（替代按工具调用数 scaling，更稳定）
        # 这样长轨迹和短轨迹的梯度贡献更均衡
        token_counts = torch.tensor(
            [t.total_tokens for t in valid_trajectories],
            device=self.model.device,
            dtype=torch.float32
        )
        
        # 6. 标准化 advantages
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages = advantages.detach()
        
        # 7. 切换到 train 模式，计算 log_prob 和 loss
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        
        for traj, advantage, num_tokens in zip(valid_trajectories, advantages, token_counts):
            # 在 train 模式下计算 log_prob
            traj_log_prob = torch.tensor(0.0, device=self.model.device, requires_grad=True)
            
            for data in traj.turn_data:
                input_ids = data["input_ids"]
                generated_ids = data["generated_ids"]
                input_len = input_ids.shape[1]
                
                # 拼接完整序列
                full_ids = torch.cat([input_ids[0], generated_ids]).unsqueeze(0)
                
                # Forward pass 计算 logits（在 train 模式下）
                logits = self.model(full_ids).logits
                
                # 取生成部分的 log_prob
                gen_logits = logits[0, input_len-1:-1, :]
                log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)
                token_log_probs = log_probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)
                
                # 使用 mean 而不是 sum，避免溢出
                traj_log_prob = traj_log_prob + token_log_probs.mean()
            
            # REINFORCE: -A * log π
            # 使用 mean log_prob，不需要再按 token 数 normalize
            loss = -advantage * traj_log_prob
            total_loss = total_loss + loss
        
        total_loss = total_loss / len(valid_trajectories)
        
        # 8. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 9. 梯度裁剪（博客: 稳定性）
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # 10. 更新参数
        self.optimizer.step()
        
        self.total_steps += 1
        
        # 统计
        all_rewards = [t.reward for t in all_trajectories]
        format_error_rate = sum(1 for t in all_trajectories if t.format_error) / len(all_trajectories)
        avg_tool_calls = sum(t.tool_calls_count for t in all_trajectories) / len(all_trajectories)
        
        return {
            "loss": total_loss.item(),
            "reward": sum(all_rewards) / len(all_rewards),
            "reward_valid": rewards.mean().item(),
            "baseline": baselines.mean().item() if n > 1 else 0.0,
            "num_trajectories": len(all_trajectories),
            "valid_trajectories": len(valid_trajectories),
            "format_error_rate": format_error_rate,
            "avg_tool_calls": avg_tool_calls,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }
    
    def train(
        self,
        train_data: List[Dict],
        num_steps: int = 100,
        batch_size: int = 4,
        eval_every: int = 20,
        save_every: int = 50,
        output_dir: str = "outputs/rl_v1",
        start_step: int = 0,
    ):
        """训练循环"""
        from datetime import datetime
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 日志文件（按时间戳区分，避免覆盖）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if start_step == 0:
            # 新训练：创建新的日志文件
            log_file = output_path / f"training_log_{timestamp}.jsonl"
        else:
            # 继续训练：找最新的日志文件或创建新的
            existing_logs = list(output_path.glob("training_log_*.jsonl"))
            if existing_logs:
                log_file = max(existing_logs, key=lambda x: x.stat().st_mtime)
            else:
                log_file = output_path / f"training_log_{timestamp}.jsonl"
        
        print(f"\n开始 RL 训练: {num_steps} steps (从 step {start_step} 开始)")
        print(f"  Batch size: {batch_size}")
        print(f"  Rollouts per instance: {self.num_rollouts}")
        print(f"  日志文件: {log_file}")
        
        step = start_step
        total_reward = 0
        
        pbar = tqdm(total=num_steps, initial=start_step, desc="Training")
        
        while step < num_steps:
            # 随机采样 batch
            import random
            batch = random.sample(train_data, min(batch_size, len(train_data)))
            
            # 训练一步
            stats = self.train_step(batch)
            
            total_reward += stats["reward"]
            step += 1
            
            pbar.update(1)
            pbar.set_postfix({
                "reward": f"{stats['reward']:.3f}",
                "loss": f"{stats.get('loss', 0):.4f}",
                "valid": stats.get("valid_trajectories", 0),
            })
            
            # 记录日志（完整统计信息）
            log_entry = {
                "step": step,
                "reward": stats["reward"],
                "reward_valid": stats.get("reward_valid", stats["reward"]),
                "avg_reward": total_reward / (step - start_step),
                "loss": stats.get("loss", 0),
                "num_trajectories": stats.get("num_trajectories", 0),
                "valid_trajectories": stats.get("valid_trajectories", 0),
                "format_error_rate": stats.get("format_error_rate", 0),
                "avg_tool_calls": stats.get("avg_tool_calls", 0),
                "grad_norm": stats.get("grad_norm", 0),
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # 保存 checkpoint
            if step % save_every == 0:
                save_path = output_path / f"checkpoint-{step}"
                save_path.mkdir(parents=True, exist_ok=True)
                
                # 1. 保存模型
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                
                # 2. 保存 optimizer 状态
                torch.save(self.optimizer.state_dict(), save_path / "optimizer.pt")
                
                # 3. 保存训练状态
                state = {
                    "step": step,
                    "total_reward": total_reward,
                    "avg_reward": total_reward / (step - start_step) if step > start_step else 0,
                }
                with open(save_path / "train_state.json", "w") as f:
                    json.dump(state, f, indent=2)
                
                # 4. 只保留最近 3 个 checkpoint（节省空间）
                all_ckpts = sorted(output_path.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
                for old_ckpt in all_ckpts[:-3]:
                    import shutil
                    shutil.rmtree(old_ckpt)
                
                print(f"\n  保存: {save_path}")
        
        pbar.close()
        
        # 最终保存
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"\n训练完成，模型保存到: {output_path}")


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="一键 RL 训练")
    
    # 模型
    parser.add_argument("--sft_model", type=str, required=True, help="SFT 模型路径")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B", help="基座模型")
    parser.add_argument("--output_dir", type=str, default="outputs/rl_v1", help="输出目录")
    
    # 数据
    parser.add_argument("--split", type=str, default="lite", choices=["lite", "full"], help="SWE-Bench 版本")
    parser.add_argument("--data_file", type=str, default=None, help="自定义数据文件（覆盖 split）")
    parser.add_argument("--max_samples", type=int, default=100, help="最大训练样本数")
    
    # 训练参数
    parser.add_argument("--num_steps", type=int, default=100, help="训练步数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--num_rollouts", type=int, default=4, help="每个样本的 rollout 数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--max_turns", type=int, default=4, help="最大对话轮数")
    parser.add_argument("--max_parallel_calls", type=int, default=8, help="每轮最大并行工具调用数")
    
    # 量化设置
    parser.add_argument("--quantization", type=str, default="4bit", 
                        choices=["4bit", "8bit", "none"], help="量化方式")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"], help="计算精度")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 稳定性参数
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--max_traj_length", type=int, default=32768, help="最大轨迹长度（token数）")
    parser.add_argument("--scale_by_tool_calls", action="store_true", default=True, help="按工具调用数缩放advantage")
    parser.add_argument("--no_scale_by_tool_calls", action="store_false", dest="scale_by_tool_calls")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_every", type=int, default=50, help="每N步保存一次")
    parser.add_argument("--debug", action="store_true", help="调试模式：打印详细信息")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 继续训练，如 outputs/rl_v1/checkpoint-100")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LightningGrep RL 训练")
    print("=" * 60)
    
    # 设置随机种子
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 检查 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 未检测到 GPU，训练会很慢！")
    
    # 打印配置
    print(f"\n配置:")
    print(f"  量化: {args.quantization}")
    print(f"  精度: {args.dtype}")
    print(f"  温度: {args.temperature}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  学习率: {args.learning_rate}")
    
    # 1. 加载数据
    print("\n[1/4] 加载 SWE-Bench 数据...")
    if args.data_file:
        # 使用自定义数据文件
        print(f"  从自定义文件加载: {args.data_file}")
        with open(args.data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = load_swebench_data(args.split)
    
    # 限制样本数
    if args.max_samples and len(data) > args.max_samples:
        data = random.sample(data, args.max_samples)
    
    print(f"  使用 {len(data)} 条数据")
    
    # 2. 初始化仓库管理器
    print("\n[2/4] 初始化仓库管理器...")
    repo_manager = RepoManager()
    print(f"  缓存目录: {repo_manager.cache_dir}")
    print(f"  已缓存仓库: {len(repo_manager.cloned_repos)}")
    
    # 3. 加载模型
    print("\n[3/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    
    # 设置精度
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    compute_dtype = dtype_map[args.dtype]
    
    # 量化配置
    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif args.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:  # none
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # 加载 SFT LoRA 并合并
    model = PeftModel.from_pretrained(base_model, args.sft_model)
    model = model.merge_and_unload()
    
    # 创建新的 LoRA 用于 RL
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 训练
    print("\n[4/4] 开始 RL 训练...")
    
    # 检查是否从 checkpoint 继续
    start_step = 0
    resume_optimizer_path = None
    if args.resume:
        resume_path = Path(args.resume)
        state_file = resume_path / "train_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            start_step = state["step"]
            print(f"  从 checkpoint 继续: step {start_step}")
            
            # 重新加载模型：基座 + checkpoint LoRA（不需要再创建新 LoRA）
            if args.quantization == "4bit":
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            elif args.quantization == "8bit":
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    torch_dtype=compute_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            
            # 先合并 SFT LoRA
            sft_model = PeftModel.from_pretrained(base_model, args.sft_model)
            merged_model = sft_model.merge_and_unload()
            
            # 加载 checkpoint 的 RL LoRA
            model = PeftModel.from_pretrained(merged_model, args.resume)
            model.print_trainable_parameters()
            
            # 检查 optimizer 状态
            if (resume_path / "optimizer.pt").exists():
                resume_optimizer_path = resume_path / "optimizer.pt"
                print(f"  加载 optimizer 状态")
        else:
            print(f"  ⚠️ 未找到 train_state.json，从头开始")
    
    trainer = SWEGrepTrainer(
        model=model,
        tokenizer=tokenizer,
        repo_manager=repo_manager,
        learning_rate=args.learning_rate,
        num_rollouts=args.num_rollouts,
        max_turns=args.max_turns,
        max_parallel_calls=args.max_parallel_calls,
        temperature=args.temperature,
        max_traj_length=args.max_traj_length,
        grad_clip=args.grad_clip,
        scale_by_tool_calls=args.scale_by_tool_calls,
    )
    
    # 加载 optimizer 状态
    if resume_optimizer_path:
        trainer.optimizer.load_state_dict(torch.load(resume_optimizer_path))
    
    # 调试模式
    if args.debug:
        trainer.debug = True
    
    trainer.train(
        train_data=data,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_every=args.save_every,
        start_step=start_step,
    )
    
    # 打印完成信息
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存到: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
