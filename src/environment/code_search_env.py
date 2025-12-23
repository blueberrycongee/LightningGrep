"""
Code Search 环境
用于 RL 训练的交互式代码搜索环境
"""
import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SearchResult:
    """搜索结果"""
    files_found: List[str]
    lines_found: Dict[str, List[Tuple[int, int]]]  # file -> [(start, end), ...]
    tool_calls: int
    turns: int


class CodeSearchEnv:
    """代码搜索环境"""
    
    def __init__(
        self,
        repo_path: str,
        max_turns: int = 4,
        max_parallel_calls: int = 8,
        max_grep_results: int = 50,
        max_read_lines: int = 100,
    ):
        self.repo_path = Path(repo_path)
        self.max_turns = max_turns
        self.max_parallel_calls = max_parallel_calls
        self.max_grep_results = max_grep_results
        self.max_read_lines = max_read_lines
        
        self.current_turn = 0
        self.total_tool_calls = 0
        self.files_found = set()
        self.lines_found = {}
        self.history = []
        self.submitted_answer = None  # 模型提交的最终答案
    
    def reset(self):
        """重置环境"""
        self.current_turn = 0
        self.total_tool_calls = 0
        self.files_found = set()
        self.lines_found = {}
        self.history = []
        self.submitted_answer = None
    
    def step(self, tool_calls: List[Dict]) -> Tuple[List[Dict], bool]:
        """
        执行一轮 tool calls
        
        Args:
            tool_calls: [{"name": "grep", "arguments": {...}}, ...]
        
        Returns:
            results: 每个 tool call 的结果
            done: 是否结束
        """
        if self.current_turn >= self.max_turns:
            return [], True
        
        # 限制并行数
        tool_calls = tool_calls[:self.max_parallel_calls]
        
        results = []
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}
            
            result = self._execute_tool(name, args)
            results.append({
                "tool_call_id": call.get("id", ""),
                "content": result
            })
            self.total_tool_calls += 1
        
        self.current_turn += 1
        self.history.append({"tool_calls": tool_calls, "results": results})
        
        done = self.current_turn >= self.max_turns
        return results, done
    
    def _execute_tool(self, name: str, args: Dict) -> str:
        """执行单个工具"""
        try:
            if name == "grep":
                return self._grep(args.get("query", ""), args.get("path", "."))
            elif name == "read":
                return self._read(
                    args.get("file", ""),
                    args.get("start", 1),
                    args.get("end", 50)
                )
            elif name == "glob":
                return self._glob(args.get("pattern", "*"))
            elif name == "find":
                return self._find(args.get("name", ""))
            elif name == "submit":
                return self._submit(args.get("results", []))
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _submit(self, results: List[Dict]) -> str:
        """
        处理模型提交的最终答案
        
        Args:
            results: [{"file": "path/to/file", "start_line": 10, "end_line": 20}, ...]
        
        Returns:
            确认信息
        """
        # 限制最多 8 个结果
        results = results[:8]
        
        # 验证并存储
        valid_results = []
        for r in results:
            if isinstance(r, dict) and "file" in r:
                file_path = r.get("file", "")
                start_line = r.get("start_line", 1)
                end_line = r.get("end_line", start_line + 10)
                
                # 验证文件存在
                full_path = self.repo_path / file_path
                if full_path.exists():
                    valid_results.append({
                        "file": file_path,
                        "start_line": int(start_line),
                        "end_line": int(end_line)
                    })
        
        self.submitted_answer = valid_results
        
        if valid_results:
            summary = "\n".join([
                f"  {i+1}. {r['file']}:{r['start_line']}-{r['end_line']}"
                for i, r in enumerate(valid_results)
            ])
            return f"Submitted {len(valid_results)} results:\n{summary}"
        else:
            return "No valid results submitted"
    
    def _grep(self, query: str, path: str = ".") -> str:
        """
        搜索文本，返回 文件:行号:内容 格式
        
        这样模型能看到匹配的具体内容，即使目录树没显示该文件
        """
        if not query:
            return "Error: empty query"
        
        search_path = self.repo_path / path
        if not search_path.exists():
            search_path = self.repo_path
        
        try:
            # 使用 ripgrep，返回 文件:行号:内容
            cmd = [
                "rg", "-n",  # 显示行号
                "--max-count", "3",  # 每个文件最多3个匹配
                "-M", "200",  # 每行最多200字符
                query, 
                str(search_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split("\n")[:self.max_grep_results]
                
                # 转换为相对路径并记录
                output_lines = []
                for line in lines:
                    # 格式: /abs/path/file.py:42:content
                    if ":" in line:
                        parts = line.split(":", 2)
                        if len(parts) >= 2:
                            try:
                                abs_path = Path(parts[0])
                                rel_path = abs_path.relative_to(self.repo_path)
                                self.files_found.add(str(rel_path))
                                
                                # 记录行号
                                if len(parts) >= 2 and parts[1].isdigit():
                                    line_num = int(parts[1])
                                    if str(rel_path) not in self.lines_found:
                                        self.lines_found[str(rel_path)] = []
                                    self.lines_found[str(rel_path)].append((line_num, line_num))
                                
                                # 重新组合为相对路径
                                content = parts[2] if len(parts) > 2 else ""
                                output_lines.append(f"{rel_path}:{parts[1]}:{content}")
                            except:
                                output_lines.append(line)
                    else:
                        output_lines.append(line)
                
                return "\n".join(output_lines)
            else:
                return "No matches found"
        except subprocess.TimeoutExpired:
            return "Search timeout"
        except FileNotFoundError:
            # fallback to grep
            try:
                cmd = ["grep", "-r", "-n", query, str(search_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.stdout:
                    lines = result.stdout.strip().split("\n")[:self.max_grep_results]
                    return "\n".join(lines)
                return "No matches found"
            except:
                return "grep not available"
    
    def _read(self, file: str, start: int = 1, end: int = 50) -> str:
        """读取文件"""
        file_path = self.repo_path / file
        
        if not file_path.exists():
            return f"File not found: {file}"
        
        if not file_path.is_file():
            return f"Not a file: {file}"
        
        # 限制行数
        end = min(end, start + self.max_read_lines)
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            start = max(1, start)
            end = min(end, len(lines))
            
            content = []
            for i in range(start - 1, end):
                content.append(f"{i + 1}: {lines[i].rstrip()}")
            
            # 记录找到的文件和行范围
            self.files_found.add(file)
            if file not in self.lines_found:
                self.lines_found[file] = []
            self.lines_found[file].append((start, end))
            
            return "\n".join(content)
        except Exception as e:
            return f"Error reading file: {e}"
    
    def _glob(self, pattern: str) -> str:
        """列出匹配文件"""
        try:
            matches = list(self.repo_path.glob(pattern))[:self.max_grep_results]
            if not matches:
                return "No matches found"
            
            result = []
            for m in matches:
                try:
                    rel = m.relative_to(self.repo_path)
                    result.append(str(rel))
                except:
                    result.append(str(m))
            
            return "\n".join(result)
        except Exception as e:
            return f"Error: {e}"
    
    def _find(self, name: str) -> str:
        """按名称查找文件"""
        try:
            matches = []
            for p in self.repo_path.rglob(f"*{name}*"):
                if p.is_file():
                    try:
                        rel = p.relative_to(self.repo_path)
                        matches.append(str(rel))
                    except:
                        pass
                if len(matches) >= self.max_grep_results:
                    break
            
            if not matches:
                return "No matches found"
            return "\n".join(matches)
        except Exception as e:
            return f"Error: {e}"
    
    def get_result(self) -> SearchResult:
        """获取搜索结果"""
        return SearchResult(
            files_found=list(self.files_found),
            lines_found=self.lines_found,
            tool_calls=self.total_tool_calls,
            turns=self.current_turn
        )
    
    def compute_reward(
        self,
        ground_truth_files: List[str],
        ground_truth_lines: Dict[str, List[Tuple[int, int]]] = None,
        beta: float = 0.5,
        use_submission: bool = True
    ) -> float:
        """
        计算 reward (Weighted F1, β=0.5 偏向 Precision)
        
        博客设计: reward = (file_f1 + line_f1) / 2
        
        Args:
            ground_truth_files: 真实需要找到的文件
            ground_truth_lines: 真实需要找到的行范围 {file: [(start, end), ...]}
            beta: F-beta 的 beta 值，<1 偏向 precision，>1 偏向 recall
            use_submission: 是否优先使用 submit 工具的答案
        
        Returns:
            reward: F-beta score
        """
        if not ground_truth_files:
            return 0.0
        
        # 如果有 submit 的答案，优先使用
        if use_submission and self.submitted_answer:
            return self._compute_reward_from_submission(
                ground_truth_files, ground_truth_lines, beta
            )
        
        # 否则使用隐式收集的结果
        # 1. 文件级 F1
        file_f1 = self._compute_file_f1(ground_truth_files, beta)
        
        # 2. 行级 F1（如果有 ground truth）
        if ground_truth_lines:
            line_f1 = self._compute_line_f1(ground_truth_lines, beta)
            # 博客: 文件和行的平均
            return (file_f1 + line_f1) / 2
        
        return file_f1
    
    def _compute_reward_from_submission(
        self,
        ground_truth_files: List[str],
        ground_truth_lines: Dict[str, List[Tuple[int, int]]] = None,
        beta: float = 0.5
    ) -> float:
        """
        基于 submit 工具的答案计算 reward
        
        submitted_answer: [{"file": "path", "start_line": 10, "end_line": 20}, ...]
        """
        if not self.submitted_answer:
            return 0.0
        
        # 提取提交的文件和行
        submitted_files = set()
        submitted_lines = {}
        
        for r in self.submitted_answer:
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
            
            correct_lines = len(gt_lines & found_lines)
            
            precision = correct_lines / len(found_lines) if found_lines else 0
            recall = correct_lines / len(gt_lines) if gt_lines else 0
            
            if precision + recall == 0:
                line_f1 = 0.0
            else:
                beta_sq = beta ** 2
                line_f1 = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
            
            return (file_f1 + line_f1) / 2
        
        return file_f1
    
    def _compute_file_f1(self, ground_truth_files: List[str], beta: float) -> float:
        """计算文件级 F-beta"""
        gt_set = set(ground_truth_files)
        found_set = self.files_found
        
        if not found_set:
            return 0.0
        
        correct = len(gt_set & found_set)
        
        precision = correct / len(found_set) if found_set else 0
        recall = correct / len(gt_set) if gt_set else 0
        
        if precision + recall == 0:
            return 0.0
        
        beta_sq = beta ** 2
        return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    
    def _compute_line_f1(
        self, 
        ground_truth_lines: Dict[str, List[Tuple[int, int]]], 
        beta: float
    ) -> float:
        """
        计算行级 F-beta
        
        使用行覆盖率：预测的行范围与 ground truth 行范围的重叠
        """
        # 将行范围转换为行集合
        gt_lines = set()
        for file, ranges in ground_truth_lines.items():
            for start, end in ranges:
                for line in range(start, end + 1):
                    gt_lines.add((file, line))
        
        found_lines = set()
        for file, ranges in self.lines_found.items():
            for start, end in ranges:
                for line in range(start, end + 1):
                    found_lines.add((file, line))
        
        if not found_lines or not gt_lines:
            return 0.0
        
        correct = len(gt_lines & found_lines)
        
        precision = correct / len(found_lines) if found_lines else 0
        recall = correct / len(gt_lines) if gt_lines else 0
        
        if precision + recall == 0:
            return 0.0
        
        beta_sq = beta ** 2
        return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


# 工具定义（供模型使用）
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for text patterns in files",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search pattern"},
                    "path": {"type": "string", "description": "Directory to search in", "default": "."}
                },
                "required": ["query"]
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
                    "start": {"type": "integer", "description": "Start line", "default": 1},
                    "end": {"type": "integer", "description": "End line", "default": 50}
                },
                "required": ["file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "List files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., **/*.py)"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Find files by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "File name to search"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit your final answer with the most relevant files and line ranges. Call this in your last turn. Maximum 8 files, ordered by relevance (most relevant first).",
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "description": "List of relevant code locations, ordered by importance",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string", "description": "File path relative to repo root"},
                                "start_line": {"type": "integer", "description": "Start line number"},
                                "end_line": {"type": "integer", "description": "End line number"}
                            },
                            "required": ["file", "start_line", "end_line"]
                        },
                        "maxItems": 8
                    }
                },
                "required": ["results"]
            }
        }
    }
]
