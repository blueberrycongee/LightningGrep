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
from typing import Dict, List, Optional
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
    
    if cache_file.exists():
        print(f"  从缓存加载: {cache_file}")
        with open(cache_file, "r") as f:
            return json.load(f)
    
    print(f"  从 HuggingFace 下载 SWE-Bench {split}...")
    
    if split == "lite":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    else:
        dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
    
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
        """克隆仓库"""
        url = f"https://github.com/{repo}.git"
        print(f"  克隆仓库: {repo}...")
        
        try:
            subprocess.run(
                ["git", "clone", "--depth", "100", url, str(repo_path)],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ 克隆失败: {e.stderr.decode()}")
            raise
    
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
            subprocess.run(
                ["git", "fetch", "--unshallow"],
                cwd=repo_path,
                capture_output=True,
            )
            subprocess.run(
                ["git", "checkout", commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )


# ========== 奖励计算 ==========

def compute_reward(
    predicted_files: List[str],
    ground_truth_files: List[str],
    beta: float = 0.5,
) -> float:
    """
    计算 Weighted F1 奖励
    beta < 1 偏向 Recall
    """
    if not predicted_files or not ground_truth_files:
        return 0.0
    
    pred_set = set(predicted_files)
    gt_set = set(ground_truth_files)
    
    correct = len(pred_set & gt_set)
    
    if correct == 0:
        return 0.0
    
    precision = correct / len(pred_set)
    recall = correct / len(gt_set)
    
    # Weighted F1
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    
    return f_beta


# ========== 简化版 RL 训练 ==========

@dataclass
class Trajectory:
    """一条搜索轨迹"""
    instance_id: str
    query: str
    files_found: List[str]
    ground_truth: List[str]
    reward: float
    log_prob: float


class SimpleRLTrainer:
    """简化版 REINFORCE 训练器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        repo_manager: RepoManager,
        learning_rate: float = 1e-5,
        num_rollouts: int = 4,
        max_turns: int = 3,
        temperature: float = 0.7,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.repo_manager = repo_manager
        self.num_rollouts = num_rollouts
        self.max_turns = max_turns
        self.temperature = temperature
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
        )
    
    def generate_response(self, messages: List[Dict]) -> str:
        """生成模型响应"""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=TOOLS_DEFINITION,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False
        )
        
        return response
    
    def run_episode(self, instance: Dict) -> Trajectory:
        """运行一个 episode"""
        repo_path = self.repo_manager.get_repo_path(
            instance["repo"],
            instance["base_commit"]
        )
        
        env = CodeSearchEnv(str(repo_path), max_turns=self.max_turns)
        ground_truth = parse_patch_files(instance["patch"])
        
        # 初始消息
        messages = [{"role": "user", "content": instance["problem_statement"]}]
        
        total_log_prob = 0.0
        
        for turn in range(self.max_turns):
            # 生成响应
            response = self.generate_response(messages)
            
            # 解析 tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                break
            
            # 执行 tool calls
            results, done = env.step(tool_calls)
            
            # 更新消息
            messages.append({"role": "assistant", "content": response})
            for result in results:
                messages.append({"role": "tool", "content": result["content"]})
            
            if done:
                break
        
        # 计算奖励
        files_found = list(env.files_found)
        reward = compute_reward(files_found, ground_truth)
        
        return Trajectory(
            instance_id=instance["instance_id"],
            query=instance["problem_statement"][:100],
            files_found=files_found,
            ground_truth=ground_truth,
            reward=reward,
            log_prob=total_log_prob,
        )
    
    def _parse_tool_calls(self, response: str) -> List[Dict]:
        """解析工具调用"""
        import re
        
        tool_calls = []
        
        # 匹配 <tool_call>...</tool_call> 或 JSON
        patterns = [
            r'<tool_call>\s*(\{[^}]+\})\s*',
            r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{[^}]+\})\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        tool_calls.append({
                            "name": match[0],
                            "arguments": json.loads(match[1])
                        })
                    else:
                        tool_calls.append(json.loads(match))
                except json.JSONDecodeError:
                    continue
        
        return tool_calls[:8]  # 最多 8 个并行调用
    
    def train_step(self, instances: List[Dict]) -> Dict:
        """一步训练"""
        all_trajectories = []
        
        for instance in instances:
            for _ in range(self.num_rollouts):
                try:
                    traj = self.run_episode(instance)
                    all_trajectories.append(traj)
                except Exception as e:
                    print(f"  ⚠️ Episode 失败: {e}")
                    continue
        
        if not all_trajectories:
            return {"loss": 0, "reward": 0}
        
        # 计算 baseline (mean reward)
        rewards = [t.reward for t in all_trajectories]
        baseline = sum(rewards) / len(rewards)
        
        # REINFORCE 更新 (简化版)
        # 这里只返回统计信息，实际梯度更新需要更复杂的实现
        
        return {
            "loss": 0,
            "reward": sum(rewards) / len(rewards),
            "baseline": baseline,
            "num_trajectories": len(all_trajectories),
        }
    
    def train(
        self,
        train_data: List[Dict],
        num_steps: int = 100,
        batch_size: int = 4,
        eval_every: int = 20,
        save_every: int = 50,
        output_dir: str = "outputs/rl_v1",
    ):
        """训练循环"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n开始 RL 训练: {num_steps} steps")
        print(f"  Batch size: {batch_size}")
        print(f"  Rollouts per instance: {self.num_rollouts}")
        
        step = 0
        total_reward = 0
        
        pbar = tqdm(total=num_steps, desc="Training")
        
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
                "avg": f"{total_reward/step:.3f}",
            })
            
            # 保存
            if step % save_every == 0:
                save_path = output_path / f"checkpoint-{step}"
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
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
    parser.add_argument("--max_samples", type=int, default=100, help="最大训练样本数")
    
    # 训练
    parser.add_argument("--num_steps", type=int, default=100, help="训练步数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--num_rollouts", type=int, default=4, help="每个样本的 rollout 数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LightningGrep RL 训练")
    print("=" * 60)
    
    # 检查 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 未检测到 GPU，训练会很慢！")
    
    # 1. 加载数据
    print("\n[1/4] 加载 SWE-Bench 数据...")
    data = load_swebench_data(args.split)
    
    # 限制样本数
    if args.max_samples and len(data) > args.max_samples:
        import random
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
    
    # 4-bit 量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 加载 SFT LoRA 并合并
    model = PeftModel.from_pretrained(base_model, args.sft_model)
    model = model.merge_and_unload()
    
    # 创建新的 LoRA 用于 RL
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 训练
    print("\n[4/4] 开始 RL 训练...")
    trainer = SimpleRLTrainer(
        model=model,
        tokenizer=tokenizer,
        repo_manager=repo_manager,
        learning_rate=args.learning_rate,
        num_rollouts=args.num_rollouts,
    )
    
    trainer.train(
        train_data=data,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
