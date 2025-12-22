# Fast Parallel Retrieval 研究笔记

> 超快并行检索模型的研究现状和技术分析
> 
> 最后更新：2025-12-22

---

## 目录

- [一、背景](#一背景)
- [二、Windsurf SWE-grep](#二windsurf-swe-grep)
- [三、学术界研究](#三学术界研究)
- [四、技术对比](#四技术对比)
- [五、开源资源](#五开源资源)
- [六、未知信息](#六未知信息)

---

## 一、背景

### 1.1 问题定义

传统的代码/文档检索 Agent 存在以下问题：
- **串行执行**：工具调用是串行的，效率低
- **延迟高**：每次工具调用都需要等待 LLM 响应
- **成本高**：大量的 LLM API 调用

### 1.2 解决方向

**并行检索**：让模型学会识别可以并行执行的工具调用，同时执行多个搜索操作。

---

## 二、Windsurf SWE-grep

### 2.1 基本信息

**来源**：Cognition AI（Windsurf/Devin 团队）官方博客
- 博客链接：https://cognition.ai/blog/swe-grep
- 发布时间：2024年10月16日
- 产品：Windsurf IDE 的 Fast Context 功能

### 2.2 传统检索方法的问题

**官方博客指出的两种传统方法及其缺陷**：

#### 方法 1：Embedding Search (RAG)

**优点**：
- 索引完成后，查询速度快

**缺点**：
- 结果不准确，特别是对于需要跨代码库多次跳转的复杂查询（如追踪执行路径）
- Embedding 可能适得其反，Agent 可能给不相关信息过多权重

#### 方法 2：Agentic Search（传统串行）

**优点**：
- 更灵活
- Claude Code 和 Cline 都认为效果好

**缺点**：
- 速度慢，需要数十次串行往返（用户设备 ↔ 推理端点）
- 强制 Agent 处理数万个不相关的 Token
- 导致速度慢的同时，还会"污染上下文"（context poisoning），显著降低答案质量

**观察数据**：
- 在 Windsurf 和 Devin 中，Agent 轨迹的第一轮通常有 >60% 的时间用于检索上下文

**结论**：
- 速度与智能的权衡似乎无法避免
- 直到 SWE-grep 的出现：匹配前沿模型的检索能力，同时快一个数量级

### 2.3 为什么需要 Fast Context Subagent

**官方博客给出的三个理由**：

1. **节省主 Agent 的上下文预算和智能**
   - 主 Agent 将检索委托给子 Agent
   - 节省宝贵的 Agent Token
   - 避免用不相关信息污染主 Agent 的上下文
   - 主 Agent 只需关注相关 Token
   - 避免"上下文失败"（参考 Drew Breunig 的 "How Contexts Fail"）

2. **检索是通用且广泛有用的能力**
   - AI 辅助编码栈的所有层都能受益
   - 自动补全模型：在给出建议前看到什么
   - Cascade：在实现更改前
   - Devin：在大型 PR 期间
   - 检索子 Agent 是智能模型和快速模型之间的完美"交接点"

3. **检索是可验证的任务**
   - 子 Agent 通常会总结发现给主 Agent，但有两个缺点：
     - 快速模型的总结可能得出错误结论，误导智能模型
     - 难以评分自由形式的总结
   - Fast Context 的设计：检索文件列表 + 行范围
   - 可以定义客观的 Ground Truth 数据集
   - 可以计算清晰的确定性奖励来做 RL

### 2.4 模型架构

**SWE-grep 模型族**：
1. **SWE-grep**：高智能变体，用于复杂检索任务
2. **SWE-grep-mini**：超高速变体，吞吐量超过 2,800 tokens/s

**模型规模**：
- 官方未公开具体参数量
- 未公开基座模型

### 2.5 SWE-grep 的速度优势来源

**官方博客指出的三个关键因素**：

1. **并行工具调用 + 限制串行轮数**
   - 传统 Agentic Search：通常需要 10-20 次串行轮数
   - SWE-grep：只需 4 次串行轮数
   - 通过高度并行的工具调用实现
   - 每轮最多 8 个并行工具调用（grep、glob、read 等）
   - 模型在几秒内完成深度搜索，同时探索代码库的不同部分

2. **快速工具调用**
   - 单个工具调用的时间变得很重要
   - 优化了工具调用的执行方式：
     - 索引优化
     - 多线程
     - 精心限制的工具集
   - 与 SWE-grep 模型协同设计

3. **快速推理**
   - 与 Cerebras（最快推理提供商）合作
   - 部署和优化定制的 SWE-grep 模型
   - 速度对比：
     - SWE-grep-mini：2,800+ tokens/s
     - SWE-grep：650+ tokens/s
     - Claude Haiku 4.5：140 tokens/s
     - SWE-grep-mini 比 Haiku 4.5 快 20 倍
     - SWE-grep 比 Haiku 4.5 快 4.5 倍

**并行度的影响**：
- 通过消融实验发现：
- 将并行度从 4 提升到 8 个搜索/轮
- 可以将搜索轮数从 6 轮减少到 4 轮
- 同时保持相同的性能

### 2.6 训练方法

**已知信息**（来自官方博客）：

1. **强化学习算法**：
   - 使用 Policy Gradient
   - 采用 Per-Sequence Importance Sampling（不是 Per-Token）
   - 使用 Leave-One-Out Baseline 减少方差

2. **奖励函数**：
   ```
   R = weighted_F1(files, lines)
   
   其中：
   - β = 0.5（偏向 Precision）
   - 文件级别 F1
   - 行级别 F1
   ```

3. **稳定性技巧**：
   - Mask 过长轨迹
   - Mask 极端 Importance Sampling Ratio
   - 移除 Format Reward
   - 错误格式直接打断，给 0 奖励
   - 按平均工具调用数缩放 Advantage

4. **环境 Token 处理**：
   - Retrieved Token Masking
   - 在计算 Loss 时 Mask 掉工具返回的内容

### 2.7 并行能力

**并行度**：
- 每轮最多 8 个并行工具调用
- 最多 4 轮（3 轮探索 + 1 轮答案）

**工具集**：
- `grep`：搜索关键词
- `read`：读取文件片段
- `glob`：列出文件
- 跨平台兼容（包括 Windows）

### 2.8 评估指标和方法

**评估指标**（来自官方博客）：

1. **Weighted F1 Score**
   - F-β，其中 β = 0.5
   - 文件检索的 F1
   - 行范围检索的 F1
   - 优先考虑 Precision 而不是 Recall

2. **端到端延迟**

**为什么优先考虑 Precision**：
- 发现上下文污染很重要
- 污染主 Agent 的上下文比遗漏一些上下文更有害
- Agent 通常只需几次搜索就能恢复任何剩余的上下文

**评估设置**：
- 允许每个模型 4 轮
- 每轮最多 8 个并行工具调用（搜索、读取等）

### 2.9 性能指标

**速度**：
- SWE-grep-mini：2,800+ tokens/s
- SWE-grep：650+ tokens/s
- 对比：Claude Haiku 4.5 为 140 tokens/s

**效果**（在 SWE-Bench Verified 子集上）：
- 使用 Fast Context 后，完成相同数量任务
- 端到端时间显著降低
- 搜索步骤减少

**推理提供商**：
- 使用 Cerebras（最快推理提供商）
- 进行了定制优化

### 2.10 数据集

**名称**：Cognition CodeSearch Eval

**来源**：
- 真实世界的代码仓库
- 用户查询
- 标注的 Ground Truth（相关文件和行范围）
- 来自最困难的 Bug Report 和内部测试

**标注**：
- 相关文件列表
- 相关行范围
- Ground Truth

**规模**：未公开

### 2.11 下游任务评估

**编码任务**（SWE-Bench Verified）：
- 使用随机选择的困难任务子集
- 主模型：Sonnet 4.5
- 使用 Fast Context 子 Agent 后：
  - 完成相同数量的任务
  - 端到端时间显著降低
  - 搜索文件步骤减少（包括 grep、文件读取、glob 搜索等）

**代码库问答**：
- 在开源仓库上的示例查询
- 端到端延迟对比：
  - Fast Context Agent（Windsurf 中使用）
  - vs Claude Code
  - vs Cursor CLI
- 测量端到端延迟

**对比设置**：
- 在各自的 Modal 容器中托管两个 Agent
- Fast Context Agent（从 Windsurf 中剥离）
- 原生 Claude Code
- 通过 stdin/stdout 传输输入/输出
- 旨在反映本地使用每个 Agent 的体验
- 不是极其严格的基准测试，而是演示体验

### 2.12 产品哲学：Flow Window

**目标**：让用户保持"心流"（Flow）状态
- 定义：完全沉浸在活动中的状态（Mihaly Csikszentmihalyi）

**Flow Window**：
- Windsurf 的目标：5 秒内响应
- 估计：等待 Agent 响应时，每过一秒，打破心流的概率几何增长 10%
- 阈值因请求的感知复杂度而异

**行业趋势观察**：
- 当前：编码 Agent 以 2-30 小时的自主性为卖点
- 营销激励：让 Agent 更慢，不是更快
- Cognition 的观点：趋势会反转
- 原因：他们看到了大规模同步和异步代码 Agent 的实际用户行为

**避免"Semi-Async Valley of Death"**：
- 同时研究两个方向：
  1. 推动编码 Agent 自主性的前沿
  2. 在"足够好"的标准下让它们更快

### 2.13 开源状态

❌ **未开源**
- 模型权重：未公开
- 训练代码：未公开
- 数据集：未公开
- 仅通过 Windsurf 产品使用

---

## 三、学术界研究

### 3.1 ParallelSearch（NVIDIA）

**论文信息**：
- 标题：ParallelSearch: Train your LLMs to Decompose Query and Search Sub-queries in Parallel with Reinforcement Learning
- arXiv ID：2508.09303
- 发布时间：2025年8月12日
- 机构：NVIDIA

**作者**：
1. Shu Zhao（宾夕法尼亚州立大学，实习于 NVIDIA）
2. Tan Yu（NVIDIA）
3. Anbang Xu（NVIDIA）
4. Japinder Singh（NVIDIA）
5. Aaditya Shukla（NVIDIA）
6. Rama Akkiraju（NVIDIA）

**核心思想**：
- 问题：现有搜索 Agent（如 Search-R1）只能串行搜索
- 解决：训练 LLM 识别可并行查询，并行执行
- 方法：查询分解 + 并行搜索执行 + RL 训练

**训练方法**：
- 使用 PPO（Proximal Policy Optimization）
- 基于 vLLM 和 Ray 构建
- 分布式训练和推理

**奖励函数**（详细）：

总奖励公式：
```
r = r_o + r_d + r_s + r_f
```

其中：
- `r_o`：结果奖励（Outcome Reward）
- `r_d`：分解奖励（Decomposition Reward）
- `r_s`：搜索计数奖励（Search Efficiency Reward）
- `r_f`：格式奖励（Format Reward）

**1. 分解奖励 `r_d`**：
```
r_d = 
  α * λ_d     如果 q ∈ P 且 D(q)=True   (可并行且分解了：奖励)
  -λ_d        如果 q ∉ P 且 D(q)=True   (不可并行却分解了：惩罚)
  0           其他情况
```
- `q ∈ P`：Query 是否属于"可并行问题"
- `D(q)`：检测是否包含分解符（如 `##`）
- `α > 1`：放大系数（可并行问题较少，需加大奖励）
- `λ_d = 0.15`（论文最佳参数）

**2. 搜索效率奖励 `r_s`**：
```
r_s = 
  -count(search) * λ_s           如果 q ∈ P 或 q ∈ S (可并行或单跳)
  -min(count(search), 2) * λ_s   否则 (不可并行的多跳问题)
  -2 * λ_s                       如果 count(search)=0 (惩罚幻觉)
```
- 对简单/可并行问题：每多搜索一轮就扣分，迫使并行
- 对复杂串行问题：惩罚有上限（capped at 2），防止过度惩罚
- `λ_s = 0.35`（论文最佳参数）

**3. 结果奖励 `r_o`**：
```
r_o = EM(a_gold, a_pred)
```
- Exact Match：完全匹配得 1 分，否则 0 分

**4. 格式奖励 `r_f`**：
```
r_f = 
  -λ_f    如果答对了但格式乱了 (惩罚)
  +λ_f    如果答错了但格式对了 (鼓励)
  0       其他情况
```
- 设计意图：即使答错，格式对也给分，稳住结构输出能力
- `λ_f ≈ 0.1`（通常设置）

**关键超参数**（论文 Ablation Study 验证）：
- `λ_d = 0.15`（分解权重）
- `λ_s = 0.35`（搜索惩罚权重）
- `λ_f ≈ 0.1`（格式权重）

**评估数据集**：
- Natural Questions (NQ)
- HotpotQA
- MultihopRAG
- 共 7 个 QA 数据集

**性能提升**：
- 平均性能提升：2.9%（7 个数据集）
- 可并行查询：12.7% 性能提升
- LLM 调用减少：30.4%
- 只需要 69.6% 的 LLM 调用次数

**应用场景**：
- NVInfo AI（NVIDIA 内部知识助手）
- 服务 30,000+ 员工

**开源状态**：
- 论文：✅ 公开（arXiv）
- 代码：❌ GitHub 仓库存在但无法访问（https://github.com/Tree-Shu-Zhao/ParallelSearch）
- 模型：❌ 未在 HuggingFace 找到
- 项目主页：https://shuzhao.me/ParallelSearchProject/

**模型规模**（已确认）：
- 参数量：7B（70亿参数）
- 基座模型：Qwen-2.5-7B（Base 和 Instruct 两个版本）

**训练数据**：
- 合并 Natural Questions (NQ) 和 HotpotQA 的训练集

**未公开信息**：
- 训练数据规模（具体条数）
- 训练成本
- 训练时间

### 3.2 GAP（Graph-Based Agent Planning）

**论文信息**：
- 标题：GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning
- arXiv ID：2510.25320
- 发布时间：2024年10月
- GitHub：https://github.com/WJQ7777/Graph-Agent-Planning

**核心思想**：
- 解决传统智能体（如 ReAct）只能"串行"思考和执行工具的瓶颈
- 训练模型先构建任务依赖图（DAG）
- 识别哪些子任务是独立的，可以并行调用工具
- 从 "想一步→做一步" 进化到 "全局规划依赖关系→并行执行无依赖任务"

**关键创新**：
- 不是让模型隐式地"感觉"哪些任务可以并行
- 而是显式地要求模型生成结构化的图数据
- 通过 SFT 教会模型一种"结构化思维"

**模型参数**：
- 参数量：3B（30亿参数）
- 基座模型：Qwen2.5-3B-Instruct
- 优势：证明了较小规模模型通过特定训练范式也能在复杂推理任务上达到很好效果

**训练数据与流程**：

1. **数据合成与 SFT（有监督微调）**
   - 数据来源：Natural Questions (NQ) 和 HotpotQA 数据集
   - 数据构建：使用 GPT-4o 合成约 7,000 条高质量的"基于图的规划路径"数据
   - 过滤机制：
     - 去除过于简单的样本（少于 3 次搜索）
     - 去除过于冗长的样本（超过 2000 token）
     - 保留需要并行检索的复杂任务

2. **强化学习 (RL)**
   - 算法：DAPO（Direct Alignment from Preference Optimization）
   - 在 SFT 之后进行端到端强化学习
   - 奖励函数：基于最终答案的正确性给予二元奖励（0 或 1）
   - 优化目标：在并行探索和上下文窗口限制之间权衡

**依赖图构建流程**（核心机制）：

GAP 的依赖图构建发生在规划阶段，分为三个步骤：

1. **子任务识别（Sub-task Identification）**
   - 模型分析用户问题，拆解为"原子级"子任务
   - 示例：问题"法国和德国首都的人口分别是多少？"
     - s1: 查询法国首都在哪
     - s2: 查询德国首都在哪
     - s3: 查询 s1 结果的人口
     - s4: 查询 s2 结果的人口

2. **依赖分析（Dependency Analysis）**
   - 通过分析子任务的输入-输出关系判断依赖性
   - 逻辑：如果 sj 的输入需要依赖 si 的输出，则 sj 依赖于 si
   - 判定：
     - s3 需要知道"巴黎"（s1 的结果），所以 s3 依赖 s1
     - s1 和 s2 都不依赖任何前置信息，可以并行

3. **图结构生成（Graph Construction）**
   - 模型将分析结果转化为 DAG，并输出文本表示
   - 输出格式（XML 风格）：
     ```xml
     <graph>
         <node id="s1">search("capital of France")</node>
         <node id="s2">search("capital of Germany")</node>
         <node id="s3" depends="s1">search("population of {s1}")</node>
         <node id="s4" depends="s2">search("population of {s2}")</node>
     </graph>
     ```
   - 格式说明：
     - `id`：每个子任务的唯一编号
     - `depends`：显式声明依赖关系，为空表示无依赖（可并行）
     - `{s1}`：占位符，稍后填入 s1 的执行结果

**实际推理中的输出格式**：
```
<plan>
Task 1: Search for John Frankenheimer's occupations and career
- Dependencies: none

Task 2: Search for Tiffanie DeBartolo's occupations and career
- Dependencies: none

Task 3: Compare their occupations to identify shared ones
- Dependencies: Task 1, Task 2
</plan>
```

**从图到执行：拓扑排序**：
- 构建好图后，使用拓扑排序将任务分层（Algorithm 1）
- Level 0（初始层）：找出所有没有入边（依赖为 None）的节点
  - 执行策略：这一层的所有工具调用打包成一个 Batch，并行发送给环境执行
- Level 1+（后续层）：找出依赖仅在 Level 0 的节点
  - 执行策略：等待 Level 0 的结果返回后，填入占位符，再并行执行这一层

**核心价值**：
- 不使用复杂的图神经网络
- 通过 SFT 教会 LLM 一种"结构化思维"
- 强制模型在行动前输出带有 `id` 和 `depends` 属性的文本
- 显式的结构化输出使系统可以解析出并行逻辑
- 打破了传统 ReAct "想一步做一步"的串行限制

**应用场景**：
- 主要场景：多跳问答 (Multi-Hop QA) 和复杂信息检索
- 具体任务：需要调用外部工具（如搜索引擎）来解决的复杂问题

**并行能力**：
- 每轮最多 8 个并行工具调用
- 最多 4 轮

**评估数据集**（7 个）：
- 单跳 QA：NQ, TriviaQA, PopQA
- 多跳 QA：HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle

**对比基线**：
- Search-R1（当前较强的检索增强模型）
- ZeroSearch
- StepSearch
- Chain of Agents
- Qwen2.5-3B-Instruct（原始模型）

**性能指标**（SOTA）：

1. **准确率提升**
   - 在 4 个多跳 QA 数据集上，比之前的 SOTA 基线平均提升 0.9%
   - HotpotQA：
     - GAP-3B：42.5%
     - Search-R1：37.6%
     - Qwen2.5-3B（原始）：9.9%

2. **效率提升**
   - 交互轮数减少：
     - HotpotQA：减少 21.6%
     - 2Wiki：减少 33.4%
   - 回复长度缩短：减少 24.9% 的 token 数量（推理成本更低）
   - 推理时间：HotpotQA 上减少 32.3%

**核心价值**：
- 在保持 3B 小参数规模的同时
- 在处理复杂检索任务时比传统 SOTA 模型更快、更准
- 通过并行执行减少了"来回折腾"

**开源状态**：
- 论文：✅ 公开
- 代码：✅ 开源
- 模型：未明确说明是否有预训练模型

### 3.3 Search-R1

**论文信息**：
- 标题：Search-R1: Training LLMs to Reason and Leverage External Knowledge
- arXiv ID：2503.09516
- 发布时间：2025年3月

**核心思想**：
- RL 训练 LLM 自主决定搜索时机
- 使用特殊 Token：`<search>`、`</search>`、`<think>`、`<information>`
- Outcome-based Reward（只看最终答案）
- Retrieved Token Masking（稳定训练）

**训练方法**：
- PPO / GRPO
- 只奖励最终正确性
- Mask 掉检索内容的 Loss

**开源状态**：
- 论文：✅ 公开
- 代码/模型：未明确说明

### 3.4 其他相关工作

**ReZero**：
- 奖励"再试一次"的行为
- 条件奖励（只有成功才给奖励）
- 鼓励探索不同查询策略

**Reasoning Agentic RAG Survey**：
- arXiv ID：2506.10408
- 发布时间：2025年6月
- 综述了 Agentic Retrieval 的各种方法

---

## 四、技术对比

### 4.1 方法对比

| 方法 | 依赖建模 | 并行度 | 训练方法 | 基座模型 | 开源 |
|------|---------|--------|---------|---------|------|
| **SWE-grep** | 隐式 | 8/轮，4轮 | Policy Gradient + Per-Seq IS | 未公开 | ❌ |
| **ParallelSearch** | 隐式 | `##` 分隔并行 | PPO | Qwen2.5-7B | ❌ |
| **GAP** | 显式（DAG） | 8/轮，4轮 | SFT + DAPO | Qwen2.5-3B | ✅ |
| **Search-R1** | 隐式 | 串行 | PPO/GRPO | 未明确 | ❓ |
| **LightningGrep** | 隐式 | 动态并行 | SFT + RL | Qwen3-1.7B | ✅（计划） |

### 4.2 应用场景对比

| 方法 | 应用场景 | 数据类型 | 行号召回 |
|------|---------|---------|---------|
| **SWE-grep** | 代码检索 | 代码库 | ✅ |
| **ParallelSearch** | 通用问答 | 文本、知识库 | ❌ |
| **GAP** | 多跳推理 | 问答数据集 | ❌ |
| **LightningGrep** | 通用 QA（后扩展代码） | 问答数据集 | ✅ |

### 4.3 奖励函数对比

| 方法 | 正确性奖励 | 效率奖励 | 格式奖励 | 置信度奖励 |
|------|-----------|---------|---------|-----------|
| **SWE-grep** | Weighted F1 | ❓ | ❓ | ❌ |
| **ParallelSearch** | EM | r_s + r_d | r_f | ❌ |
| **GAP** | 二元（0/1） | 隐式 | ❌ | ❌ |
| **LightningGrep** | 三层 F1（课程学习） | r_s + 动态终止 | r_f | ✅ |

---

## 五、开源资源

### 5.1 已开源

1. **GAP**
   - GitHub：https://github.com/WJQ7777/Graph-Agent-Planning
   - 论文：https://arxiv.org/abs/2510.25320

2. **相关框架**
   - vLLM：https://github.com/vllm-project/vllm
   - Ray：https://github.com/ray-project/ray
   - VERL：RL 训练框架

### 5.2 未开源但有论文

1. **SWE-grep**
   - 博客：https://cognition.ai/blog/swe-grep
   - 无论文、无代码、无模型

2. **ParallelSearch**
   - 论文：https://arxiv.org/abs/2508.09303
   - 项目主页：https://shuzhao.me/ParallelSearchProject/
   - GitHub 存在但无法访问
   - 无模型权重

3. **Search-R1**
   - 论文：https://arxiv.org/abs/2503.09516

### 5.3 数据集

**公开数据集**：
- Natural Questions (NQ)
- HotpotQA
- MultihopRAG
- MHQA（Multi-Hop Question Answering）
- SWE-Bench Verified

**未公开数据集**：
- SWE-grep 的训练数据
- ParallelSearch 的训练数据

---

## 六、未知信息

### 6.1 SWE-grep 未知信息

- ❓ 模型参数量
- ❓ 基座模型
- ❓ 训练数据规模
- ❓ 训练成本
- ❓ 训练时间
- ❓ 具体的超参数
- ❓ 数据生成方法
- ❓ 评估数据集的详细信息

### 6.2 ParallelSearch 未知信息

- ❓ 模型参数量
- ❓ 基座模型
- ❓ 训练数据规模
- ❓ 训练成本
- ❓ 具体的训练超参数
- ❓ 为什么代码和模型还未公开

### 6.3 GAP 未知信息

- ❓ 是否有预训练模型可下载
- ❓ 训练成本
- ❓ 训练时间

---

## 七、技术要点总结

### 7.1 共同特点

1. **并行工具调用**
   - 都支持每轮多个工具调用
   - 都限制在 4 轮左右

2. **强化学习训练**
   - 都使用 RL 训练
   - 都有奖励函数设计
   - 都需要处理环境 Token

3. **检索优化**
   - 都关注检索效率
   - 都减少 LLM 调用次数
   - 都提升准确率

### 7.2 关键差异

1. **依赖建模**
   - SWE-grep、ParallelSearch：隐式学习
   - GAP：显式 DAG

2. **应用场景**
   - SWE-grep：代码检索
   - ParallelSearch：通用问答
   - GAP：多跳推理

3. **开源程度**
   - SWE-grep：完全闭源
   - ParallelSearch：论文公开，代码未公开
   - GAP：完全开源

---

## 八、参考资料

### 8.1 官方博客

- Cognition AI - SWE-grep: https://cognition.ai/blog/swe-grep

### 8.2 论文

- ParallelSearch: https://arxiv.org/abs/2508.09303
- GAP: https://arxiv.org/abs/2510.25320
- Search-R1: https://arxiv.org/abs/2503.09516
- Reasoning Agentic RAG Survey: https://arxiv.org/abs/2506.10408

### 8.3 项目主页

- ParallelSearch: https://shuzhao.me/ParallelSearchProject/
- GAP: https://github.com/WJQ7777/Graph-Agent-Planning

---

## 九、LightningGrep 设计方案

### 9.1 项目定位

**目标**：开源的、生产可用的并行检索模型

**差异化**：
| 维度 | SWE-grep | ParallelSearch | GAP | LightningGrep |
|------|----------|---------------|-----|---------------|
| 开源 | ❌ | ❌ | ✅ | ✅ |
| 模型规模 | 未知 | 7B | 3B | 1.7B |
| 行号召回 | ✅ | ❌ | ❌ | ✅ |
| 置信度 | ❌ | ❌ | ❌ | ✅ |
| 动态终止 | ❌ | ❌ | ❌ | ✅ |

### 9.2 基座模型

- **选择**：Qwen3-1.7B
- **理由**：小型化、快速推理、工具调用支持好

### 9.3 训练数据

- **来源**：HotpotQA + NQ
- **构建方法**：GPT-4o 合成
- **规模**：SFT 5,000-10,000 条，RL 20,000-50,000 episodes
- **成本估算**：$50-100 (GPT-4o API)

### 9.4 奖励函数设计

基于 ParallelSearch 扩展，总奖励公式：
```
R = r_o_new + r_d + r_s + r_f + r_conf
```

**复用 ParallelSearch**：
- `r_d`：分解奖励（λ_d = 0.15）
- `r_s`：搜索效率奖励（λ_s = 0.35）
- `r_f`：格式奖励（λ_f ≈ 0.1）

**创新点 1：行级别奖励替换 EM**
```python
# 原 ParallelSearch
r_o = EM(a_gold, a_pred)  # 0 或 1

# LightningGrep（三层 F1）
r_o_new = w_file * F1_file + w_block * F1_block + w_line * F1_line

# 课程学习权重调度
if progress < 0.3:
    weights = {"file": 0.7, "block": 0.2, "line": 0.1}
elif progress < 0.7:
    weights = {"file": 0.3, "block": 0.4, "line": 0.3}
else:
    weights = {"file": 0.1, "block": 0.2, "line": 0.7}
```

**创新点 2：置信度校准奖励**
```python
# 置信度奖励
r_conf = calibration_score(predicted_confidence, actual_correctness)

# 效率奖励（动态终止）
efficiency_bonus = (max_rounds - rounds_used + 1) / max_rounds
```

### 9.5 模型输出格式

```xml
<round id="1">
  <actions>
    <search query="capital of France"/>
    <search query="capital of Germany"/>
  </actions>
  <confidence>0.3</confidence>
</round>

<round id="2">
  <actions>
    <read file="wiki/Paris.txt" lines="10-20"/>
    <read file="wiki/Berlin.txt" lines="15-25"/>
  </actions>
  <confidence>0.92</confidence>
</round>

<terminate reason="high_confidence"/>
<answer>
  Paris: 2.1 million, Berlin: 3.6 million
  <sources>
    <source file="wiki/Paris.txt" lines="15-17"/>
    <source file="wiki/Berlin.txt" lines="20-22"/>
  </sources>
</answer>
```

### 9.6 评测计划

**数据集**：
- 主攻：HotpotQA（可与 GAP 直接对比）
- 扩展：2WikiMultiHopQA, NQ, Musique

**对比基线**：
- GAP-3B（开源 SOTA）
- Search-R1
- Qwen3-1.7B 原始模型

**指标**：
- 准确率：Exact Match / F1
- 效率：平均轮数、LLM 调用次数
- 行号精度：Line-level F1（β=0.5）

### 9.7 研究阶段

```
阶段 1：数据准备（2 周）
├── 下载 HotpotQA 数据集
├── 设计 GPT-4o 合成 Prompt
├── 合成 10,000 条并行轨迹 + 置信度 + 行号
└── 质量检查和过滤

阶段 2：SFT 冷启动（1 周）
├── 格式化训练数据
├── 在 Qwen3-1.7B 上 SFT
└── 验证模型能输出正确格式

阶段 3：RL 训练（2-3 周）
├── 实现三层奖励函数
├── 实现课程学习调度
├── 实现置信度校准奖励
└── 训练和调参

阶段 4：评测和开源（1 周）
├── 在 HotpotQA, 2Wiki 等评测
├── 与 GAP, Search-R1 对比
├── 撰写技术报告
└── 开源模型和代码
```

---

## 十、其他研究方向

基于以上已知信息，其他可探索的方向：

1. **代码检索场景扩展**
   - 在通用 QA 验证后迁移到代码
   - 利用 SWE-Bench 构建代码检索数据

2. **更大规模模型**
   - 如果 1.7B 效果不够，可扩展到 3B/7B

3. **多模态检索**
   - 结合图片、表格等非文本信息

4. **垂直领域优化**
   - 法律、医疗等专业领域

---

**注意**：本文档只记录已确认的信息，所有未明确说明的内容都标注为"未公开"或"未明确说明"。不包含任何推测或猜测。
