# Agent 规划能力设计

## 1. 核心定性

本质上，**Agent 规划**是一个为了将高层目标分解为可执行动作序列，通过**状态空间搜索 + 启发式评估**实现的序列决策过程。

---

## 2. 具体流程

1. **目标解析**：将自然语言指令转化为形式化的目标状态表示（如谓词逻辑、符号状态）。
2. **动作建模**：定义动作的前置条件（Precondition）与执行后效果（Effect），构建状态转移图。
3. **搜索求解**：在状态空间中搜索从初始状态到目标状态的最短路径，输出动作序列。

---

## 3. 数学基础

### 状态表示
状态 $s \in S$，动作 $a \in A$，转移函数：
$$\delta: S \times A \rightarrow S$$

### 规划问题形式化
四元组定义经典 STRIPS 规划：
$$\Pi = \langle S, A, s_0, g \rangle$$

其中：
- $S$: 所有可能状态的集合
- $A$: 可用动作集合，每个动作 $a = \langle pre(a), add(a), del(a) \rangle$
- $s_0 \in S$: 初始状态
- $g \subseteq S$: 目标状态集合

### 启发式搜索（A* 规划）
$$f(n) = g(n) + h(n)$$

其中：
- $g(n)$: 从初始状态到节点 $n$ 的实际代价
- $h(n)$: 启发函数，估计 $n$ 到目标的代价（如 **忽略删除列表启发式** $h^{add}$ 或 **最大启发式** $h^{max}$）

### 现代 LLM-Agent 规划：ReAct 形式化
$$\text{Thought}_t, \text{Action}_t = \text{LLM}(\text{Prompt}, \text{Obs}_{<t})$$
$$\text{Obs}_t = \text{Env}(\text{Action}_t)$$

目标：最大化累积奖励或最小化到达目标的步数。

---

## 4. 工程考量

| 维度 | Trade-off | 致命弱点 |
|------|-----------|----------|
| **完备性 vs 效率** | 全搜索（如 A*）保证最优但指数爆炸；贪心/束搜索牺牲完备性换速度 | 状态空间爆炸：$O(b^d)$，$b$ 为分支因子，$d$ 为解深度 |
| **符号 vs 神经** | 符号规划（PDDL）可解释但难处理开放域；LLM 规划灵活但幻觉严重 | **复合错误累积**：每步微小误差在多步规划中被指数级放大 |
| **全局 vs 局部** | 全局规划一次生成全序列（脆弱，环境变化需重算）；局部规划（ReAct）动态调整但短视 | 长程依赖断裂：超过 5-7 步后，LLM 规划成功率断崖下降 |

**关键瓶颈**：
- **回溯成本**：现实世界动作不可逆（如发送邮件、删除数据），与搜索算法的"试错"本质冲突。
- **启发式质量**：$h(n)$  admissible 才能保证 A* 最优，但设计 domain-independent 的 admissible 启发式是 NP-hard 问题。

---

## 5. 工业映射

| 方法 | 工业落地场景 |
|------|-------------|
| **STRIPS/PDDL** | NASA 深空探测器任务规划（自主故障恢复）、AWS Step Functions 工作流编排 |
| **HTN（分层任务网络）** | 游戏 AI（如《星际争霸》Bot）、工业机器人任务分解（ROSPlan） |
| **ReAct / Reflexion** | OpenAI ChatGPT Plugins、AutoGPT、LangChain AgentExecutor 的推理-行动循环 |
| **MCTS + LLM** | AlphaCode 代码生成、DeepMind 的 FunSearch 数学发现 |
| **ToT（思维树）** | 复杂数学推理（如 GSM8K 竞赛题）、代码多路径搜索（如 GitHub Copilot X） |

> **实例**：LangChain 的 `Plan-and-Execute` Agent 本质上是 **全局规划器（LLM 生成大纲）+ 局部执行器（工具调用链）** 的两层架构，正是 HTN 思想在 LLM 时代的神经化实现。
