# ReAct 框架实现与提示设计

## 1. 核心定性

**本质上，ReAct 是一个通过交错生成"推理轨迹(Reasoning)"与"动作执行(Action)"来解决多步决策问题的协同式 Agent 架构。**

---

## 2. 具体流程

1. **Thought → Action → Observation 循环**：LLM 先生成内部推理(Thought)，基于推理输出工具调用(Action)，执行后获取环境反馈(Observation)，再进入下一轮循环。

2. **状态维护**：每一步的输出作为下一步的输入上下文，形成链式记忆，直到 LLM 生成 `Finish` 动作终止循环。

3. **工具路由决策**：LLM 自主决定"何时调用工具、调用哪个工具、传入什么参数"，而非预定义流程。

---

## 3. 数学基础

ReAct 可建模为**部分可观察马尔可夫决策过程(POMDP)** 的近似求解：

$$
\pi(a_t | h_t) = \text{LLM}(h_t; \theta), \quad h_t = \{s_0, a_0, o_1, s_1, ..., o_t\}
$$

其中：
- $h_t$: 到时刻 $t$ 为止的完整历史轨迹（包含 Thought + Action + Observation）
- $s_t$: 当前状态（即 LLM 的上下文表示）
- $a_t$: 动作（工具调用或终止）
- $o_t$: 环境观察结果

**核心提示模板结构**：

```markdown
# ReAct Loop Prompt Template

You are an agent. Solve the task by alternating between Thought and Action.

Available Tools:
{tool_descriptions}

Task: {user_query}

Loop Format:
Thought: [你的推理过程，分析当前状态]
Action: [工具名(参数)] 或 Finish(最终答案)
Observation: [工具执行结果，由系统填充]

Begin!
{history}
Thought:
```

---

## 4. 工程考量

| Trade-off | 说明 |
|-----------|------|
| **灵活性 vs 可控性** | LLM 自主决策带来强大泛化能力，但也可能出现"幻觉式工具调用"或无限循环 |
| **上下文膨胀** | 每轮循环追加历史，长任务会导致上下文长度爆炸，需配合 `summarize` 或 `retrieve` 工具做记忆压缩 |
| **延迟累积** | 每步需一次 LLM 调用，多步任务总延迟 = N × TTFT，实时性要求高的场景难以承受 |

**致命弱点**：
- **错误传播不可恢复**：早期 Thought 错误会导致后续 Action 连锁错误，ReAct 本身无内置回滚机制
- **工具边界模糊**：LLM 可能在"信息不足时应继续检索"与"信息已够应直接回答"之间摇摆，陷入死循环

---

## 5. 工业映射

在工业界，ReAct 范式被直接应用于：
- **LangChain Agent** 的 `zero-shot-react-description` 模板
- **OpenAI Function Calling** 的多轮工具调用模式
- **AutoGPT** 的核心决策循环
- **扣子(Coze)** / **Dify** 等可视化 Agent 平台的"思考-执行-观察"工作流编排

> **降维金句**：ReAct = 把 CoT(思维链) 从"一次性生成"改造成"交互式执行"，让 LLM 从一个"做题家"进化成"能动手查资料的工程师"。
