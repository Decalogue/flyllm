# 大模型函数调用 (Function Calling) 原理与训练

## 1. 核心定性

本质上，Function Calling 是为了解决**大模型与外部世界交互**的问题，通过**将工具描述注入上下文，训练模型生成结构化参数**来实现的一种**解码时工具选择机制**。

---

## 2. 具体流程

1. **工具描述注入**: 将可用函数的 Schema (名称、参数、描述) 以 System Prompt 形式嵌入上下文
2. **自回归决策**: 模型在解码过程中，基于用户 Query 与工具描述，通过注意力机制计算选择概率，生成函数名与参数 JSON
3. **执行-观察闭环**: 外部系统执行函数，将返回值 (Observation) 再次输入模型，支持多轮工具调用

---

## 3. 数学基础

函数选择可形式化为条件概率最大化问题：

$$f^* = \arg\max_{f \in \mathcal{F}} P(f \mid Q, \mathcal{D}_{tools})$$

参数生成则通过自回归建模：

$$P(\theta \mid f^*, Q) = \prod_{t=1}^{T} P(\theta_t \mid \theta_{<t}, f^*, Q)$$

其中：
- $Q$: 用户 Query
- $\mathcal{F}$: 候选函数集合
- $\mathcal{D}_{tools}$: 工具描述 (Name, Description, Parameters Schema)
- $\theta$: 函数参数字符串序列

**关键训练目标** (基于指令微调):

$$\mathcal{L}_{tool} = -\mathbb{E}_{(x,y)\sim \mathcal{D}} \left[ \sum_{t=1}^{|y|} \log P_\theta(y_t \mid y_{<t}, x) \right]$$

$x$ 为带工具描述的指令，$y$ 为期望的函数调用 JSON。

---

## 4. 工程考量

| Trade-off | 说明 |
|-----------|------|
| **Schema 长度 vs 上下文** | 工具描述过长会挤占可用上下文，需做摘要或检索筛选 |
| **准确性与召回率** | 严格 Schema 约束提高准确性，但可能漏选边界可用工具 |
| **延迟与多轮调用** | 多轮 Function Call 会线性增加推理延迟 |

**致命弱点**:
- **幻觉参数**: 模型可能生成看似合理但违反 Schema 的参数值
- **错误选择**: 语义相似工具间容易混淆，缺乏显式判别信号
- **级联错误**: 前一步工具返回错误结果，模型难以自我纠正

---

## 5. 工业映射

| 方案 | 工业应用 |
|------|----------|
| **Prompt-based** (GPT-4, Claude) | OpenAI Function Calling API，直接通过 System Prompt 注入工具描述 |
| **LoRA 专用适配器** | 为 Tool Use 训练低秩适配模块，保持基座能力同时增强函数选择 |
| **工具检索 + Rerank** | 先检索 Top-K 候选工具，再精排选择，解决千级工具库问题 |

在工业界，该机制被直接应用于 **OpenAI Assistants API / LangChain / AutoGPT** 中，用于实现 Agent 与外部 API、数据库、搜索引擎的无缝交互。
