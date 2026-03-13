# 记忆分层架构：短期/工作/长期记忆的边界划分与选型

## 1. 核心定性

本质上，**AI记忆分层** 是一个为了解决「上下文窗口有限 vs 信息无限」矛盾，通过 **时间衰减 + 重要性过滤 + 向量索引** 机制实现的分级存储-检索系统。

---

## 2. 具体流程

1. **感知阶段**：用户输入进入 **短期记忆 (Short-Term Memory, STM)**，作为原始感知缓冲区，仅保留最近 $K$ 轮对话 token。

2. **加工阶段**：STM 内容经注意力机制筛选，高价值信息被提取到 **工作记忆 (Working Memory, WM)** 进行深度推理加工。

3. **固化阶段**：WM 中经重要性评分 $I > \theta$ 的信息，编码为向量写入 **长期记忆 (Long-Term Memory, LTM)**，实现跨会话持久化。

---

## 3. 数学基础

### 3.1 记忆状态转移方程

$$
M_{t+1} = \underbrace{\alpha \cdot \text{Decay}(M_t)}_{\text{遗忘衰减}} + \underbrace{\beta \cdot \text{Compress}(S_t)}_{\text{新信息编码}}
$$

其中：
- $M_t$: 时刻 $t$ 的记忆状态向量
- $\alpha = e^{-\lambda \Delta t}$: 时间衰减系数，$\lambda$ 为遗忘率
- $S_t$: 当前感知输入
- $\text{Compress}(\cdot)$: 摘要压缩函数，通常采用 LLM-based summarization

### 3.2 重要性评分函数 (决定能否进入 LTM)

$$
I(x) = \underbrace{\gamma \cdot \text{Novelty}(x)}_{\text{新颖性}} + \underbrace{(1-\gamma) \cdot \text{Relevance}(x, \text{Goal})}_{\text{目标相关性}}
$$

### 3.3 记忆检索的向量相似度计算

$$
\text{Score}(q, m_i) = \underbrace{\cos(\mathbf{E}(q), \mathbf{E}(m_i))}_{\text{语义相似度}} \times \underbrace{e^{-\eta \cdot \text{Age}(m_i)}}_{\text{时间衰减因子}} \times \underbrace{\text{RecencyBoost}(m_i)}_{\text{近因加权}}
$$

其中 $\mathbf{E}(\cdot)$ 为 embedding 编码器。

---

## 4. 工程考量

| 层级 | 存储介质 | 检索方式 | 容量 | 延迟 | 一致性 |
|------|----------|----------|------|------|--------|
| **STM** | 上下文窗口 (Context Window) | 直接索引 $O(1)$ | 128K tokens | <1ms | 强 |
| **WM** | GPU HBM / 高性能缓存 (Redis) | 注意力查询 $O(n)$ | 1K-10K tokens | 1-10ms | 强 |
| **LTM** | 向量数据库 (Milvus/Pinecone) | ANN 近似最近邻 $O(\log N)$ | 无上限 | 10-100ms | 最终一致 |

### Trade-off 分析

| 维度 | 牺牲 | 换取 |
|------|------|------|
| **STM → WM** | 完整信息保留 | 推理效率与相关性聚焦 |
| **WM → LTM** | 实时一致性 | 跨会话持久化与海量存储 |
| **精确检索 → ANN** | 召回率 100% | 查询延迟从 $O(N)$ 降至 $O(\log N)$ |

### 致命弱点

1. **灾难性遗忘 (Catastrophic Forgetting)**：LTM 写入新信息时，若 embedding 空间重叠，可能覆盖旧记忆的向量表示。
2. **检索噪声**：ANN 在高维稀疏向量上的近似误差，可能导致召回无关记忆污染上下文。
3. **上下文碎片化**：多轮对话后 STM 被 system message / tool calls 挤占，导致用户意图理解断裂。

---

## 5. 工业映射

在工业界，该三层架构被直接应用于：

| 项目 | 实现方案 |
|------|----------|
| **OpenAI GPTs** | STM = 128K 上下文窗口；LTM 通过 `file_search` 连接向量存储 |
| **LangChain Memory** | `ConversationBufferMemory` (STM) → `VectorStoreRetrieverMemory` (LTM) |
| **Claude 3** | 200K 上下文作为 STM；配合 RAG 架构实现 LTM |
| **MemGPT** | 显式区分 WM (OS 管理的虚拟上下文) 与 LTM (外部数据库分页) |
| **ChatGPT Custom Instructions** | 作为 **语义化的 LTM 冷启动**，直接注入 system prompt |

### 选型决策树

```
是否需要跨会话持久化?
├── 否 → 仅用 STM (上下文窗口足够)
└── 是 → 是否需要复杂查询?
    ├── 否 → Redis Hash (Key-Value)
    └── 是 → 向量数据库 (Milvus/Pinecone/Weaviate)
        → 高并发? 选 Milvus
        → 云原生? 选 Pinecone
        → 边缘部署? 选 Chroma
```

---

**一句话总结**：STM 是 GPU 显存里的热数据，WM 是注意力 spotlight 下的聚焦区域，LTM 是向量数据库里的冷存储——三者通过 **时间衰减过滤 + 重要性评分** 实现自动分层，核心矛盾在于 **容量-延迟-一致性** 的不可能三角。
