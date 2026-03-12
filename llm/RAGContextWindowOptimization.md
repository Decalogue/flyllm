# RAG 上下文窗口限制与长文档优化

## 1. 核心定性

本质上，RAG 的上下文窗口限制是一个**检索精度与生成质量的联合优化问题**，通过**分块-重排序-截断/压缩**的三级策略，在**信息密度**与**位置偏见**之间寻找帕累托最优。

---

## 2. 具体流程

1. **语义分块**：将长文档按语义边界切分为 chunk，保证每个 chunk 的语义完整性，同时控制长度 $L_{chunk} \leq \frac{1}{3} L_{context}$，为后续检索留出余量。

2. **检索与重排**：使用向量相似度召回 Top-K chunks，再通过 Cross-Encoder 重排序筛选出最相关的 $k_{final}$ 个，确保有限窗口承载最高密度信息。

3. **上下文构造**：采用 **LongContextReorder** 策略——将最重要的 chunk 放在窗口两端（缓解 LLM 的 "Lost in the Middle" 效应），或进行 **摘要压缩**（Map-Reduce / Refine）后再注入。

---

## 3. 数学基础

**召回-重排联合优化目标**：

$$\mathcal{L} = \underbrace{\sum_{i=1}^{k} \mathbb{I}(d_i \in D_{rel}) \cdot \text{sim}(q, d_i)}_{\text{召回精度}} + \lambda \cdot \underbrace{\text{RankNet}(d_1, ..., d_k)}_{\text{重排序损失}}$$

**上下文利用率上界**：

$$U_{ctx} = \frac{\sum_{j=1}^{m} |c_j|}{L_{max}} \leq \eta_{util}$$

其中：
- $D_{rel}$：与查询 $q$ 相关的文档集合
- $\text{sim}(q, d_i)$：查询与文档的语义相似度（余弦/dot-product）
- $\text{RankNet}$：基于 pairwise 比较的重排序损失函数
- $c_j$：第 $j$ 个 chunk 的有效信息 token 数
- $L_{max}$：上下文窗口最大长度（如 4K/8K/128K）
- $\eta_{util}$：利用率阈值（通常取 0.7~0.8，预留推理生成长度）

**关键伪代码（重排序阶段）**：

```python
# Stage 1: 向量检索（Bi-Encoder，快速粗排）
candidate_chunks = vector_store.similarity_search(query, k=K_RETRIEVE)

# Stage 2: Cross-Encoder 精排
scores = cross_encoder.predict([(query, c.text) for c in candidate_chunks])
top_chunks = [c for _, c in sorted(zip(scores, candidate_chunks), reverse=True)[:K_RERANK]]

# Stage 3: 长上下文重排（LongContextReorder）
# 缓解 "Lost in the Middle": 最重要文档放两端
ordered_chunks = [top_chunks[0]] + top_chunks[-1:0:-1]  # 头 + 逆序尾

context = "\n\n".join([c.text for c in ordered_chunks])
```

---

## 4. 工程考量

### Trade-offs

| 策略 | 牺牲 | 换取 |
|------|------|------|
| **减小 chunk_size** | 检索召回率下降（语义碎片化） | 上下文利用粒度更精细 |
| **增大 chunk_overlap** | 存储成本↑、冗余计算 | chunk 边界语义连续性↑ |
| **Cross-Encoder 重排** | 推理延迟 +$O(k \cdot L_{chunk})$ | 精度提升 10%~30% |
| **Map-Reduce 压缩** | 信息损失（多次摘要的蒸馏损耗） | 适应超长文档（无限长度） |

### 致命弱点

1. **"Lost in the Middle" 诅咒**：即使窗口够大，LLM 对中间位置的注意力显著衰减。实验证明，关键信息放在 4K 上下文中间位置时，召回率下降 **~40%**。

2. **重排序瓶颈**：Cross-Encoder 的 $O(n^2)$ 复杂度在大规模召回时成为瓶颈；当 $k > 100$ 时，延迟不可接受。

3. **语义漂移**：Map-Reduce 多级摘要导致"传话游戏"效应，长链依赖的信息在压缩中逐渐失真。

---

## 5. 工业映射

在工业界，该机制被直接应用于 **LangChain 的 `ParentDocumentRetriever`** 和 **LlamaIndex 的 `SentenceWindowNodeParser`** 中：

- **ParentDocumentRetriever**：以小块（child chunk）检索，返回大块（parent document）作为上下文，平衡检索精度与上下文完整性，用于 **GitHub Copilot 的代码问答** 场景。

- **Anthropic Claude 的 Contextual Retrieval**：在 chunk 前注入文档级上下文摘要，缓解分块后的语义孤岛问题，支撑 **Claude 200K 长文档分析** 的高准确率。

- **Google DeepMind 的 ReadAgent**：采用"记忆压缩 + 分页读取"策略，将超长文档分块后按需加载，模拟人类阅读行为，突破固定窗口限制，应用于 **Gemini 的 1M token 长上下文推理**。
