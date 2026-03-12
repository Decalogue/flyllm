# Hybrid Search

## 1. 核心定性
本质上，Hybrid Search 是为了解决**纯向量搜索的语义漂移问题**，通过**融合向量相似度与关键词匹配分数**实现的**多路召回融合算法**。

## 2. 具体流程
1. **双路并行召回**：Query 同时送入 Embedding 模型生成稠密向量，以及分词/权重计算模块生成稀疏向量；
2. **独立打分排序**：Dense Retrieval 计算余弦相似度得到分数 $S_{vec}$，Sparse Retrieval（BM25/TF-IDF）计算文本匹配分数 $S_{text}$；
3. **融合重排（Rerank）**：通过线性加权或 RRF 算法合并两路分数，生成最终排序返回 Top-K。

## 3. 数学基础

### 线性加权融合（Linear Combination）
$$S_{final} = \alpha \cdot S_{vec} + (1-\alpha) \cdot S_{text}$$

其中：
- $S_{vec}$: 向量检索归一化后的相似度分数（范围 $[0, 1]$）
- $S_{text}$: 关键词检索归一化后的匹配分数（范围 $[0, 1]$）
- $\alpha \in [0, 1]$: **稠密向量权重**，平衡语义相似性与字面匹配精度

### Reciprocal Rank Fusion（RRF，业界更鲁棒）
$$S_{RRF}(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}$$

其中：
- $R$: 检索源集合（如 {向量检索, 关键词检索}）
- $rank_r(d)$: 文档 $d$ 在检索源 $r$ 中的排序位次
- $k$: 常数平滑因子（通常取 60，用于削弱低排名噪声）

## 4. 工程考量

| Trade-off | 描述 |
|-----------|------|
| **精确 vs 召回** | $\alpha$ 增大提升语义泛化能力但削弱专有名词匹配；减小则相反 |
| **计算成本** | 双路召回 → 2 倍召回开销 + 融合排序的 CPU 计算 |
| **归一化难题** | $S_{vec}$ 与 $S_{text}$ 分布差异巨大（余弦相似度 vs TF-IDF 原始值），必须做 **Min-Max/Softmax 归一化** 才能加权 |

**致命弱点**：
- **领域漂移灾难**：跨领域数据上，$\alpha$ 的"黄金值"会剧烈偏移（医学文献 $\alpha \approx 0.3$，开放问答 $\alpha \approx 0.7$），**不存在普适静态权重**
- **长尾 Query 崩溃**：当 Query 生僻词过多时，Embedding 的语义空间覆盖不足，若 $\alpha > 0.5$ 会直接漏召回

## 5. 工业映射
在工业界，该机制被直接应用于 **Elasticsearch 8.x 的 `semantic_text` 字段** 和 **Weaviate 的 Hybrid Search API** 中，用于应对 **企业知识库检索场景**（如 Confluence 文档 + 向量索引），其中 Elasticsearch 采用 RRF 作为默认融合策略。

---

**权重平衡实战建议**：先离线标注 100 条难例 Query，用 Grid Search 在 $[0.2, 0.8]$ 区间以 0.1 步长扫描 $\alpha$，选择 NDCG@10 最高的值作为业务最优权重。
