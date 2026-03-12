# Dense Retrieval vs Sparse Retrieval & BM25 vs DPR

---

## 一、核心定性

**Sparse Retrieval**：基于词汇匹配的离散表示检索，通过精确词项重叠计算相关性，本质是"关键词命中游戏"。

**Dense Retrieval**：基于语义向量的连续表示检索，通过嵌入空间中的向量相似度捕捉语义相关性，本质是"语义近邻搜索"。

---

## 二、具体流程

### Sparse Retrieval (以 BM25 为例)
1. **构建倒排索引**：对每个文档建立词项 → 文档列表的倒排表
2. **查询解析**：将查询分词为词项集合 $Q = \{q_1, q_2, ..., q_n\}$
3. **评分计算**：对每个候选文档计算 BM25 得分并排序返回

### Dense Retrieval (以 DPR 为例)
1. **编码器预训练**：用双塔 BERT 分别编码查询和文档为固定维度向量
2. **向量索引构建**：将文档向量存入 FAISS/ANN 索引库
3. **语义检索**：查询向量化后，在向量空间中寻找最近邻文档

---

## 三、数学基础

### BM25 评分公式

$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

其中：
- $f(q_i, D)$：词项 $q_i$ 在文档 $D$ 中的词频
- $|D|$：文档 $D$ 的长度（词数）
- $\text{avgdl}$：语料库中文档平均长度
- $k_1 \in [1.2, 2.0]$：词频饱和度控制参数
- $b \in [0, 1]$：文档长度归一化参数（通常取 0.75）
- $\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)$：逆文档频率，$N$ 为总文档数，$n(q_i)$ 为包含 $q_i$ 的文档数

### DPR 相似度计算

$$\text{sim}(q, d) = \mathbf{E}_Q(q)^\top \cdot \mathbf{E}_D(d)$$

其中：
- $\mathbf{E}_Q, \mathbf{E}_D \in \mathbb{R}^{d \times h}$：查询和文档的 BERT 编码器
- $q, d$：查询和文档文本
- 输出维度：通常为 768 维（BERT-base）

**训练目标**（对比学习）：

$$\mathcal{L} = -\log\frac{e^{\text{sim}(q, d^+)/\tau}}{\sum_{d \in \{d^+, d_1^-, ..., d_n^-\}} e^{\text{sim}(q, d)/\tau}}$$

其中 $d^+$ 为正样本，$d_i^-$ 为负样本，$\tau$ 为温度系数。

---

## 四、工程考量

| 维度 | Sparse (BM25) | Dense (DPR) |
|------|---------------|-------------|
| **索引空间** | 倒排索引，稀疏存储，空间小 | 稠密向量，需 $(N \times d \times 4)$ 字节，空间大 |
| **检索速度** | 毫秒级，倒排表跳转极快 | 依赖 ANN（FAISS/HNSW），亚毫秒~毫秒级 |
| **语义理解** | 无法处理同义词、语义漂移 | 强大的语义泛化能力 |
| **冷启动** | 零参数，即插即用 | 需领域数据微调，否则性能断崖 |
| **可解释性** | 高，可追踪具体命中词项 | 低，向量空间黑盒 |

**致命弱点**：
- **BM25**：词汇失配（"苹果" vs "Apple"）、长尾词稀疏、无法理解语义相近但词汇不同的查询
- **DPR**：领域迁移能力差（开箱即用的 DPR 在垂直领域常比 BM25 差）、对短查询敏感、需要大规模负样本训练

**Trade-off**：
- BM25 牺牲了**语义理解**换取了**零成本部署**和**确定性行为**
- DPR 牺牲了**可解释性**和**存储效率**换取了**语义泛化能力**

---

## 五、工业映射

**工业界实战**：现代 RAG 系统（如 Bing、OpenAI Retrieval、LangChain）普遍采用 **Hybrid Retrieval（混合检索）**：

$$\text{Score}_{\text{hybrid}} = \lambda \cdot \text{BM25}(q, d) + (1-\lambda) \cdot \text{cos}(\mathbf{v}_q, \mathbf{v}_d)$$

- **Elasticsearch 8.0+**：原生支持 `text_similarity`（BM25）+ `dense_vector`（kNN）混合查询
- **Azure Cognitive Search**：集成语义重排序（Semantic Ranker）在 BM25 初筛后接 Cross-Encoder 精排
- **Meta Faiss + BM25**：工业级 pipeline 中，BM25 负责召回 Top-K（高召回率），DPR 负责语义重排（高精度）

---

> **面试金句**："Sparse 是符号主义的遗产，Dense 是连接主义的胜利。生产环境中，二者不是替代关系，而是互补——BM25 做粗排保召回，DPR 做精排保语义。"
