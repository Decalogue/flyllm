# RAG Embedding 模型选择

## 1. 核心定性

本质上，**RAG Embedding 模型选择**是一个在**语义对齐能力、推理成本、语言/领域适配**之间做权衡的决策问题：通过选对编码器，使查询向量与文档向量在同一语义空间内具备高区分度，从而最大化检索召回率与精排前的粗排质量。

选型错误会导致：语义鸿沟（同义不召）、维度浪费（高维低效）、领域漂移（通用模型在垂直领域 MRR 暴跌）。

---

## 2. 主流模型一览

### 2.1 中文场景优先

| 模型 | 维度 | 最大长度 | 特点 | 典型用途 |
|------|------|----------|------|----------|
| **bge-large-zh-v1.5** | 1024 | 512 token | C-MTEB 中文榜首级，中英双语均衡 | 中文 RAG 生产首选 |
| **bge-small-zh-v1.5** | 512 | 512 token | 轻量、延迟低 | 资源受限/高 QPS |
| **M3E-base / M3E-large** | 768 | 512 token | 中文优化，开源易部署 | 纯中文知识库 |
| **GTE-Qwen2-1.5B** | 1536 | 32K token | 长上下文、多语言 | 长文档、多语混合 |
| **text2vec-base-chinese** | 768 | 512 token | 早期中文 SOTA，生态成熟 | 存量项目兼容 |

### 2.2 多语言 / 长文本

| 模型 | 维度 | 最大长度 | 特点 | 典型用途 |
|------|------|----------|------|----------|
| **BGE-M3** | 1024 | 8192 token | 稀疏+稠密+多向量三合一，100+ 语言 | 多语 RAG、替代 BM25+向量 |
| **E5-mistral-7B-instruct** | 4096 | 32K token | 长文本、指令式 query 优化 | 长文档、复杂 query |
| **multilingual-e5-large** | 1024 | 512 token | 多语言均衡，MTEB 前列 | 多语种统一索引 |
| **GTE-large** | 1024 | 512 token | 多语言 MTEB 表现优 | 中英/多语生产 |

### 2.3 云 API（免部署）

| 模型 | 维度 | 最大长度 | 特点 | 典型用途 |
|------|------|----------|------|----------|
| **OpenAI text-embedding-3-small/large** | 1536/3072 | 8K token | 多语言、稳定、按量计费 | 快速验证、多语 SaaS |
| **Cohere embed-v3** | 1024 | 512 token | 多语言、支持 input_type 优化 | 检索/重排一体化 |
| **Voyage-3** | 1024 | 16K token | 长文档、领域模型可选 | 长文档、领域 RAG |

---

## 3. 选型决策流程

### 3.1 按语言与数据规模

```
                     ┌─────────────────────┐
                     │  主要语种？          │
                     └──────────┬──────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
    仅中文                中英/多语                仅英文
        │                       │                       │
        ▼                       ▼                       ▼
  bge-large-zh         BGE-M3 / GTE-large      E5 / text-embedding-3
  或 M3E-large          或 multilingual-e5
  （资源紧用 small）     （长文档用 BGE-M3）
```

### 3.2 按延迟与成本

- **极致延迟 / 边缘部署**：bge-small-zh、M3E-base、E5-small；或 384 维量化模型 + IVF-PQ。
- **平衡**：bge-large-zh、GTE-base、text-embedding-3-small。
- **精度优先**：BGE-M3 稠密、E5-mistral-7B、text-embedding-3-large；配合 Cross-Encoder 重排。

### 3.3 按文档长度

- **Chunk ≤ 512 token**：上述 512 窗口模型均可。
- **长 chunk（2K～8K）**：BGE-M3（8192）、GTE-Qwen2-1.5B（32K）。
- **超长（32K+）**：E5-mistral-7B-instruct、Voyage-3；需评估单次编码成本与索引更新频率。

---

## 4. 数学与评测基础

### 4.1 相似度与检索目标

**余弦相似度**（常用，与向量范数解耦）：
$$\text{sim}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\| \|\mathbf{e}_d\|}$$

**点积**（当向量已归一化时等价余弦；部分 API 默认返回点积）：
$$\text{score}(q, d) = \mathbf{e}_q^\top \mathbf{e}_d \quad \text{（}\|\mathbf{e}\|=1\text{ 时 } \equiv \cos\theta\text{）}$$

检索目标：从文档集 $D$ 中取与查询 $q$ 相似度最高的 Top-K：
$$\mathcal{R}(q) = \underset{d \in D}{\text{top-}k}\; \text{sim}(E(q), E(d))$$

### 4.2 常用评测基准

| 基准 | 范围 | 用途 |
|------|------|------|
| **MTEB** | 8 类任务、58+ 数据集、多语言 | 通用语义向量能力 |
| **C-MTEB** | 中文检索、分类、聚类等 | 中文模型选型主参考 |
| **BEIR** | 零样本检索、多领域 | 泛化与领域迁移 |
| **MIRACL** | 多语言检索 | 多语模型对比 |

选型时优先看**检索类**子任务（Retrieval）的 nDCG@10、MRR@10、Recall@k，其次看分类/聚类是否与业务相关。

### 4.3 维度与索引的权衡

- 维度 $D$ 越大，表达力潜力越高，但 HNSW/IVF 的建索与查询成本上升，且高维下 Recall 易退化。
- 经验：**768～1024** 为 RAG 常用甜点；>2048 需配合量化或降维，否则 Recall@10 可能明显下降（参见 [RAGRetrievalDesign](./RAGRetrievalDesign.md)）。

---

## 5. 工程考量

| 维度 | Trade-off | 建议 |
|------|-----------|------|
| **中文 vs 多语** | 专用中文模型（bge-large-zh）在 C-MTEB 上常优于「全能」多语模型 | 仅中文场景不必强上 BGE-M3 |
| **维度** | 高维≈更强表达，但索引更大、ANN 召回更敏感 | 生产常用 768/1024；云 API 可按需降维 |
| **最大长度** | 长上下文模型支持大 chunk，但单次编码贵、延迟高 | chunk 控制在 512～1024 token 时，512 窗口模型即可 |
| **稀疏+稠密** | BGE-M3 一类可同时出稀疏向量，替代独立 BM25 | 多语/长文档且希望统一管道时考虑 |
| **指令式 query** | E5/GTE 等支持「Query: …」前缀，提升 query 侧表达 | 使用官方推荐 prompt 模板 |
| **领域偏移** | 通用模型在医疗/法律等垂直领域可能 MRR 掉 20%+ | 有数据时做领域微调或选领域 API（如 Voyage 领域模型） |

---

## 6. 工业映射与黄金法则

- **LangChain / LlamaIndex**：通过 `HuggingFaceEmbeddings` 或厂商 SDK 切换上述模型；混合检索见 `EnsembleRetriever`（向量 + BM25）+ 重排。
- **向量库**：Milvus / Qdrant / Elasticsearch dense_vector 等均以 768/1024 维为常见配置；选型时注意模型维度与索引算法（HNSW/IVF）的匹配。
- **重排**：Embedding 负责粗排；精排建议用 Cross-Encoder（如 bge-reranker、Cohere rerank），召回率可再提升 15%～40%。

**黄金法则**：
- **中文 RAG 首选**：bge-large-zh-v1.5（或 small 做成本/延迟权衡）；纯中文可选 M3E-large。
- **多语 / 长文档**：BGE-M3 或 GTE-large；需长 chunk 时选 BGE-M3 / GTE-Qwen2 / E5-mistral。
- **快速验证 / 多语 SaaS**：OpenAI text-embedding-3-small 或 Cohere embed-v3。
- **生产标配**：向量检索（选型后的 Embedding）+ BM25 混合 + Cross-Encoder 重排（参见 [RAGRetrievalDesign](./RAGRetrievalDesign.md)）。
