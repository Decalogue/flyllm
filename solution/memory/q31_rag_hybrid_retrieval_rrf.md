# RAG 混合检索 RRF 融合

## 1. 核心定性

本质上，RAG混合检索的RRF（Reciprocal Rank Fusion）融合是通过倒数排名加权将BM25（关键词）、Dense（向量）和Reranker（重排序）的多路检索结果合并，使用公式score = Σ 1/(rank_i + k)实现无监督效果提升的排名融合算法。

## 2. 具体流程

1. **多路检索**: BM25、Dense、Sparse向量三路并行检索top-k
2. **RRF融合**: 对每路结果计算倒数排名分并加权求和
3. **重排序**: 使用Cross-Encoder对top-n精排
4. **结果返回**: 合并去重后返回最终检索结果

## 3. 数学基础

**RRF融合公式**:
```python
RRF_score(doc) = Σ_{method∈M} w_m × 1 / (rank_m(doc) + k)

其中：
- M = {BM25, Dense, Sparse}  # 检索方法集合
- rank_m(doc): doc在方法m中的排名（从1开始）
- w_m: 方法m的权重（通常w_bm25=0.3, w_dense=0.5, w_sparse=0.2）
- k: 平滑常数（通常60，防止排名第一得分过大）
```

**示例计算**:
```python
# 文档D在三路检索中的排名
doc = "doc_123"
rank_bm25(doc) = 3   # BM25排名第3
rank_dense(doc) = 1  # Dense排名第1
rank_sparse(doc) = 5 # Sparse排名第5

# 权重配置
w = {BM25: 0.3, Dense: 0.5, Sparse: 0.2}
k = 60

# RRF得分计算
score = 0.3×1/(3+60) + 0.5×1/(1+60) + 0.2×1/(5+60)
      = 0.3/63 + 0.5/61 + 0.2/65
      = 0.00476 + 0.00820 + 0.00308
      = 0.01604

# 对比单一方法最高分（Dense第1名）
dense_only = 1/(1+60) = 0.01639
# RRF通过多路信息融合，让综合表现好的文档得分更高
```

**重排序（Reranking）**:
```python
# Cross-Encoder精排（更准但更慢）
rerank_score(doc, query) = CrossEncoder(f"{query}[SEP]{doc}")

# 最终得分
final_score(doc) = α×RRF_score(doc) + β×rerank_score(doc)

# 通常α=0.7, β=0.3（RRF为主，rerank为辅）
```

**Hybrid Search权重学习**:
```python
# 权重优化目标
minimize Σ_{query,doc} (final_score(doc) - relevance_label)²

# 梯度更新
w_m ← w_m - η·∂loss/∂w_m

# 约束条件
Σ w_m = 1, w_m ≥ 0
```

**去重策略**:
```python
# 多路结果可能返回相同文档
docs_set = set()
for method in methods:
    for doc in results[method]:
        key = doc.id + doc.chunk_id  # 使用ID+块ID去重
        if key not in docs_set:
            unique_docs.append(doc)
            docs_set.add(key)

# 内容重复检测（语义）
if cosine_similarity(doc1.content, doc2.content) > 0.95:
    merge_docs(doc1, doc2)  # 合并重复内容
```

## 4. 工程考量

**Trade-off**:
- 增加：计算延迟（三路并行+RRF+Rerank）
- 换取：检索质量（Recall@10提升15-25%）
- 平衡：延迟与质量的选择

**致命弱点**:
- **参数敏感性**:
  ```python
  # k值选择影响大
  # k过小：top-ranked权重过大
  # k过大：排名差异不明显

  # 解决方案：自动调优
  best_k = argmax(k→validation_mAP(k))
  # 通常k∈[20, 100]，数据集相关
  ```

- **权重不均衡**:
  ```python
  # 三路质量差异大
  # BM25召回率高但精度低
  # Dense精度高但覆盖不足

  # 解决方案：权重自适应
  w_bm25 = 0.3 if sparse_query else 0.2
  w_dense = 0.5 if semantic_query else 0.7
  w_sparse = 0.2 if dense_retriever_weak else 0.1
  ```

- **延迟累积**:
  ```python
  # 三路并行检索 + Reranker串行执行
  # 延迟 = max(T_bm25, T_dense, T_sparse) + T_rerank
  # 可能超过200ms

  # 优化方案：流水线
  # Step 1: BM25快速返回（10ms）
  # Step 2: Dense异步返回（50ms）
  # Step 3: Rerank前50名（30ms）
  ```

- **结果不一致**:
  ```python
  # 相同查询不同时间结果不同
  # 向量索引更新导致dense排名变化
  # 影响用户体验

  # 解决方案：结果缓存 + RRF缓存
  if query in cache:
      return cached_result
  else:
      result = hybrid_search(query)
      cache[query] = result  # 缓存10分钟
```

**性能优化**:
```python
# 1. 检索结果缓存
@lru_cache(maxsize=10000)
def cached_search(query, method):
    return retriever[method].search(query)

# 2. 并行执行
with ThreadPoolExecutor() as executor:
    futures = {
        "bm25": executor.submit(bm25.search, query),
        "dense": executor.submit(dense.search, query),
        "sparse": executor.submit(sparse.search, query)
    }
    results = {k: v.result() for k, v in futures.items()}

# 3. 向量量化和索引优化
# HNSW索引：搜索从O(N)降到O(logN)
# PQ量化：内存从4GB降到1GB

# 4. 轻量级rerank
# 先用bi-encoder粗排，只用cross-encoder重排top-10
# 节省90%计算量
```

## 5. 工业映射

在工业界，该机制被直接应用于Pinecone的Hybrid Search，自动融合dense和sparse检索。Weaviate的Multi-Vector Indexing支持多向量存储和RRF融合。Azure AI Search的Semantic Ranking结合BM25和Reranker（基于Microsoft的DeBERTA），在文档检索中准确率提升40%。YouTube的搜索使用三阶段（字面值→语义→个性化）RRF融合，视频检索相关性提升30%。在RAG场景中，LlamaIndex的Ensemble Retriever支持多路检索自动融合，在文档问答中召回率提升20%。最新的LangChain的Contextual Compression使用基础检索+LLM过滤，在保持质量同时减少50%检索量。Google的Vertex AI Search的Enterprise RAG使用Hybrid Lexical+Neural Search + Reranker，在内部测试中排名第一结果点击率提升25%。
