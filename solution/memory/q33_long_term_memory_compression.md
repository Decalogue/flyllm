# 长期记忆的建模与压缩

## 1. 核心定性

本质上，长期记忆的建模与压缩是通过摘要提取（Extractive）和抽象生成（Abstractive）将冗余记忆压缩为紧凑表示，结合重要性评分和过期淘汰机制，在有限存储空间内保留高价值信息的记忆管理系统。

## 2. 具体流程

1. **冗余检测**: 计算新旧记忆的语义相似度，识别重复内容
2. **摘要压缩**: 对重复/相似记忆进行压缩（保留核心，舍弃细节）
3. **重要性评分**: 评估记忆价值，决定压缩强度
4. **过期淘汰**: 移除低价值旧记忆，释放存储空间

## 3. 数学基础

**记忆价值评分**:
```python
value(M) = α·importance + β·recency + γ·frequency + δ·compression_ratio

importance(M) = LLM_evaluate("How important is this?", M.content)
recency(M) = e^{-λ·Δt(M)}  # 时间衰减
frequency(M) = log(1 + access_count)
compression_ratio(M) = original_size / compressed_size

# 权重配置（可调整）
α,β,γ,δ = 0.4, 0.2, 0.2, 0.2
```

**冗余度计算**:
```python
# 语义相似度检测
redundancy(M₁, M₂) = cosine_similarity(embed(M₁), embed(M₂))

# Jaccard相似度（关键词级别）
jaccard(M₁, M₂) = |keywords(M₁) ∩ keywords(M₂)| / |keywords(M₁) ∪ keywords(M₂)|

# 综合冗余判断
is_redundant = redundancy > 0.9 and jaccard > 0.7
```

**压缩策略选择**:
```python
# 基于价值的压缩强度
if value(M) > 0.8:
    strategy = "extractive"  # 保留核心（抽取式）
    compression_ratio = 2x
elif value(M) > 0.5:
    strategy = "abstractive"  # 总结概括（抽象式）
    compression_ratio = 5x
else:
    strategy = "compress_vs_expire"
    if compression_ratio < 3x:
        expire(M)  # 价值低且难压缩，直接淘汰
    else:
        compress_heavy(M)  # 高度压缩
```

**抽取式压缩（Extractive）**:
```python
# TextRank提取关键句
importance(sentence) = (1-d) + d × Σ(PR(sentence→other) / out_degree(other))

# 选择top-20%关键句
sorted_sentences = sorted(sentences, key=importance)
extracted = sorted_sentences[:len(sentences)//5]
```

**抽象式压缩（Abstractive）**:
```python
# 使用LLM生成摘要
summary = LLM(f"""
Summarize the following memory in 2-3 sentences:
{M.content}

Key points to retain:
- Who, what, when, where
- User's preferences and constraints
- Action outcomes
""")

# 摘要质量评估
coherence = BERTScore(summary, M.content)
if coherence < 0.7:
    retry_summary()  # 质量不足，重新生成
```

**压缩率计算**:
```python
# 存储空间压缩
original_size = len(M.content)  # 原始字符数
compressed_size = len(M.compressed)

compression_ratio = original_size / compressed_size

# 信息保留度（ROUGE分数）
retention = ROUGE(M.compressed, M.original)

# 压缩效率
if retention > 0.8 and compression_ratio > 3:
    quality = "high"
elif retention > 0.6 and compression_ratio > 2:
    quality = "medium"
else:
    quality = "poor"  # 压缩效率低，考虑直接淘汰
```

## 4. 工程考量

**Trade-off**:
- 增加：计算成本（摘要生成需要LLM调用）
- 换取：存储节省（压缩率3-10倍）
- 平衡：压缩速度与质量

**致命弱点**:
- **压缩信息损失**:
  ```python
  # 过度压缩导致关键信息丢失
  # 示例："用户对产品A的投诉：功能问题，数据丢失"
  # 压缩后："用户有意见"

  # 解决方案：重要性保护
  important_entities = extract_named_entities(M)
  in_summary = all(entity in summary for entity in important_entities)

  if not in_summary:
      regenerate_with_constraints(keep=important_entities)
  ```

- **压缩成本**:
  ```python
  # LLM摘要生成成本高
  # $0.01 × 1000条 = $10/压缩批次

  # 解决方案：分层压缩
  if M.age > 7_days:  # 只有旧数据压缩
      compress()

  if value(M) < 0.5:  # 只有低价值数据压缩
      compress()
      expire_if_compressed_size_too_large()
  ```

- **压缩冲突**:
  ```python
  # 同一段记忆被多次压缩
  # 导致质量下降

  # 解决方案：版本控制
  M.compressed_version += 1
  if M.compressed_version > 3:
      # 压缩3次后，考虑保留原文
      keep_original_instead()

  # 或使用增量压缩
  new_compressed = compress(M.original)
  # 而不是在M.compressed上继续压缩
  ```

**自适应压缩**:
```python
# 根据存储压力调整压缩策略
storage_pressure = used_space / total_space

if storage_pressure > 0.8:  # >80%使用率
    compression_level = "aggressive"  # 激进压缩
    target_ratio = 5
elif storage_pressure > 0.6:
    compression_level = "moderate"   # 适中压缩
    target_ratio = 3
else:
    compression_level = "light"      # 轻度压缩
    target_ratio = 2
```

**分层存储**:
```python
# 根据价值分层存储
class StorageTier:
    HOT = "memory"       # 高价值，不压缩
    WARM = "disk"        # 中等价值，轻度压缩
    COLD = "s3/glacier"  # 低价值，高度压缩

# 自动分层
if value(M) > 0.8:
    store(HOT)
elif value(M) > 0.4:
    compress(M, ratio=2)
    store(WARM)
else:
    compress(M, ratio=5)
    if compressed_size < 1000:
        store(COLD)
    else:
        expire(M)  # 低价值+大体积=淘汰
```

## 5. 工业映射

在工业界，该机制被直接应用于Notion的页面历史压缩，旧版本自动摘要节省70%存储。GitHub的issue评论压缩使用抽取式摘要展示关键讨论。Slack的消息归档对超过90天的频道进行压缩。在RAG场景中，Pinecone的TextSplitter结合摘要提取优化chunk质量。Hugging Face的Datasets库对大型语料库自动压缩，减少50%存储。最新的Claude的Long Context使用动态记忆压缩，在100K上下文场景下通过智能摘要保持95%的信息完整性。
