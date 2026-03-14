# 记忆检索的复合得分函数

## 1. 核心定性

本质上，记忆检索复合得分函数是将语义相似度、时间衰减、重要性和访问频率四个维度加权融合，通过动态权重调整实现相关性与新鲜度的平衡，模拟人类记忆回忆的多因素决策过程。

## 2. 具体流程

1. **向量搜索**: 使用embedding检索top-k候选记忆
2. **特征计算**: 计算每个记忆的时效性、重要性、频率特征
3. **得分融合**: 加权求和得到综合得分
4. **重排序**: 按得分重排并返回top-n结果

## 3. 数学基础

**复合得分公式**:
```python
Score(M|Q) = α·cos(M,Q) + β·e^{-λ·Δt} + γ·Importance(M) + δ·log(Freq(M))

其中：
- M: 记忆片段
- Q: 查询
- cos(M,Q): 向量相似度（余弦距离）
- Δt: 现在与记忆创建的时间差
- Importance(M): LLM评估的重要性（0-1）
- Freq(M): 访问频率
- α,β,γ,δ: 权重参数
```

**各维度定义**:

**1. 语义相似度**:
```python
cos(M,Q) = (embed(M) · embed(Q)) / (||embed(M)|| × ||embed(Q)||)

# 使用Cross-Encoder精排（比双塔更准）
cos_exact(M,Q) = LLM("Is M relevant to Q? <sep> M={M} Q={Q}") → 0-1 score
```

**2. 时间衰减**:
```python
recency(M) = e^{-λ·Δt}

Δt = (current_time - M.timestamp) / time_unit
λ = 0.1 ~ 0.5  # 衰减系数

# 不同场景调整λ
λ_conversation = 0.3  # 对话记忆衰减快
λ_knowledge = 0.05    # 知识记忆衰减慢
λ_episodic = 0.2      # 事件记忆中等
```

**3. 重要性评分**:
```python
importance(M) = LLM_rank("How important is this memory?", M)

# 简化版本（无需每次调用LLM）
importance(M) =
    0.9 if M.type == "core_preference"  # 核心偏好
    0.7 if M.type == "key_decision"     # 关键决策
    0.5 if M.type == "general_fact"     # 普通事实
    0.3 if M.type == "ephemeral"        # 临时信息

# 带时间衰减的重要性
importance(M,t) = importance(M) × e^{-λ₂·Δt}
```

**4. 访问频率**:
```python
frequency(M) = log(1 + access_count(M))

# 使用对数平滑
frequency(M) = log(1 + count) / log(1 + max_count)

# 时间加权访问频率（考虑访问时间分布）
frequency_weighted(M) = Σ w(tᵢ) where tᵢ ∈ access_times
w(t) = e^{-μ·(now - t)}  # 最近访问权重更高
```

**权重调优**:
```python
# 动态权重调整
α + β + γ + δ = 1

# 基于查询类型调整
if Q.type == "recency_important":
    α,β,γ,δ = 0.2, 0.5, 0.2, 0.1  # 加强时间权重
elif Q.type == "factual_query":
    α,β,γ,δ = 0.6, 0.1, 0.2, 0.1  # 加强语义权重
elif Q.type == "preference_query":
    α,β,γ,δ = 0.3, 0.2, 0.4, 0.1  # 加强重要性

# 在线学习优化
loss = MSE(click_feedback, predicted_score)
∇α,∇β,∇γ,∇δ = gradient_descent(loss, [α,β,γ,δ])
```

**检索质量评估**:
```python
# 使用MAP或NDCG评估
MAP = Σ_{k=1}^n P(k) × rel(k) / |relevant|

P(k) = precision at k
rel(k) = 1 if item_k is relevant else 0

# 相关性判断（用户反馈）
rel(M) =
    1 if clicked and time_spent > 30s
    0.5 if clicked
    0 if ignored
    -1 if explicitly_rejected
```

## 4. 工程考量

**Trade-off**:
- 增加：计算开销（多特征计算）
- 换取：检索质量（相关性提升20-40%）
- 平衡：维度权重调优复杂度

**致命弱点**:
- **权重选择困难**:
  ```python
  # 固定权重无法适应所有场景
  # 对话历史需要强时序性
  # 知识检索需要强语义性

  # 解决方案：查询意图识别
  intent = classify_query(Q)
  weights = lookup_weights(intent)  # 查表获取

  # 多组权重并行计算后选择最优
  scores = [compute_score(Q, params, w) for w in weight_configs]
  best = argmax(validate_on_dev_set(scores))
  ```

- **冷启动问题**:
  ```python
  # 新记忆无访问频率、无重要性评分
  # 导致永远无法被检索到

  # 解决方案：新记忆boost
  if M.is_new and cos(M,Q) > 0.7:
      score_boost = 0.2  # 新记忆加分
  else:
      score_boost = 0

  # 新记忆强制展示几次
  if M.age < timedelta(hours=1):
      force_include(M, top_k=3)
  ```

- **计算延迟**
  ```python
  # 四维度计算导致检索慢（100ms+）

  # 解决方案：分层计算
  phase1: 向量检索（1ms） → 粗排top-100
  phase2: 计算三特征（10ms） → 精排top-20
  phase3: LLM重排（100ms） → 最终top-5

  # 缓存加速
  @lru_cache(maxsize=10000)
def compute_score_cached(Q_hash, M_hash):
      return compute_score(Q, M)
  ```

- **维度冲突**:
  ```python
  # 高相似度但旧记忆 vs 低相似度但新记忆
  # 难以平衡

  # 解决方案：帕累托前沿
  candidates = vector_search(Q, top_k=50)
  pareto_optimal = find_pareto_frontier(
      candidates,
      dimensions=["similarity", "recency"]
  )
  # 选择最均衡的
  selected = max(pareto_optimal, key=balance_score)
  ```

**多级索引优化**:
```python
# L1: 时间索引（快速过滤旧数据）
# L2: 类型索引（记忆类型）
# L3: 向量索引（语义相似度）
# L4: 重要性索引（记忆优先级）

def hierarchical_search(Q, filters):
    # Step 1: 时间过滤（如果查询需要新数据）
    if Q.needs_recent:
        candidates = time_index.filter(days=7)

    # Step 2: 类型过滤
    if Q.type:
        candidates = type_index.filter(type=Q.type)

    # Step 3: 相似度检索
    similars = vector_index.search(Q, top_k=100)

    # Step 4: 重要性过滤
    return [m for m in similars if m.importance > 0.3]
```

## 5. 工业映射

在工业界，该机制被直接应用于Google的Personalized Search，使用点击熵和停留时间加权搜索结果。Pinterest的多目标排序结合语义相似度、时效性和流行度。Notion AI的记忆功能使用LFU（Least Frequently Used）+时间衰减。Duolingo的记忆系统根据艾宾浩斯遗忘曲线优化复习时间，提升30%记忆保持率。在对话系统中，Replika的情感记忆使用重要性权重优先保留高频情感话题。Foursquare的位置记忆使用距离衰减（recency + distance）推荐用户可能想去的地方。最新的Claude的Memory层根据对话的参与度（engagement）自动评估重要性，优于人工标注，让个性化回复更精准。
