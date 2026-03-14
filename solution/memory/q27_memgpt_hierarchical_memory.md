# MemGPT 的分层记忆架构

## 1. 核心定性

本质上，MemGPT的分层记忆架构是模拟操作系统的虚拟内存管理，通过Working Context（上下文窗口）→ Main Context（核心记忆）→ Archival Store（归档存储）三级结构，结合page_in/out机制突破LLM上下文窗口限制的渐进式记忆管理系统。

## 2. 具体流程

1. **Working Context**: 当前对话窗口（受LLM context长度限制）
2. **Main Context**: 核心记忆空间（LLM可访问的"内存"）
3. **Archival Store**: 持久化存储（外部向量数据库）
4. **Page管理**: LRU-K算法在三级之间移动记忆片段

## 3. 数学基础

**三级存储架构**:
```python
M_total = M_working + M_main + M_archival

M_working = C_LLm  # 受限于模型context长度（如4K）
M_main = α·C_LLm   # 可访问的核心记忆（如8K）
M_archival = ∞     # 无限归档存储

α = 2  # Main Context通常是Working的2倍
```

**Page数据结构**:
```python
Page = {
    "id": str,
    "content": str,
    "virtual_address": (segment, offset),  # 虚拟地址
    "last_access": timestamp,
    "frequency": int,  # 访问次数
    "importance": float  # LLM评分
}

Segment = working | main | archival
```

**地址映射**:
```python
# 虚拟地址到物理地址映射
page_table = {
    page_id: {
        "virtual_addr": (seg, offset),
        "physical_addr": db_id | memory_location,
        "present": bool,  # 是否在Working/Main Context
        "dirty": bool     # 是否被修改
    }
}

# 地址转换开销
T_address_translation =
    0.1ms if present  # 内存访问
    5ms   if in archival  # 向量检索
    100ms if page_out_need # 需要page_out换入
```

**Page评分函数（LRU-K）**:
```python
score(page) = w₁·recency + w₂·frequency + w₃·importance

recency = exp(-λ·(now - page.last_access))  # 时间衰减
frequency = log(1 + page.access_count)
importance = LLM_evaluate("How important is this?", page.content)

# LRU-K权重（通常K=2，考虑最近两次访问）
w = [0.5, 0.3, 0.2]
λ = 0.1  # 衰减系数
```

**Page_in/out策略**:
```python
# Page_out（从Working到Main/Archival）
if len(working_memory) > C_working_threshold:
    victim = min(working_pages, key=score)
    if victim.frequency >= 2:
        page_to_main(victim)  # 热数据到Main
    else:
        page_to_archival(victim)  # 冷数据到存储

# Page_in（从Main到Working）
if task_needs_info(info_id):
    page = retrieve_from_main_or_archival(info_id)
    if working_memory.full():
        page_out()  # 先换出
    page_to_working(page)  # 换入
    self.page_table[page.id]["present"] = True
```

**虚拟内存公式**:
```python
# 有效内存容量
M_effective = M_working + M_main + α·M_archival
α = retrieval_accuracy × relevancy_filter

# 示例
M_working = 4K tokens
M_main = 8K tokens
M_archival = 1M tokens
retrieval_accuracy = 0.85
α = 0.85
M_effective = 4K + 8K + 0.85M ≈ 862K tokens
```

## 4. 工程考量

**Trade-off**:
- 增加：检索延迟（page_in/out开销）
- 换取：记忆容量提升（200倍以上）
- 平衡：命中率和检索质量

**致命弱点**:
- **检索准确性不足**: page_in换入不相关的信息
  ```python
  # 解决方案：双阶段检索
  # 1. 粗排：向量检索召回top-100
  candidates = vec_search(query, top_k=100)
  # 2. 精排：LLM重排top-5（cross-encoder）
  pages_to_load = llm_rerank(candidates, top_k=5)
  ```

- **换入换出风暴**:
  ```python
  # 频繁在不同记忆间切换导致性能下降
  # 例如：多轮对话中反复page_in/out

  # 解决方案：预取和合并
  # 预取：预测接下来可能需要的记忆
  predicted = next_likely_queries(history)
  prefetch_pages(predicted)

  # 合并：批量page_in减少次数
  if len(page_in_queue) > threshold:
      batch_page_in(page_in_queue)  # 合并向量检索
  ```

- **数据一致性**:
  ```python
  # 同一段记忆在Working和Archival有两个版本
  # working: "用户喜欢蓝色"
  # archival: "用户喜欢红色"（旧数据）

  # 解决方案：版本向量和冲突检测
  version_vector = {
      "working": 10,
      "main": 9,
      "archival": 8
  }

  if version_conflict_detected:
      resolved = llm_resolve_conflict(
          working_version,
          archival_version,
          context="哪个信息更新？"
      )
  ```

- **上下文污染**:
  ```python
  # 无关记忆污染Working Context
  # 导致LLM关注错误信息

  # 解决方案：重要性过滤
  if page.importance < threshold:
      skip_page_in()  # 不重要的不载入
      add_to_auxiliary(page)  # 放入辅助上下文
  ```

**性能优化**:
```python
# 1. 分层检索
retrieval_strategy = {
    "working": "直接访问",
    "main": "内存搜索（暴力）",
    "archival": "向量检索（FAISS）+ 重排"
}

# 2. 预取策略
if query_patterns.predictable:
    # 例如：问及A后80%概率问B
    prefetch_pages(get_related_pages(query))

# 3. 压缩存储
compressed_main = llm_summarize(main_memory)  # 2×压缩率
main_memory.store(compressed_main)

# 4. 冷热分离
hot_pages = [p for p in pages if p.accessed_recently]
cold_pages = [p for p in pages if not p.accessed_recently]
page_out(cold_pages)  # 优先换出冷数据
```

## 5. 工业映射

在工业界，该机制被直接应用于MemGPT开源项目，在文档问答中支持百万字长文档。Google的Bard对话系统使用三层记忆架构（短期对话→中期上下文→长期知识库）提升多轮一致性。Anthropic的Claude-100K使用类似page机制，从知识库检索相关信息注入context。在代码助手场景，GitHub Copilot维护最近编辑文件的Working Memory，函数签名和文档在Main Memory，历史repo在Archival Storage。OpenAI的Assistant API的thread功能类似Main Context，持久存储对话历史，vector store作为Archival Store。最新推出的Claude的Contextual Retrieval结合MemGPT思想，通过prompt guidance让LLM自动识别重要信息并保存，实现更智能的page管理，在长文档分析任务中准确率提升15%。
