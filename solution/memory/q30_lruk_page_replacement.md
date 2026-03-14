# 分层记忆的 LRU-K 页面替换

## 1. 核心定性

本质上，分层记忆的LRU-K页面替换是在LRU（最近访问）基础上增加访问频率因素，通过Score = w₁·R + w₂·1/Δt + w₃·I计算每个记忆的保留优先级，动态管理Working/Main/Archival三层存储的内存管理算法。

## 2. 具体流程

1. **统计访问**: 记录每个记忆的访问次数（Kth-to-last access）
2. **计算得分**: 综合最近访问时间、访问频率、重要性计算优先级
3. **页面替换**: 内存满时替换score最低的记忆到下层
4. **动态调整**: 根据命中率反馈调整权重参数

## 3. 数学基础

**LRU-K评分**:
```python
Score(M) = w₁·R(M) + w₂·(1/Δt(M)) + w₃·I(M)

其中：
- R(M): Recency，最近一次访问至现在的时间间隔
- Δt(M): 第K次最近访问的时间间隔（访问频率）
- I(M): Importance，重要性评分
- w₁, w₂, w₃: 权重（通常0.4, 0.3, 0.3）
```

**时间间隔计算**:
```python
# 记录最近K次访问时间
access_history[M] = [t₁, t₂, t₃, ..., t_K]  # 按时间倒序

current_time = now()
R(M) = current_time - access_history[M][0]  # 最近访问间隔

if len(access_history[M]) >= K:
    # 第K次访问间隔（越小说明频率越高）
    Δt(M) = access_history[M][0] - access_history[M][K-1]
else:
    # 访问不足K次，给默认大值
    Δt(M) = float('inf')
```

**LRU-K vs LRU差异**:
```python
# 示例：两种访问模式
# Pattern A: a,a,a,a,b,b,b,c,c  （集中访问a→b→c）
# Pattern B: a,b,c,a,b,c,a,b,c   （交替访问）

# LRU（仅考虑最近访问）:
# Pattern A: 访问c后，a被淘汰（尽管a总访问次数多）
# Pattern B: 访问c后，a被淘汰（交替模式下表现差）

# LRU-K（考虑K次访问频率）:
# Pattern A: K=2时，Δt(a)=10, Δt(b)=10, Δt(c)=∞
#           score(a)最高，优先保留（总访问多）
# Pattern B: Δt(a)=Δt(b)=Δt(c)=20
#           频率相同，退化为LRU
```

**分层替换策略**:
```python
# 三层：Working → Main → Archival
# 命中率：Working(90%) > Main(70%) > Archival(50%)

# Working层满 → 换出到Main层
if working_memory.full():
    victim = min(working_memory, key=Score)
    demote(victim, to="main")

# Main层满 → 换出到Archival层
if main_memory.full():
    victim = min(main_memory, key=Score_w)

    # Working层权重不同
    Score_w = working_w₁·R + working_w₂·1/Δt + working_w₃·I
    Score_m = main_w₁·R + main_w₂·1/Δt + main_w₃·I

    if Score_w(victim) < threshold_cold:
        demote(victim, to="archival")  # 真正冷数据
    else:
        keep_in_main(victim)  # 还保留在Main
```

**虚拟地址映射**:
```python
# 虚拟地址空间（连续）→ 物理地址（分散）
virtual_memory_space = [0, MAX_ADDRESS]

page_table = {
    virtual_page_id: {
        "physical_location": db_id | ram_address,
        "present": bool,      # 是否在内存
        "modified": bool,     # 是否修改
        "score": float        # LRU-K评分
    }
}

# 地址转换开销
T_translation =
    0ms if TLB_hit  # Translation Lookaside Buffer
    1ms if in_memory
    10ms if in_storage
```

**自适应K值**:
```python
# K不是固定值，根据访问模式动态调整
K_optimal = argmin(K → prediction_accuracy(K))

# 计算在不同K下的命中率
for K in range(1, 5):
    hits[K] = simulate_lru_k(memory_trace, K)

best_K = argmax(hits)

# 典型值：K=2（平衡最近性和频率）
```

## 4. 工程考量

**Trade-off**:
- 增加：空间开销（记录访问历史）
- 换取：命中率提升（LRU-K比LRU高10-20%）
- 平衡：K值选择（K越大，内存开销越大）

**致命弱点**:
- **K值选择困难**:
  ```python
  # K=1：退化为LRU，不考虑频率
  # K=2：平衡最近性和频率
  # K>2：需要更久历史数据，冷启动问题

  # 解决方案：自适应K
  if memory.full():
      K = 2  # 保守值
  elif len(unique_access_patterns) > threshold:
      K = 3  # 复杂模式需要更大K
  else:
      K = 2
  ```

- **内存开销**:
  ```python
  # K=2时，每个记忆需记录2次访问时间
  overhead_per_memory = K × sizeof(timestamp) = 16 bytes

  # 100K记忆 → 1.6MB开销（可接受）

  # 压缩存储
  store_delta = True  # 存时间间隔而非绝对时间
  access_interval = t₁ - t₂  # 差值通常较小，压缩效率高
  ```

- **扫描抗性不足**:
  ```python
  # 顺序扫描（scan）导致所有页面被访问
  # 错误地提升所有页面的频率

  # 解决方案：频率上限
  MAX_FREQUENCY_CAP = 100
  access_history[M] = access_history[M][-MAX_FREQUENCY_CAP:]

  # 或使用LRU-2（只考虑倒数第二次访问）
  if scan_pattern_detected():
      reduce_weight_frequency()  # 降低频率权重
  ```

- **突发访问**:
  ```python
  # 短时间内高频访问（如循环）
  # 误认为是热数据

  # 解决方案：访问间隔归一化
  Δt_normalized = Δt / (current_time - first_access)
  # 真实热数据的Δt_normalized会较小
  ```

**预取优化**:
```python
# 在换入数据时，预取可能访问的相关数据
if page_load(M):
    related = find_related_pages(M)  # 相似度>0.8
    for r in related[:3]:  # 预取3个相关
        if r not in working_memory:
            prefetch_queue.put(r)

# 批量换入
if len(prefetch_queue) >= batch_size:
    batch_page_in(prefetch_queue)
```

**冷热数据分离**:
```python
# 独立的冷热链表
hot_list = LRU(working_memory)     # 热数据
warm_list = LRU_K(main_memory)     # 温数据（LRU-K管理）
cold_storage = archival_storage    # 冷数据

# 数据流动
# hot → warm: 在hot中停留超过阈值
# warm → cold: LRU-K评分持续低于阈值
# cold → warm: 被重新访问，频率回升
```

## 5. 工业映射

在工业界，该机制被直接应用于操作系统虚拟内存管理，Linux使用LRU近似算法管理page cache。Redis的eviction策略（volatile-lru, allkeys-lru）使用LRU淘汰键。MySQL的InnoDB buffer pool使用改进LRU（分young和old区）管理数据页。在LLM场景中，vLLM的PagedAttention使用类似LRU的块管理，回收最久未使用的KV cache。MongoDB的WiredTiger存储引擎使用LRU管理cache，优化读写性能。最新推出的Mongodb智能内存管理使用LRU-K识别工作集（working set），在OLTP场景下提升随机读性能40%。
