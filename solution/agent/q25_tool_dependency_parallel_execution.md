# 工具调用的依赖图并行执行

## 1. 核心定性

本质上，工具调用依赖图并行执行是通过拓扑排序识别可并行调用，使用异步执行框架并发提交独立工具调用，最后按依赖顺序合并结果，将串行执行时间从O(N)降低到O(critical_path)的调度优化系统。

## 2. 具体流程

1. **依赖分析**: 解析工具调用中的参数依赖（一个调用的输出作为另一个输入）
2. **拓扑排序**: 构建DAG并计算可并行执行批次
3. **并发执行**: 提交独立调用到线程池/协程池
4. **结果合并**: 收集输出并注入依赖调用，继续下一轮执行

## 3. 数学基础

**依赖图构建**:
```python
G = (V, E)
V = {call₁, call₂, ..., callₙ}  # 工具调用节点
E = {(callᵢ, callⱼ) | callᵢ.output ∈ callⱼ.inputs}  # 依赖边

# 入度计算
indegree(callⱼ) = |{callᵢ | (callᵢ, callⱼ) ∈ E}|
```

**拓扑排序批次**:
```python
# Kahn算法计算可并行批次
batches = []
queue = {call | indegree(call) == 0}  # 初始可执行节点

while queue:
    batch = queue  # 当前批次全部可并行
    batches.append(batch)

    next_queue = set()
    for call in batch:
        for neighbor in G[call]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                next_queue.add(neighbor)

    queue = next_queue

# 批次数量 = 关键路径长度
num_batches = len(batches)
```

**执行时间优化**:
```python
# 串行执行时间（假设每个调用耗时t）
T_serial = n · t

# 并行执行时间（基于拓扑批次）
T_parallel = Σ_{batch ∈ batches} max_{call ∈ batch} t(call)

# 加速比
speedup = T_serial / T_parallel = n / num_batches

# 示例：5个调用，依赖链 call₁→call₂, call₃→call₄
# 可并行批次：[{call₁, call₃}, {call₂, call₄}, {call₅}]
# speedup = 5/3 ≈ 1.67
```

**资源约束调度**:
```python
# 当并行调用有资源限制时
max_concurrency = min(resource_limit, len(batch))

# 最优批次划分（装箱问题变种）
minimize Σ batches
subject to: |batch_i| ≤ max_concurrency

# 贪心算法
for call in topological_order:
    assigned = False
    for batch in batches:
        if all_dependencies_met(call, batch) and len(batch) < max_concurrency:
            batch.append(call)
            assigned = True
            break
    if not assigned:
        batches.append([call])
```

**动态关键路径调整**:
```python
# 执行时间不确定时动态调整
actual_duration(call) = estimated_duration + δ

# 重新计算后续批次
if actual_duration > threshold:
    # 长任务阻塞，启动备用方案
    if has_alternative(call):
        spawn_alternative(call)  # 同时执行备选方案
        use_first_completed()
```

## 4. 工程考量

**Trade-off**:
- 牺牲：内存（需要存储中间结果）
- 换取：执行时间（通常加速2-5倍）
- 增加：实现复杂度（异步编程和状态管理）

**致命弱点**:
- **循环依赖**:
  ```python
  # 错误：call₁依赖call₂，call₂依赖call₁
  call₁.input = call₂.output
  call₂.input = call₁.output

  # 检测与处理
  if has_cycle(G):
      strongly_connected = find_scc(G)
      for scc in strongly_connected:
          if len(scc) > 1:
              # 将SCC合并为单节点（需要LLM合并逻辑）
              merged_call = llm_merge_calls(scc)
              replace_with_merged(G, scc, merged_call)
  ```

- **资源竞争**:
  ```python
  # 多个调用竞争同一资源（如数据库连接）
  # 导致实际并发度远低于理论值

  # 解决方案：资源感知调度
  resource_groups = group_calls_by_resource(batch)
  for resource, calls in resource_groups.items():
      semaphore = get_resource_semaphore(resource)
      await execute_with_limit(calls, semaphore)
  ```

- **动态参数**:
  ```python
  # 参数在运行时才能确定（如需要根据call₁输出决定call₂参数）
  # 无法在启动前构建完整DAG

  # 解决方案：两阶段执行
  phase1_calls = {call for call in V if all_params_static(call)}
  phase2_calls = V - phase1_calls

  results = await execute_batch(phase1_calls)
  updated_graph = resolve_dynamic_params(phase2_calls, results)
  await execute_batch(updated_graph)
  ```

- **错误级联**:
  ```python
  # 一个调用失败导致依赖链全部失败
  # 示例：search失败 → write_report失败

  # 解决方案：断点续传 + 降级策略
  for call in topological_order:
      try:
          result = await execute(call)
      except Exception as e:
          if is_critical(call):  # 关键调用
              raise  # 整体失败
          else:
              # 非关键调用，使用默认值继续
              result = get_default_output(call)
              mark_as_failed(call)
  ```

**性能优化**:
```python
# 1. IO密集型调用使用异步
async def execute_batch(calls):
    tasks = [asyncio.create_tool_task(call) for call in calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(zip(calls, results))

# 2. CPU密集型调用使用线程池
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_data, call) for call in batch]
    results = [f.result() for f in futures]

# 3. 缓存加速
@lru_cache(maxsize=1024)
def cached_execution(tool_name, params_hash):
    return execute_tool(tool_name, params)

# 4. 结果预取
prefetch_calls = predict_next_needed_calls(current_batch)
asyncio.create_task(execute_batch(prefetch_calls))  # 后台执行
```

## 5. 工业映射

在工业界，该机制被直接应用于LangChain的AsyncTools，通过调用链条自动识别可并行步骤。Hugging Face的Inference Endpoints在batch inference时并行调用多个模型，将多模型推理时间从线性降低。GitHub Actions的job并行化使用DAG定义workflow，自动识别可并行执行的任务。在数据科学平台中，Kubeflow Pipelines基于DAG并行执行数据预处理、训练和评估步骤，将ML pipeline从小时级优化到分钟级。Uber的Cadence Workflows协调微服务调用，通过依赖分析并行执行独立查询，提升整体API性能3倍。最新的Temporal.io引入workflow task并行化，在订单处理系统中同时执行库存检查、价格计算和运费估算，将下单延迟从800ms降低到200ms。
