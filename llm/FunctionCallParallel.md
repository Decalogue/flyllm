# Function Call 并行调用实现与管理

## 1. 核心定性

**本质上，Function Call 的并行调用是一个通过 `并发槽位分配 + 异步 I/O 多路复用` 机制，在单请求生命周期内最大化工具吞吐量的调度结构。**

---

## 2. 具体流程

1. **意图解析**：LLM 生成包含多个 `tool_calls` 的响应，每个调用块携带独立 `id`（如 `call_abc123`）和参数
2. **并行分发**：执行层将各调用映射到线程池/协程池，通过 `asyncio.gather` 或 `Promise.all` 并发执行独立工具
3. **聚合回填**：收集各调用结果，按原 `id` 回填至上下文，触发二次推理生成最终回复

---

## 3. 数学基础

### 并行调度模型

$$
T_{total} = T_{parse} + \max_{i \in [1,n]}(T_{exec_i}) + T_{aggregate}
$$

其中：
- $T_{parse}$: LLM 生成 tool_calls 的推理耗时
- $T_{exec_i}$: 第 $i$ 个工具的执行耗时（IO-bound 或 CPU-bound）
- $n$: 并行调用数量
- $T_{aggregate}$: 结果聚合与二次推理耗时

### 资源约束不等式

$$
\sum_{i=1}^{n} M_i \leq M_{max} \quad \land \quad \forall i, T_{exec_i} \leq T_{timeout}
$$

- $M_i$: 第 $i$ 个调用的内存/连接资源占用
- $M_{max}$: 系统资源上限
- $T_{timeout}$: 单调用超时时限

### 核心伪代码结构

```python
async def parallel_execute(tool_calls: List[ToolCall]) -> List[Result]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)  # 限流保护
    
    async def bounded_execute(call):
        async with semaphore:
            return await execute_tool(call)
    
    # 并发执行 + 超时熔断
    results = await asyncio.gather(
        *[bounded_execute(c) for c in tool_calls],
        return_exceptions=True
    )
    return [r for r in results if not isinstance(r, Exception)]
```

---

## 4. 工程考量

| 维度 | Trade-off | 致命弱点 |
|------|-----------|----------|
| **并发度** | 高并发提升吞吐，但加剧资源争抢 | 下游服务限流/雪崩时，级联超时导致整请求失败 |
| **一致性** | 并行缩短延迟，但丢失调用间顺序语义 | 存在依赖关系的工具（如 B 依赖 A 的结果）无法并行，强行并行导致数据竞态 |
| **容错性** | `return_exceptions=True` 实现部分成功 | 无重试策略时，单点失败即整盘皆输 |
| **上下文爆炸** | 多结果回填丰富上下文，但加速 token 消耗 | 工具返回大对象（如数据库全表）时，触发上下文截断，关键信息丢失 |

---

## 5. 工业映射

在工业界，该机制被直接应用于：

- **OpenAI API** 的 `tools` / `functions` 模式：单次响应可携带最多 **128 个** `tool_calls`，由客户端并发执行
- **LangChain 的 `RunnableParallel`**：通过 `asyncio.gather` 编排多工具链路，用于 RAG 场景下的 "向量检索 + 网页抓取 + SQL 查询" 并行化
- **Claude Computer Use**：多工具（`bash`、`edit`、`browser`）并发执行，通过 `tool_use_id` 精准回填执行结果
- **K8s Operator 的 Reconcile 循环**：并行调用多 API（CRD 状态查询 + Secret 读取 + Pod List），最终聚合决策
