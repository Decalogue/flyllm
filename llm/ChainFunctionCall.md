# Function Call 链式调用与依赖图设计

## 1. 核心定性

**本质上，Function Call 链式调用是一个为了解决"多步推理任务中工具依赖编排"问题，通过"有向无环图(DAG)调度 + 拓扑排序执行"机制实现的异步执行框架。**

---

## 2. 具体流程

1. **依赖建模**：将每个 Function Call 抽象为 DAG 节点，边表示数据依赖（A 的输出 → B 的输入）
2. **拓扑执行**：通过 Kahn 算法或 DFS 检测环并生成拓扑序，就绪节点入队，异步并行调度
3. **上下文注入**：每步执行结果通过 `call_id` 回填到上下文，LLM 根据中间态决定后续调用路径

---

## 3. 数学基础

### 依赖图结构定义

$$G = (V, E, \lambda, \tau)$$

其中：
- $V = \{v_1, v_2, ..., v_n\}$：Function Call 节点集合，$n = |V|$
- $E \subseteq V \times V$：依赖边，$(v_i, v_j) \in E$ 表示 $v_j$ 依赖 $v_i$ 的输出
- $\lambda: V \rightarrow \Sigma^*$：节点到函数签名的映射
- $\tau: V \rightarrow \mathbb{N}$：节点执行耗时估计

### 拓扑排序合法性判定

$$\forall (v_i, v_j) \in E: \text{pos}(v_i) < \text{pos}(v_j)$$

执行并行度由**反链长度**决定：

$$\text{Parallelism} = \max_{k} |\{v \in V : \text{indegree}(v) = 0 \land \text{layer}(v) = k\}|$$

### 状态转移方程

$$
\text{State}(v) = 
\begin{cases} 
\text{PENDING} & \text{if } \exists (u,v) \in E: \text{State}(u) \neq \text{SUCCESS} \\
\text{RUNNING} & \text{if ready and assigned to worker} \\
\text{SUCCESS} & \text{if execution completed} \\
\text{FAILED} & \text{if error or timeout}
\end{cases}
$$

### 核心调度伪代码

```python
class DependencyGraph:
    def execute(self) -> Dict[CallID, Result]:
        in_degree = {v: len(self.deps[v]) for v in self.nodes}
        queue = deque([v for v in self.nodes if in_degree[v] == 0])
        results = {}
        
        while queue:
            ready_batch = list(queue)  # 同层并行
            queue.clear()
            
            # 并行执行就绪节点
            futures = {v: self.executor.submit(v.func, v.args) for v in ready_batch}
            
            for v, future in futures.items():
                results[v.id] = future.result()
                
                # 拓扑更新：释放下游依赖
                for downstream in self.reverse_deps[v]:
                    in_degree[downstream] -= 1
                    if in_degree[downstream] == 0:
                        queue.append(downstream)
                        
        return results
```

---

## 4. 工程考量

| Trade-off | 选择 | 代价 |
|-----------|------|------|
| **DAG vs 顺序执行** | 并行化提升吞吐 | 引入死锁风险，需严格环检测 |
| **同步 vs 异步等待** | 异步非阻塞 | 状态机复杂度↑，调试难度↑ |
| **静态 vs 动态图** | 运行时动态构建 | LLM 推理延迟增加，但灵活性高 |

**致命弱点**：
- **循环依赖死锁**：LLM 生成的 Call Graph 若存在隐式循环依赖（A→B→C→A），拓扑排序失效
- **级联失效**：单节点超时/报错会导致整个子图失败，需设计**断路器 + 降级策略**
- **上下文爆炸**：长链调用导致上下文长度指数增长，触发 token 限制

---

## 5. 工业映射

在工业界，该机制被直接应用于：
- **OpenAI Assistants API** 的 `run` 对象中，通过 `required_action` 状态机处理多步 Tool Call
- **LangChain** 的 `RunnableParallel` 和 `RunnableMap` 实现 DAG 编排
- **Dify/Coze** 等工作流引擎的"节点依赖线"可视化编排

用于应对**复杂 Agent 任务**（如：查天气→推荐餐厅→预订座位→发送日历邀请）中的**多工具协调与高并发执行**场景。
