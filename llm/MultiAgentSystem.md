# Multi-Agent 系统架构与协作机制

---

## 1. 核心定性

**本质上，Multi-Agent 是一个为了解决"复杂任务分解与并行执行"问题，通过"消息总线 + 状态共享"机制实现的分布式认知计算架构。**

---

## 2. 具体流程

1. **任务路由 (Task Routing)**：调度器 (Orchestrator) 接收用户请求，基于 Agent 能力签名 $\phi_i$ 进行任务分解与分配；
2. **并行执行 (Parallel Execution)**：各 Agent 在隔离沙箱中异步执行子任务，通过共享状态空间 $S$ 交换中间结果；
3. **结果聚合 (Result Aggregation)**：Orchestrator 收集各 Agent 输出，进行一致性校验与冲突消解，输出最终答案。

---

## 3. 数学基础

### 3.1 Agent 能力建模

每个 Agent 抽象为一个能力函数：

$$\mathcal{A}_i: (S_t, M_{in}) \rightarrow (S_{t+1}, M_{out}, \tau_i)$$

其中：
- $S_t$: 时刻 $t$ 的全局共享状态（向量空间 $\mathbb{R}^d$）
- $M_{in}$: 输入消息（JSON / protobuf 序列化）
- $M_{out}$: 输出消息
- $\tau_i$: Agent $i$ 的执行延迟（随机变量，服从分布 $\mathcal{D}_i$）

### 3.2 协作拓扑的邻接矩阵

系统拓扑用有向图 $G=(V, E)$ 描述，邻接矩阵 $A \in \{0,1\}^{n \times n}$：

$$A_{ij} = \begin{cases} 1 & \text{if Agent } i \text{ 可向 } j \text{ 发送消息} \\ 0 & \text{otherwise} \end{cases}$$

### 3.3 一致性收敛条件

对于需要共识的 Multi-Agent 系统（如投票、协商），状态收敛要求：

$$\forall i,j: \lim_{t \to T} \| S_t^{(i)} - S_t^{(j)} \|_2 < \epsilon$$

即各 Agent 的本地状态副本在有限轮次 $T$ 内达成 $\epsilon$-一致。

### 3.4 核心通信伪代码

```python
class MessageBus:
    def publish(self, topic: str, payload: Event) -> None:
        # At-least-once delivery semantics
        for subscriber in self.subscriptions[topic]:
            subscriber.inbox.put(payload)  # async queue

class Agent:
    def run(self, state: SharedState) -> Tuple[Action, DeltaState]:
        # Read-Compute-Write cycle
        local_view = state.snapshot()
        action = self.llm.plan(local_view, self.system_prompt)
        delta = self.execute(action)
        return action, delta
```

---

## 4. 工程考量

| 维度 | Trade-off | 致命弱点 |
|------|-----------|----------|
| **通信机制** | 同步 RPC (强一致) vs 异步消息队列 (高吞吐) | 消息风暴：当 $n$ 个 Agent 全连接广播时，复杂度 $O(n^2)$，网络瞬间打满 |
| **状态共享** | 集中式 State Store (易调试) vs 分布式 CRDT (去中心) | 脑裂风险：网络分区时，各 Agent 持有冲突状态副本，无法自动合并 |
| **编排策略** | 静态 Workflow (确定性) vs 动态规划 (灵活性) | 循环依赖：Agent A 等 B，B 等 C，C 等 A，死锁无法自愈 |
| **容错设计** | 重试 + 超时 (简单) vs  Saga 补偿 (事务性) | 级联故障：单个 Agent 超时重试引发雪崩，拖垮整个系统 |

---

## 5. 工业映射

在工业界，该机制被直接应用于：

| 项目 | 核心机制 | 应用场景 |
|------|----------|----------|
| **LangGraph** | 图结构编排 + 状态机转换 | LangChain 生态中的复杂 Agent 工作流（循环、条件分支） |
| **AutoGen (Microsoft)** | Conversational Programming + 代码执行沙箱 | 多 Agent 协作编程、代码生成与自动调试 |
| **CrewAI** | 角色扮演 (Role-Playing) + 层级委托 | 模拟企业组织架构（Manager → Worker Agent）完成研究、写作任务 |
| **OpenAI Swarm** | 轻量级 Handoff 机制 | 快速原型验证、简单多 Agent 路由 |

**面试绝杀金句**：  
> *"Multi-Agent 的核心不是 Agent 数量，而是**状态一致性边界**与**通信复杂度**的博弈——当 $n>5$ 时，任何全连接拓扑都会在生产环境暴雷，必须用分层或 Pub/Sub 降维。"*

---
