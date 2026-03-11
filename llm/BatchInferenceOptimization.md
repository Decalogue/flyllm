# 批量推理优化与动态批处理策略

## 1. 核心定性

本质上，批量推理优化是为了解决 **GPU 内存带宽瓶颈** 与 **请求长度异构** 的矛盾，通过**动态请求聚合**与**流水线化调度**，在延迟与吞吐之间做帕累托最优的动态平衡。

## 2. 具体流程

1. **请求分桶准入**：按输入/输出长度分布将请求划分到不同 Bucket，Padding 对齐至桶内最大长度后入队
2. **Continuous/Inflight 调度**：在前向传播迭代间隙，动态重组 Batch —— 新请求插入、已完成请求剔除，保持 GPU 持续饱和
3. **内存复用与驱逐**：采用 PagedAttention 机制物理离散存储 KV Cache，请求完成立即释放页框，空缺由等待队列填充

## 3. 数学基础

### 3.1 Batch 填充效率

$$\eta = \frac{\sum_{i=1}^{B} L_i}{B \cdot L_{max}} \times 100\%$$

其中：
- $B$: 当前 Batch 大小
- $L_i$: 第 $i$ 个请求的实际序列长度
- $L_{max}$: Batch 内最大序列长度（含 Padding）

### 3.2 吞吐与延迟权衡

$$\text{Throughput} = \frac{B_{eff}}{\frac{L_{max}}{P} + T_{sched}}$$

$$\text{Latency}_{p99} = Q_{wait} + \frac{L_{max}}{P} \cdot N_{iter}$$

其中：
- $B_{eff} = \sum_{i=1}^{B} \mathbb{1}_{active(i)}$: 有效并发数（排除已完成的空转槽位）
- $P$: GPU 峰值算力 (tokens/s)
- $T_{sched}$: 调度开销
- $Q_{wait}$: 排队等待时间
- $N_{iter}$: 生成总迭代次数

### 3.3 动态批处理核心状态机

```
State: RUNNING / WAITING / FINISHED
Transition:
  - WAITING → RUNNING:  当 GPU 内存充足且 Batch 槽位空闲
  - RUNNING → FINISHED: 当 request.output_len ≥ request.max_tokens 或 EOS
  - RUNNING → RUNNING:  每轮迭代后，若 Batch 未满，尝试从 WAITING 队列填充
```

### 3.4 PagedAttention 内存管理

$$\text{Physical Blocks} = \lceil \frac{\sum_{j=1}^{M} S_j \cdot D \cdot N_{layer} \cdot 2}{B_{size}} \rceil$$

其中：
- $M$: 并发请求数
- $S_j$: 第 $j$ 个请求的序列长度
- $D$: 注意力头维度 (hidden_size / num_heads)
- $N_{layer}$: 层数
- $B_{size}$: KV Cache 块大小（通常 16 tokens）
- 系数 2: K 和 V 双份存储

## 4. 工程考量

### Trade-off
| 收益 | 代价 |
|------|------|
| 吞吐提升 5-20x | 单请求延迟上升（Head-of-Line Blocking） |
| GPU 利用率接近 100% | 内存碎片增加，需 PagedAttention 缓解 |
| 隐藏传输延迟（Async H2D/D2H） | 调度复杂度激增，CPU 瓶颈风险 |

### 致命弱点
1. **Straggler 拖尾效应**：Batch 中只要有一个长序列请求，所有短序列请求必须等待至其完成，导致尾部延迟恶化
2. **Prefill vs Decode 争抢**：Prefill 阶段 compute-bound，Decode 阶段 memory-bound，同 Batch 混合调度会导致资源错配
3. **显存 OOM 雪崩**：动态批处理在高峰期可能超量接纳，当 KV Cache 累积至临界点，触发 OOM 导致整批崩溃

## 5. 工业映射

| 机制 | 开源实现 | 应用场景 |
|------|----------|----------|
| **Continuous Batching** | vLLM (`vllm/core/scheduler.py`) | 高并发在线服务，支持 decode 阶段动态换入换出 |
| **Inflight Batching** | TensorRT-LLM (`batch_manager`) | NVIDIA GPU 极致性能，支持 CUDA Graph 优化 |
| **PagedAttention** | vLLM (`vllm/attention/`) | 解决 KV Cache 内存碎片，提升 2-4x 并发 |
| **Split-Fuse** | DeepSeek/Moonshot 内部框架 | Prefill 与 Decode 分离调度，消除资源争抢 |
