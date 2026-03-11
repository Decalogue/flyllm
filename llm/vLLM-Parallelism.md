# vLLM 并行机制详解

## 1. 核心定性

**本质上，vLLM 的并行架构是一个为了在单卡显存受限下实现大模型推理吞吐最大化，通过「Token-Level 调度 + 三维并行解耦计算与存储」实现的分布式推理引擎。**

---

## 2. 具体流程

1. **请求入队 → PagedAttention 分配物理块**: 使用 Copy-on-Write 的块表管理 KV Cache，避免传统静态分配的显存浪费。
2. **调度器按 Token 粒度组 Batch**: 连续批处理 (Continuous Batching) 动态将解码阶段的新 Token 与预填充请求拼接，提高 GPU 利用率。
3. **三维并行分布式执行**: TP 切分 Layer 内矩阵计算，PP 切分 Layer 间流水线，DP 复制多实例承载并发，通过 NCCL 通信原语协同。

---

## 3. 数学基础

### 3.1 PagedAttention 显存分配

$$
\text{显存占用} = \lceil \frac{S_{\text{max}}}{B} \rceil \times B \times L \times H \times d_{\text{head}} \times 2 \times \text{sizeof(dtype)}
$$

其中：
- $S_{\text{max}}$: 最大序列长度
- $B$: 块大小 (Block Size, 默认 16)
- $L$: 模型层数
- $H$: 注意力头数
- $d_{\text{head}}$: 每个头的维度
- 因子 2: 分别存储 K 和 V

### 3.2 连续批处理吞吐公式

$$
\text{Throughput} = \frac{\sum_{i=1}^{N} T_i}{\Delta t}, \quad \text{其中 } T_i \text{ 为请求 } i \text{ 在批次中生成的 Token 数}
$$

### 3.3 Tensor Parallel 通信量

$$
\text{AllReduce 数据量} = 2 \times (L-1) \times \text{hidden\_size} \times \text{batch\_size} \times \text{seq\_len} \times \text{sizeof(dtype)}
$$

---

## 4. 工程考量

| 并行策略 | Trade-off | 致命弱点 |
|---------|-----------|---------|
| **Tensor Parallel (TP)** | 牺牲通信带宽换取单卡显存减负 | TP 度>8 时，AllReduce 成为瓶颈；小 batch 下通信开销 > 计算收益 |
| **Pipeline Parallel (PP)** | 牺牲流水线气泡 (Bubble) 换取层间解耦 | 当 `num_micro_batch` 不足或序列长度差异大时，Bubble 占比可达 30%+ |
| **Data Parallel (DP)** | 牺牲显存冗余换取并发吞吐 | DP 实例间 KV Cache 不共享，长上下文场景下显存爆炸 |

**vLLM 的致命约束**: 
- **TP 与 PP 互斥于单节点**: TP 要求 NVLink 级带宽，跨节点 TP 会使 latency 恶化 10x+
- **调度器单点瓶颈**: 当 QPS > 1000 时，Python GIL 下的调度逻辑成为天花板
- **Prefix Caching 碎片**: 共享前缀的 Copy-on-Write 在 high 并发变长输入下产生严重显存碎片

---

## 5. 工业映射

在工业界，该机制被直接应用于 **vLLM 的 `LLMEngine` + `Worker` 架构** 中：

- **Tensor Parallel**: 通过 `torch.distributed` + `tensor_parallel` 模块，将 Linear 层切分为 Column/Row Parallel，配合 AllReduce/AllGather 完成 MHA/MLP 计算
- **Pipeline Parallel**: 通过 `PipelineParallelRunner` 将模型按层均分，使用 `recv_forward/send_forward` 进行 stage 间点对点通信
- **Data Parallel**: 通过多 `Worker` 实例独立持有一份模型权重，由中央 `Scheduler` 统一调度，配合 `Ray` 或 ` multiprocessing` 实现进程隔离

**典型部署范式**: 
- 8xA100 节点内: TP=8, PP=1, DP=N (按并发需求扩展节点)
- 跨节点大模型 (70B+): TP=8 (节点内) × PP=2~4 (跨节点) × DP 按需扩展
