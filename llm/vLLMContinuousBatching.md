# vLLM Continuous Batching 实现原理与动态调度机制

---

## 1. 核心定性

**本质上，Continuous Batching 是为了解决 LLM 推理中"每个请求生成长度差异巨大"导致的 GPU 算力空转问题，通过将传统 Batch 粒度从"请求级"拆解到"Token 级"实现的细粒度流水线调度结构。**

---

## 2. 具体流程

1. **拆解执行单元**：当某个请求在 Batch 中完成生成（遇到 EOS 或达到长度上限），vLLM 立即将其踢出 Batch，从等待队列拉入新请求补位，而非等待 Batch 中所有请求完成。

2. **动态内存调度**：基于 PagedAttention 的 Block Table 机制，将 KV Cache 分页管理，新请求可复用已释放请求的物理显存块，无需等待 Batch 完全清空再统一重分配。

3. **迭代级调度（Iteration-Level Scheduling）**：每次 Forward Pass 结束后，Scheduler 重新评估 Batch 组成，根据当前 GPU 显存占用和等待队列长度，决定下一轮的 Batch Size 和成员。

---

## 3. 数学基础

### 状态机模型

定义系统状态 $S_t$ 在第 $t$ 个迭代步：

$$S_t = \langle \mathcal{B}_t, \mathcal{Q}_t, \mathcal{M}_t \rangle$$

其中：
- $\mathcal{B}_t$：当前正在运行的请求集合（Batch），$|\mathcal{B}_t| = B_t$
- $\mathcal{Q}_t$：等待队列中的请求集合
- $\mathcal{M}_t$：Block Table 映射，$\mathcal{M}_t: r_i \mapsto \{b_1, b_2, ..., b_k\}$，将请求 $r_i$ 映射到其占用的物理块集合

### 调度决策函数

每次迭代后，Scheduler 执行：

$$\mathcal{B}_{t+1} = (\mathcal{B}_t \setminus \mathcal{D}_t) \cup \mathcal{N}_t$$

其中：
- $\mathcal{D}_t = \{r \in \mathcal{B}_t \mid \text{finished}(r) = \text{true}\}$：本迭代完成生成的请求
- $\mathcal{N}_t \subseteq \mathcal{Q}_t$：从等待队列新加入的请求，需满足显存约束：
  $$\sum_{r \in \mathcal{B}_{t+1}} \text{Mem}(r) \leq M_{\text{max}}$$

### 吞吐率对比

**静态批处理吞吐率**：
$$\text{Throughput}_{\text{static}} = \frac{\sum_{i=1}^{B} L_i}{B \cdot \max(L_i) \cdot T_{\text{iter}}}$$

**Continuous Batching 吞吐率**：
$$\text{Throughput}_{\text{cont}} = \frac{\sum_{r \in \mathcal{R}} L_r}{T_{\text{total}} \cdot T_{\text{iter}}}$$

其中 $L_i$ 为请求 $i$ 的生成长度，$T_{\text{iter}}$ 为单次迭代耗时。当生成长度方差 $\text{Var}(L)$ 增大时，Continuous Batching 优势呈指数级放大。

---

## 4. 工程考量

### Trade-offs

| 维度 | 收益 | 代价 |
|------|------|------|
| **GPU 利用率** | 消除"短请求等长请求"的 Bubble，利用率从 ~40% 提升至 ~90% | 调度器 CPU 开销增加，每次迭代需遍历 Batch 做状态判断 |
| **显存效率** | PagedAttention 实现 Block 级细粒度复用，显存碎片率 < 5% | 需维护 Block Table 映射表，增加元数据管理复杂度 |
| **延迟公平性** | 全局吞吐最优 | 单个请求 P99 延迟不可控，长尾请求可能被饥饿 |

### 致命弱点

1. **前缀匹配的局部性陷阱**：若新请求与当前 Batch 中请求的前缀相似度低，无法复用 Prefix Cache，导致大量重复计算
2. **Chunked-Prefill 的调度耦合**：在 Decode 阶段插入新请求的 Prefill 会阻塞整批 Decode，产生 "Prefill-Decode Interference"
3. **显存碎片极限**：当请求的 Context Length 分布极度离散时（如 1K vs 128K），Block Table 的伙伴分配器会产生不可合并的碎片

---

## 5. 工业映射

**在 vLLM 中**，该机制被直接实现于 `scheduler.py` 的 `_schedule()` 方法与 `model_runner.py` 的 `execute_model()` 中：

- **Orchestration 层**：`Scheduler` 类维护 `self.running`（当前 Batch）、`self.swapped`（被抢占到 CPU 的请求）、`self.waiting`（等待队列）三个队列，每次迭代执行 `self._schedule()` 重新决策
- **Execution 层**：`Worker` 通过 `execute_model()` 执行 Forward，返回的 `SamplerOutput` 包含每个请求的 `finish_reason`，Scheduler 据此决定哪些请求踢出 Batch
- **Memory 层**：`BlockManagerV2` 维护 `self.block_tables: Dict[str, BlockTable]`，实现物理 Block 的引用计数与延迟释放

**被用于应对**：
- 生产环境中 "短 Prompt + 长生成" 与 "长 Prompt + 短生成" 请求混合的高吞吐在线服务场景（如 ChatGPT 类 API）
- 峰值流量下 GPU 显存利用率最大化需求，实测在 ShareGPT 负载下相比静态批处理吞吐提升 **3-5x**
