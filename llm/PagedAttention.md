# PagedAttention: vLLM 的 KV Cache 内存管理机制

## 1. 核心定性

**本质上，PagedAttention 是为了解决 LLM 推理中 KV Cache 内存碎片化和预留浪费问题，通过借鉴 OS 虚拟内存的页式管理机制，实现非连续、动态分配的 KV Cache 存储结构。**

---

## 2. 具体流程

1. **逻辑分页**：将每个序列的 KV Cache 划分为固定大小的逻辑块（Block），每个 Block 存储固定数量的 Token 的 Key 和 Value 向量。

2. **物理页映射**：维护一个逻辑块到物理内存块的映射表（Block Table），物理块从预分配的 Block Engine 池中动态申请，逻辑上连续的块物理上可以不连续。

3. **Attention 计算时动态拼装**：通过 Block Table 在 GPU Kernel 中按逻辑顺序读取非连续的物理块，使用自定义 CUDA Kernel 完成分页内存上的 Attention 计算。

---

## 3. 数学基础

**Block 地址映射公式：**

$$\text{PhysicalAddr}(b, t) = \text{BlockTable}[b] \times B + (t \bmod B)$$

其中：
- $b$: 逻辑块索引（Block Index）
- $t$: Token 在序列中的绝对位置
- $B$: 每个 Block 容纳的 Token 数量（通常为 16）
- $\text{BlockTable}$: 逻辑块 → 物理块索引的映射数组

**内存分配状态（核心数据结构）：**

```cpp
struct Block {
    int ref_count;           // 引用计数（支持 Copy-on-Write）
    torch::Tensor data;      // [2, num_heads, head_size, block_size] (K/V)
};

struct BlockTable {
    vector<int> physical_blocks;  // 逻辑块到物理块的映射
    int num_tokens;               // 当前序列实际长度
};
```

**PagedAttention Kernel 核心逻辑：**

```python
# 简化示意
for block_idx in logical_blocks:
    phy_block = block_table[block_idx]
    k_block = key_cache[phy_block]      # [num_heads, head_size, block_size]
    v_block = value_cache[phy_block]
    # 计算当前 block 内的 attention scores
    scores = softmax(q @ k_block.T / sqrt(d_k))
    out += scores @ v_block
```

---

## 4. 工程考量

| Trade-off | 说明 |
|-----------|------|
| **空间碎片化 ↓ vs 额外元数据 ↑** | 消除了内存碎片，但需要维护 Block Table（~0.1% 额外开销） |
| **动态扩容 ✓ vs 连续内存访问 ✗** | 支持动态增长，但 GPU Kernel 需多次非连续内存访问（通过 L2 Cache 缓解） |
| **Copy-on-Write 共享 ↓ vs 引用计数开销 ↑** | Beam Search/并行解码时共享 KV Cache，但需原子操作维护 ref_count |

**致命弱点：**

1. **短序列 overhead**：当序列长度 $L \ll B$ 时，Block 内大量空闲槽位，内存利用率反而下降
2. **GQA/MQA 适配复杂度**：Grouped Query Attention 中 K/V 头数减少，需要调整 Block 布局以避免 bank conflict
3. **Prefix Caching 的 Hash 冲突**：长共享前缀场景下，Block 级别的 Hash 键可能导致伪命中

---

## 5. 工业映射

在工业界，该机制被直接应用于 **vLLM 的 LLMEngine / Worker 模块** 中，用于应对 **高并发、变长序列推理** 场景。

- **vLLM**: `vllm/worker/cache_engine.py` 实现 Block Allocator，`vllm/attention/ops/paged_attn.py` 实现 CUDA Kernel
- **SGLang**: 在 PagedAttention 基础上引入 **RadixAttention**，用 Radix Tree 管理共享前缀，实现更激进的 KV Cache 复用
- **TensorRT-LLM**: 引入 **Inflight Batching + Paged KV Cache**，支持单 batch 内不同序列长度的高效并行

---
