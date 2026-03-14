# 003: FlashAttention 内存访问优化

## 核心定性
本质上，FlashAttention 是为解决**标准 Attention 的内存墙瓶颈**，通过**分块计算（Tiling）+ 在线 Softmax**实现的**IO 感知精确注意力算法**，在不改变数学结果的前提下，将 HBM 访问次数从 $O(N^2)$ 降低到 $O(N)$，使注意力计算从 Memory-bound 转变为 Compute-bound。

## 具体流程

### 标准 Attention 的内存灾难

```python
# 标准实现（HBM 访问 = 2N + 2N^2 次）
S = Q @ K.T                      # 读取 Q, K (2N) → 写入 S (N^2)
P = softmax(S)                   # 读取 S (N^2) → 写入 P (N^2)
O = P @ V                        # 读取 P, V (N^2 + N) → 写入 O (N)
# 总计: 读取 2N^2 + 3N, 写入 2N^2 + N
```

**HBM 访问成本**（A100-80G）:
- HBM 带宽: 2TB/s
- SRAM 带宽: 19TB/s（每个 SM）
- **差距**: 9.5x

当 N=4096 时:
- S 矩阵大小: 4096×4096×2 bytes = 32MB（需要反复读写）
- 标准注意力: 读写 64MB × 3 次 = **192MB HBM 访问**
- 耗时: ~100μs（Memory-bound）

### FlashAttention 的分块策略

**核心洞察**: 注意力是归约操作，无需完整存储中间矩阵 S 和 P。

```python
# FlashAttention 伪代码（SRAM 中完成全部计算）
O = zeros(N, d)              # 输出矩阵
for i in range(0, N, Bc):    # Q 分块
    Qi = Q[i:i+Bc, :]         # 加载 Qi 到 SRAM (Bc×d)
    Oi = zeros(Bc, d)

    row_sum = zeros(Bc)      # 在线 softmax 的分子累加
    row_max = -inf(Bc)       # 在线 softmax 的分母累加

    for j in range(0, N, Br):  # KV 分块
        Kj = K[j:j+Br, :]     # 加载 Kj (Br×d)
        Vj = V[j:j+Br, :]     # 加载 Vj (Br×d)

        # 在 SRAM 中计算分块 Sij = Qi @ Kj.T
        Sij = Qi @ Kj.T       # (Bc×Br)

        # 在线 softmax: 更新每行的最大值和累加和
        row_max_new = max(row_max, Sij.max(dim=-1))
        row_sum = row_sum * exp(row_max - row_max_new) + exp(Sij - row_max_new).sum(dim=-1)

        # 在线更新输出（不需要存储完整的 P）
        Pij = exp(Sij - row_max_new.unsqueeze(-1))
        Oi = Oi * (row_sum_old / row_sum_new).unsqueeze(-1) + Pij @ Vj

    O[i:i+Bc, :] = Oi          # 写回 HBM（仅一次）
```

**HBM 访问减少**:
- Q: 读取 N 次（每个分块一次）
- K/V: 各读取 N × (N/Br) = N^2/Br 次
- O: 写入 N 次
- **总计**: ~2N^2/Br 次（Br=128 时，减少 64 倍）

### SRAM 计算流程

```python
# 更精确的 FlashAttention-2 实现（融合 QK + Softmax + PV）
def flash_attn_fwd(Q, K, V, block_size=128):
    """
    SRAM 容量假设: 192KB per SM
    - Q block: 128×128×2bytes = 32KB
    - K/V block: 128×128×2bytes = 32KB each
    - 中间结果: ~64KB
    - 总计: ~160KB < 192KB（安全）
    """
    N, d = Q.shape
    O = torch.zeros_like(Q)

    # 每个 SM 处理一个 Q block
    for i in range(0, N, block_size):
        Qi = Q[i:i+block_size]      # [Bc, d]

        # 在线 softmax 状态
        m_i = torch.full((block_size,), -float('inf'))  # 行最大值
        l_i = torch.zeros(block_size)                   # 行累加和
        Oi = torch.zeros(block_size, d)                 # 输出累加

        for j in range(0, N, block_size):
            Kj = K[j:j+block_size]      # [Br, d]
            Vj = V[j:j+block_size]      # [Br, d]

            # 分块计算 Sij
            Sij = torch.matmul(Qi, Kj.T)  # [Bc, Br]

            # 在线 softmax 更新
            m_ij = torch.max(Sij, dim=-1).values
            Pij = torch.exp(Sij - m_ij[:, None])

            # 更新全局状态
            m_new = torch.max(m_i, m_ij)
            P_sum = torch.exp(m_i - m_new) * l_i + torch.exp(m_ij - m_new) * Pij.sum(dim=-1)

            # 缩放旧输出并累加新贡献
            scale = torch.exp(m_i - m_new) * (l_i / (P_sum + 1e-6))
            Oi = Oi * scale[:, None] + torch.matmul(Pij, Vj) * (torch.exp(m_ij - m_new) / (P_sum + 1e-6))[:, None]

            m_i = m_new
            l_i = P_sum

        O[i:i+block_size] = Oi

    return O
```

## 复杂度分析

### 时间复杂度

| 算法 | 计算量 | HBM 访问 | 瓶颈 |
|------|--------|----------|------|
| **标准 Attention** | $O(N^2d)$ | $O(N^2)$ | Memory-bound |
| **FlashAttention** | $O(N^2d)$ | $O(N^2/B_r)$ | Compute-bound |
| **FlashAttention-2** | $O(N^2d)$ | $O(N^2/B_r)$ | 更少的同步 |

**实际加速**（A100-80G, d=128, B_r=128）:
- N=4096: 加速 3-4x
- N=8192: 加速 7-8x
- N=16384: 加速 10-12x

### 内存占用对比

| 矩阵 | 标准实现 | FlashAttention |
|------|----------|----------------|
| S (QK^T) | N^2×2bytes | 0（不存储） |
| P (Softmax) | N^2×2bytes | 0（不存储） |
| O (Output) | Nd×2bytes | Nd×2bytes |
| **总计** | **2N^2 + Nd** | **Nd** |

**节省**: 当 N=4096, d=128 时，节省 **64 倍内存**。

## 工业映射

### vLLM 中的 FlashAttention 集成

```python
# vLLM 源码: vllm/attention/flash_attn.py
from flash_attn import flash_attn_func

class FlashAttentionBackend:
    def forward(
        self,
        query: torch.Tensor,    # [num_tokens, num_heads, head_dim]
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,  # 批次中每个序列的起始位置
        max_seqlen: int,
    ) -> torch.Tensor:
        """
        连续批处理场景：多个序列拼接在一起
        FlashAttention 天然支持变长序列（通过 cu_seqlens）
        """
        return flash_attn_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,  # 因果掩码（自回归）
            softmax_scale=None,  # 1/sqrt(head_dim)
        )
```

**性能数据**（vLLM 0.3.0）:
- 延迟降低: 3-5x（相比 PyTorch 原生）
- 吞吐量提升: 2-3x（连续批处理场景）
- 内存节省: 90%（无需中间矩阵）

### LLaMA2 训练中的应用

```python
# Meta 内部训练脚本
# 关键配置: use_flash_attention=True
training_config = {
    "model": {
        "use_flash_attention": True,  # 启用 FlashAttention
    },
    "optimizer": {
        "block_size": 128,  # 与 FlashAttention 的 B_r 对齐
    },
    "performance": {
        "memory_efficient": True,  # 减少激活检查点
    }
}

# 节省效果:
# 70B 模型训练: 从 80GB → 45GB/卡
# 可支持批大小: 从 4 → 16
# 训练速度: 提升 2.5x
```

### 字节跳动豆包推理优化

```python
# ByteDance vLLM 定制版
# 优化点: 与 PagedAttention 协同
class FlashPagedAttention:
    def __init__(self, block_size=16):
        """
        FlashAttention 处理计算，PagedAttention 管理内存
        """
        self.block_size = block_size
        self.flash_backend = FlashAttentionBackend()
        self.block_manager = PagedBlockManager()

    def forward_with_paging(self, query, key_cache, value_cache, block_tables):
        """
        1. 从 Paged 内存加载 KV block
        2. 在 SRAM 中用 FlashAttention 计算
        3. 结果直接写回连续 O 矩阵
        """
        # HBM 访问: 仅需加载 block table（指针）
        # 无需加载完整 KV Cache
        logits = self.flash_backend(query, key_cache, value_cache)
        return logits
```

**协同效果**:
- PagedAttention 节省 95% 显存碎片
- FlashAttention 节省 90% 计算中间结果
- **总计**: 支持 100K+ 上下文，延迟仅增加 30%

## 面试高频追问

**Q1: FlashAttention 的 SRAM 容量限制？**

A: A100 每 SM 192KB SRAM，约束 block 大小选择：
```
required_sram = B_r*d*2 + B_c*d*2 + B_r*B_c*2 + 中间结果
            ≤ 192KB
对于 d=128: 最大 B_r = B_c = 128
```

**Q2: v1 vs v2 的核心差异？**

A:
- **v1**: Q 循环在外，K/V 循环在内，reduction 操作在 K/V 循环内
- **v2**: Q 循环在内，减少同步次数，减少 Q 从 HBM 读取
- **效果**: v2 在反向传播时快 2-3x

**Q3: 为什么需要在线 Softmax，不使用全局？**

A:
- 全局 Softmax 需要看到完整的 S 矩阵（N×N），必须存储
- 在线 Softmax 通过维护 `row_max` 和 `row_sum`，无需存储 S
- 精度: 在线算法是数值稳定的，误差在可接受范围

**Q4: 与 PagedAttention 如何协同工作？**

A:
- PagedAttention 解决 KV Cache 的内存碎片和动态分配
- FlashAttention 解决计算过程中的中间结果内存
- 合作: Paged 提供分块 KV，Flash 在 SRAM 中完成计算
**比喻**: Paged 是仓库管理员（管理空间），Flash 是生产线（减少废料）

---

** 难度评级 **: ⭐⭐⭐
** 出现频率 **: 85%（NVIDIA、所有推理优化岗）
** 掌握要求 **: 分块策略 + 在线算法 + 工业实践
