# 001: MHA → GQA → MQA 的演进与改造

## 核心定性
本质上，MHA→GQA→MQA 是**在注意力机制中逐步压缩 KV 头数量**的演进，通过**牺牲少量模型表达能力**来**显著降低推理时的内存带宽和计算量**，是解决长上下文推理瓶颈的关键工程优化。

## 具体流程
1. **MHA（标准多头注意力）**：每个 Query 头都有独立的 KV 头，参数量和计算量最大
2. **GQA（分组查询注意力）**：将 Query 头分组，每组共享一个 KV 头，减少 KV Cache 8-16 倍
3. **MQA（多查询注意力）**：极端情况，所有 Query 头共享一个 KV 头，最大化内存节省

## 数学基础

### 内存复杂度对比

**MHA（Multi-Head Attention）**:
```
KV Cache = 2 * batch * seq_len * num_heads * head_dim * bytes
         = 2 * b * n * h * d * 2
```

**GQA（Grouped Query Attention）**:
```
num_kv_heads = num_heads // 8  # 典型值: 8 组
KV Cache = 2 * batch * seq_len * num_kv_heads * head_dim * bytes
         = 2 * b * n * (h/8) * d * 2
         = MHA 的 1/8
```

**MQA（Multi-Query Attention）**:
```
num_kv_heads = 1
KV Cache = 2 * batch * seq_len * head_dim * bytes
         = MHA 的 1/h (h 通常为 32-128)
```

### 精度损失量化

**表达能力度量**：
$$Expressiveness = \log(\text{num_kv_heads})$$

- MHA: $\log(h)$
- GQA: $\log(h/8) = \log(h) - \log(8)$ → 损失 2.1 nats
- MQA: $\log(1) = 0$ → 损失 $\log(h)$ nats

**实验数据**：LLaMA2-70B
- GQA (h=64→8): 困惑度增加 2-3%，但吞吐量提升 8x
- MQA: 困惑度增加 5-8%，吞吐量提升 64x

## 关键代码实现

### PyTorch 实现：MHA → GQA 改造

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads  # MHA 兼容
        self.head_dim = d_model // n_heads

        # Query 投影：保持 num_heads
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)

        # KV 投影：使用 num_kv_heads
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)

        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        # 1. 线性投影
        Q = self.W_q(x)  # [batch, seq_len, n_heads * head_dim]
        K = self.W_k(x)  # [batch, seq_len, n_kv_heads * head_dim]
        V = self.W_v(x)  # [batch, seq_len, n_kv_heads * head_dim]

        # 2. 重塑维度
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 3. KV 重复（核心改造点）
        repeat_factor = self.n_heads // self.n_kv_heads
        K = K.repeat_interleave(repeat_factor, dim=1)  # [batch, n_heads, seq_len, head_dim]
        V = V.repeat_interleave(repeat_factor, dim=1)

        # 4. 标准注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        # 5. 输出投影
        out = torch.matmul(attn, V)  # [batch, n_heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.W_o(out)
```

### MHA/GQA/MQA 配置对比

```python
configs = {
    "MHA": {"n_heads": 64, "n_kv_heads": 64},      # h=64, h_kv=64
    "GQA-8": {"n_heads": 64, "n_kv_heads": 8},    # h=64, h_kv=8 (LLaMA2-70B)
    "GQA-4": {"n_heads": 64, "n_kv_heads": 16},   # h=64, h_kv=16 (LLaMA2-13B)
    "MQA": {"n_heads": 64, "n_kv_heads": 1},       # h=64, h_kv=1 (PaLM)
}
```

## 复杂度分析

| 指标 | MHA | GQA (h/8) | MQA |
|------|-----|-----------|-----|
| **KV Cache 内存** | 2bnhd | 2bnd(h/8) | 2bnd |
| **相对大小** | 1x | 1/8x | 1/hx |
| **计算量 (QK^T)** | bn²h | bn²h | bn²h |
| **带宽 (加载 KV)** | bnhd | bnd(h/8) | bnd |
| **精度损失 (PPL)** | 0% | +2-3% | +5-8% |
| **吞吐量提升** | 1x | 8x | 64x |

**关键洞察**:
- GQA 在 **内存减少 8 倍** 的同时，**精度损失仅 2-3%**，是最佳平衡点
- MQA 内存节省极致，但 **精度损失明显**（5-8%）
- **GQA-8** 已成为 70B+ 模型的标配（LLaMA2-70B, GPT-4）

## 工业映射

### LLaMA2-70B 的 GQA 实践

```python
# 配置文件
model_config = {
    "n_heads": 64,          # Q 头数量
    "n_kv_heads": 8,        # KV 头数量（8 组）
    "head_dim": 128,        # 每个头 128 维
    "d_model": 8192,        # 总维度 64*128
}

# 显存节省计算
# 原始 MHA: 2 * 4 * 4096 * 64 * 128 * 2 = 1,073,741,824 bytes ≈ 1GB
# GQA 后:   2 * 4 * 4096 * 8 * 128 * 2   = 134,217,728 bytes ≈ 128MB
# 节省: 87.5%
```

**推理性能提升**: 在 A100-80G 上
- **MHA**: 批大小=4, seq_len=4096 → 显存耗尽
- **GQA**: 批大小=32, seq_len=4096 → 推理吞吐提升 8x

### GPT-4 的混合策略

- **前 48 层**: GQA-8（平衡精度与速度）
- **后 16 层**: MQA（极致加速，层越深影响越小）
- **动态切换**: 根据输入长度自动调整

### 面试高频追问

**Q1: 为什么选择每组 8 个 Query 头，而不是 4 或 16？**

A: 经验法则 `n_kv_heads = n_heads // 8` 的数学依据：
- 精度损失与内存节省的帕累托最优
- 实验数据：h/8 时困惑度增加在 3% 以内，超过可接受阈值
- 通信效率：All-Reduce 时 8 的倍数在 NCCL 中效率最高

**Q2: 改造后的注意力矩阵形状变化？**

A:
```python
# MHA: Q [b, h, n, d], K [b, h, n, d] → scores [b, h, n, n]
# GQA: Q [b, h, n, d], K [b, h/8, n, d] → K_repeat [b, h, n, d]
# MQA: Q [b, h, n, d], K [b, 1, n, d] → K_repeat [b, h, n, d]
```

**Q3: GQA 在反向传播时的梯度计算？**

A: KV 重复的梯度累加：
```python
# dL/dK_shared = Σ dL/dK_repeat over 8 repeats
dK_shared = dK_repeat.view(b, h/8, 8, n, d).sum(dim=2)
```

---

**难度评级**: ⭐⭐⭐
**出现频率**: 100%（所有 Transformer 相关岗位）
**掌握要求**: 代码 + 内存计算 + 工程权衡
