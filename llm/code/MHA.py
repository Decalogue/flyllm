"""
Multi-Head Attention 完整实现（MHA/MQA/GQA）

三个独立实现，展示KV投影的核心差异
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    标准的点积注意力计算

    参数:
        Q: (batch, h, seq_len, d_k) - Query
        K: (batch, h, seq_len, d_k) - Key
        V: (batch, h, seq_len, d_k) - Value
        mask: Optional mask
        dropout: dropout rate

    返回:
        output: (batch, h, seq_len, d_k)
        attn_weights: (batch, h, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    # 计算QK^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=True)

    output = torch.matmul(attn_weights, V)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    标准Multi-Head Attention (MHA)

    核心特征：每个head有独立的KV
    - W_K: d_model → h * d_k
    - W_V: d_model → h * d_k
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout

        # 注意：这里是d_model -> d_model (h * d_k)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)  # 每个head独立KV
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播步骤详解（含每一步shape变化）

        示例: batch_size=2, seq_len=10, d_model=512, num_heads=8
              → d_k = 512/8 = 64
        """
        batch_size, seq_len, _ = X.size()
        # X shape: (batch_size, seq_len, d_model) = (2, 10, 512)

        # ==================== 步骤1: Q/K/V线性投影 ====================
        # 1.1 Q投影并reshape为多头
        # X -> Linear -> (2, 10, 512)
        Q = self.W_Q(X)  # (2, 10, 512)
        # view: 将最后维度拆分为(num_heads, d_k) -> (2, 10, 8, 64)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        # transpose: 交换维度1和2，将head维提前 -> (2, 8, 10, 64)
        Q = Q.transpose(1, 2)  # final: (batch, num_heads, seq_len, d_k)
        # Q: (2, 8, 10, 64) ✨

        # 1.2 K投影并reshape（和Q完全相同）
        K = self.W_K(X)  # (2, 10, 512)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # K: (2, 8, 10, 64) ✨ 每个head有独立的K

        # 1.3 V投影并reshape（和Q完全相同）
        V = self.W_V(X)  # (2, 10, 512)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # V: (2, 8, 10, 64) ✨ 每个head有独立的V
        # 总结：Q/K/V都有独立的(batch, 8, 10, 64)

        # ==================== 步骤2: Scaled Dot-Product Attention ====================
        # 输入: Q/K/V均为 (batch, num_heads, seq_len, d_k) = (2, 8, 10, 64)
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        # attn_output: (2, 8, 10, 64) - 每个head的输出
        # attn_weights: (2, 8, 10, 10) - 每个head的注意力权重矩阵

        # ==================== 步骤3: 合并heads并输出投影 ====================
        # 3.1 transpose: 将head维度换回来 -> (2, 10, 8, 64)
        attn_output = attn_output.transpose(1, 2)

        # 3.2 contiguous + view: 合并head和d_k维度 -> (2, 10, 512)
        # contiguous()确保内存连续，便于view操作
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)
        # attn_output: (2, 10, 512) ✨

        # 3.3 最后的线性投影
        output = self.W_O(attn_output)  # (2, 10, 512) -> (2, 10, 512)

        # 返回最终输出和注意力权重（用于可视化或分析）
        return output, attn_weights
        # output: (batch, seq_len, d_model)
        # attn_weights: (batch, num_heads, seq_len, seq_len)


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA)

    核心特征：所有head共享1组KV
    - W_K: d_model → d_k (1 * d_k)  ⭐ 核心差异：不是d_model!
    - W_V: d_model → d_k (1 * d_k)
    - W_Q: d_model → d_model (h * d_k)

    实现技巧：unsqueeze + expand
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout

        # Q投影到所有head
        self.W_Q = nn.Linear(d_model, d_model)  # d_model -> h*d_k

        # K和V只投影到1个head的维度！
        self.W_K = nn.Linear(d_model, self.d_k)  # d_model -> d_k (1*head)
        self.W_V = nn.Linear(d_model, self.d_k)  # d_model -> d_k

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播步骤详解（含每一步shape变化）

        示例: batch_size=2, seq_len=10, d_model=512, num_heads=8
              → d_k = 512/8 = 64

        与MHA的核心差异：KV在投影时只有1个head，然后通过broadcast复制到所有head
        这带来了h倍的参数量减少和KV Cache显存减少
        """
        batch_size, seq_len, _ = X.size()
        # X shape: (batch_size, seq_len, d_model) = (2, 10, 512)

        # ==================== 步骤1: Q投影（同MHA） ====================
        # Q的投影和reshape与MHA完全相同
        Q = self.W_Q(X)  # (2, 10, 512)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q: (2, 8, 10, 64) ✨ 8个head，每个64维

        # ==================== 步骤2: KV投影（核心差异！） ====================
        # 2.1 K投影（只有1个head）
        # 差异：W_K输出维度是d_k=64，不是d_model=512！
        K = self.W_K(X)  # (2, 10, 64) ⭐ 只有1个head的维度！
        # unsqueeze: 在dim=1插入head维度 -> (2, 1, 10, 64)
        K = K.unsqueeze(1)
        # K: (2, 1, 10, 64) ✨

        # 2.2 V投影（同K）
        V = self.W_V(X)  # (2, 10, 64)
        V = V.unsqueeze(1)  # -> (2, 1, 10, 64)
        # V: (2, 1, 10, 64) ✨

        # 2.3 广播到所有head（核心操作！）
        # expand: 将dim=1的head复制8次，不增加内存占用
        # 因为expand是broadcast操作，只改变stride，不拷贝数据
        K = K.expand(batch_size, self.num_heads, seq_len, self.d_k)
        V = V.expand(batch_size, self.num_heads, seq_len, self.d_k)
        # K/V: (2, 8, 10, 64) ✨ 从(2,1,10,64)广播到(2,8,10,64)

        # 总结：Q有8个独立的head，而K/V是1个head广播到8份
        # 这就是MQA能压缩h倍参数量的关键！

        # ==================== 步骤3: Scaled Dot-Product Attention ====================
        # 输入: Q (2, 8, 10, 64), K/V (2, 8, 10, 64) - broadcast后的shape相同
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        # attn_output: (2, 8, 10, 64)
        # attn_weights: (2, 8, 10, 10)

        # ==================== 步骤4: 合并heads并输出投影（同MHA） ====================
        # 4.1 transpose: (2, 8, 10, 64) -> (2, 10, 8, 64)
        attn_output = attn_output.transpose(1, 2)

        # 4.2 contiguous + view: -> (2, 10, 512)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)
        # attn_output: (2, 10, 512) ✨

        # 4.3 最后的线性投影: (2, 10, 512) -> (2, 10, 512)
        output = self.W_O(attn_output)

        # 返回最终输出和注意力权重
        return output, attn_weights
        # output: (batch, seq_len, d_model)
        # attn_weights: (batch, num_heads, seq_len, seq_len)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA)

    核心特征：g个KV组（1 < g < h）⭐ LLaMA2-70B用8组
    - W_K: d_model → g * d_k  ⭐ 核心差异：g*d_k (不是d_model!)
    - W_V: d_model → g * d_k
    - 每组KV服务 h/g 个head

    实现：先unsqueeze到(g, group_size, ...)，再view回h
    """
    def __init__(self, d_model: int, num_heads: int, num_kv_groups: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        assert num_heads % num_kv_groups == 0, "num_heads必须能被num_kv_groups整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.d_k = d_model // num_heads
        self.group_size = num_heads // num_kv_groups  # 每组服务的head数
        self.dropout = dropout

        # Q投影到所有head
        self.W_Q = nn.Linear(d_model, d_model)  # d_model -> num_heads*d_k

        # KV投影到g个组: d_model -> num_kv_groups*d_k
        self.W_K = nn.Linear(d_model, num_kv_groups * self.d_k)
        self.W_V = nn.Linear(d_model, num_kv_groups * self.d_k)

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播步骤详解（含每一步shape变化）

        示例: batch_size=2, seq_len=10, d_model=512, num_heads=8, num_kv_groups=2
              → d_k = 512/8 = 64, group_size = 8/2 = 4

        与MHA/MQA的核心差异：KV在投影时有g个组（不是1个也不是h个）
        每组KV服务group_size个head，通过unsqueeze+expand+view实现
        """
        batch_size, seq_len, _ = X.size()
        # X shape: (batch_size, seq_len, d_model) = (2, 10, 512)

        # ==================== 步骤1: Q投影（同MHA） ====================
        # Q的投影和reshape与MHA完全相同
        Q = self.W_Q(X)  # (2, 10, 512)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q: (2, 8, 10, 64) ✨ 8个head，每个64维

        # ==================== 步骤2: KV投影（核心差异！） ====================
        # 2.1 K投影（g个组）
        # 差异：W_K输出维度是num_kv_groups*d_k=2*64=128，不是d_model=512！
        K = self.W_K(X)  # (2, 10, 128) ⭐ g*d_k（不是h*d_k）!
        # view: 拆分为(num_kv_groups, d_k) -> (2, 10, 2, 64)
        K = K.view(batch_size, seq_len, self.num_kv_groups, self.d_k)
        # transpose: 交换维度1和2 -> (2, 2, 10, 64)
        K = K.transpose(1, 2)
        # K: (2, 2, 10, 64) ✨ 2个KV组，每组10个token，64维

        # 2.2 V投影（同K）
        V = self.W_V(X)  # (2, 10, 128)
        V = V.view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)
        # V: (2, 2, 10, 64) ✨
        # 总结：有g=2个KV组，而不是8个独立head或1个共享head

        # ==================== 步骤3: repeat到num_heads个head（核心操作！） ====================
        # 3.1 unsqueeze: 在dim=2插入group_size维度 -> (2, 2, 1, 10, 64)
        K = K.unsqueeze(2)
        V = V.unsqueeze(2)
        # K/V: (batch, num_kv_groups, 1, seq_len, d_k)

        # 3.2 expand: 将dim=2复制group_size=4次
        # 每个KV组服务4个head（因为8个head / 2个组 = 4）
        K = K.expand(batch_size, self.num_kv_groups, self.group_size, seq_len, self.d_k)
        V = V.expand(batch_size, self.num_kv_groups, self.group_size, seq_len, self.d_k)
        # K/V: (2, 2, 4, 10, 64) ✨ 第一个维度：2个KV组；第二个维度：每组4个head
        # 物理意义：KV1服务head1-4，KV2服务head5-8

        # 3.3 contiguous + view: reshape为(num_heads, seq_len, d_k)
        # contiguous()确保内存连续，因为expand后stride可能不连续
        K = K.contiguous().view(batch_size, self.num_heads, seq_len, self.d_k)
        V = V.contiguous().view(batch_size, self.num_heads, seq_len, self.d_k)
        # K/V: (2, 8, 10, 64) ✨ 从(2,2,4,10,64)reshape为(2,8,10,64)
        # 现在shape和MHA/MQA相同，都是(batch, num_heads, seq_len, d_k)

        # 总结：通过unsqueeze+expand+view，实现了分组共享KV
        # 8个head被分为2组，每组4个head共享1个KV

        # ==================== 步骤4: Scaled Dot-Product Attention ====================
        # 输入: Q (2, 8, 10, 64), K/V (2, 8, 10, 64) - 相同shape
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        # attn_output: (2, 8, 10, 64)
        # attn_weights: (2, 8, 10, 10)

        # ==================== 步骤5: 合并heads并输出投影（同MHA/MQA） ====================
        # 5.1 transpose: (2, 8, 10, 64) -> (2, 10, 8, 64)
        attn_output = attn_output.transpose(1, 2)

        # 5.2 contiguous + view: -> (2, 10, 512)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)
        # attn_output: (2, 10, 512) ✨

        # 5.3 最后的线性投影: (2, 10, 512) -> (2, 10, 512)
        output = self.W_O(attn_output)

        # 返回最终输出和注意力权重
        return output, attn_weights
        # output: (batch, seq_len, d_model)
        # attn_weights: (batch, num_heads, seq_len, seq_len)


# ================ 核心差异对比 ================
"""
**面试必背：三个类的3处关键差异**

| 维度 | MHA | MQA | GQA |
|------|-----|-----|-----|
| **W_K/W_V输出维度** | d_model (h*d_k) | d_k (1*d_k) | g*d_k |
| **KV heads数量** | h个 (独立) | 1个 | g个 (1<g<h) |
| **参数计算方式** | d_model × d_model | d_model × d_k | d_model × (g*d_k) |
| **实现技巧** | reshape | unsqueeze + expand | unsqueeze + reshape |
| **压缩比** | 1× (基准) | h×压缩 | h/g×压缩 |
| **适用模型** | BERT/GPT-3 | T5/ChatGLM | LLaMA2-70B |
| **推理速度** | 基准 | +3-4倍 | +2-3倍 |
| **质量损失** | 0 | 微小(-1%) | 几乎无损 |

**代码速记（3行差异）**:
```python
# 差异1: W_K/W_V的Linear输出
MHA: self.W_K = nn.Linear(d_model, d_model)  # (h*d_k)
MQA: self.W_K = nn.Linear(d_model, d_k)     # (1*d_k)  ⭐
GQA: self.W_K = nn.Linear(d_model, g*d_k)   # (g*d_k)  ⭐

# 差异2: KV的reshape
MHA: K.view(batch, seq_len, h, d_k)      # h个head
MQA: K.unsqueeze(1)                      # 1个head + expand  ⭐
GQA: K.view(batch, seq_len, g, d_k)      # g个head + repeat  ⭐

# 差异3: 参数量
MHA: 2 * d_model * d_model
MQA: 2 * d_model * d_k       (压缩h倍)  ⭐
GQA: 2 * d_model * (g*d_k)   (压缩h/g倍)  ⭐
```

**70B模型实测对比**:
- d_model=8192, h=64, d_k=128
- MHA KV参数量: 2 * 8192 * 8192 = 134M
- MQA KV参数量: 2 * 8192 * 128 = 2M (压缩67×) ⭐
- GQA KV参数量: 2 * 8192 * (8*128) = 16M (压缩8.4×)

**KV Cache显存** (seq_len=4096, batch=1, FP16):
- MHA: 2 * 64 * 4096 * 128 * 2 = 134 GB
- MQA: 2 * 1 * 4096 * 128 * 2 = 4.2 GB (压缩32×) ⭐
- GQA: 2 * 8 * 4096 * 128 * 2 = 33.5 GB (压缩4×)

**面试回答框架**: "根据场景选择：
1. 训练时或质量优先 → MHA
2. 大模型推理，求平衡 → GQA (LLaMA2选8组)
3. 极致速度，可牺牲一点质量 → MQA (T5在用)"
"""

# ================ 测试代码 ================
if __name__ == "__main__":
    # 配置
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    num_kv_groups = 2  # GQA用2组

    X = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, num_heads, seq_len, seq_len)

    print("=" * 60)
    print("测试 Multi-Head Attention")
    print("=" * 60)
    mha = MultiHeadAttention(d_model, num_heads)
    output, weights = mha(X, mask)
    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重: {weights.shape}")
    print(f"参数量: {sum(p.numel() for p in mha.parameters()) / 1e6:.2f}M")
    print()

    print("=" * 60)
    print("测试 Multi-Query Attention")
    print("=" * 60)
    mqa = MultiQueryAttention(d_model, num_heads)
    output, weights = mqa(X, mask)
    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重: {weights.shape}")
    print(f"参数量: {sum(p.numel() for p in mqa.parameters()) / 1e6:.2f}M")
    print()

    print("=" * 60)
    print("测试 Grouped-Query Attention")
    print("=" * 60)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_groups)
    output, weights = gqa(X, mask)
    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重: {weights.shape}")
    print(f"参数量: {sum(p.numel() for p in gqa.parameters()) / 1e6:.2f}M")
    print()

    # KV Cache显存对比
    seq_len, d_k = 4096, d_model // num_heads
    print("=" * 60)
    print("KV Cache 显存对比 (batch=1, seq_len=4096)")
    print("=" * 60)
    print(f"MHA: {2 * num_heads * seq_len * d_k * 2 / 1024**2:.2f} MB")
    print(f"MQA: {2 * 1 * seq_len * d_k * 2 / 1024**2:.2f} MB")
    print(f"GQA(num_kv_groups={num_kv_groups}): {2 * num_kv_groups * seq_len * d_k * 2 / 1024**2:.2f} MB")
    print()

    # 参数量对比
    print("=" * 60)
    print("参数量对比 (只算K和V投影)")
    print("=" * 60)
    mha_kv_params = 2 * d_model * d_model  # W_K + W_V: d_model × d_model
    mqa_kv_params = 2 * d_model * d_k      # d_model × d_k
    gqa_kv_params = 2 * d_model * (num_kv_groups * d_k)  # d_model × g×d_k

    print(f"MHA KV参数量: {mha_kv_params / 1e6:.2f}M (baseline)")
    print(f"MQA KV参数量: {mqa_kv_params / 1e6:.2f}M (压缩{mha_kv_params / mqa_kv_params:.1f}×)")
    print(f"GQA KV参数量: {gqa_kv_params / 1e6:.2f}M (压缩{mha_kv_params / gqa_kv_params:.1f}×)")
