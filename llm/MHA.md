# 多头注意力机制（Multi-Head Attention）手撕代码

## 一、标准多头注意力实现

### 核心公式

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ)W^O

其中 headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)

Attention(Q, K, V) = softmax(QK^T / √dₖ)V
```

### 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    标准多头注意力机制
    
    Args:
        d_model: 模型维度（通常是 512, 768, 1024 等）
        num_heads: 注意力头数（通常是 8, 12, 16 等）
        dropout: Dropout 比率
        bias: 是否使用偏置
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        self.scale = 1.0 / math.sqrt(self.d_k)  # 缩放因子 1/√d_k
        
        # Q, K, V 的线性投影层
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # 输出投影层
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None, return_attention=False):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_k, d_model]
            mask: [batch_size, seq_len_q, seq_len_k] 或 [batch_size, 1, seq_len_k]
            return_attention: 是否返回注意力权重
        
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k] (可选)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # 1. 线性投影并重塑为多头形式
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k]
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k)
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k)
        V = self.W_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k)
        
        # 2. 转置以便批量计算注意力
        # [batch_size, num_heads, seq_len, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 3. 计算注意力分数
        # [batch_size, num_heads, seq_len_q, d_k] @ [batch_size, num_heads, d_k, seq_len_k]
        # -> [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 4. 应用 mask（如果有）
        if mask is not None:
            # mask: [batch_size, seq_len_q, seq_len_k] 或 [batch_size, 1, seq_len_k]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            # mask 中 True/1 表示需要屏蔽的位置
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 6. 加权求和
        # [batch_size, num_heads, seq_len_q, seq_len_k] @ [batch_size, num_heads, seq_len_k, d_k]
        # -> [batch_size, num_heads, seq_len_q, d_k]
        context = torch.matmul(attention_weights, V)
        
        # 7. 拼接多头
        # [batch_size, num_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, num_heads, d_k]
        context = context.transpose(1, 2).contiguous()
        # -> [batch_size, seq_len_q, d_model]
        context = context.view(batch_size, seq_len_q, self.d_model)
        
        # 8. 输出投影
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output


# 使用示例
if __name__ == "__main__":
    # 参数设置（类似 GPT-2 Small）
    d_model = 768
    num_heads = 12
    batch_size = 2
    seq_len = 128
    
    # 创建模型
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 创建输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = mha(query, key, value)
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    
    # 带 mask 的前向传播
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, seq_len//2:] = 0  # 屏蔽后半部分
    output_masked = mha(query, key, value, mask=mask)
    
    # 返回注意力权重
    output, attn_weights = mha(query, key, value, return_attention=True)
    print(f"Attention weights shape: {attn_weights.shape}")  # [2, 12, 128, 128]
```

---

## 二、优化版本：融合计算（Fused Attention）

### 优化思路

标准实现中，Q、K、V 分别投影后需要多次 reshape 和 transpose。优化版本可以：
1. 一次性投影 QKV
2. 减少中间张量的创建
3. 使用更高效的内存布局

```python
class FusedMultiHeadAttention(nn.Module):
    """
    融合计算的多头注意力（减少内存分配和转置操作）
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # 融合 QKV 投影（一次计算三个）
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 或 [batch_size, 1, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 一次性计算 QKV
        qkv = self.W_qkv(x)  # [batch_size, seq_len, 3*d_model]
        
        # 2. 分离并重塑
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 3. 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        
        # 4. 拼接并输出
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output
```

---

## 三、分组查询注意力（GQA - Grouped Query Attention）

### 背景

GQA 是 Google 在 PaLM 2 中提出的优化，通过减少 K、V 的头数来降低内存和计算量。

**核心思想**：多个查询头共享同一组 K、V 头。

```python
class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（GQA）
    
    Args:
        d_model: 模型维度
        num_query_heads: 查询头数（通常等于 num_heads）
        num_kv_heads: K/V 头数（通常 < num_query_heads，如 num_heads // 2）
    """
    def __init__(self, d_model: int, num_query_heads: int, num_kv_heads: int, 
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % num_query_heads == 0
        assert num_query_heads % num_kv_heads == 0, "num_query_heads 必须能被 num_kv_heads 整除"
        
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_query_heads
        self.num_groups = num_query_heads // num_kv_heads  # 每个 KV 头对应的 Q 头数
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Q: [batch_size, seq_len_q, num_query_heads, d_k]
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_query_heads, self.d_k)
        Q = Q.transpose(1, 2)
        
        # K, V: [batch_size, seq_len_k, num_kv_heads, d_k]
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_kv_heads, self.d_k)
        K = K.transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len_k, self.num_kv_heads, self.d_k)
        V = V.transpose(1, 2)
        
        # 关键：将 K, V 广播到与 Q 相同的头数
        # [batch_size, num_kv_heads, seq_len_k, d_k] -> [batch_size, num_query_heads, seq_len_k, d_k]
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        
        # 计算注意力（与标准 MHA 相同）
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        
        # 输出
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len_q, self.d_model)
        output = self.W_o(context)
        
        return output
```

**优势**：
- 内存减少：K、V 投影参数量减少 `num_kv_heads / num_query_heads` 倍
- 计算减少：K、V 的计算量减少
- 性能：在长序列上效果显著（如 LLaMA 2 使用 GQA）

---

## 四、旋转位置编码（RoPE - Rotary Position Embedding）

### 背景

RoPE 是 Meta 在 LLaMA 中提出的位置编码方法，通过旋转矩阵将位置信息编码到注意力计算中。

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    应用旋转位置编码
    
    Args:
        q, k: [batch_size, num_heads, seq_len, d_k]
        cos, sin: [seq_len, d_k // 2] 或 [max_seq_len, d_k // 2]
        position_ids: [batch_size, seq_len] 位置索引
    """
    def rotate_half(x):
        """将 x 的后半部分取负并交换前后部分"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    if position_ids is not None:
        # 根据 position_ids 选择对应的 cos/sin
        cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, d_k//2]
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_k//2]
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    # 应用旋转：q_rot = q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class RoPEMultiHeadAttention(nn.Module):
    """
    带 RoPE 的多头注意力
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048, 
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # 预计算 RoPE 的 cos 和 sin
        self.register_buffer('cos_cached', None)
        self.register_buffer('sin_cached', None)
        self.max_seq_len = max_seq_len
        self._build_rope_cache(max_seq_len)
        
    def _build_rope_cache(self, max_seq_len):
        """构建 RoPE 缓存"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        t = torch.arange(max_seq_len, device=inv_freq.device).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, d_k//2]
        
        # 缓存 cos 和 sin
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, d_k]
        self.cos_cached = emb.cos()[None, None, :, :]  # [1, 1, max_seq_len, d_k]
        self.sin_cached = emb.sin()[None, None, :, :]
        
    def forward(self, x, mask=None, position_ids=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. 计算 QKV
        qkv = self.W_qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 2. 应用 RoPE
        cos = self.cos_cached[:, :, :seq_len, :self.d_k//2]
        sin = self.sin_cached[:, :, :seq_len, :self.d_k//2]
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # 3. 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
        
        # 4. 输出
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output
```

---

## 五、完整示例：结合 GQA + RoPE

```python
class OptimizedMultiHeadAttention(nn.Module):
    """
    优化版多头注意力：GQA + RoPE
    类似 LLaMA 2 的实现
    """
    def __init__(self, d_model: int, num_query_heads: int, num_kv_heads: int,
                 max_seq_len: int = 2048, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert d_model % num_query_heads == 0
        assert num_query_heads % num_kv_heads == 0
        
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_query_heads
        self.num_groups = num_query_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE 缓存
        self._build_rope_cache(max_seq_len)
        
    def _build_rope_cache(self, max_seq_len):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
        
    def forward(self, x, mask=None, position_ids=None):
        batch_size, seq_len, _ = x.shape
        
        # QKV 投影
        Q = self.W_q(x).view(batch_size, seq_len, self.num_query_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # RoPE
        cos = self.cos_cached[:, :, :seq_len, :self.d_k//2]
        sin = self.sin_cached[:, :, :seq_len, :self.d_k//2]
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
        
        # GQA: 广播 K, V
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        
        # 输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context)
```

---

## 六、面试要点总结

### 1. 核心公式
- **注意力分数**：`QK^T / √d_k`
- **缩放原因**：防止点积过大导致 softmax 梯度消失
- **多头拼接**：`Concat(head₁, ..., headₕ)W^O`

### 2. 关键细节
- **维度检查**：`d_model % num_heads == 0`
- **Mask 处理**：`masked_fill(mask == 0, float('-inf'))` 在 softmax 前
- **Dropout 位置**：在 softmax 之后，加权求和之前
- **转置顺序**：`transpose(1, 2)` 将 head 维度提前以便批量计算

### 3. 优化技巧
- **融合计算**：QKV 一次投影减少内存分配
- **GQA**：减少 K、V 头数降低内存和计算
- **RoPE**：相对位置编码，支持外推

### 4. 复杂度分析
- **时间复杂度**：O(n²d)，其中 n 是序列长度，d 是模型维度
- **空间复杂度**：O(n²) 用于存储注意力矩阵
- **Flash Attention**：通过分块计算将空间复杂度降至 O(n)

---

## 七、与 Flash Attention 的区别

**标准实现**：需要存储完整的注意力矩阵 `[batch, heads, seq_len, seq_len]`

**Flash Attention**：
- 分块计算，不存储完整注意力矩阵
- 使用在线 softmax 算法
- 空间复杂度从 O(n²) 降至 O(n)
- 需要 CUDA 实现，PyTorch 原生版本较慢

**面试建议**：
- 能写出标准实现 + GQA + RoPE 已经足够
- Flash Attention 了解原理即可（分块、在线 softmax）
- 除非明确要求，否则不需要手写 Flash Attention 的 CUDA 代码

---

## 八、测试代码

```python
def test_mha():
    """测试标准多头注意力"""
    d_model, num_heads = 768, 12
    batch_size, seq_len = 2, 128
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = mha(x, x, x)
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ 标准 MHA 测试通过")

def test_gqa():
    """测试 GQA"""
    d_model, num_q_heads, num_kv_heads = 768, 12, 4
    batch_size, seq_len = 2, 128
    
    gqa = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = gqa(x, x, x)
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ GQA 测试通过")

def test_rope():
    """测试 RoPE"""
    d_model, num_heads = 768, 12
    batch_size, seq_len = 2, 128
    
    rope_mha = RoPEMultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = rope_mha(x)
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ RoPE MHA 测试通过")

if __name__ == "__main__":
    test_mha()
    test_gqa()
    test_rope()
    print("\n所有测试通过！")
```

---

## 总结

这份代码涵盖了大厂面试中常见的多头注意力实现要求：

1. **标准实现**：完整的 Multi-Head Attention，包含所有细节
2. **优化版本**：融合计算、GQA、RoPE
3. **工程实践**：mask 处理、dropout、维度检查
4. **面试要点**：复杂度分析、关键细节说明

**面试建议**：
- 优先掌握标准实现（必须）
- 了解 GQA 和 RoPE 的原理和实现（加分项）
- Flash Attention 了解原理即可，不需要手写 CUDA 代码

