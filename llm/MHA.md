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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
    def forward(self, query, key, value, mask=None, return_attention=False, kv_cache=None):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_k, d_model]
            mask: [batch_size, seq_len_q, seq_len_k] 或 [batch_size, 1, seq_len_k]
            return_attention: 是否返回注意力权重
            kv_cache: dict，包含 'k' 和 'v'，形状为 [batch_size, num_heads, cached_len, d_k]
                     用于自回归生成时缓存历史 K/V，将复杂度从 O(t²) 降到 O(t)
        
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k] (可选)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # 1. 线性投影并重塑为多头形式
        # [batch_size, seq_len_q/k, d_model] -> [batch_size, seq_len_q/k, num_heads, d_k]
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k)
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k)
        V = self.W_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k)
        
        # 2. 转置以便批量计算注意力
        # [batch_size, num_heads, seq_len, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 2.5. KV Cache 处理（关键：在 seq_len 维度拼接）
        if kv_cache is not None:
            if 'k' in kv_cache and 'v' in kv_cache and kv_cache['k'] is not None:
                # 拼接缓存的 K/V 和当前的 K/V（在 dim=2，即 seq_len 维度）
                K = torch.cat([kv_cache['k'], K], dim=2)  # [batch_size, num_heads, cached_len + seq_len_k, d_k]
                V = torch.cat([kv_cache['v'], V], dim=2)
            
            # 更新缓存（存储完整的 K/V，包括新计算的）
            kv_cache['k'] = K
            kv_cache['v'] = V
            # 更新 seq_len_k 为拼接后的长度
            seq_len_k = K.shape[2]
        
        # 3. 计算注意力分数
        # [batch_size, num_heads, seq_len_q, d_k] @ [batch_size, num_heads, d_k, seq_len_k]
        # -> [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 4. 应用 mask（如果有）
        if mask is not None:
            # mask: [batch_size, seq_len_q, seq_len_k] 或 [batch_size, 1, seq_len_k]
            # mask 中 1 表示有效位置，0 表示需要屏蔽的位置（padding mask 约定）
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            # 将需要屏蔽的位置（mask == 0）的分数设为 -inf，softmax 后变为 0
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
            # mask 中 1 表示有效位置，0 表示需要屏蔽的位置
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

## 三、旋转位置编码（RoPE - Rotary Position Embedding）

RoPE（Rotary Position Embedding）通过旋转矩阵将位置信息编码到注意力计算中。

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    应用旋转位置编码
    
    Args:
        q, k: [batch_size, num_heads, seq_len, d_k]
        cos, sin: [seq_len, d_k // 2] 或 [batch_size, seq_len, d_k // 2] 或 [batch_size, num_heads, seq_len, d_k // 2]
        position_ids: [batch_size, seq_len] 位置索引（可选，通常 cos/sin 已经根据 position_ids 选择，此参数保留用于未来扩展）
    
    注意：position_ids 参数在当前实现中未使用，因为 cos/sin 已经在调用前根据 position_ids 索引过了
    """
    def rotate_half(x):
        """将 x 的后半部分取负并交换前后部分"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    # 调整 cos/sin 的维度以匹配 q, k
    # q, k 的形状是 [batch_size, num_heads, seq_len, d_k]
    # 需要将 cos/sin 扩展到 [batch_size, num_heads, seq_len, d_k//2]
    if cos.dim() == 2:  # [seq_len, d_k//2]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_k//2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_k//2]
    elif cos.dim() == 3:  # [batch_size, seq_len, d_k//2]
        cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, d_k//2]
        sin = sin.unsqueeze(1)  # [batch_size, 1, seq_len, d_k//2]
    # 如果已经是 4 维，假设形状已经是 [batch_size, num_heads, seq_len, d_k//2] 或类似
    
    # 应用旋转：q_rot = q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
```

## 四、完整示例：结合 MHA + RoPE

```python
class RoPEMultiHeadAttention(nn.Module):
    """
    带 RoPE 的多头注意力
    类似 LLaMA 的实现
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048, 
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # 延迟构建 RoPE 缓存（在第一次 forward 时根据设备构建）
        self.max_seq_len = max_seq_len
        self.cos_cached = None
        self.sin_cached = None
        
    def _build_rope_cache(self, max_seq_len, device):
        """构建 RoPE 缓存"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
        t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, d_k//2]
        
        # 缓存 cos 和 sin（形状：[max_seq_len, d_k//2]）
        # 使用 register_buffer 注册，确保能正确移动到设备
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
        
    def forward(self, x, mask=None, position_ids=None, return_attention=False, kv_cache=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 或 [batch_size, 1, seq_len]
            position_ids: [batch_size, seq_len] 位置索引（可选）
            return_attention: 是否返回注意力权重
            kv_cache: dict，包含 'k' 和 'v'，形状为 [batch_size, num_heads, cached_len, d_k]
                     用于自回归生成时缓存历史 K/V
        """
        batch_size, seq_len, _ = x.shape
        
        # 延迟构建 RoPE 缓存（如果尚未构建或设备不匹配）
        if self.cos_cached is None or self.cos_cached.device != x.device:
            self._build_rope_cache(self.max_seq_len, device=x.device)
        
        # 1. 计算 QKV
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 应用 RoPE
        # 注意：使用 KV Cache 时，只对当前 token 的 Q/K 应用 RoPE，缓存的 K 已经应用过 RoPE
        if position_ids is not None:
            # 如果提供了 position_ids，使用它来索引 cos/sin
            # position_ids: [batch_size, seq_len] -> cos/sin: [batch_size, seq_len, d_k//2]
            cos = self.cos_cached[position_ids]  # [batch_size, seq_len, d_k//2]
            sin = self.sin_cached[position_ids]  # [batch_size, seq_len, d_k//2]
        else:
            # 如果没有提供 position_ids，根据是否有 KV Cache 来决定起始位置
            if kv_cache is not None and kv_cache.get('k') is not None:
                # 有 KV Cache：从缓存长度开始的位置
                cached_len = kv_cache['k'].shape[2]
                pos_start = cached_len
                cos = self.cos_cached[pos_start:pos_start + seq_len]  # [seq_len, d_k//2]
                sin = self.sin_cached[pos_start:pos_start + seq_len]  # [seq_len, d_k//2]
            else:
                # 无 KV Cache：从 0 开始的连续位置
                cos = self.cos_cached[:seq_len]  # [seq_len, d_k//2]
                sin = self.sin_cached[:seq_len]  # [seq_len, d_k//2]
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
        
        # 2.5. KV Cache 处理（在应用 RoPE 之后）
        if kv_cache is not None:
            if 'k' in kv_cache and 'v' in kv_cache and kv_cache['k'] is not None:
                # 拼接缓存的 K/V 和当前的 K/V（在 dim=2，即 seq_len 维度）
                K = torch.cat([kv_cache['k'], K], dim=2)  # [batch_size, num_heads, cached_len + seq_len, d_k]
                V = torch.cat([kv_cache['v'], V], dim=2)
            
            # 更新缓存（存储完整的 K/V，包括新计算的）
            kv_cache['k'] = K
            kv_cache['v'] = V
            # 更新 seq_len 为拼接后的长度（用于后续计算）
            seq_len_k = K.shape[2]
        else:
            seq_len_k = seq_len
        
        # 3. 计算注意力
        # 注意：使用 KV Cache 时，Q 的 seq_len 可能小于 K/V 的 seq_len（自回归生成）
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len_k]
        
        if mask is not None:
            # mask 中 1 表示有效位置，0 表示需要屏蔽的位置
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        elif kv_cache is not None:
            # 使用 KV Cache 时，通常不需要额外的 mask（causal mask 已通过缓存长度保证）
            # 但如果有 padding，仍需要 mask
            pass
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # 4. 输出
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output
```

---

## 五、KV Cache 支持（推理加速）

### 核心原理

KV Cache 将自回归生成的复杂度从 **O(t²) 降到 O(t)**：
- **Prefill 阶段**：计算 prompt 所有 token 的 K/V，O(n)
- **生成阶段**：每步只计算当前 token 的 K/V，O(1)
- **总复杂度**：O(n) + O(t) = O(t)

### 实现要点

1. **只缓存 K/V，不缓存 Q**：Q 每次生成都不同，K/V 历史不变
2. **在 seq_len 维度拼接**：`torch.cat([kv_cache['k'], K], dim=2)`
3. **自回归生成**：每次只传入最后一个 token（`seq_len=1`）

### 使用示例

```python
# Prefill 阶段：计算 prompt 的 KV Cache
prompt = torch.randn(1, 10, d_model)  # [batch_size, prompt_len, d_model]
kv_cache = {'k': None, 'v': None}
_ = rope_mha(prompt, kv_cache=kv_cache)  # 初始化缓存

# 生成阶段：每次只传入最后一个 token
for step in range(max_new_tokens):
    current_token = torch.randn(1, 1, d_model)  # [batch_size, 1, d_model]
    output = rope_mha(current_token, kv_cache=kv_cache)  # 复用缓存的 K/V
    # ... 采样新 token ...
```

### 显存占用计算

```
KV Cache 显存 = n_layers × batch_size × seq_len × n_heads × d_head × 2 × bytes_per_element
```

**示例（LLaMA-7B，FP16）：**
- 24 层，32 头，d_head=128，seq_len=2048
- `24 × 1 × 2048 × 32 × 128 × 2 × 2B = 805MB`

详细内容请参考 `KVCache.md`。

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
- 能写出标准实现 + RoPE 已经足够
- Flash Attention 了解原理即可（分块、在线 softmax）
- 除非明确要求，否则不需要手写 Flash Attention 的 CUDA 代码
- GQA 相关内容请参考 GQA.md

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

def test_rope():
    """测试 RoPE"""
    d_model, num_heads = 768, 12
    batch_size, seq_len = 2, 128
    
    rope_mha = RoPEMultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = rope_mha(x)
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ RoPE MHA 测试通过")

def test_kv_cache():
    """测试 KV Cache"""
    d_model, num_heads = 768, 12
    batch_size = 1
    
    rope_mha = RoPEMultiHeadAttention(d_model, num_heads)
    
    # Prefill 阶段：计算 prompt 的 KV Cache
    prompt_len = 10
    prompt = torch.randn(batch_size, prompt_len, d_model)
    kv_cache = {'k': None, 'v': None}
    output_prefill = rope_mha(prompt, kv_cache=kv_cache)
    assert output_prefill.shape == (batch_size, prompt_len, d_model)
    assert kv_cache['k'] is not None
    assert kv_cache['k'].shape[2] == prompt_len  # 缓存长度 = prompt_len
    
    # 生成阶段：每次只传入最后一个 token
    for step in range(3):
        current_token = torch.randn(batch_size, 1, d_model)
        output = rope_mha(current_token, kv_cache=kv_cache)
        assert output.shape == (batch_size, 1, d_model)
        # 缓存长度应该递增
        assert kv_cache['k'].shape[2] == prompt_len + step + 1
    
    print("✓ KV Cache 测试通过")

if __name__ == "__main__":
    test_mha()
    test_rope()
    test_kv_cache()
    print("\n所有测试通过！")
```

---

## 总结

这份代码涵盖了大厂面试中常见的多头注意力实现要求：

1. **标准实现**：完整的 Multi-Head Attention，包含所有细节
2. **优化版本**：融合计算、RoPE
3. **工程实践**：mask 处理、dropout、维度检查
4. **面试要点**：复杂度分析、关键细节说明

**面试建议**：
- 优先掌握标准实现（必须）
- 了解 RoPE 的原理和实现（加分项）
- Flash Attention 了解原理即可，不需要手写 CUDA 代码
- GQA 相关内容请参考 GQA.md

