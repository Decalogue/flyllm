# 多头注意力机制（Multi-Head Attention）手撕代码

"手撕"多头注意力机制是深度学习面试和实际理解 Transformer 架构的必修课。

为了让你透彻理解，我将提供**多个版本**的代码：
1. **PyTorch 版本（工业级/面试推荐）**：利用 `einops` 和矩阵乘法，简洁高效，符合现代深度学习开发习惯。
2. **NumPy 版本（底层原理版）**：手动实现矩阵分块和拼接，帮助你从数学公式层面理解数据流动。
3. **RoPE 版本**：结合旋转位置编码，符合主流模型（LLaMA、Qwen 等）的实现。
4. **融合计算版本**：优化内存和计算效率。

---

## 一、核心公式回顾

在写代码前，先明确数学定义。多头注意力本质上是将输入 $X$ 分别投影到 Query ($Q$), Key ($K$), Value ($V$)，计算缩放点积注意力，最后输出结果。

对于第 $i$ 个头 ($h$)：
$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

其中 Attention 函数：
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

最终多头输出是将所有头拼接起来再过一次线性层：
$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O $$

---

## 二、PyTorch 标准版（使用 einops，推荐）

这个版本使用了 `einops` 库，它能让张量重排变得非常直观。如果不使用 `einops`，代码中的 `view` 和 `transpose` 操作会非常难以阅读。

**依赖安装**：`pip install torch einops`

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """
    标准多头注意力机制
    
    Args:
        d_model: 模型维度（通常是 512, 768, 1024 等）
        num_heads: 注意力头数（通常是 8, 12, 16 等）
        dropout: Dropout 比率
        bias: 是否使用偏置（Transformer 原论文中通常不使用 bias）
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = False):
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
                  mask 中 1 表示有效位置，0 表示需要屏蔽的位置（padding mask 约定）
            return_attention: 是否返回注意力权重
            kv_cache: dict，包含 'k' 和 'v'，形状为 [batch_size, num_heads, cached_len, d_k]
                     用于自回归生成时缓存历史 K/V，将复杂度从 O(t²) 降到 O(t)
        
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k] (可选)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # 1. 线性投影
        Q = self.W_q(query)  # [batch, seq_len_q, d_model]
        K = self.W_k(key)    # [batch, seq_len_k, d_model]
        V = self.W_v(value)  # [batch, seq_len_k, d_model]
        
        # 2. 拆分多头并调整维度顺序（使用 einops 简化）
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h=self.num_heads)
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads)
        
        # 2.5. KV Cache 处理（关键：在 seq_len 维度拼接）
        if kv_cache is not None:
            if 'k' in kv_cache and 'v' in kv_cache and kv_cache['k'] is not None:
                # 拼接缓存的 K/V 和当前的 K/V（在 dim=2，即 seq_len 维度）
                K = torch.cat([kv_cache['k'], K], dim=2)  # [batch, num_heads, cached_len + seq_len_k, d_k]
                V = torch.cat([kv_cache['v'], V], dim=2)
            
            # 更新缓存（存储完整的 K/V，包括新计算的）
            kv_cache['k'] = K
            kv_cache['v'] = V
            # 更新 seq_len_k 为拼接后的长度
            seq_len_k = K.shape[2]
        
        # 3. 计算注意力分数
        # [batch, num_heads, seq_len_q, d_k] @ [batch, num_heads, d_k, seq_len_k]
        # -> [batch, num_heads, seq_len_q, seq_len_k]
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
        # [batch, num_heads, seq_len_q, seq_len_k] @ [batch, num_heads, seq_len_k, d_k]
        # -> [batch, num_heads, seq_len_q, d_k]
        context = torch.matmul(attention_weights, V)
        
        # 7. 合并多头
        # [batch, num_heads, seq_len_q, d_k] -> [batch, seq_len_q, num_heads, d_k] -> [batch, seq_len_q, d_model]
        context = rearrange(context, 'b h s d -> b s (h d)')
        
        # 8. 输出投影
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output


# --- 测试代码 ---
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 512
    num_heads = 8
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    mha = MultiHeadAttention(d_model, num_heads)
    output, attn = mha(x, x, x, return_attention=True)
    
    print(f"Input shape:      {x.shape}")
    print(f"Output shape:     {output.shape}")
    print(f"Attention shape:  {attn.shape}")  # 应该是 [2, 8, 5, 5]
```

### 代码关键点解析

1. **维度拆分**：核心在于 `rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)`。这句话将 `d_model` (512) 拆解为 `num_heads` (8) 和 `d_k` (64)，并直接调整到 `[batch, heads, seq_len, d_k]` 的格式，比传统的 `view` + `transpose` 更直观。
2. **矩阵乘法**：为了计算所有词之间的相似度，我们需要 Q 的形状为 `[batch, heads, seq_len, d_k]`，K 转置后为 `[batch, heads, d_k, seq_len]`。这样 `matmul` 的结果才是 `[batch, heads, seq_len, seq_len]` 的相似度矩阵。
3. **缩放因子**：除以 $\sqrt{d_k}$ 非常关键，否则当维度很高时，点积结果会很大，导致 softmax 梯度消失（进入饱和区）。
4. **Mask 处理**：mask 中 1 表示有效位置，0 表示需要屏蔽的位置。使用 `masked_fill(mask == 0, float('-inf'))` 在 softmax 前屏蔽无效位置。

---

## 三、融合计算版本（Fused Attention）

优化版本可以一次性投影 QKV，减少内存分配和转置操作。

```python
class FusedMultiHeadAttention(nn.Module):
    """
    融合计算的多头注意力（减少内存分配和转置操作）
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = False):
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
        
        # 2. 分离并重塑（使用 einops 简化）
        # [batch, seq_len, 3*d_model] -> [batch, seq_len, 3, num_heads, d_k] -> [3, batch, num_heads, seq_len, d_k]
        qkv = rearrange(qkv, 'b s (three h d) -> three b h s d', three=3, h=self.num_heads)
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
        context = rearrange(context, 'b h s d -> b s (h d)')
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output
```

---

## 四、旋转位置编码（RoPE - Rotary Position Embedding）

RoPE（Rotary Position Embedding）通过旋转矩阵将位置信息编码到注意力计算中。RoPE 最初在 RoFormer 中提出，后被 LLaMA、Qwen 等主流模型采用。

```python
def rotate_half(x):
    """将 x 的后半部分取负并交换前后部分"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    应用旋转位置编码
    
    Args:
        q, k: [batch_size, num_heads, seq_len, d_k]
        cos, sin: [seq_len, d_k // 2] 或 [batch_size, seq_len, d_k // 2] 或 [batch_size, num_heads, seq_len, d_k // 2]
        position_ids: [batch_size, seq_len] 位置索引（可选，通常 cos/sin 已经根据 position_ids 选择）
    
    注意：position_ids 参数在当前实现中未使用，因为 cos/sin 已经在调用前根据 position_ids 索引过了
    """
    # 调整 cos/sin 的维度以匹配 q, k
    # q, k 的形状是 [batch_size, num_heads, seq_len, d_k]
    # 需要将 cos/sin 扩展到 [batch_size, num_heads, seq_len, d_k//2]
    if cos.dim() == 2:  # [seq_len, d_k//2]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_k//2]
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:  # [batch_size, seq_len, d_k//2]
        cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, d_k//2]
        sin = sin.unsqueeze(1)
    # 如果已经是 4 维，假设形状已经是 [batch_size, num_heads, seq_len, d_k//2] 或类似
    
    # 应用旋转：q_rot = q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
```

---

## 五、完整示例：结合 MHA + RoPE

```python
class RoPEMultiHeadAttention(nn.Module):
    """
    带 RoPE 的多头注意力
    类似 LLaMA、Qwen 的实现
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048, 
                 dropout: float = 0.1, bias: bool = False):
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
        
        # 1. 计算 QKV（使用 einops 简化）
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. 拆分多头并调整维度
        # [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h=self.num_heads)
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads)
        
        # 3. 应用 RoPE
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
        
        # 3.5. KV Cache 处理（在应用 RoPE 之后）
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
        
        # 4. 计算注意力
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
        
        # 5. 输出
        context = rearrange(context, 'b h s d -> b s (h d)')
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output
```

---

## 六、KV Cache 支持（推理加速）

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
rope_mha = RoPEMultiHeadAttention(d_model, num_heads)
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

---

## 七、NumPy 原理版（面试手写/教学用）

如果你在面试中被要求"不用框架手写"，或者想理解底层循环逻辑，看这个版本。我们不使用矩阵乘法的黑盒，而是展示分头计算的过程。

```python
import numpy as np

def softmax(x):
    """数值稳定的 softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class MultiHeadAttentionNumpy:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 随机初始化权重 (Xavier 初始化的简化版)
        scale = 1 / np.sqrt(d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 1. 线性投影
        # (N, S, D) @ (D, D) -> (N, S, D)
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # 2. 拆分多头
        # (N, S, H, Dk)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 3. 转置以适配矩阵乘法
        # Q: (N, H, S, Dk)
        # K: (N, H, Dk, S)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 3, 1)  # 交换最后两维
        V = V.transpose(0, 2, 1, 3)  # (N, H, S, Dk)
        
        # 4. 缩放点积
        # Score: (N, H, S, S)
        scores = (Q @ K) / np.sqrt(self.d_k)
        
        # 5. Softmax
        attn_weights = softmax(scores)
        
        # 6. 加权求和
        # (N, H, S, S) @ (N, H, S, Dk) -> (N, H, S, Dk)
        context = attn_weights @ V
        
        # 7. 合并多头
        # (N, H, S, Dk) -> (N, S, H, Dk) -> (N, S, D)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # 8. 输出投影
        output = context @ self.W_o
        
        return output

# --- 测试 ---
if __name__ == "__main__":
    np.random.seed(42)
    B, S, D = 2, 5, 512
    H = 8
    x_np = np.random.randn(B, S, D)
    
    mha_np = MultiHeadAttentionNumpy(D, H)
    out_np = mha_np.forward(x_np)
    print("Numpy Output shape:", out_np.shape)
```

---

## 八、面试要点总结

### 1. 核心公式
- **注意力分数**：`QK^T / √d_k`
- **缩放原因**：防止点积过大导致 softmax 梯度消失
- **多头拼接**：`Concat(head₁, ..., headₕ)W^O`

### 2. 关键细节
- **维度检查**：`d_model % num_heads == 0`
- **Mask 处理**：`masked_fill(mask == 0, float('-inf'))` 在 softmax 前
- **Dropout 位置**：在 softmax 之后，加权求和之前
- **转置顺序**：使用 `rearrange` 或 `transpose(1, 2)` 将 head 维度提前以便批量计算

### 3. 优化技巧
- **融合计算**：QKV 一次投影减少内存分配
- **RoPE**：相对位置编码，支持外推，被 LLaMA、Qwen 等主流模型采用
- **KV Cache**：推理加速，将复杂度从 O(t²) 降到 O(t)

### 4. 复杂度分析
- **时间复杂度**：O(n²d)，其中 n 是序列长度，d 是模型维度
- **空间复杂度**：O(n²) 用于存储注意力矩阵
- **Flash Attention**：通过分块计算将空间复杂度降至 O(n)

### 5. 面试加分项

在面试或实际写代码时，如果提到以下几点，会给面试官留下深刻印象：

1. **掩码机制**：解释在 Decoder 中为什么要加 `masked_fill`（防止看到未来的信息，即因果掩码）以及 Padding 掩码（防止模型关注无效的填充字符）。
2. **数值稳定性**：提到 Softmax 之前为什么通常需要做 `scores - max(scores)`（在 `softmax` 函数内部实现），以防止 `exp` 溢出。
3. **内存优化**：指出标准的实现会计算 $N \times N$ 的注意力矩阵，显存消耗随序列长度平方级增长。可以提到 FlashAttention 的分块计算思想。
4. **偏差**：提到在 Attention 的线性投影（$W_q, W_k, W_v$）中通常不加 bias，但在最后的输出投影（$W_o$）中通常会加 bias（虽然 PyTorch 的 `MultiheadAttention` 默认都不加，但 BERT 等 Transformer 变体中 $W_o$ 是带 bias 的）。
5. **RoPE 优势**：相对位置编码，支持外推，计算效率高，被主流模型广泛采用。

---

## 九、与 Flash Attention 的区别

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

## 十、测试代码

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

1. **标准实现**：完整的 Multi-Head Attention，使用 `einops.rearrange` 简化代码，包含所有细节
2. **优化版本**：融合计算、RoPE（符合主流模型实现）
3. **工程实践**：mask 处理、dropout、维度检查、KV Cache
4. **面试要点**：复杂度分析、关键细节说明、加分项

**面试建议**：
- 优先掌握标准实现（必须）
- 了解 RoPE 的原理和实现（加分项，主流模型都使用）
- Flash Attention 了解原理即可，不需要手写 CUDA 代码
- GQA 相关内容请参考 GQA.md
