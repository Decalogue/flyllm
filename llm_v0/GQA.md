# 分组查询注意力（Grouped Query Attention, GQA）

## 一、背景与核心思想

### 背景

GQA（Grouped Query Attention）是 Google 在 PaLM 2 中提出的优化，通过减少 K、V 的头数来降低内存和计算量。后来被 LLaMA 2 采用，成为现代大模型的重要优化技术。

### 核心思想

**多个查询头共享同一组 K、V 头**。

- **标准 MHA**：Q、K、V 都有 `num_heads` 个头
- **GQA**：Q 有 `num_query_heads` 个头，K、V 只有 `num_kv_heads` 个头（通常 `num_kv_heads < num_query_heads`）
- **分组比例**：`num_groups = num_query_heads // num_kv_heads`，表示每个 KV 头对应多少个 Q 头

### 优势

- ✅ **内存减少**：K、V 投影参数量减少 `num_kv_heads / num_query_heads` 倍
- ✅ **计算减少**：K、V 的计算量减少
- ✅ **性能**：在长序列上效果显著，几乎不影响模型性能
- ✅ **实际应用**：LLaMA 2 使用 GQA（如 32 个 Q 头，8 个 KV 头）

---

## 二、完整实现

### 基础版本

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（GQA）
    
    Args:
        d_model: 模型维度
        num_query_heads: 查询头数（通常等于 num_heads）
        num_kv_heads: K/V 头数（通常 < num_query_heads，如 num_heads // 2）
        dropout: Dropout 比率
        bias: 是否使用偏置
    """
    def __init__(self, d_model: int, num_query_heads: int, num_kv_heads: int, 
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % num_query_heads == 0, "d_model 必须能被 num_query_heads 整除"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads 必须能被 num_kv_heads 整除"
        
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_query_heads  # 每个头的维度
        self.num_groups = num_query_heads // num_kv_heads  # 每个 KV 头对应的 Q 头数
        self.scale = 1.0 / math.sqrt(self.d_k)  # 缩放因子 1/√d_k
        
        # Q 投影：输出维度为 d_model（num_query_heads 个头）
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        # K, V 投影：输出维度为 num_kv_heads * d_k（只有 num_kv_heads 个头）
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        # 输出投影
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
            kv_cache: dict，包含 'k' 和 'v'，形状为 [batch_size, num_kv_heads, cached_len, d_k]
                     用于自回归生成时缓存历史 K/V，将复杂度从 O(t²) 降到 O(t)
        
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_query_heads, seq_len_q, seq_len_k] (可选)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # 1. Q 投影并重塑为多头形式
        # [batch_size, seq_len_q, d_model] -> [batch_size, seq_len_q, num_query_heads, d_k]
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_query_heads, self.d_k)
        Q = Q.transpose(1, 2)  # [batch_size, num_query_heads, seq_len_q, d_k]
        
        # 2. K, V 投影并重塑（注意：只有 num_kv_heads 个头）
        # [batch_size, seq_len_k, d_model] -> [batch_size, seq_len_k, num_kv_heads, d_k]
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_kv_heads, self.d_k)
        K = K.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len_k, d_k]
        
        V = self.W_v(value).view(batch_size, seq_len_k, self.num_kv_heads, self.d_k)
        V = V.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len_k, d_k]
        
        # 3. KV Cache 处理（关键：在 seq_len 维度拼接，在广播之前）
        if kv_cache is not None:
            if 'k' in kv_cache and 'v' in kv_cache and kv_cache['k'] is not None:
                # 拼接缓存的 K/V 和当前的 K/V（在 dim=2，即 seq_len 维度）
                # 注意：缓存中的 K/V 形状是 [batch_size, num_kv_heads, cached_len, d_k]
                K = torch.cat([kv_cache['k'], K], dim=2)  # [batch_size, num_kv_heads, cached_len + seq_len_k, d_k]
                V = torch.cat([kv_cache['v'], V], dim=2)
            
            # 更新缓存（存储完整的 K/V，包括新计算的）
            kv_cache['k'] = K
            kv_cache['v'] = V
            # 更新 seq_len_k 为拼接后的长度
            seq_len_k = K.shape[2]
        
        # 4. 关键步骤：将 K, V 广播到与 Q 相同的头数
        # [batch_size, num_kv_heads, seq_len_k, d_k] -> [batch_size, num_query_heads, seq_len_k, d_k]
        # repeat_interleave 会将每个 KV 头重复 num_groups 次
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        
        # 5. 计算注意力分数（与标准 MHA 相同）
        # [batch_size, num_query_heads, seq_len_q, d_k] @ [batch_size, num_query_heads, d_k, seq_len_k]
        # -> [batch_size, num_query_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 6. 应用 mask（如果有）
        if mask is not None:
            # mask: [batch_size, seq_len_q, seq_len_k] 或 [batch_size, 1, seq_len_k]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            # mask 中 1 表示有效位置，0 表示需要屏蔽的位置（padding mask 约定）
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 7. Softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 8. 加权求和
        # [batch_size, num_query_heads, seq_len_q, seq_len_k] @ [batch_size, num_query_heads, seq_len_k, d_k]
        # -> [batch_size, num_query_heads, seq_len_q, d_k]
        context = torch.matmul(attention_weights, V)
        
        # 9. 拼接多头
        # [batch_size, num_query_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, num_query_heads, d_k]
        context = context.transpose(1, 2).contiguous()
        # -> [batch_size, seq_len_q, d_model]
        context = context.view(batch_size, seq_len_q, self.d_model)
        
        # 10. 输出投影
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output


# 使用示例
if __name__ == "__main__":
    # 参数设置（类似 LLaMA 2）
    d_model = 4096
    num_query_heads = 32
    num_kv_heads = 8  # 4:1 的分组比例
    batch_size = 2
    seq_len = 2048
    
    # 创建模型
    gqa = GroupedQueryAttention(d_model, num_query_heads, num_kv_heads)
    
    # 创建输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = gqa(query, key, value)
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    
    # 带 mask 的前向传播
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, seq_len//2:] = 0  # 屏蔽后半部分
    output_masked = gqa(query, key, value, mask=mask)
    
    # 返回注意力权重
    output, attn_weights = gqa(query, key, value, return_attention=True)
    print(f"Attention weights shape: {attn_weights.shape}")  # [2, 32, 2048, 2048]
    
    # 计算参数量对比
    d_k = d_model // num_query_heads
    standard_mha_params = 4 * d_model * d_model  # Q, K, V, O
    gqa_params = (d_model * d_model +  # Q
                  2 * d_model * (num_kv_heads * d_k) +  # K, V
                  d_model * d_model)  # O
    reduction = (1 - gqa_params / standard_mha_params) * 100
    print(f"\n参数量对比：")
    print(f"标准 MHA: {standard_mha_params:,}")
    print(f"GQA: {gqa_params:,}")
    print(f"减少: {reduction:.1f}%")
```

---

## 三、关键实现细节

### 1. 维度约束

```python
assert d_model % num_query_heads == 0  # Q 头维度必须整除
assert num_query_heads % num_kv_heads == 0  # 分组比例必须为整数
```

### 2. 投影层维度

- **Q 投影**：`d_model -> d_model`（`num_query_heads` 个头）
- **K, V 投影**：`d_model -> num_kv_heads * d_k`（只有 `num_kv_heads` 个头）
- **O 投影**：`d_model -> d_model`（与标准 MHA 相同）

### 3. KV Cache 处理（如果使用）

```python
# 关键：在 seq_len 维度拼接缓存的 K/V 和当前的 K/V
# 注意：缓存中的 K/V 形状是 [batch_size, num_kv_heads, cached_len, d_k]
if kv_cache is not None:
    if kv_cache['k'] is not None:
        K = torch.cat([kv_cache['k'], K], dim=2)  # 在 dim=2 拼接
        V = torch.cat([kv_cache['v'], V], dim=2)
    kv_cache['k'] = K
    kv_cache['v'] = V
```

**重要**：KV Cache 处理在广播之前进行，因为缓存中存储的是 `num_kv_heads` 个头的 K/V。

### 4. 广播操作

```python
# 关键：将 K, V 从 num_kv_heads 广播到 num_query_heads
K = K.repeat_interleave(self.num_groups, dim=1)
V = V.repeat_interleave(self.num_groups, dim=1)
```

`repeat_interleave` 会将每个 KV 头重复 `num_groups` 次，使得：
- 前 `num_groups` 个 Q 头共享第 1 个 KV 头
- 接下来 `num_groups` 个 Q 头共享第 2 个 KV 头
- 以此类推

**处理顺序**（基础版本）：投影 → KV Cache（可选）→ 广播 → 注意力计算

**处理顺序**（带 RoPE 版本）：投影 → RoPE → KV Cache（可选）→ 广播 → 注意力计算

### 5. 注意力计算

广播后的 K、V 与 Q 的维度匹配，后续计算与标准 MHA 完全相同。

---

## 四、参数量与计算量分析

### 参数量对比

假设 `d_model = 4096`，`num_query_heads = 32`，`num_kv_heads = 8`：

**标准 MHA**：
- Q: `4096 × 4096 = 16,777,216`
- K: `4096 × 4096 = 16,777,216`
- V: `4096 × 4096 = 16,777,216`
- O: `4096 × 4096 = 16,777,216`
- **总计**: `67,108,864` 参数

**GQA**：
- Q: `4096 × 4096 = 16,777,216`
- K: `4096 × (8 × 128) = 4,194,304`
- V: `4096 × (8 × 128) = 4,194,304`
- O: `4096 × 4096 = 16,777,216`
- **总计**: `41,943,040` 参数
- **减少**: `37.5%`

### 计算量对比

- **K, V 投影计算量减少**: `num_kv_heads / num_query_heads` 倍
- **注意力矩阵计算**: 与标准 MHA 相同（因为广播后维度相同）
- **内存占用减少**: K, V 的中间张量更小

---

## 五、实际应用案例

### LLaMA 2 配置

- **LLaMA 2 7B**: `num_query_heads = 32`, `num_kv_heads = 32`（标准 MHA）
- **LLaMA 2 13B**: `num_query_heads = 40`, `num_kv_heads = 40`（标准 MHA）
- **LLaMA 2 70B**: `num_query_heads = 64`, `num_kv_heads = 8`（GQA，8:1 比例）

### 性能影响

- **模型性能**: 几乎无影响（在大多数任务上性能相当）
- **推理速度**: 提升约 20-30%（取决于序列长度）
- **内存占用**: 减少约 30-40%（K, V 缓存更小）

---

## 六、测试代码

```python
def test_gqa():
    """测试 GQA"""
    d_model, num_q_heads, num_kv_heads = 768, 12, 4
    batch_size, seq_len = 2, 128
    
    gqa = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = gqa(x, x, x)
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ GQA 测试通过")
    
    # 测试注意力权重
    output, attn_weights = gqa(x, x, x, return_attention=True)
    assert attn_weights.shape == (batch_size, num_q_heads, seq_len, seq_len)
    print("✓ GQA 注意力权重形状正确")
    
    # 测试 mask
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, seq_len//2:] = 0
    output_masked = gqa(x, x, x, mask=mask)
    assert output_masked.shape == (batch_size, seq_len, d_model)
    print("✓ GQA mask 测试通过")

def test_gqa_rope():
    """测试 GQA + RoPE"""
    d_model, num_q_heads, num_kv_heads = 768, 12, 4
    batch_size, seq_len = 2, 128
    
    gqa_rope = GQAWithRoPE(d_model, num_q_heads, num_kv_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = gqa_rope(x)
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ GQA + RoPE 测试通过")
    
    # 测试 position_ids
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    output = gqa_rope(x, position_ids=position_ids)
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ GQA + RoPE position_ids 测试通过")
    
    # 测试返回注意力权重
    output, attn_weights = gqa_rope(x, return_attention=True)
    assert attn_weights.shape == (batch_size, num_q_heads, seq_len, seq_len)
    print("✓ GQA + RoPE 注意力权重形状正确")

def test_gqa_kv_cache():
    """测试 GQA + RoPE + KV Cache"""
    d_model, num_q_heads, num_kv_heads = 768, 12, 4
    batch_size = 1
    
    gqa_rope = GQAWithRoPE(d_model, num_q_heads, num_kv_heads)
    
    # Prefill 阶段：计算 prompt 的 KV Cache
    prompt_len = 10
    prompt = torch.randn(batch_size, prompt_len, d_model)
    kv_cache = {'k': None, 'v': None}
    output_prefill = gqa_rope(prompt, kv_cache=kv_cache)
    assert output_prefill.shape == (batch_size, prompt_len, d_model)
    assert kv_cache['k'] is not None
    assert kv_cache['k'].shape[1] == num_kv_heads  # 注意：是 num_kv_heads，不是 num_q_heads
    assert kv_cache['k'].shape[2] == prompt_len  # 缓存长度 = prompt_len
    print("✓ GQA + RoPE + KV Cache Prefill 测试通过")
    
    # 生成阶段：每次只传入最后一个 token
    for step in range(3):
        current_token = torch.randn(batch_size, 1, d_model)
        output = gqa_rope(current_token, kv_cache=kv_cache)
        assert output.shape == (batch_size, 1, d_model)
        # 缓存长度应该递增
        assert kv_cache['k'].shape[2] == prompt_len + step + 1
        # 缓存头数应该是 num_kv_heads
        assert kv_cache['k'].shape[1] == num_kv_heads
    
    print("✓ GQA + RoPE + KV Cache 生成阶段测试通过")

if __name__ == "__main__":
    test_gqa()
    test_gqa_rope()
    test_gqa_kv_cache()
    print("\n所有测试通过！")
```

---

## 七、KV Cache 支持（推理加速）

### 核心原理

GQA 的 KV Cache 与标准 MHA 类似，但有一个重要区别：**只缓存 num_kv_heads 个 K/V 头**，而不是 num_query_heads 个。

KV Cache 将自回归生成的复杂度从 **O(t²) 降到 O(t)**：
- **Prefill 阶段**：计算 prompt 所有 token 的 K/V，O(n)
- **生成阶段**：每步只计算当前 token 的 K/V，O(1)
- **总复杂度**：O(n) + O(t) = O(t)

### GQA 中的 KV Cache 优势

由于 GQA 只有 `num_kv_heads` 个 K/V 头（通常 `num_kv_heads < num_query_heads`），KV Cache 的显存占用更小：

**标准 MHA KV Cache 显存**：
```
n_layers × batch_size × seq_len × num_heads × d_head × 2 × bytes_per_element
```

**GQA KV Cache 显存**：
```
n_layers × batch_size × seq_len × num_kv_heads × d_head × 2 × bytes_per_element
```

**显存减少比例**：`num_kv_heads / num_query_heads`

### 实现要点

1. **只缓存 K/V，不缓存 Q**：Q 每次生成都不同，K/V 历史不变
2. **在 seq_len 维度拼接**：`torch.cat([kv_cache['k'], K], dim=2)`
3. **缓存形状**：`[batch_size, num_kv_heads, cached_len, d_k]`（注意是 `num_kv_heads`，不是 `num_query_heads`）
4. **广播时机**：在 KV Cache 拼接之后，再进行 GQA 的广播操作

### 使用示例

```python
# Prefill 阶段：计算 prompt 的 KV Cache
prompt = torch.randn(1, 10, d_model)  # [batch_size, prompt_len, d_model]
kv_cache = {'k': None, 'v': None}
_ = gqa_rope(prompt, kv_cache=kv_cache)  # 初始化缓存

# 生成阶段：每次只传入最后一个 token
for step in range(max_new_tokens):
    current_token = torch.randn(1, 1, d_model)  # [batch_size, 1, d_model]
    output = gqa_rope(current_token, kv_cache=kv_cache)  # 复用缓存的 K/V
    # ... 采样新 token ...
```

### 显存占用计算（GQA）

**示例（LLaMA 2 70B，FP16）：**
- 80 层，64 个 Q 头，8 个 KV 头，d_head=128，seq_len=2048
- **标准 MHA**：`80 × 1 × 2048 × 64 × 128 × 2 × 2B = 2.68GB`
- **GQA**：`80 × 1 × 2048 × 8 × 128 × 2 × 2B = 335MB`
- **减少**：`87.5%`（8:1 比例）

详细内容请参考 `KVCache.md`。

---

## 八、与标准 MHA 的对比

| 特性 | 标准 MHA | GQA |
|------|---------|-----|
| Q 头数 | `num_heads` | `num_query_heads` |
| K 头数 | `num_heads` | `num_kv_heads` |
| V 头数 | `num_heads` | `num_kv_heads` |
| K, V 参数量 | `2 × d_model²` | `2 × d_model × (num_kv_heads × d_k)` |
| 内存占用 | 高 | 低（减少 30-40%） |
| 计算量 | 高 | 低（K, V 投影减少） |
| 模型性能 | 基准 | 几乎相同 |

---

## 九、面试要点总结

### 核心概念

1. **分组思想**：多个 Q 头共享同一组 K、V 头
2. **分组比例**：`num_groups = num_query_heads // num_kv_heads`
3. **广播操作**：使用 `repeat_interleave` 将 K、V 广播到与 Q 相同的头数

### 关键实现

1. **维度约束**：`num_query_heads % num_kv_heads == 0`
2. **投影层**：K、V 投影输出维度为 `num_kv_heads * d_k`
3. **广播时机**：在 KV Cache 拼接之后，计算注意力分数之前进行广播
4. **KV Cache**：只缓存 `num_kv_heads` 个 K/V 头，显存占用更小

### 优势与局限

**优势**：
- ✅ 显著减少参数量和计算量
- ✅ 降低内存占用（特别是 K、V 缓存，KV Cache 显存减少 `num_kv_heads / num_query_heads` 倍）
- ✅ 几乎不影响模型性能
- ✅ 支持 KV Cache，推理加速效果更明显

**局限**：
- ⚠️ 理论上表达能力略弱于标准 MHA（但实际影响很小）
- ⚠️ 需要 `num_query_heads` 能被 `num_kv_heads` 整除

---

## 十、总结

GQA 是现代大模型（如 LLaMA 2）中重要的优化技术，通过减少 K、V 头数来降低内存和计算成本，同时几乎不影响模型性能。

**关键要点**：
1. ✅ 多个 Q 头共享 K、V 头
2. ✅ 使用 `repeat_interleave` 进行广播
3. ✅ 参数量减少约 30-40%
4. ✅ KV Cache 显存减少 `num_kv_heads / num_query_heads` 倍
5. ✅ 实际应用中性能几乎无损失

**面试建议**：
- 理解 GQA 的核心思想（共享 K、V 头）
- 掌握广播操作的实现细节
- 了解参数量和计算量的减少比例
- 理解 KV Cache 在 GQA 中的优势（显存减少更多）
- 知道实际应用案例（如 LLaMA 2 70B）

---

## 十一、完整示例：结合 GQA + RoPE

### RoPE 辅助函数

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

### GQA + RoPE 完整实现

```python
class GQAWithRoPE(nn.Module):
    """
    分组查询注意力 + 旋转位置编码
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
            kv_cache: dict，包含 'k' 和 'v'，形状为 [batch_size, num_kv_heads, cached_len, d_k]
                     用于自回归生成时缓存历史 K/V
        """
        batch_size, seq_len, _ = x.shape
        
        # 延迟构建 RoPE 缓存（如果尚未构建或设备不匹配）
        if self.cos_cached is None or self.cos_cached.device != x.device:
            self._build_rope_cache(self.max_seq_len, device=x.device)
        
        # 1. QKV 投影
        Q = self.W_q(x).view(batch_size, seq_len, self.num_query_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        
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
        
        # 2.5. KV Cache 处理（在应用 RoPE 之后，在广播之前）
        if kv_cache is not None:
            if 'k' in kv_cache and 'v' in kv_cache and kv_cache['k'] is not None:
                # 拼接缓存的 K/V 和当前的 K/V（在 dim=2，即 seq_len 维度）
                # 注意：缓存中的 K/V 形状是 [batch_size, num_kv_heads, cached_len, d_k]
                K = torch.cat([kv_cache['k'], K], dim=2)  # [batch_size, num_kv_heads, cached_len + seq_len, d_k]
                V = torch.cat([kv_cache['v'], V], dim=2)
            
            # 更新缓存（存储完整的 K/V，包括新计算的）
            kv_cache['k'] = K
            kv_cache['v'] = V
            # 更新 seq_len 为拼接后的长度（用于后续计算）
            seq_len_k = K.shape[2]
        else:
            seq_len_k = seq_len
        
        # 3. GQA: 广播 K, V 到与 Q 相同的头数
        # [batch_size, num_kv_heads, seq_len_k, d_k] -> [batch_size, num_query_heads, seq_len_k, d_k]
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        
        # 4. 计算注意力
        # 注意：使用 KV Cache 时，Q 的 seq_len 可能小于 K/V 的 seq_len（自回归生成）
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, num_query_heads, seq_len, seq_len_k]
        
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
        context = torch.matmul(attention_weights, V)  # [batch_size, num_query_heads, seq_len, d_k]
        
        # 5. 输出
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output


# 使用示例
if __name__ == "__main__":
    # 参数设置（类似 LLaMA 2 70B）
    d_model = 4096
    num_query_heads = 64
    num_kv_heads = 8  # 8:1 的分组比例
    batch_size = 2
    seq_len = 2048
    
    # 创建模型
    gqa_rope = GQAWithRoPE(d_model, num_query_heads, num_kv_heads)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = gqa_rope(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 带 position_ids 的前向传播
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    output = gqa_rope(x, position_ids=position_ids)
    
    # 返回注意力权重
    output, attn_weights = gqa_rope(x, return_attention=True)
    print(f"Attention weights shape: {attn_weights.shape}")  # [2, 64, 2048, 2048]
```

### 关键实现细节

1. **处理顺序**：RoPE → KV Cache → 广播 → 注意力计算
   - 先对 Q 和 K 应用 RoPE
   - 然后进行 KV Cache 拼接（如果使用）
   - 再进行 GQA 的广播操作（将 K/V 从 `num_kv_heads` 广播到 `num_query_heads`）
   - 最后计算注意力
2. **RoPE 缓存**：预计算 cos 和 sin 值，避免每次前向传播时重复计算
3. **位置编码**：支持通过 `position_ids` 指定位置，适用于处理不同长度的序列
4. **KV Cache 优势**：只缓存 `num_kv_heads` 个 K/V 头，显存占用比标准 MHA 更小

### 性能优势

- **GQA**：减少 K、V 的参数量和计算量
- **RoPE**：相对位置编码，支持外推到更长的序列
- **结合使用**：在 LLaMA 2 等现代大模型中广泛应用，兼顾效率和性能
