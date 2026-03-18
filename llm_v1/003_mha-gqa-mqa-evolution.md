---
concept: "MHA→GQA→MQA演进"
template: "问题解决型 + 对比矩阵"
difficulty: ⭐⭐⭐⭐
importance: 🌟🌟🌟🌟🌟
user_mastery: 0.0
prerequisites: ["self_attention", "multi_head_attention"]
related_concepts: ["flash_attention", "inference_optimization", "kv_cache"]
category: "Transformer"
module: "LLM"
generated_at: "2026-03-16"
---

# MHA→GQA→MQA演进详解

## 1. 演进时间线与痛点驱动

```
2017: Multi-Head Attention (MHA) ← Transformer原版
    ↓ 问题：推理时KV缓存爆炸
2022: Multi-Query Attention (MQA) ← Google PaLM
    ↓ 问题：精度损失2-3%
2023: Grouped-Query Attention (GQA) ← LLaMA-2
    ↓ 平衡：显存减半，精度损失<0.5%
2024: MLA (Multi-head Latent Attention) ← DeepSeek-V2
    ↓ 更激进：KV缓存再降90%
```

## 2. MHA的问题（推理时KV缓存爆炸）

**场景**: 70B模型，batch_size=1，推理生成长度=8000

**KV缓存 per token**:
- Key: [1, 64, 1, 128] = 64 × 128 = 8,192 parameters
- Value: 同样 = 8,192 parameters
- **总计**: 16,384 parameters/token

**长序列推理**:
- 8000 tokens × 16KB = **131M parameters ≈ 500MB**
- 100个并发 = 50GB → **OOM崩溃！**

**结论**: MHA在训练时完美，推理时不可接受。

## 3. MQA：激进优化

### 3.1 MQA核心思想

```python
def multi_query_attention(x, h=64):
    """
    MQA: Multi-Query Attention
    64个Query，1个Key/Value共享
    """
    batch, seq_len, d_model = x.shape
    d_kv = d_model // h
    
    # 1. Query: 64个头
    Q = linear(x, d_model, h * d_kv)
    Q = Q.reshape(batch, seq_len, h, -1).transpose(1, 2)  # [B, 64, S, 128]
    
    # 2. Key/Value: 1个！
    K = linear(x, d_model, d_kv)      # [B, S, 128]
    V = linear(x, d_model, d_kv)      # [B, S, 128]
    
    # 3. 广播到所有头
    K = K.unsqueeze(1).expand(-1, h, -1, -1)
    V = V.unsqueeze(1).expand(-1, h, -1, -1)
    
    # 4. Attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_kv)
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    
    return output
```

### 3.2 MQA的权衡

**收益**: KV缓存减少8x，推理速度提升2-4x

**代价**: **精度损失2-3%**（GLUE基准）

## 4. GQA：平衡方案

### 4.1 GQA核心思想

```python
def grouped_query_attention(x, h=64, g=8):
    """
    GQA: Grouped-Query Attention
    h个Query，g个KV组
    """
    batch, seq_len, d_model = x.shape
    d_kv = d_model // h
    heads_per_group = h // g
    
    # Query: h个头
    Q = linear(x, d_model, h * d_kv)
    Q = Q.reshape(batch, seq_len, h, -1).transpose(1, 2)
    
    # Key/Value: g个组
    K = linear(x, d_model, g * d_kv)
    V = linear(x, d_model, g * d_kv)
    K = K.reshape(batch, seq_len, g, -1).transpose(1, 2)
    V = V.reshape(batch, seq_len, g, -1).transpose(1, 2)
    
    # 广播K/V
    K = K.unsqueeze(2).expand(-1, -1, heads_per_group, -1, -1)
    K = K.reshape(batch, h, seq_len, d_kv)
    V = V.unsqueeze(2).expand(-1, -1, heads_per_group, -1, -1)
    V = V.reshape(batch, h, seq_len, d_kv)
    
    # Attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_kv)
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    
    return output
```

### 4.2 GQA的优势

**精度**: <0.5%下降（vs MQA的2-3%）

**KV缓存**: 同MQA的8x减少

**推理速度**: 提升2-3x

**工业界首选**: LLaMA-2 70B, CodeLLaMA等都采用

## 5. 对比总结

| 维度 | MHA | MQA | GQA |
|------|-----|-----|-----|
| Query | 64 | 64 | 64 |
| Key/Value | 64 | 1 | 8 |
| KV缓存 | 100% | 12.5% | 12.5% |
| 精度损失 | 0% | 2-3% | <0.5% |
| 速度提升 | 1x | 2-4x | 2-3x |

## 6. 实战案例

**LLaMA-2 70B部署**:
- MHA: 需要4张A100，800ms/request
- GQA: 只需2张A100，320ms/request（2.5x提升）
- **成本节省50%**

## 7. 面试要点

### 必背
1. MHA在**训练**完美，**推理**瓶颈
2. MQA: 8x缓存，精度损失2-3%
3. GQA: 8x缓存，精度损失<0.5%，工业首选
4. g=8是经验最优值（h/g≈8）

### 追问
"GQA的g=8如何选出的？"
→ 实验驱动，精度+速度的sweet spot

"精度损失如何评估？"
→ MMLU、HumanEval、GSM8K等多个维度

## 8. 下阶段

推荐学习：RoPE位置编码 + FlashAttention优化

---

*掌握度: 待评估*
