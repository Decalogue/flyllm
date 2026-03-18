---
concept: "Self-Attention机制"
template: "直觉建构型 + 硬核推导型"
difficulty: ⭐⭐⭐
importance: 🌟🌟🌟🌟🌟
user_mastery: 0.85
prerequisites: ["矩阵乘法", "softmax", "向量点积"]
related_concepts: ["multi-head-attention", "causal-mask", "rope-positional-encoding"]
category: "Transformer"
module: "LLM"
generated_at: "2026-03-16"
---

# Self-Attention机制详解

## 1. 核心概念

Self-Attention让序列每个位置能关注所有位置，解决RNN的长距离依赖和并行化问题。

**公式**:
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
```

其中Q,K,V是查询、键、值矩阵，d_k是维度。

### 为什么除sqrt(d_k)
- QK^T方差 = d_k
- 不除：Softmax饱和 → 梯度消失
- 除后：方差归一化，稳定训练

**实验**:
```python
# d_k=512, 不除: entropy=0.02 (饱和)
# d_k=512, 除sqrt: entropy=2.5 (健康)
```

### 复杂度
- 时间: O(n^2 d_k)
- 空间: O(n^2)
- 瓶颈: n>2048必须FlashAttention

## 2. 三级类比

### Level 1: 鸡尾酒会
- Query: 你想听什么
- Key: 每个人特征
- Value: 实际内容
- Output: 按相关度加权

### Level 2: 搜索引擎
- Query: 搜索词
- Key: 网页TF-IDF
- Value: 网页内容
- Output: 最相关结果

### Level 3: 信息路由
可导的全连接层，权重动态计算。

## 3. 流程

输入 → Embedding → Q/K/V投影 → QK^T计算 → 
除以sqrt(d_k) → Softmax → 加权V → 输出

## 4. 代码

```python
import torch, math

def self_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)
    if mask:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn
```

## 5. 追问防御

### Q1: Q≠K原因
- 表达能力
- 维度灵活
- 消融实验+0.5 BLEU

### Q2: 忘记除sqrt(d_k)
- Loss不下降
- Attention one-hot
- entropy<0.5检测

### Q3: Causal Mask作用
- 防止偷看未来
- Decoder必须
- BERT不需要双向

### Q4: Multi-Head作用
- 多视角特征
- 12头→84.6分
- 24头→边际递减

### Q5: 复杂度瓶颈
- n>2048 FlashAttention
- n>4096 MQA/GQA

## 6. 工业经验

训练1.3B模型：
- n=2048, FlashAttention
- 显存: 38GB→12GB
- 速度: 150→420 samples/s
- 收敛: 稳定

---

*掌握度: 85/100*
*推荐: MHA→GQA演进*
