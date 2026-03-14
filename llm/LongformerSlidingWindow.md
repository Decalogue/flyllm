# Longformer 的滑动窗口注意力是怎么实现的？窗口大小怎么选？

## 1. 核心定性
本质上，**Longformer 的滑动窗口注意力**通过将全局密集注意力矩阵稀疏化为带状结构（banded matrix），仅计算中心对角线及两侧固定宽度 $w$ 内的注意力，将复杂度从 $O(n^2)$ 降至 $O(n w)$，使 Transformer 可处理 4096+ 长度的文档，同时通过 dilation 和全局 token 机制捕获长距离依赖。

## 2. 具体流程
1. **固定窗口实现**: 对每个位置 $i$，只计算 $[i-w, i+w]$ 范围内的注意力，通过精心设计的 mask 矩阵屏蔽窗口外位置
2. **Dilated 窗口**: 在窗口内跳过固定步长（dilation $d$），如 $i, i+d, i+2d,...$ 形成稀疏感受野，单层捕获更长距离
3. **全局 token**: 指定特殊位置（如 [CLS]、句子首）为全局 token，可关注全序列，打破纯局部限制

## 3. 数学基础

**滑动窗口掩码**:
$$M_{ij} = \begin{cases}
0 & |i - j| \leq w \\
-\infty & \text{otherwise}
\end{cases}$$

**带 dilation 的窗口**:
$$M_{ij} = \begin{cases}
0 & |i - j| \leq w \text{ and } (i - j) \mod d = 0 \\
-\infty & \text{otherwise}
\end{cases}$$

**全局 token**: 
若 $i \in G$（全局 token 位置集）:
$$M_{ij} = 0 \quad \forall j$$

若 $j \in G$:
$$M_{ij} = 0 \quad \forall i$$

**复杂度**:
- **时间**: $O(n w d)$，$w=256, d=1$ 时约为 dense 的 1/8
- **空间**: $O(n w)$，KV-cache 减少 $n/w$ 倍

**感受野增长**:
单层: $[i-w, i+w]$
多层 ($L$ 层): $[i-Lw, i+Lw]$

Dilated ($d$): $[i-Lwd, i+Lwd]$

**理论保证**:
给定 $L=12, w=256$，最大捕获距离 $= 12 \times 256 = 3072$ tokens

## 4. 工程考量

**窗口大小 $w$ 选择**:

**经验法则**:
$$w = k \times L_{\text{avg}}$$

其中 $L_{\text{avg}}$ 是平均句长（通常 30-50 tokens），$k=4-10$。

**推荐值**:
| 任务 | 推荐 $w$ | 说明 |
|------|---------|------|
| NER | 64-128 | 实体局部依赖 |
| 文本分类 | 128-256 | 句子级信息 |
| 问答 | 256-512 | 跨句推理 |
| 文档理解 | 512+ | 段落级关联 |

**LLaMA 2 在长文本中的应用**:
- 4096 长度模型: $w=256$（覆盖率 6.25%）
- 16 层后感受野: $16 \times 256 = 4096$（全覆盖）
- Dilation $d=4$ 时: $16 \times 256 \times 4 = 16384$（可处理更长）

**Dilation 策略**:
底层: $d=1$（密集局部）
中层: $d=2$（稀疏中等范围）
顶层: $d=4$（稀疏长距离）

**全局 Token 放置**:
- **CLS token**: 位置 0
- **句首**: 每 64-128 tokens 插入一个
- **锚点**: 对关键实体位置手动标记

**实现技巧**:

**PyTorch 实现**:
```python
import torch.nn.functional as F

def sliding_window_attention(q, k, v, window_size=256):
    batch, heads, seq_len, dim = q.shape
    
    # Unfold to create windowed views
    k_windows = k.unfold(2, window_size, 1)
    v_windows = v.unfold(2, window_size, 1)
    
    # Compute attention within each window
    scores = torch.matmul(q.unsqueeze(-2), k_windows.transpose(-1, -2))
    scores = scores / math.sqrt(dim)
    
    # Apply causal mask within window
    mask = torch.triu(torch.ones(window_size, window_size), diagonal=1)
    scores = scores.masked_fill(mask.bool(), float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v_windows)
    
    return output.squeeze(-2)
```

**Triton 优化**:
- 编写自定义 kernel 融合 unfold + matmul
- 共享内存中计算，减少 HBM 访问
- 速度比 naive PyTorch 快 3-5x

**质量 vs 速度 Trade-off**:

- $w=128$: 速度最快，但长距离依赖丢失，QA F1 下降 3-5%
- $w=256$: 平衡（推荐），速度提升 2-3x，质量损失 0.5-1%
- $w=512$: 接近 MHA 质量，速度提升 1.5x
- $w=1024$: 几乎无损，速度提升 1.2x

**训练挑战**:

1. **感受野限制**: 纯局部无法捕获超长依赖（>Lw）
   - 解决: 插入全局 token（[CLS]）或 dilation
   
2. **边界伪影**: 窗口边缘 token 信息流动受阻
   - 解决: 重叠窗口（stride < window_size）
   
3. **层次传播**: 需要足够深（12+ 层）才能捕获全局信息
   - 浅层模型（6 层）不适用

**与 FastAttention 结合**:

Longformer 的滑动窗口 + FlashAttention 的 IO 优化：
- 内存: $O(nw)$ → $O(wd)$（极致）
- 速度: 批处理时比 dense 快 4-6x
- 质量: 几乎无损

## 5. 工业映射

**AllenAI Longformer**:
- 实现: `longformer-base-4096`, $w=512$, dilation=[1,2,4]
- 在 TriviaQA (4096 长度) 上 F1=68.5（vs BERT 512 长度的 63.1）
- 内存: 16GB → 4GB（减少 75%）
- 速度: 推理 2.5x

**BigScience BLOOM**:
- Longformer 模式用于 2048+ 长度文档
- 窗口 $w=256$ + 全局 attention 在句首
- 在 XLSum 上 Rouge-L 42.1 vs dense 40.8（提升）

**Processing 长文本推理**:
- vLLM + Longformer 滑动窗口
- 70B 模型处理 16k 长度:
  - MHA: OOM
  - Longformer $w=256$: 在 A100-80GB 运行，速度 25 tok/s
  - 质量: Perplexity 8.3 vs 8.1（dense）（损失 2%）

**领域适应**:

**医疗文献 (PubMed)**:
- 平均长度 2500 tokens
- $w=512$, dilation=2
- NER F1 85.3% vs BERT 512 的 82.1%

**法律合同**:
- 平均长度 8000 tokens
- $w=1024$, 每 128 tokens 插入全局 token
- 条款分类准确率 91.5%（vs 截断到 512 的 78.3%）

**与 Recurrence 结合**:

Transformer-XL + Longformer:
- 每层 $w=256$ + 段间循环
- 有效长度无限，内存 $O(wd)$
- 语言建模 perplexity 7.8（vs 纯 Longformer 8.5）

**实现建议**:
- **标准文本**: $w=256$, dilation=1, 每 128 tokens 全局 token
- **长依赖重要**: $w=512$, dilation=[1,2]
- **极致长度**: $w=1024$, 递归 + 全局锚点
- **推理优先**: $w=128$, batch 处理缩短延迟

**选择 vs 其他稀疏方法**:

| 方法 | 实现复杂度 | 速度 | 质量 | 适用 |
|------|-----------|------|------|------|
| Longformer | 中 | ★★★★☆ | ★★★★☆ | 通用 |
| BigBird | 高 | ★★★★☆ | ★★★★★ | 全局重要 |
| Linformer | 低 | ★★★★★ | ★★★☆☆ | 近似场景 |
| Performer | 中 | ★★★★★ | ★★★☆☆ | 快速近似 |

**未来趋势**:

- **Longformer 2**: 自适应窗口，重要位置自动扩大窗口
- **Sparse-Dense 混合**: 局部区域用 dense，全局用 Longformer
- **Retrieval-Augmented**: 滑动窗口 + 外部检索机制

**结论**: Longformer 的滑动窗口是实现长序列 Transformer 的经典方法。$w=256$ 是通用推荐值，配合适当的全局 token 和 dilation 策略，可在保证质量的同时显著降低计算成本。当前已被 FlashAttention 部分取代，但在 CPU 和不支持 CUDA 的环境中仍是首选
