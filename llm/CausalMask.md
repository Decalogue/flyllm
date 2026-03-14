# 因果掩码（Causal Mask）是什么？为什么自回归模型需要它？怎么实现？

## 1. 核心定性
本质上，**因果掩码（Causal Mask）** 是一种上三角负无穷屏蔽矩阵，通过强制注意力矩阵的对角线以上位置为 $-\infty$，阻止当前位置关注未来信息，确保自回归生成符合时间因果律，实现从左到右的无泄漏预测。

## 2. 具体流程
1. **掩码生成**：构造矩阵 $M \in \mathbb{R}^{n \times n}$，其中 $M_{ij} = 0$ 当 $i \geq j$（当前位置可看到历史和自身），$M_{ij} = -\infty$ 当 $i < j$（当前位置不能看到未来）
2. **注意力计算**：在 softmax 前将掩码加到注意力分数：$S = QK^T / \sqrt{d_k} + M$，被屏蔽位置变为极小的数
3. **Softmax 归零**：在 softmax 后，被屏蔽位置的权重 $A_{ij} = 0$（因 $e^{-\infty} = 0$），确保输出只依赖前缀，符合自回归约束

## 3. 数学基础
**掩码矩阵定义**：
$$M_{ij} = \begin{cases}
0 & i \geq j \\
-\infty & i < j
\end{cases}$$

**带掩码的注意力**：
$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)$$

**Masked Softmax 结果**：
$$A_{ij} = \begin{cases}
\frac{\exp(S_{ij})}{\sum_{k=1}^{i} \exp(S_{ik})} & j \leq i \\
0 & j > i
\end{cases}$$

**矩阵形式（下三角）**：
$$A = \text{LowerTriangular}(\text{softmax}(S))$$

**因果实现**：在生成第 $t$ 个 token 时，query $q_t$ 只能访问前 $t$ 个 key/value：
$$o_t = \sum_{j=1}^{t} A_{tj} v_j$$

**梯度传播**：
由于掩码，梯度只能向前传播（backprop to prefix）：
$$\frac{\partial \mathcal{L}}{\partial x_j} = 0 \quad \text{if } j > t$$

**计算优化**：
在实现中，不构造完整的 $n \times n$ 掩码，而是对 key/value 序列缓存（KV-cache）：
$$\text{Cache}_t = \{k_1, k_2, ..., k_t\}$$

生成时只计算：
$$A_{t*} = \text{softmax}\left(q_t \text{Cache}_t^T / \sqrt{d_k}\right)$$

复杂度从 $O(n^2)$ 优化到 $O(n)$（累计），仅需计算新的注意力行。

**PyTorch 实现**：
```python
import torch
endef get_causal_mask(seq_len):
    """生成因果掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))

# 或使用布尔掩码
mask = torch.tril(torch.ones(seq_len, seq_len))
attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
```

**FlashAttention 实现**：
在分块计算中，每块只计算合法位置：
$$\text{for } i \text{ in range}(0, n, B):\\
    \text{for } j \text{ in range}(0, i+B, B):\\
        \text{compute block if } j \leq i$$

## 4. 工程考量
**为什么需要 Causal Mask**：

1. **自回归约束**：语言生成是条件概率链：
$$p(x_t \mid x_{1:t-1})$$
若未掩码，$p(x_t)$ 会泄露 $x_{t+1:n}$ 的信息，违反条件独立性，模型学会"偷看答案"

2. **训练-推理一致**：训练时使用 causal mask，推理时逐 token 生成，确保行为一致。若训练无 mask，推理时需额外处理，性能暴跌 50%+

3. **并行训练**：尽管生成是串行，训练时可一次构造所有位置的 mask，使 teacher forcing 并行化，矩阵乘法一次性计算所有 $n$ 个位置，速度提升 1000 倍

**实现细节**：
- **内存布局**：在计算图中，mask 作为常量节点，梯度不更新
- **底层优化**：CUDA kernel 中，掩码通过条件判断跳过计算：
```c
if (col > row) return -INFINITY;
```
避免实际存储 mask 矩阵

**Trade-off**:
- **因果约束 vs 双向理解**：causal mask 限制模型只能看历史，无法像 BERT 那样理解全文上下文，在需要全局规划的任务（如摘要）上效果差 2-3 ROUGE
- **计算浪费**：在生成长序列时，早期 token 反复参与所有后续 token 计算，冗余度高。通过 caching 缓解但无法消除

**致命弱点**:
- **长度外推**：绝对位置编码的 sin/cos 在超出训练长度时失效，需使用 ALiBi 或 RoPE 等相对位置编码。若编码失效，因果 mask 无法阻止注意力崩溃
- **注意力稀释**：长度 $n=32k$ 时，每个位置的权重 $1/n$ 极小，梯度消失，深层 Transformer 无法训练，需引入门控机制或残差缩放
- **位置泄露**：若 batch 中的序列 padding 后做 mask，未正确处理 padding mask（padding mask + causal mask）会导致不同序列间信息泄露，损失异常下降但泛化差

## 5. 工业映射
在工业界，causal mask 是 **所有自回归 LLM 的基石**：
- **PyTorch 原生支持**：`torch.nn.Transformer` 提供 `attn_mask` 参数，通过布尔或浮点掩码实现 causal mask。在 GPT-2 实现中，mask 在 `forward()` 中生成一次并缓存，避免重复创建
- **Hugging Face**：`GPT2LMHeadModel` 在 `prepare_inputs_for_generation()` 中自动创建 causal mask，支持静态缓存（`use_cache=True`），生成时只计算当前 token 与历史的注意力，速度提升 50%
- **FlashAttention-2**：无需显式构造 mask，在 kernel 内部判断位置跳过计算，因果模式比密集模式快 1.3 倍，内存从不存储掩码。在 LLaMA 2 70B 推理时，32k 上下文仅占用 24GB 显存，比标准实现少 80%
- **vLLM 连续批处理**：因果 mask 与动态批尺寸结合，每个请求的已生成长度 $t_i$ 不同，mask 为变长下三角。vLLM 通过 PagedAttention 的块级 mask，支持不同 $t_i$ 的请求在同一批次，无需 padding，吞吐量提升 3-5 倍
- **BERT vs GPT 对比**：BERT 使用双向注意力（无 mask），在 GLUE 上微调效果好 2-3 点，但无法生成。GPT 虽牺牲双向理解，但因果 mask 使生成成为可能，在文本完成、代码生成上 HumanEval 可达 85%，双向模型仅 30%
- **训练-推理棚架**：在 GPT-4 API 中，系统 prompt 作为 Encoder（双向理解），用户 prompt 和生成使用因果 mask，二者通过 prefix-LM 结构融合，既利用双向理解又保持生成因果，在 MT-Bench 上得分 8.5，优于纯因果的 7.9 和纯双向的 7.2

**实现陷阱**：忘记添加 causal mask 是常见 bug，导致训练 loss 异常低（模型泄露），但推理 perplexity 奇高。检测方法：训练时监控每个位置的 attention entropy，若所有位置接近 0（one-hot），则 mask 可能失效。正确实现应在位置 $i$ 处有 $i$ 个有效注意力权重，entropy 随 $i$ 递增
