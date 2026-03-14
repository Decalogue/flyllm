# Transformer 的核心架构是什么？为什么它成为所有 LLM 的基础？Encoder 和 Decoder 的区别在哪里？

## 1. 核心定性
本质上，**Transformer** 是一种基于自注意力机制的序列建模架构，通过摒弃循环/卷积结构并引入多头自注意力和位置前馈网络，实现了 O(1) 并行度的全局依赖捕获，为大规模语言模型的可扩展性奠定了理论和工程基础。

## 2. 具体流程
1. **编码流程**：输入序列经过词嵌入 + 位置编码后，送入 N 层（通常 6-96 层）的 Transformer Block，每层包含多头自注意力（捕获序列内依赖）和前馈网络（FFN）进行非线性变换，残差连接和层归一化稳定训练
2. **解码流程**：在编码器之上，解码器增加编码器-解码器交叉注意力层，自注意力使用因果掩码（causal mask）保证自回归生成，每一步只能关注已生成的 token，交叉注意力则关注编码器的所有输出以融合源信息
3. **并行化革命**：传统 RNN 的 O(n) 串行计算变为 Transformer 的 O(1) 并行，通过矩阵乘法 $QK^T$ 一次性计算所有位置的注意力权重，在 GPU 上实现 100-1000 倍加速

## 3. 数学基础
**自注意力层**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $Q, K, V \in \mathbb{R}^{n \times d}$，$M$ 是掩码矩阵（下三角为 0，上三角为 $-\infty$）

**多头机制**:
$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

$W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d \times d_k}$，$W^O \in \mathbb{R}^{hd_v \times d}$，通常 $h=8-64$，$d_k = d_v = d/h$

**前馈网络**:
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
$W_1 \in \mathbb{R}^{d \times 4d}$，$W_2 \in \mathbb{R}^{4d \times d}$，放大维度 4 倍再压缩

**层归一化**:
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$
$$\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2}$$

**残差连接**:
$$x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))$$

**Encoder-Decoder 差异**:
- Encoder：双向注意力，掩码 $M=0$（全可见）
- Decoder：因果掩码，$M_{ij} = \begin{cases}0 & i \geq j \\-\infty & i < j\end{cases}$

**复杂度分析**:
- 自注意力：$O(n^2 d)$ 内存和计算
- FFN：$O(nd^2)$
- 总每层：$O(n^2 d + nd^2)$

## 4. 工程考量
**为什么是 LLM 基础**:
- **可扩展性**：参数量从 BERT 的 110M 到 GPT-4 的 1.8T 仅通过增加层数/宽度，理论并行度不变，Transformer 架构无瓶颈
- **硬件友好**：矩阵乘法占 70-90% 计算，完美映射到 GPU/TPU 的张量核心，实现 90%+ 硬件利用率。RNN 的串行依赖仅 30% 利用率
- **长距离依赖**：自注意力理论捕获任意距离依赖，而 LSTM 的梯度传播呈指数衰减，在 512+ 长度时失效

**Encoder vs Decoder 差异**:

| 特性 | Encoder | Decoder |
|------|---------|---------|
| 注意力方向 | 双向（全可见） | 单向（因果）|
| 预训练任务 | MLM（掩码语言模型）| 自回归（CLM）|
| 应用任务 | 分类、NER、问答 | 生成、翻译、摘要 |
| 计算效率 | n^2 一次完成 | n^2 但需缓存 KV |
| 因果性 | 无（可观察全文） | 严格（仅看历史）|

**Trade-off**:
- **全局信息 vs 串行效率**：Encoder 的并行度高但不能生成，Decoder 支持自回归但推理需逐步解码，速度比 Encoder-only 慢 50-100 倍
- **层数深渊**：Transformer 的深层梯度消失问题，在 48+ 层时需引入 Post-LN 改 Pre-LN，或使用残差缩放因子 $\alpha = 1/\sqrt{L}$ 维持稳定

**致命弱点**:
-   **位置编码外推**  ：绝对位置编码 sin/cos 无法外推到训练长度之外，ALiBi 和 RoPE 虽可外推但性能衰减严重，超过 4x 训练长度后困惑度上升 50%+
-   **平方复杂度杀手**  ：自注意力的 $n^2$ 内存消耗在长上下文（100k+）上成为瓶颈，64k 长度需要 16GB 显存仅存储注意力矩阵，触发 OOM

## 5. 工业映射
在工业界，Transformer 是 **LLM 的标准化模块**：
- **Hugging Face Transformers**: 所有模型（BERT, GPT, T5, LLaMA）共享统一 `TransformerBlock` 接口，`AutoModel` 根据 config 自动实例化 Encoder/Decoder。在 1.6B 参数规模下，训练速度比 RNN 快 120 倍，通过 `torch.compile()` 进一步优化到 150 倍
- **GPT 系列（Decoder-only）**：OpenAI 使用标准 Transformer Decoder，96 层，每层隐藏维度 12288，引入稀疏注意力局部窗口降低复杂度。在推理时，KV-cache 预分配和连续批处理（continuous batching）使吞吐量达到 100 req/s，RNN 仅为 2-3 req/s
- **BERT 系列（Encoder-only）**：Google 的 BERT-base 使用 12 层双向编码器，在 GLUE 上达到 SOTA，通过 MLM 和 NSP 预训练，`[CLS]` 标记聚合全局信息。在搜索引擎中，BERT Encoder 部署在 CPU 通过量化（INT8）实现 10ms/query 的延迟
- **T5 / BART（Encoder-Decoder）**：用于翻译和摘要，在 WMT'14 英-德上 BLEU 达 29.8，通过交叉注意力融合源和目标信息。Encoder 侧支持完整双向上下文而 Decoder 保持生成能力，在 beam search 大小为 4 时比纯 Decoder 生成质量高 1-2 BLEU
- **架构演进**：现代 LLM 普遍采用 Decoder-only，因实证显示在相同参数量下，Encoder-Decoder 在生成任务上的性能增益低于复杂度增加（2 倍交叉注意力），LLaMA 2 70B 通过纯 Decoder + RoPE 在 MT-Bench 上超越 T5-11B
