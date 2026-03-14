# Cross-Attention 和 Self-Attention 有什么区别？分别用在什么场景？

## 1. 核心定性
本质上，**Cross-Attention** 是跨两个不同序列的注意力机制，其中 Query 来自目标序列而 Key/Value 来自源序列，实现信息的条件融合；**Self-Attention** 则是单序列的内部注意力，Q/K/V 同源，用于捕获序列内依赖。

## 2. 具体流程
1. **Self-Attention**：输入序列 $X$ 同时生成 Q, K, V（同源），计算 $X$ 内每对位置的依赖，如 "The cat sat on the mat" 中 "cat" 与 "sat" 的关联
2. **Cross-Attention**：在解码器生成时，当前 token 的查询 $Q_{\text{dec}}$ 从编码器输出 $K_{\text{enc}}, V_{\text{enc}}$ 中提取相关信息，如翻译时法语的 "chat" 查询从英文编码器中提取 "cat" 的信息
3. **场景选择**：Encoder-only 模型仅用 Self-Attention（BERT），Decoder-only 模型用 Causal Self-Attention（GPT），Encoder-Decoder 模型交替使用 Self-Attention（理解）和 Cross-Attention（融合），如 T5, BART

## 3. 数学基础
**Self-Attention**：
$$\text{SelfAttn}(X) = \text{softmax}\left(\frac{XW^Q (XW^K)^T}{\sqrt{d_k}}\right) XW^V$$

其中 $Q, K, V$ 同源：$Q = XW^Q$，$K = XW^K$，$V = XW^V$

**Cross-Attention**：
$$\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{Q_{\text{dec}}W^Q (K_{\text{enc}}W^K)^T}{\sqrt{d_k}}\right) V_{\text{enc}}W^V$$

其中：
- $Q_{\text{dec}}$：来自解码器的隐藏状态（生成侧）
- $K_{\text{enc}}, V_{\text{enc}}$：来自编码器的输出（源信息侧）

**维度定义**：
- $Q_{\text{dec}} \in \mathbb{R}^{n_{\text{dec}} \times d}$（目标序列长度）
- $K_{\text{enc}}, V_{\text{enc}} \in \mathbb{R}^{n_{\text{enc}} \times d}$（源序列长度）

**注意力的输出**：
$$O \in \mathbb{R}^{n_{\text{dec}} \times d_v}$$

**复杂度对比**：
- Self-Attention：$O(n^2 d)$（$n_{\text{dec}} = n_{\text{enc}}$）
- Cross-Attention：$O(n_{\text{dec}} \cdot n_{\text{enc}} \cdot d)$

## 4. 工程考量
**主要区别**:

| 特性 | Self-Attention | Cross-Attention |
|------|---------------|----------------|
| Q, K, V 来源 | 同一序列 | Q 来自目标，K/V 来自源 |
| 输入长度相关性 | 源=目标 | 源≠目标 |
| 位置关系 | 捕获内部依赖 | 捕获跨序列对齐 |
| 计算复杂度 | $O(n^2)$ | $O(n_{\text{dec}} \cdot n_{\text{enc}})$ |
| 使用场景 | 理解、编码 | 生成、翻译、融合 |

**为什么需要 Cross-Attention**：
- **序列到序列映射**：翻译时源语言长度 $n_{\text{enc}}$ 与目标长度 $n_{\text{dec}}$ 不同，需要动态对齐。Self-Attention 无法引入外部信息
- **解码器信息瓶颈**：Decoder 只能看到已生成的 token（causal mask），需从编码器获取完整的源语义，Cross-Attention 是唯一的跨序列通道

**Trade-off**:
- **双向理解 vs 条件生成**：Self-Attention 提供上下文理解能力，Cross-Attention 提供条件依赖能力，二者结合使模型能先理解再生成
- **计算开销**：Cross-Attention 增加额外的 $n_{\text{dec}} \cdot n_{\text{enc}} \cdot d$ 计算，在生成时每一步都要重新计算，使解码比纯 Decoder 慢 2 倍

**致命弱点**:
- **对齐错误**：在长度差异大时（$n_{\text{enc}} \gg n_{\text{dec}}$），Cross-Attention 可能过度关注源序列的局部区域，导致漏翻或错翻，需引入覆盖机制（coverage）惩罚重复关注
- **梯度传播路径**：Cross-Attention 引入从解码器到编码器的梯度回传路径，在深层 Transformer（48+层）时可能造成梯度消失或爆炸，需使用 Pre-LN 或残差缩放

## 5. 工业映射
在工业界，Cross-Attention 是 **Seq2Seq 任务的标配**：
- **T5 / mT5**：Google 的 Text-to-Text 框架，Encoder 12 层 Self-Attention 理解输入，Decoder 12 层交替 Self-Attention（理解已生成内容）和 Cross-Attention（融合源信息）。在 WMT'14 英德翻译上，Cross-Attention 使 BLEU 从 20（仅 Decoder）提升至 29，证明条件融合的价值
- **Transformer 翻译**：原始论文中 Encoder-Decoder 架构，Cross-Attention 位于解码器每层，对齐源目标。在 fairseq 实现中，Cross-Attention 与 Self-Attention 共享 KV-cache 内存，通过 `torch.bmm` 批量计算，在 GPU 上吞吐达 1000 token/s
- **BART / Pegasus**：摘要任务中，Cross-Attention 让解码器从编码器的完整文档表示中提取关键信息，结合 beam search 长度惩罚，避免重复和遗漏。在 CNN/DailyMail 上，Cross-Attention 层数从 6 增加到 12，ROUGE-2 提升 1.5 点
- **GPT 代码生成**：尽管是 Decoder-only，GitHub Copilot 在提示（prompt）和生成间隐式使用 Cross-Attention：prompt 部分可双向 Self-Attention，生成部分 Causal Self-Attention，类似 Encoder-Decoder 的简化。实现时通过 `past_key_values` 缓存，将 prompt 的 KV 作为常量，生成时 query 从 cache 中提取信息，达到 Cross-Attention 效果但无需显式编码器
- **多模态融合**：CLIP 的图像编码器与文本解码器间使用 Cross-Attention，图像嵌入作为 Key/Value，文本 token 作为 Query，在 DALL-E 2 中实现文本到图像生成的细粒度控制，交叉注意力图可视化显示模型对齐文本描述与图像区域
