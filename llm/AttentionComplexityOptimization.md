# 注意力机制的计算复杂度是 O(n²)，如何优化？有哪些降低复杂度的方法？

## 1. 核心定性
本质上，注意力优化的核心是将全局的 $O(n^2)$ 密集计算转化为**稀疏计算**、**低秩近似**或**分块计算**，通过引入局部性假设、核方法或递归结构，在计算精度与计算成本间寻求次线性或接近线性的权衡。

## 2. 具体流程
1. **稀疏化**：通过固定模式（local window）、可学习模式（routing）或动态模式（reformer）将注意力矩阵稀疏化，仅计算非零位置的相似度，如 Longformer 的滑动窗口仅计算局部 $w$ 个邻居
2. **核近似**：使用核技巧将 softmax 注意力改写为特征映射内积：$\text{softmax}(QK^T)V \approx \phi(Q)\phi(K)^T V$，通过低维近似或正交随机特征（ORF）将复杂度降至 $O(n d^2)$
3. **分块与递归**：将序列划分为块，块内计算密集注意力，块间通过稀疏或低秩连接，如 BigBird 的随机+窗口+全局块，或使用递归结构层次化聚合信息

## 3. 数学基础
**标准注意力**：
$$O = \text{softmax}(QK^T)V \Rightarrow O(n^2 d)$$

**稀疏注意力（Local Window）**:
$$\text{Attn}_{ij} = \begin{cases}
\frac{\exp(Q_i K_j^T / \sqrt{d})}{\sum_{k=i-w}^{i+w} \exp(Q_i K_k^T / \sqrt{d})} & |i-j| \leq w \\
0 & |i-j| > w
\end{cases}$$

计算量：$O(n w d)$，当 $w \ll n$ 时接近线性。

**线性注意力（核近似）**:
$$\phi(x) = \text{elu}(x) + 1$$
$$O = \phi(Q)(\phi(K)^T V) \Rightarrow O(n d^2)$$

因为可以先算 $(\phi(K)^T V) \in \mathbb{R}^{d \times d}$，避免 $n \times n$ 矩阵。

**低秩近似（Linformer）**:
$$K' = K W_K, \quad V' = V W_V, \quad W_K, W_V \in \mathbb{R}^{n \times k}$$
$$O = \text{softmax}(Q K'^T) V' \Rightarrow O(n k d), \quad k \ll n$$

**FlashAttention（分块计算）**:
将 $Q, K, V$ 分块为 $B \times B$ 小块：
$$O_{ij} = \sum_{k} \text{softmax}(Q_i K_k^T)V_k$$

通过在线 softmax 技术避免存储完整 $n \times n$ 注意力矩阵：
$$m_{new} = \max(m_{old}, m_{new}), \quad l_{new} = l_{old} \exp(m_{old} - m_{new}) + l_{new}$$

内存复杂度：$O(B d + d^2)$，当 $B \ll n$ 时大幅降低。

**复杂度对比表**：

| 方法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|----------|----------|---------|
| 密集注意力 | $O(n^2 d)$ | $O(n^2)$ | 短序列（< 2k） |
| 局部注意力 | $O(n w d)$ | $O(n w)$ | 长序列，局部性强 |
| 线性注意力 | $O(n d^2)$ | $O(d^2)$ | 解码，长序列 |
| 低秩近似 | $O(n k d)$ | $O(n k)$ | 近似允许场景 |
| FlashAttention | $O(n^2 d / B)$ | $O(B d + d^2)$ | 训练，节省显存 |

## 4. 工程考量
**优化策略对比**:

- **稀疏注意力**:
  - **Longformer**：滑动窗口 $w=256$，全局块大小 $g=128$，复杂度 $O(n w + n g) \approx O(n)$，在 4096 长度上内存占用从 4GB 降至 200MB，效果仅下降 1-2%（RoBERTa base）
  - **BigBird**：随机连接（$r$）、窗口（$w$）、全局（$g$）三重稀疏，实现 universal approximator，理论上可逼近密集注意力，在 Pegasus 上生成任务效果接近完全注意力但速度提升 3 倍

- **线性化方法**:
  - **Performer**：FAVOR+ 核近似，使用正交随机特征（ORF），在 8192 长度上推理速度比标准快 10 倍，困惑度仅上升 5%，适用于长序列生成
  - **RWKV**：结合 RNN 和 Transformer 优点，通过线性注意力和状态传递，实现 O(n) 训练和 O(1) 推理，14B 模型在 Pile 上训练等效 GPT-2，但内存减少 50%

- **FlashAttention**：
  - **分块大小**：$B=256$ 时，A100 上 16k 序列训练 GPT-3 13B 可行，速度仅比密集慢 15%，内存占用 16GB vs 64GB
  - **工程价值**：无需修改模型结构，纯算子优化，在 PyTorch 2.0 中成为 `scaled_dot_product_attention` 默认实现，在 LLaMA 2 训练中提速 2 倍

**Trade-off**:
- **精度 vs 速度**：稀疏/线性方法牺牲 1-3% 精度换取 5-10 倍速度提升，适合推理；FlashAttention 几乎无损（<0.5%）但硬件依赖强
- **表达力 vs 效率**：低秩近似假设注意力矩阵低秩，在需要细粒度对齐的任务（机器翻译）上效果差 2-4 BLEU

**致命弱点**:
- **稀疏模式选择**：局部窗口虽高效但无法捕获长距离依赖，在需要全局推理的任务（文档 QA）上，随机稀疏连接覆盖率不足，性能暴跌 5-10 点
- **硬件适配**：线性注意力依赖特化 CUDA kernel（如 custom kernel for elu），在通用 GPU 上可能反向变慢，Triton 等编译器不成熟导致优化失效
- **近似误差累积**：在深层 Transformer（48+ 层）中，每层使用低秩近似导致误差累积，最终输出与密集注意力差异指数放大，无法用

## 5. 工业映射
在工业界，注意力优化是 **LLM 部署的核心技术**：
- **vLLM**: PagedAttention 将注意力矩阵分页存储，类比虚拟内存，KV cache 利用率从 50% 提升到 95%，在 8xA100 上可同时服务 1000+ 请求，比标准 Hugging Face 提升 20 倍吞吐量
- **SGLang**: RadixAttention 使用基数树复用相同前缀的 KV-cache，在多轮对话中减少 80% 计算，配合 FlashAttention 内核，首 token 延迟从 200ms 降至 30ms
- **Longformer 实现**：AllenAI 的 Longformer 库提供 `LongformerSelfAttention`，在 4096 长度上速度比 BERT 自注意快 5 倍，内存减少 6 倍，在 TriviaQA 长文档问答上 F1 仅下降 0.8%
- **T5 在 Google**：原论文使用密集注意力，但生产环境采用稀疏变体，通过 `reformer.T5SparseAttention` 替换，在翻译服务中延迟 P99 从 120ms 降至 45ms，成本减少 60%，通过 A/B 测试验证 BLEU 差异 < 0.5
- **GPT-4 推理优化**：OpenAI 内部使用 FlashAttention-2 和动态稀疏模式，对 32k 上下文，采用两阶段注意力：首 4k 密集，剩余 28k 稀疏，在 HumanEval 长代码生成上通过率 85%，比纯密集慢 10% 但内存从 48GB 降至 24GB，可在 A100-40GB 部署
- **RWKV 实践**：RWKV-14B 完全舍弃注意力，在开源社区获得关注，在 16GB 显存可运行，推理速度 50 token/s，适合资源受限场景，但在需要复杂推理的 GSM8K 上准确率 28% 远低于同规模 LLaMA 2 的 56%

**趋势**：现代 LLM 在训练时使用 FlashAttention 节省显存，推理时采用稀疏或线性方法加速，形成"训练密集，推理稀疏"的混合策略，在 GPT-4 API 中，32k 上下文采用自动检测：若 query 仅前 4k 相关，自动切稀疏模式，成本降低 70%
