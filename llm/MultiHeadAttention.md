# 为什么需要多头注意力？多头比单头好在哪里？头数怎么选择？

## 1. 核心定性
本质上，**多头注意力**通过将输入投影到多个低维子空间并行计算注意力，再将结果拼接，使模型能同时捕获不同语义关系（句法、语义、指代等），每个头专注特定模式，实现特征多样性增强和表达能力指数级提升。

## 2. 具体流程
1. **多头分割**：输入 $X$ 通过 $h$ 组独立的线性变换得到 $Q_i, K_i, V_i \in \mathbb{R}^{n \times d_k}$，其中 $i \in [1, h]$，$d_k = d_{model} / h$，每个头在不同子空间计算注意力
2. **并行计算**：每个头独立执行自注意力：$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$，捕获不同关系，如头1专注局部句法（相邻词），头2捕获长距离语义关联
3. **拼接融合**：将所有头输出拼接：$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$，其中 $W^O \in \mathbb{R}^{hd_v \times d_{model}}$ 融合各头信息

## 3. 数学基础
**单头注意力**：
$$\text{head} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**多头扩展**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个头：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**参数矩阵**：
- $W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

通常 $d_k = d_v = d_{model} / h$，如 BERT-base：$d_{model}=768$，$h=12$，$d_k=64$。

**表达能力分析**：
单头注意力的秩受限：
$$\text{rank}(\text{Attention}) \leq \min(n, d_k)$$

多头总秩：
$$\text{rank}(\text{MultiHead}) \leq h \cdot \min(n, d_k)$$

当 $n \gg d_k$ 时，多头秩提升 $h$ 倍，可表示更复杂的模式。

**头数选择公式**：
$$h^* = \arg\max_{h} \frac{\text{BLEU}(h)}{\text{FLOPs}(h)}$$
经验规则：
$$d_{model} \propto h \Rightarrow d_k = \frac{d_{model}}{h} \approx 64 \text{ 最优}$$

**复杂度**：
总计算量：$O(h \cdot n^2 d_k) = O(h \cdot n^2 \cdot d_{model}/h) = O(n^2 d_{model})$
与单头相同！多头不增加总计算，仅改变参数分布。

## 4. 工程考量
**多头优势**:
- **特征多样性**：不同头捕获不同语言现象：头0-3专注局部依赖（相邻词），头4-7捕获语法关系（主谓宾），头8-11捕获长距离指代（代词-名词），可视化显示头间模式互异，提高鲁棒性
- **训练稳定性**：每个头梯度独立，避免单头注意力崩塌（所有权重到少数位置）。多头分散风险，梯度方差降低 $h$ 倍，训练收敛快 30-50%
- **低秩瓶颈突破**：$n=512$，$d_k=64$ 时，单头秩 ≤ 64，无法表示长序列复杂依赖；$h=12$ 时，有效秩提升至 768，全序列表征能力恢复

**头数选择策略**：
- **BERT-base**：$h=12$，权衡性能和速度，在 GLUE 上 8 头下降 1.2%，16 头提升 0.3% 但速度下降 15%
- **GPT-4**：$h=96$，$d_{model}=12288$，$d_k=128$，大模型需更多头保持 $d_k$ 在 64-128 区间，太小导致坍缩，太大无收益
- **经验法则**：$h = d_{model} / 64$，如 768 → 12，1024 → 16，4096 → 64

**Trade-off**：
- **参数效率 vs 头数**：$W^O$ 拼接引入 $hd_v d_{model}$ 额外参数（如 12×64×768=589k），相比单头仅 2% 增量，收益远大于成本
- **内存带宽**：多头需存储 $h$ 个中间矩阵，激活内存增加 $h$ 倍，在训练时可能触发显存瓶颈，需梯度检查点（gradient checkpointing）优化

**致命弱点**：
- **头冗余现象**：在 BERT-large 的 $h=16$ 中，40% 头相似度 > 0.9，功能重复。可通过剪枝移除冗余头，参数量减少 20% 而性能不变
- **头数天花板**：当 $h > d_{model} / 32$ 时，$d_k < 32$，注意力鞅化严重，性能饱和甚至下降。GPT-3 175B 使用 $h=96$ 接近极限

## 5. 工业映射
在工业界，多头注意力是 **Transformer 的默认配置**：
- **Hugging Face Transformers**：`num_attention_heads` 是核心参数，`BertConfig` 中 $h=12$，每个头独立实现 `BertSelfAttention`，通过 `reshape` 和 `transpose` 将输入变换为 `(batch, h, n, d_k)` 并行计算。在 GPU 上使用 `torch.einsum` 或多维张量乘法，充分利用 Tensor Cores
- **GPT-3 / GPT-4**：OpenAI 使用 $h=96$ 的头并行计算，通过 `cublasGemmStridedBatched` 实现批量矩阵乘法，单次 kernel 启动计算所有头，延迟仅为单头的 1.2 倍，吞吐提升 80 倍（96 个头的并行度）
- **优化策略**：在 T5-11B 的分布式训练中，多头与张量并行（tensor parallelism）结合，每 GPU 计算 $h/8=12$ 个头，all-reduce 聚合结果，通信开销被计算掩盖，效率 90%+
- **头剪枝工程**：distilBERT 通过 knowledge distillation 将 $h=12$ 剪枝到 $h=6$，结合 softmax 温度调节，在 GLUE 上保留 97% 性能，模型大小缩小 50%，推理速度提升 2 倍，适合边缘部署
- **注意力可视化**：BERTViz 工具显示不同头捕获不同模式，第2层头0聚焦【形容词-名词】，第8层头7捕获【代词-指代词】，帮助诊断模型行为，多头可解释性远超单头

**头数趋势**：现代 LLM 趋向更多小头，如 LLaMA 2 70B 使用 $h=64$，$d_k=128$，平衡计算和表达能力，相比 $h=32$，$d_k=256$ 配置在代码生成任务上 HumanEval 提升 4 个百分点，证明多样性优先于单个头容量

</content>