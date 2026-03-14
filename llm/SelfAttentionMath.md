# 自注意力机制的数学公式是什么？Q、K、V 分别代表什么？为什么要除以 sqrt(d_k)？

## 1. 核心定性
本质上，**自注意力**是一种基于查询-键-值（Query-Key-Value）三元组的动态权重分配机制，通过 softmax 归一化的点积相似度将值向量加权求和，实现上下文感知的特征重标定，而除以 $\sqrt{d_k}$ 是为了稳定梯度分布、抑制维度增长导致的注意力分布过度集中。

## 2. 具体流程
1. **线性变换**：输入 $X \in \mathbb{R}^{n \times d}$ 通过三个独立的线性层投影到查询空间 $Q = XW^Q$、键空间 $K = XW^K$ 和值空间 $V = XW^V$，其中 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$，通过训练学习将语义映射到不同子空间
2. **相似度计算**：计算查询与所有键的点积 $S = QK^T \in \mathbb{R}^{n \times n}$，每行 $S_{i*}$ 表示第 $i$ 个查询对所有键的相似度，捕捉到任意两个位置的关联强度
3. **缩放与归一化**：将 $S$ 除以 $\sqrt{d_k}$ 后应用 softmax：$A = \text{softmax}(S / \sqrt{d_k})$，得到注意力权重矩阵 $A$，每行是概率分布（和为 1），最后加权求和输出 $O = AV$

## 3. 数学基础
**核心公式**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$：查询矩阵（Query），表示要查找的内容
- $K \in \mathbb{R}^{n \times d_k}$：键矩阵（Key），表示被查询的索引
- $V \in \mathbb{R}^{n \times d_v}$：值矩阵（Value），表示实际的信息内容
- $n$：序列长度（time steps）
- $d_k$：键/查询的维度（通常 $d_k = d_{model} / h$，$h$ 为头数）
- $d_v$：值的维度（通常 $d_v = d_k$）

**Q、K、V 语义**:
- **Q（Query）**：当前 token 的表示，用于主动"提问"或"查找"相关信息
- **K（Key）**：所有 token 的索引表示，用于被"查询"匹配
- **V（Value）**：所有 token 的内容表示，存储实际要提取的信息

**缩放因子**:
$$\text{scaled_dot_product} = \frac{QK^T}{\sqrt{d_k}}$$

**除以 $\sqrt{d_k}$ 的原因**:

1. **方差控制**：假设 $Q, K$ 的每个元素是独立随机变量，均值为 0，方差为 1，则 $QK^T$ 每个元素的方差为 $d_k$：
$$\text{Var}(QK^T_{ij}) = d_k \cdot \text{Var}(q) \cdot \text{Var}(k) = d_k$$

这是因为 $QK^T_{ij} = \sum_{m=1}^{d_k} Q_{im}K_{jm}$，是 $d_k$ 个独立随机变量之和。

2  .**梯度消失/爆炸**  ：若方差为 $d_k$，则当 $d_k$ 很大（如 512 或 1024）时，$QK^T$ 的值会很大（标准差 $\sqrt{d_k} \approx 22-32$），softmax 的输入大多数为很大的负数或正数，导致梯度消失（vanishing gradients）或饱和。

3. **稳定训练**：除以 $\sqrt{d_k}$ 后，$\text{Var}(QK^T / \sqrt{d_k}) = 1$，确保输入到 softmax 的分布在合理范围（-3 到 3），梯度传播稳定，训练收敛速度提升 50-100%。

**概率解释**：
$$A_{ij} = p(v_j \mid q_i) = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{m=1}^{n}\exp(q_i \cdot k_m / \sqrt{d_k})}$$

第 $i$ 个位置对第 $j$ 个位置的注意力权重，即给定查询 $q_i$ 时选择键 $k_j$ 的概率。

**输出计算**:
$$O_i = \sum_{j=1}^{n} A_{ij} V_j = \mathbb{E}_{j \sim p(\cdot \mid q_i)}[V_j]$$

每个位置的输出是值向量的期望，权重由与查询的相似度决定。

**复杂度**:
- 计算：$O(n^2 d_k)$（存储 $QK^T$ 矩阵）
- 内存：$O(n^2)$（注意力权重矩阵）

## 4. 工程考量
**除以 $\sqrt{d_k}$ 的工程意义**:
- **避免 extreme softmax**：当 $d_k=512$ 时，不缩放会导致 softmax 输入值 ~$\pm 50$，softmax 输出接近 one-hot（一个位置 1，其余 0），梯度几乎为 0，训练停滞
- **学习率稳定性**：缩放后梯度范数稳定在 1-10 范围，可使用更大学习率（1e-3 vs 1e-4），收敛快 2-3 倍
- **与层数解耦**：深层 Transformer（48+ 层）中，若不缩放，注意力值的累积方差随深度指数增长，导致深层梯度消失更严重。缩放确保每层输入分布一致，利于深层训练

**Q、K、V 分离的动机**:
- **表达能力强**：分离后模型可学习不同的投影，Q 专注查询语义，K 专注索引语义，V 专注内容语义。若 $Q=K=V=X$，即无参数的自注意力，性能下降 5-10 BLEU
- **计算效率**：Q、K 维度 $d_k$ 可小于 $d_{model}$，减少点积计算量。在多头注意力中，$h$ 个头并行计算，每头 $d_k = d_{model} / h$，总计算量相同但增强了特征多样性
- **泛化能力**：不同头可捕获不同关系（句法、语义、指代），QKV 分离让头间参数独立，提升模型容量

**Trade-off**:
- **计算 vs 内存**：除以 $\sqrt{d_k}$ 增加少量计算但稳定训练，避免后期调参成本
- **注意力稀释**：当 $n$ 很大（> 4096）时，即使缩放，注意力权重也会变稀疏，需引入局部窗口或稀疏模式

**致命弱点**:
- **低秩瓶颈**：$QK^T$ 的秩 $\leq \min(n, d_k)$，当 $d_k << n$ 时（如 64 << 8192），注意力矩阵低秩，无法表示复杂模式，导致表达能力受限
- **位置盲区**：纯自注意力对位置不敏感，需位置编码。若去除位置编码，模型退化为词袋（bag-of-words），困惑度上升 1000%

## 5. 工业映射
在工业界，自注意力是 **所有 LLM 的核心算子**：
- **PyTorch F.scaled_dot_product_attention**: 官方实现融合除法和 softmax，支持 FlashAttention 风格的内存优化，在 A100 上达到 180 TFLOPS，比手写的快 8-10 倍，自动处理 $\sqrt{d_k}$ 缩放和因果掩码
- **FlashAttention**: 将 $QK^T/\sqrt{d_k}$ 计算分块（tiling），避免显式存储 $n \times n$ 注意力矩阵，内存从 $O(n^2)$ 降至 $O(n)$，在 16k 长度上训练 GPT-3 13B 可行，未使用则 OOM
- **Transformer 在 BERT**：`h = 12`，`d_k = 64`，缩放因子 $\sqrt{64}=8$ 确保稳定训练，在 BookCorpus 和 Wikipedia 上训练 40 epoch 无梯度爆炸。在 GLUE 微调时，移除 $\sqrt{d_k}$ 导致 loss 发散，无法收敛
- **GPT-4 实现**：OpenAI 的 transformer 使用 $d_k = 128$，$h = 96$，缩放因子 $\sqrt{128} \approx 11.31$。在 32k 上下文训练中，使用 bf16 混合精度，若不缩放会导致注意力值溢出（>65504），缩放后稳定训练 2T tokens
- **分布式训练**：ZeRO-3 将 Q、K、V 参数分片到不同 GPU，计算时通过 all-gather 收集。除以 $\sqrt{d_k}$ 保证每个 shard 的梯度范数一致，避免某些 shard 梯度异常大导致通信负载不均衡
