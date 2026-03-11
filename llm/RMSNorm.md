# RMSNorm vs LayerNorm

## 1. 核心定性

**RMSNorm 本质上是一种仅通过均方根(RMS)进行缩放、无需均值归一化的归一化层，以更低计算成本获得与 LayerNorm 等效的稳定训练效果。**

---

## 2. 具体流程

1. **计算均方根**：对输入向量的每个元素平方后求平均再开根号，得到 RMS 值
2. **缩放归一化**：用可学习的权重参数 γ 对归一化后的向量进行缩放
3. **输出结果**：得到与输入同维度的归一化输出

---

## 3. 数学基础

### LayerNorm 公式：
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i$$

### RMSNorm 公式：
$$\hat{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma_i$$

其中：
$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{n}\sum_{j=1}^{n}x_j^2 + \epsilon}$$

**变量定义**：
- $\mathbf{x} \in \mathbb{R}^n$：输入特征向量
- $\mu = \frac{1}{n}\sum_{i=1}^{n}x_i$：均值（**RMSNorm 丢弃此项**）
- $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2$：方差（LayerNorm 使用）
- $\gamma \in \mathbb{R}^n$：可学习的缩放参数（gating 参数）
- $\epsilon$：数值稳定性常数（通常 $10^{-6}$）
- $n$：特征维度

---

## 4. 工程考量

| 维度 | LayerNorm | RMSNorm |
|------|-----------|---------|
| **计算量** | 需计算均值+方差（两次遍历） | 仅需计算 RMS（一次遍历） |
| **参数量** | 均值、方差统计 + γ、β | 仅 RMS 统计 + γ（无 β） |
| **收敛性** | 理论保证更稳定 | 实践表明等效稳定 |

**Trade-off**：
- RMSNorm **去除了 centering（减均值）操作**，牺牲了对输入分布均值的显式建模
- 换取 **~30% 的计算加速** 和 **略少的参数量**

**致命弱点**：
- 在输入分布均值严重偏移的场景（如某些特殊初始化或极端长尾分布）下，RMSNorm 的稳定性略逊于 LayerNorm

---

## 5. 工业映射

在工业界，**RMSNorm 被 LLaMA、LLaMA2、LLaMA3、Mistral、Qwen 等主流开源大模型直接采用**，替换掉了原始的 LayerNorm。其核心动机是：

> **大模型训练中的瓶颈是计算而非数据分布偏移**——RMSNorm 以极小的稳定性代价换取显著的计算收益，在千亿级参数规模下，每次前向传播的 FLOPs 节省积少成多，最终带来可观的训练加速。

特别是在 **T5、LLaMA 的 Decoder-only 架构** 中，RMSNorm 配合 **Pre-Norm（残差连接前置归一化）** 结构，已成为现代大模型的标准配置。

---

## 6. 为什么 LLaMA 用 RMSNorm？

**根本原因：性能与效率的最优平衡**

1. **计算效率优先**：大模型训练成本高昂，RMSNorm 减少约 30% 归一化计算量
2. **Pre-Norm 架构加持**：LLaMA 使用 Pre-Norm（归一化在残差之前），RMSNorm 的无均值特性与此配合良好
3. **实践验证**：论文《Root Mean Square Layer Normalization》证明 RMSNorm 在多个 NLP 任务上与 LayerNorm 性能相当
4. **参数精简**：无需 bias 参数（β），简化模型结构

**一句话总结**：当模型规模达到百亿/千亿级别，**每一次前向传播节省的计算都价值千金**——RMSNorm 就是这种工程理性的体现。
