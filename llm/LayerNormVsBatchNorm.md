# LayerNorm vs BatchNorm 面试精要

本质上，**LayerNorm** 是为了解决**序列数据变长导致 batch 统计不稳定**问题，通过对**单样本内部维度**做归一化的统计独立方案；而 **BatchNorm** 则是通过对**跨样本同一维度**做统计来缓解**深层梯度消失**的 batch 依赖方案。

---

## 1. 具体流程

1. **BN**: 对每个特征维度，计算跨 batch 的均值 $\mu_B$ 和方差 $\sigma^2_B$，然后 $(x - \mu_B) / \sqrt{\sigma^2_B + \epsilon}$，最后学一个 affine 变换 $\gamma x + \beta$。
2. **LN**: 对每个样本，计算其所有特征维度的均值 $\mu_L$ 和方差 $\sigma^2_L$，然后 $(x - \mu_L) / \sqrt{\sigma^2_L + \epsilon}$，同样接 affine。
3. **Transformer 选 LN 的核心原因**: 序列长度不一致时，BN 的 batch 统计量方差极大；LN 统计单样本内部，与 batch size、序列长度均解耦。

---

## 2. 数学基础

**BatchNorm（对 batch 维度求期望）:**

$$\hat{x}_{i,j} = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} \cdot \gamma_j + \beta_j$$

其中：
- $i \in [1, B]$: batch 中的样本索引
- $j \in [1, D]$: 特征维度索引
- $\mu_j = \frac{1}{B}\sum_{i=1}^{B} x_{i,j}$: 第 $j$ 维在 batch 上的均值
- $\sigma_j^2 = \frac{1}{B}\sum_{i=1}^{B}(x_{i,j} - \mu_j)^2$: 第 $j$ 维在 batch 上的方差

**LayerNorm（对特征维度求期望）:**

$$\hat{x}_{i} = \frac{x_{i} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \cdot \gamma + \beta$$

其中：
- $\mu_i = \frac{1}{D}\sum_{j=1}^{D} x_{i,j}$: 第 $i$ 个样本在所有特征上的均值
- $\sigma_i^2 = \frac{1}{D}\sum_{j=1}^{D}(x_{i,j} - \mu_i)^2$: 第 $i$ 个样本的方差
- $\gamma, \beta \in \mathbb{R}^D$: 可学习的仿射参数（与 BN 的逐维参数不同，LN 可用共享或逐维）

---

## 3. 工程考量

| 维度 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| **统计依赖** | 依赖 batch size，小 batch 时 $\sigma^2$ 估计不准 | 与 batch size 完全解耦 |
| **序列兼容** | padding 导致有效 batch 统计失真，训练和推理分布不一致（需维护 running statistics） | 无视序列长度，训练和推理逻辑完全一致 |
| **梯度流** | 通过 batch 统计引入样本间隐式关联，梯度更平滑 | 样本间完全独立，梯度方差略大 |
| **致命弱点** | **小 batch / 变长序列场景下统计量噪声大，推理时分布偏移严重** | **单样本统计可能丢失跨维度的重要关联信息** |

**关键 Trade-off**:  
BN 用 **样本间统计稳定性** 换取 **batch 依赖和序列不友好**；  
LN 用 **统计独立性** 换取 **对变长序列和分布式训练的完全兼容**。

---

## 4. 工业映射

- **Transformer（BERT/GPT/T5）**: 全部使用 **LayerNorm** 前置（Pre-LN）或后置（Post-LN），确保多卡训练时梯度稳定，且推理时不受 batch size 变化影响。
- **ResNet/VGG/CNN 视觉 backbone**: 普遍使用 **BatchNorm**，因为图像输入尺寸固定、batch size 通常较大，且 BN 的轻微正则化效果有助于泛化。
- **混合方案（LLaMA/RMSNorm）**: 在 LayerNorm 基础上移除 mean centering，仅保留 $\frac{x}{\text{RMS}(x)} \cdot \gamma$，进一步减少计算量，成为现代 LLM 的默认配置。

---

## 一句话总结

Transformer 选 LayerNorm，是因为它的核心战场是**变长序列 + 超大分布式训练**，LN 的统计独立性恰好击中 BN 的 batch 依赖和长度敏感的死穴。
