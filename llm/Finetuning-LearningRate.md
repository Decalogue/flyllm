# 微调的学习率怎么设置？初始学习率怎么选？需要 Warmup 吗？学习率调度策略如何选择？

## 1. 核心定性
本质上，**微调学习率**是控制模型在预训练基础上进行任务适应的参数更新步长，其选择需要在"快速收敛"和"不破坏预训练表征"之间进行精细权衡。过大导致灾难性遗忘和优化不稳定，过小导致收敛缓慢和欠拟合。

## 2. 具体流程
1. **初始选择**: 根据模型规模和优化器选择基础 lr，大模型（>10B）用 1e-5，小模型用 1e-4，LoRA 可增大 5-10 倍
2. **Warmup 阶段**: 从前 100-1000 step 线性/余弦增加到目标 lr，防止早期梯度爆炸
3. **调度衰减**: Warmup 后采用 cosine、linear 或 constant 衰减，在 50-90% 训练步时降到 10-20% 初始 lr

## 3. 数学基础

**学习率缩放法则**:
$$lr_{\text{eff}} = lr_{\text{base}} \times \sqrt{batch\_size}$$

对于 AdamW:
$$\theta_{t+1} = \theta_t - \frac{lr}{\sqrt{v_t + \epsilon}} m_t$$

其中 $v_t$ 是二阶矩估计。

**Warmup 形式**:

**线性 Warmup**:
$$lr_t = lr_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}}$$

**Cosine Warmup**:
$$lr_t = lr_{\text{max}} \cdot (1 - \cos(\frac{t}{T_{\text{warmup}}} \pi)) / 2$$

**调度策略**:

**Cosine 退火**:
$$lr_t = lr_{\text{max}} \cdot \frac{1}{2}(1 + \cos(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \pi))$$

**Linear 退火**:
$$lr_t = lr_{\text{max}} - (lr_{\text{max}} - lr_{\text{min}}) \cdot \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}$$

**Poly 调度**:
$$lr_t = lr_{\text{max}} \cdot (1 - \frac{t}{T})^p$$

**本征维度分析**:
学习率过大时，模型跳出预训练局部最优的概率：
$$P_{\text{escape}} \propto \exp(-d_{\text{intrinsic}} \cdot lr^2)$$

**梯度噪声缩放**:
$$\text{SNR} = \frac{\|\nabla L(\theta)\|}{\sqrt{\text{Tr}(\text{Cov}(\nabla L(\theta)))}}$$

学习率应与 SNR 成反比。

## 4. 工程考量

**初始学习率选择**:

| 模型规模 | 全量微调 | LoRA | QLoRA | 推荐值 |
|----------|----------|------|-------|--------|
| < 1B | 1e-4 | 5e-4 | 1e-4 | 5e-5 |
| 1B - 7B | 5e-5 | 1e-4 | 5e-5 | 1e-5 |
| 7B - 30B | 2e-5 | 1e-4 | 1e-5 | 1e-5 |
| 30B - 70B | 1e-5 | 1e-4 | 1e-5 | 5e-6 |
| > 70B | 5e-6 | 1e-5 | 5e-6 | 1e-6 |

**经验公式**:
$$lr_{\text{opt}} \propto 1 / \sqrt{\text{num\_parameters}}$$

**Warmup 必要性**:

1. **防止梯度爆炸**: 早期梯度大，学习率过大导致 loss 发散
2. **分布适应**: 让模型从预训练分布平滑过渡到微调分布
3. **层归一化稳定**: LayerNorm 统计量在 warmup 期间稳定

**Warmup 长度选择**:

- **短 Warmup** (100-200 step):
  - 适合任务与预训练相似（如通用文本任务）
  - 快速收敛，但可能不稳定
  
- **长 Warmup** (1000-2000 step):
  - 适合任务差异大（如代码→对话）
  - 更稳定，但收敛慢

**推荐**: $T_{\text{warmup}} = 0.1 \times T_{\text{total}}$

**调度策略对比**:

| 策略 | 收敛速度 | 最终性能 | 适用场景 |
|------|----------|----------|----------|
| Constant | 中 | 中 | 小数据集（<10K） |
| Linear | 快 | 良 | 标准微调 |
| Cosine | 中 | 优 | 大模型，长训练 |
| Polynomial (p=2) | 慢 | 优+ | 需要精细优化 |

**层特定学习率**:

- **顶层**: 5-10 × $lr_{\text{base}}$（快速适应任务）
- **底层**: 0.1-0.5 × $lr_{\text{base}}$（保留通用知识）
- **嵌入层**: 0.1 × $lr_{\text{base}}$（防止词嵌入漂移）

**梯度累积与 lr**:

实际 Batch Size = $N_{\text{acc}} \times \text{batch\_size}$
$$lr_{\text{eff}} = lr_{\text{base}} \times \sqrt{N_{\text{acc}}}$$

**关键发现**:

- **大模型需要小 lr**: 7B 模型 lr=1e-5，65B 模型 lr=5e-6
- **LoRA 可大 lr**: LoRA 只更新适配器，lr 可大 5-10 倍
- **Warmup 必不可少**: 无 warmup 时 loss 发散率 30%，有 warmup 时 <1%

**学习率调度器选择**:

**Cosine**:
- 优点：平滑衰减，性能通常最好
- 缺点：需要预先知道总步数
- 适用：标准微调，知道 epoch 数

**Linear**:
- 优点：简单，无需预设总步数
- 缺点：后期下降过快
- 适用：动态训练（如 early stopping）

**Polynomial (p=0.9)**:
- 优点：平衡 cosine 和 linear
- 缺点：超参数 p 需调优
- 适用：追求极致性能

## 5. 工业映射

**Google T5 微调**:
- lr=1e-4（base），线性 warmup 10000 step
- 后线性衰减到 0.1×
- 在 SuperGLUE 上 SOTA

**Meta LLaMA-2**:
- LoRA: lr=1e-4（比全量小 10 倍）
- Warmup=100 step
- Cosine 衰减 1000 step

**HuggingFace 推荐**:
- **小数据**（<10K）: lr=2e-5, constant
- **中数据**（10K-100K）: lr=1e-5, linear
- **大数据**（>100K）: lr=5e-6, cosine

**QLoRA 实践**:
- lr 必须比 LoRA 小 5-10 倍
- LLaMA 65B: lr=1e-5，adamw
- 无 warmup！直接开始（4-bit 量化稳定性差）

**LLaMA.cpp 微调**:
- 纯 CPU: lr=3e-4（比 GPU 大 30 倍，梯度噪声大）
- Warmup=200
- 最小化内存，无 fancy 调度

**最佳实践总结**:
1. **初始 lr**: 1e-5 for 7B, 5e-6 for 65B
2. **Warmup**: 100-1000 step，线性
3. **调度**: Cosine for >7B，linear for small
4. **LoRA**: lr=1e-4，可大 10 倍
5. **监控**: 前 100 step 的 loss，如发散立即降 lr

**常见错误**:
- ❌ lr 过大: 7B 模型用 1e-4 → loss 发散
- ❌ 无 warmup: 早期梯度爆炸
- ❌ 后期 lr 不降: 过拟合
- ❌ 不同规模用相同 lr: 65B 模型用 1e-5 反而比 1e-4 好

**调试技巧**:
1. 在 100 个 batch 上跑过拟合实验，lr 应能使 loss 降至 0.01
2. 用 Weights & Biases 监控梯度范数，应在 1-10 范围
3. 前 10 step 的 loss 应稳定或小幅下降（10-20%）

**总结**: lr 是微调最关键的参数。宁可小（收敛慢），不要大（发散浪费）。大模型 + LoRA + 10% 预训练数据混合，lr=1e-5 + cosine 是最安全的选择
