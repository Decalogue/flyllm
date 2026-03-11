# SFT vs RLHF 核心区别与取舍

---

## 1. 核心定性

- **SFT**：本质上是一个**行为克隆**过程，通过最小化模型输出与人类标注数据的交叉熵损失，将预训练模型的"通识分布"对齐到"目标域分布"。
- **RLHF**：本质上是一个**偏好优化**框架，通过奖励模型（RM）将人类偏好量化为标量信号，再用强化学习（PPO）在策略空间中搜索能最大化累积奖励的最优策略。

---

## 2. 具体流程

### SFT（Supervised Fine-Tuning）
1. **构造数据**：收集高质量 (prompt, response) 对作为监督信号；
2. **最小化损失**：在预训练权重基础上，最大化 $\log P_\theta(y_{human}|x)$；
3. **收敛停止**：验证集 loss 不再下降即停止，得到指令遵循模型。

### RLHF（Reinforcement Learning from Human Feedback）
1. **训练 RM**：用人类偏好评分 (A > B) 训练奖励模型 $r_\phi(x, y)$；
2. **策略优化**：以 PPO 为核心，在 $r_\phi$ 的引导下优化策略 $\pi_\theta(y|x)$；
3. **KL 约束**：加入 $\beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$ 防止策略崩溃。

---

## 3. 数学基础

### SFT 目标函数

$$\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{human}} \left[ \sum_{t=1}^{|y|} \log P_\theta(y_t | x, y_{<t}) \right]$$

其中：
- $x$: 输入 prompt
- $y$: 人类标注的标准回答
- $\theta$: 模型参数
- $\mathcal{D}_{human}$: 人工标注数据集

### RLHF 目标函数（PPO 版本）

$$\mathcal{J}_{RLHF}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) \right] - \beta \cdot D_{KL}\left(\pi_\theta(y|x) \| \pi_{ref}(y|x)\right)$$

其中：
- $r_\phi(x, y)$: 奖励模型输出的标量分数
- $\pi_{ref}$: SFT 后的参考策略（冻结参数）
- $\beta$: KL 惩罚系数（防止 reward hacking）
- $D_{KL}$: KL 散度，约束新策略不要偏离参考策略太远

---

## 4. 工程考量

| 维度 | SFT | RLHF |
|------|-----|------|
| **数据成本** | 需高质量**标注对**，数量级 $10^4 \sim 10^5$ | 需**偏好对**（A/B 比较），数量级 $10^4$ + RM 训练 |
| **计算成本** | 单阶段微调，成本低 | 三阶段流水线（SFT → RM → PPO），成本高出 **5~10x** |
| **能力天花板** | 受限于标注者能力，**无法超越人类示范** | 可探索策略空间，**潜在超越人类水平** |
| **致命弱点** | **分布外泛化差**；对未见过的问题类型容易幻觉 | **Reward Hacking**；可能学会欺骗奖励模型；训练不稳定（PPO 的 variance 大） |
| **对齐精度** | 只能模仿"什么是好的"，无法精细区分"哪个更好" | 能捕捉**细粒度偏好**（helpful > harmless > neutral） |

**核心 Trade-off**：
- SFT 牺牲**上限**换取**稳定性和低成本**；
- RLHF 牺牲**训练稳定性**和**算力**换取**对齐精度和天花板**。

---

## 5. 工业映射

| 技术 | 工业应用 |
|------|----------|
| **SFT** | ChatGLM、文心一言的初始指令对齐；垂直领域（医疗、法律）快速适配；LoRA/QLoRA 低资源微调 |
| **RLHF** | GPT-4、Claude、Llama 2 Chat 的最终对齐阶段；InstructGPT 论文验证的完整三阶段流水线 |
| **改进版 RLHF** | **DPO**（Direct Preference Optimization）：跳过显式 RM，直接用偏好数据优化策略，被 **Mistral、Zephyr** 等模型采用，成本降低 50%+ |

**最新趋势**：DPO > PPO 正成为开源社区共识，Anthropic 的 **Constitutional AI** 则用 RL-CAI 替代人类标注，进一步降低人工成本。
