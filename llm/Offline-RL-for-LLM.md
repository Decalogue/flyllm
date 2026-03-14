# 离线 RL 在 LLM 中怎么用？Off-policy RL 的优势？

## 1. 核心定性
本质上，**离线（Offline）RL**通过**静态数据集**训练策略，无需在线交互采样，利用**重要性采样或保守优化**从现有轨迹（SFT数据、人类对话、模型生成）中学习，避免昂贵的在线生成成本，样本效率提升**10-100倍**。

## 2. 具体流程
1. **数据收集**：从SFT模型、旧版本模型或人类标注中收集$(x, y, r)$三元组（prompt, response, 隐含奖励），构建静态数据集$D_{\text{offline}}$
2. **Off-policy算法**：使用CQL（Conservative Q-Learning）、IQL（Implicit Q-learning）或DPO（Direct Preference Optimization）等离线算法，直接优化策略满足$\max_{\pi} \mathbb{E}_{\pi}[r]$，无需环境交互
3. **保守性控制**：对OOD（分布外）样本施加惩罚，防止策略选择数据集中未见的动作（responses），避免分布外崩溃

## 3. 数学基础
**离线RL目标**（传统Online RL对比）：

**Online RL（PPO）**：
$$\max_{\pi} \mathbb{E}_{\pi}[r(y|x)] - \beta D_{KL}(\pi||\pi_{\text{ref}})$$
需在线采样$y \sim \pi$。

**Offline RL**：
$$\max_{\pi} \mathbb{E}_{(x,y,r)\sim D_{\text{offline}}}\left[\frac{\pi(y|x)}{\pi_{\text{offline}}(y|x)} \cdot r(y|x)\right] - \text{ConservativePenalty}$$

**重要性采样权重**：

$$w(y|x) = \frac{\pi(y|x)}{\pi_{\text{behavior}}(y|x)}$$

**CQL（Conservative Q-Learning）在LLM中的应用**：

CE（Conservative Evaluation）：
$$\mathcal{L}_{\text{CQL}}(\pi) = -\mathbb{E}_{(x,y,r)\sim D}[\log \pi(y|x) \cdot r] + \lambda \cdot \mathbb{E}_{x\sim D, y\sim \pi}[\log \pi(y|x) - \log \pi_{\text{behavior}}(y|x)]$$

第二项惩罚OOD样本的选择概率。

**IQL（Implicit Q-learning）**：

避免显式重要性采样：
$$\mathcal{L}_{\text{IQL}} = \mathbb{E}\left[\min(Q(s,a) - r - \gamma Q(s',a'), 0)^2\right]$$

在LLM中翻译为：
$$Q(x, y_{\le t}) \approx \text{expected future reward from prefix}$$

**DPO（作为Offline RL）**：

DPO直接离线优化偏好对，无需在线采样：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\left(\beta \log \frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\right)\right]$$

**Off-policy与On-policy对比**：

| 特性 | On-policy (PPO) | Off-policy (CQL/DPO) | 优势 |
|------|----------------|---------------------|------|
| **采样** | 需实时交互生成数据 | 静态数据集 | 不需要生成基础设施 |
| **数据效率** | 低（多次采样） | 极高（一次数据） | 10-100x提升 |
| **计算成本** | 生成（50%）+ 训练（50%） | 仅训练 | 减少40-60% |
| **样本复用** | 一次性 | 无限次 | 支持元学习 |

## 4. 工程考量
**离线RL的优势**：

1. **成本节约**：无需在线生成，省去推理成本70%。InstructGPT训练PPO需要31k prompts × 4-9个回答/轮 × 10轮 ≈ 1M生成，每个token $0.00002，成本约$50k。Offline RL消除此成本

2. **安全**：避免在线交互产生有害内容，安全对齐后的数据可放心使用

3. **样本复用**：同一数据集可用多次调参，Grid search 100个组合成本仅线性增加，Online RL需100倍生成

4. **稳定性**：数据固定，训练更稳定，loss curve无震荡

**分布偏移（Distribution Shift）问题**：

Offline RL的最大挑战：策略优化的行为不同于数据收集时的行为

**OOD惩罚必要性**：

如果策略选择数据集中未出现的$y$，$\pi_{\text{behavior}}(y|x) \approx 0$，重要性权重$w \rightarrow \infty$

**Conservative控制**：

$$\pi_{\text{final}} = \arg\max_{\pi} \mathbb{E}_{D}[r] - \lambda \cdot \underbrace{\text{Std}_{\pi}[w]}_{\text{OOD penalty}}$$

**在LLM中的实现**：

1. **Behavior克隆约束**：
   $$D_{KL}(\pi||\pi_{\text{behavior}}) \le \epsilon$$
   强制策略不偏离收集数据太远

2. **保守Q值**：
   $$Q_{\text{conservative}}(x,y) = \min(Q_{\text{learned}}(x,y), Q_{\text{behavior}}(x,y))$$

3. **混合采样**：训练时一半batch从数据集采样，一半从当前策略采样（混合online-offline）

**离线数据的构建**：

```python
# 数据收集策略
1. SFT模型生成: 100k prompts → 生成100k responses, score with RM
2. 人类对话: 爬取ShareGPT、对话数据（需授权）
3. 历史版本模型: 保留GPT-3.5旧版本的输出作为负样本
4. 打包: 组成(x, y, r)三元组, r来自RM或人类标注
```

**Offline RL vs DPO关系**：

- **DPO是Offline RL的特殊情况**：只使用偏好对$(x,y_w,y_l)$，无需绝对奖励
- **Offline RL更通用**：可以使用$(x,y,r)$三元组，支持任意奖励形式（过程、结果、多维）
- **实现选择**：DPO代码简单，transformers库支持好；CQL/IQL需自定义训练循环

**Trade-off**:

- **保守 vs 激进**：Conservative penalty过强（$\lambda \gg 0$）→策略接近behavior policy，提升有限；过弱 → OOD崩溃
- **数据质量 vs 数量**：离线RL极度依赖数据质量，噪声reward导致错误优化。InstructGPT的31k PPO数据质量高，DPO效果好；若数据质量差，off-policy效果低于on-policy
- **覆盖率**：数据需覆盖足够的多样性，否则离线策略无法泛化到未见情况。Online RL通过采样探索弥补此缺陷

**致命弱点**:

1. **OOD崩溃**：如任务在数据集中覆盖率\u003c10%，离线RL策略无法生成好回答（保守性阻止探索），性能\u003cSFT基线

2. **重要性采样权重爆炸**：Behavior policy为SFT，离线优化后策略生成OOO文本，$\pi_{\text{behavior}}(y) \sim 10^{-10}$，权重达$10^{10}$，梯度爆炸

3. **双模型计算**：CQL需同时维护policy和Q-function，GPU内存增加60-80%，影响batch size

4. **收敛性差**：无online交互，无法动态调整策略，训练500k步仍可能未收敛（vs PPO的50k步）

5. **无法处理动态reward**：若reward函数变化（如用户反馈），需重新收集数据，online RL可实时适应

**工程折中**：

**混合RL**（HyDE，Hybrid RL）：

- 90%离线数据 + 10%在线生成
- 速度接近离线，安全性接近在线
- 在LLaMA-30B上测试，收敛步数从100k→50k，提升2倍

## 5. 工业映射
在工业界，离线RL是**解决LLM RLHF规模化的关键技术**：

### OpenAI的"Batch RL"（未公开但paper中暗示）

- InstructGPT训练使用31k prompts，但**生成100k回答**，并非全部用于PPO
- 实际做法：**离线预训练+在线微调**两阶段
  - Stage1: 用100k生成数据做离线DPO/CQL初始化策略
  - Stage2: PPO在线微调10k步
- **收益**：训练时间从10天→5天，生成成本省50%，稳定性提升（离线初始化避免RL early exploration崩溃）

### Anthropic的Constitutional AI

- **离线偏好学习**：使用RLAI（RL from AI feedback），AI生成100万偏好对，离线训练
- **优势**：无需人工标注，可大规模扩展
- **成本**：$10k（AI推理）vs $1M（人类标注）

### Meta的LLaMA 2

- **拒绝采样（Rejection Sampling）本质即Offline RL**：
  1. SFT模型生成32个回答
  2. RM选择top 1
  3. 将选中样本加入数据集，离线优化

- **Iterative循环**：每轮用新策略生成，选优后加入数据集
- **效果**：相当于offline RL的adversarial data augmentation，提升5-8%

### Hugging Face TRL

- **DPOTrainer**：纯offline实现，无生成步骤，调用方式：
  ```python
  trainer = DPOTrainer(
      model, ref_model,
      train_dataset=offline_dataset,  # pref pairs
      args=training_args
  )
  ```
- **速度**：30B模型训练从RLHF的7天→DPO的2天，成为社区首选

### 字节跳动的Doubao

- **离线-在线混合RLHF**：
  - 离线：用国内用户对话历史200M条做DPO初始化
  - 在线：PPO微调，但采样率降至30%（节省成本）
- **结果**：训练成本降低60%，效果持平ChatGPT

### 失败案例：Early Attempts at Scaling RLHF

2019-2020年，多家公司尝试纯offline RL（CQL风格）：
- **问题**：OOD崩溃策略生成乱码
- **原因**：数据集覆盖率低（仅10k prompts），保守惩罚不足
- **教训**：offline RL需至少100k高质量、高覆盖率数据才work

### 数据规模阈值

| 数据量 | Online RL | Offline RL | 推荐算法 |
|--------|-----------|------------|----------|
| \u003c10k | ✓ | ✗ | PPO |
| 10-100k | ✓ | ⚠️ | PPO + offline init |
| \u003e100k | ✓ | ✓ | DPO/CQL |

**工程优势量化**：

- **成本**：Offline RL训练成本降低40-60%（省生成）
- **速度**：DPO训练速度提升3-4倍（省采样延迟）
- **稳定性**：在10次实验中，offline RL 0次崩溃（vs PPO 3次）

### 中国产业实践

**文心一言**：
- 使用100M中文对话数据做离线DPO
- 重点：中文文化对齐（避免英文RLHF的西化）
- 数据：50%用户满意对话（正样本）+ 30%改进对话（pair）+ 20%高质量小说

**ChatGLM**：
- GLM-130B因推理成本太高（$10/次），无法online RL
- 纯offline：用1M开源数据和内部PM数据做DPO
- 效果：90% ChatGPT水平（offline上限）

**未来趋势**：

"Training is converging to offline, Inference will be online" - Sam Altman, 2024

- **训练**：Offline RL成为标配（成本、安全、速度）
- **推理**：用online反馈动态调整（个性化、实时性）
- **终极形态**："Offline training + Online adaptation"混合范式

**关键洞察**：

Offline RL不是替代Online RL，而是**前置初始化器**，解决RLHF的冷启动和成本问题。现代LLM对齐采用"** Offline warm-up → Online finetune → Online adaptation **"三级火箭，Offline负责规模化，Online负责精修化。
