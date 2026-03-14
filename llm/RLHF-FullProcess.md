# RLHF 全流程是怎样的？每个阶段做什么？需要多少数据？

## 1. 核心定性
本质上，**RLHF**是**三阶段流水线**：SFT阶段让模型学会遵循指令（30k数据），RM阶段训练奖励模型捕捉人类偏好（50-100k比较对），PPO阶段用强化学习优化策略提升人类满意度（10-50万prompt），实现从"讲人话"到"合人意"的能力跃迁。

## 2. 具体流程
1. **监督微调（SFT）**：在高质量指令数据（30k-50k）上用标准交叉熵微调预训练模型，学习基础指令遵循能力，耗时1天，学习率5e-6，3 epoch
2. **奖励建模（RM）**：收集5-10万组偏好比较数据（同一prompt的2个回答，人类标注哪个好），训练奖励模型$R_\phi(x,y)$预测偏好概率，用Bradley-Terry损失，epoch 1避免过拟合
3. **PPO强化学习**：从SFT模型初始化，用RM提供奖励信号，PPO算法优化策略，10-50万prompt交互50k-100k步，KL约束防崩溃，得分提升5-15%

## 3. 数学基础
**三阶段目标**：

**阶段1 - SFT**：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{i=1}^N \sum_{t=s_i}^{|y_i|} \log \pi_\theta(y_{i,t}|x_i, y_{i,<t})$$

数据规模：$N_{\text{SFT}} = 30,000-50,000$，人工标注，质量>80%

**阶段2 - RM（Bradley-Terry模型）**：

$$p(y_w \succ y_l | x) = \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))$$

损失函数：

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)\sim D}\left[\log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))\right]$$

数据规模：$N_{\text{RM}} = 50,000-100,000$对比较，需多标注员一致性>72%

**阶段3 - PPO（带KL惩罚）**：

$$\mathcal{L}_{\text{PPO}} = \hat{\mathbb{E}}\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right] - \beta D_{KL}(\pi_{\theta_{\text{ref}}}||\pi_\theta)$$

奖励信号：

$$r_t = \begin{cases}R_\phi(x, y) & t = |y| \\ 0 & t < |y|\end{cases}$$

数据规模：$N_{\text{PPO}} = 100,000-500,000$ prompts，自我交互生成

**标注一致性检验**：

Cohen's Kappa系数：

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

其中$p_o$是观察一致性，$p_e$是随机一致性。要求$\kappa \ge 0.65$（substantial agreement）。

**标注成本计算**：

标注员时薪$15，每对比较120秒：

$$\text{人力成本}_{\text{RM}} = 100,000 \times 2 \times \frac{120}{3600} \times 15 = \$100,000$$

SFT数据成本：每样本5分钟，30k样本，总成本\$40,000-60,000

## 4. 工程考量
**各阶段详情**：

| 阶段 | 数据量 | 耗时 | 计算成本 | 关键参数 |
|------|--------|------|----------|----------|
| SFT | 30k-50k | 1天 (8xA100) | $500 | LR=5e-6, 3 epochs, batch=128 |
| RM | 50k-100k pairs | 2-4小时 (1xA100) | $50 | LR=1e-5, 1 epoch, 避免过拟合 |
| PPO | 10-50万 prompts | 3-7天 (8xA100) | $3000 | KL=0.02-0.05, batch=512, steps=50k |

**数据质量要求**：
- **SFT数据**：多样化（50+任务类型），平均质量评分>80%，长度适中（<512 token）
- **RM数据**：同一prompt生成2个不同回答（temperature=0.7和1.0），标注员需通过测试（一致性>75%）
- **PPO数据**：仅需prompt，无需标注，需多样化（覆盖领域），避免与SFT数据分布差异过大（KL爆炸）

**关键工程细节**：
- **SFT**：embedding层学习率减半（2.5e-6）防止灾难性遗忘，混入5%预训练数据
- **RM**：使用SFT模型初始化，移除最后一层，添加线性头输出标量，hidden size保持1024
- **PPO**：使用SFT+RM，参考模型=SFT（固定），价值函数训练50次/PPO 1次，学习率cosine decay

**Trade-off**:
- **数据质量 vs 数量**：InstructGPT数据集60k中，高质量20k优于低质量100k，SFT后满意度从75%提升至78%（质量）vs 仅提升1%（数量）
- **RM准确度 vs 泛化**：RM训练1 epoch（测试准确率68%）优于3 epoch（训练准确率85%，测试降至62%，过拟合）
- **PPO收益递减**：PPO训练50k步后奖励提升5-10%，再训练50k步仅提升0.5-1%，需早停

**致命弱点**:
- **RM过拟合**：在域外数据上RM准确率下降15-20%，导致RL优化后模型对未见prompt生成质量差，Al能生索引建议用\'online RL\'迭代RM
- **样本标注崩溃**：对于主观问题（写诗），标注员一致性仅40-50%，导致RM训练噪声大，RL后模型性能不稳定（提升5%±5%），需定性任务剔除或改为多个正确输出
- **PPO分布崩溃**：KL系数不当导致策略熵$H(\pi)$从4.5降至2.8（聚集到单一模式），输出多样性降低60%，需entropy bonus或KL自适应

**数据泄露检测**：

RM测试集有5k样本，若在训练集上也评估：

$$\text{准确率}_{\text{train}} - \text{准确率}_{\text{test}} > 10\%$$

则过拟合，需减少训练。InstructGPT中该差距控制在3-5%。

## 5. 工业映射
在工业界，RLHF三阶段流水线是**OpenAI、Anthropic、DeepMind等顶级LLM公司的标准操作**：
- **InstructGPT（OpenAI，2022）**：首次公开三阶段数据量，SFT=13k（高质量），RM=33k对比对（来自30k unique prompts，每个prompt生成4-9个回答），PPO=31k prompts（从提示数据集中采样）。标注团队40人，标注一致性32%（不同人），同一标注员70%。RM准确率65%，PPO训练后人类偏好率从SFT的65%提升至84%
- **ChatGPT（2023）**：数据量扩大10倍，SFT=100k（人工+模型过滤），RM=500k对比对（标注员增加到几百人），PPO=1M prompts。引入在线RM迭代（每轮PPO后重新标注困难样本），使RM准确性从65%→75%，总成本$200M（标注$50M，训练$150M），证明数据规模是竞争力的核心
- **Claude（Anthropic）**：RLAI（RL from AI feedback）替代RLHF，大幅减少人工标注，SFT阶段使用15万数据（AI生成筛选），RM训练用AI标注200万对（成本降低90%），PPO基本一致。在Helpful、Harmless上超越ChatGPT，证明AI标注可扩展
- **LLaMA 2（Meta）**：开源RLHF榜样，SFT=27k（高质量），RM=1.4M对比对（通过模型自己采样生成30个回答后人工排序），PPO=1M prompts。使用拒绝采样（Rejection Sampling）在PPO后进一步提升，证明大规模+过滤的有效性
- **DeepSpeed-RLHF（微软）**：开源RLHF系统，实现三阶段全pipeline，支持7B-530B模型，在256 GPU上训练OPT-30B仅需9小时（SFT 1小时，RM 0.5小时，PPO 7.5小时），证明工程优化可降低成本10倍

**成本演变**：
- **2022（InstructGPT）**：总成本$500k（小规模）
- **2023（ChatGPT）**：总成本$200M（大规模工业化）
- **2024趋势**：RLAIF + 合成数据使成本降至$2-5M（AI标注替代），质量保持95%

**数据规模与性能关系**：

$$\text{PrefRate} = 0.65 + 0.0005 \cdot \log(N_{\text{RM}}) + 0.001 \cdot \log(N_{\text{PPO}})$$

当$N_{\text{RM}}=100k$、$N_{\text{PPO}}=500k$时，PrefRate=84%（接近ChatGPT）

**中国实践**：
- **文心一言**：三阶段全部国产化，SFT=50万中文指令，RM=200万对比对（来自用户反馈），PPO=千万级交互，总数据量是英文模型的3-5倍（弥补中文数据稀缺）
- **ChatGLM**：SFT=100k高质量中文数据（人工+AI校验），RM=300k，PPO=50万，使用奖励模型融合（融合规则型、偏好型、安全型），总分=$\alpha R_{\text{help}} + \beta R_{\text{safe}} - \gamma R_{\text{rej}}$，中文任务效果好，英文任务需翻译数据

**趋势**：三阶段边界在模糊，DPO直接跳过显式RM和PPO，将RL问题转为监督学习，但数据需求不变（需50万+偏好对），RLHF仍是高质量对齐的黄金标准。
