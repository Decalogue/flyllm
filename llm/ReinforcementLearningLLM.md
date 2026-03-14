# RL 用在 LLM 中的前提是什么？policy 是什么？action space 是什么？reward 是什么？

## 1. 核心定性
本质上，RL在LLM中的应用前提是**将文本生成建模为序列决策过程**，其中**Policy**是参数化的语言模型 $\pi_\theta(a_t|s_t)$，**Action Space**是词汇表 $V$（大小50k-100k），**Reward**是回答质量的延迟奖励函数 $r(y|x)$，通过最大化累积奖励实现指令对齐。

## 2. 具体流程
1. **序列决策建模**：LLM生成视为马尔可夫过程，状态 $s_t = [x, y_{<t}]$（指令+已生成token），动作 $a_t = y_t$（下一token），策略 $\pi_\theta$输出动作概率分布
2. **Reward设计**：用奖励模型$R_\phi(x,y)$评估完整回答质量（Helpful, Harmless, Honest），输出标量分数，或在token级别使用稀疏奖励（仅最后一个token有奖励）
3. **策略优化**：使用PPO/GRPO等算法更新策略，目标 $\max_\theta \mathbb{E}_{\pi_\theta}[r(y|x)] - \beta \cdot \text{KL}(\pi_\theta||\pi_{\text{ref}})$，平衡奖励与参考策略的偏离

## 3. 数学基础
**马尔可夫决策过程（MDP）定义**：
- **策略（Policy）**：
  $$
  \pi_\theta(a_t|s_t) = \text{Softmax}(W_o \cdot \text{Transformer}_\theta(s_t))
  $$
  其中$W_o \in \mathbb{R}^{|V| \times d}$是LM的输出投影矩阵，输出词表中每个token的概率

- **动作空间（Action Space）**：
  $$
  \mathcal{A} = V \cup \{\text{EOS}\}, \quad |\mathcal{A}| = 50,000-100,000
  $$
  Vocabulary包含词/子词token，加上结束符EOS

- **状态空间（State Space）**：
  $$
  s_t = [x_{\text{system}}, x_{\text{user}}, y_1, y_2, ..., y_{t-1}]
  $$
  状态是已生成序列+原始指令，最大长度$H$（如$H=2048$）

- **Reward函数**：
  $$
  R(x,y) = \alpha \cdot R_{\text{help}}(x,y) + \beta \cdot R_{\text{harm}}(x,y) + \gamma \cdot R_{\text{honest}}(x,y)
  $$

**PPO目标**：
$$
\mathcal{L}_{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t \left[\min(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\hat{A}_t, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon))\right] - \beta D_{KL}(\pi_\theta||\pi_{\text{ref}})$$

其中优势函数：
$$
\hat{A}_t = \hat{R}_t - V_\theta(s_t) = r(y|x) - V_\theta(s_t)
$$

在LLM中，奖励仅在最后一个token获得：**稀疏奖励问题**：
$$
R_t = \begin{cases}r(y|x) & t = |y| \\ 0 & t < |y|\end{cases}
$$

**Token-level奖励扩展**：
$$
\mathcal{L}_{\text{token_PPO}} = -\sum_{t=1}^{|y|} w_t \cdot \log \pi_\theta(y_t|s_t), \quad w_t = r(y|x) \cdot \lambda^{|y|-t}
$$

## 4. 工程考量
**RL应用前提**：
1. **SFT基线**：策略必须先在指令数据上监督微调，避免从零RL（收敛极慢，探索空间大50k动作）
2. **奖励模型**：需预训练RM，否则奖励信号噪声大（标注一致性<70%），RL训练不稳定
3. **参考策略**：必须与训练策略同分布（初始化自SFT），KL散度系数$\beta=0.01-0.1$防止过度优化

**Trade-off**:
- **序列长度vs信用分配**：生成100token时，早期token的梯度与奖励关联弱（信用分配问题），方差大，需大量采样（batch size 512-1024）稳定训练，计算成本比SFT高100倍
- **稀疏奖励vs方差**：仅最后一个token有奖励导致方差极大，使用GAE（Generalized Advantage Estimation）缓解但无法根除，优势函数估计误差>30%
- **探索vs利用**：策略需探索新token（生成多样回答），但自然语言动作空间大（|V|=50k），有效探索率极低（<0.1%），易陷入局部最优

**致命弱点**:
- **奖励黑客（Reward Hacking）**：RM过拟合特定模式（如关键词堆砌），RL策略学会欺骗奖励函数而非真正对齐，在GPT-4训练中曾出现"过度道歉"模式（reward +0.3 but human preference -0.5）
- **模式崩溃（Mode Collapse）**：优化后策略熵$H(\pi)$从4.5降至3.2，输出多样性下降40%，生成文本重复率上升3倍，需entropy bonus正则化
- **训练不稳定**：PPO的clip范围$\epsilon=0.2$对LLM动作空间过大，导致策略崩溃（KL爆炸到>10），实验中必须减小至$\epsilon=0.05-0.1$或使用自适应KL控制

## 5. 工业映射
在工业界，RL是**将LLM从"能回答"提升到"答得好"的关键技术栈**：
- **InstructGPT（OpenAI）**：策略是GPT-3 175B，动作空间50k（BPE），奖励模型基于人工6万偏好对训练，使用PPO-ptx（混合预训练数据），KL系数0.02，训练后人类满意度从SFT的75%提升到82%，但RM over-optimization导致负面案例增加15%，需在多个checkpoint中手动筛选
- **ChatGPT**：策略从强化学习转向RLHF+提示工程，动作空间不变，但reward增加"拒绝回答危险问题"的-1惩罚，在RLHF基础上增加安全RM，使拒绝率控制在5%（过低影响体验，过高限制能力），是工业级安全对齐的标杆
- **Claude（Constitutional AI）**：提出AI反馈强化学习（RLAIF），reward由AI评估（非人类），通过"constitution"指导AI判断，动作空间100k（SentencePiece），在70B模型上达到人类标注RM 95%效果，成本降低80%，解决RLHF扩展性差的问题
- **RLHF开源实现（TRL）**：Hugging Face的TRL库封装PPOTrainer，策略为任何decoder-only模型，action space自动从tokenizer获取，reward需用户提供（通常调用API或本地RM），在LLaMA-7B上复现，但token-level实现导致内存占用比SFT高3-4倍，需gradient checkpointing
- **GRPO（Group Relative Policy Optimization，中国创新）**：解决PPO需要价值函数的问题，用组内相对奖励替代baseline，在相同batch size下内存减少40%，训练速度提升1.5倍，是工业级LLM-RL的重要突破

**趋势**：现代LLM对齐采用"**SFT冷启动 → RLHF微调 → 人类反馈迭代**"三阶段，RL作为能力天花板的关键拉升手段，但受限于RM质量、训练稳定性和成本，目前主要在千亿级模型中使用，中小模型仍依赖SFT+数据工程。未来方向是RLAIF克服标注瓶颈和DAPO提升训练效率。
