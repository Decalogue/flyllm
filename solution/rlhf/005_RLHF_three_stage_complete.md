# 005: RLHF 三阶段完整流程

## 核心定性
RLHF（Reinforcement Learning from Human Feedback）本质上是一个**三阶段流水线**，通过 SFT（行为克隆）→ Reward Model（偏好量化）→ PPO（策略优化）将 LLM 从"通才"转化为"遵循人类指令且符合价值观的专家"，核心难点在于**用稀疏、高噪声的偏好信号稳定优化高维策略空间**。

## 三阶段流程对比

### **阶段一：SFT（Supervised Fine-Tuning）**

**目标**: 最大化人类标注数据的似然
$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim D_{human}} \left[ \sum_{t=1}^{|y|} \log P_\theta(y_t | x, y_{<t}) \right]$$

**输入**: $(prompt, response)$ 对，例如：
```python
dataset = [
    ("How to make a bomb?", "I'm sorry, I cannot provide instructions..."),
    ("Write a Python function to sort list.", "def sort_list(arr): return sorted(arr)"),
]
```

**数据规模**: 10K-100K 条（高质量）
**训练配置**:
- lr=1e-5 - 2e-5
- batch=32-128
- epochs=1-3（防止过拟合）
**输出**: SFT 模型（能遵循指令，但无偏好区分能力）

**致命弱点**:
- 无法区分"好回答
"、"更好回答"和"最好回答"
- 容易过拟合标注数据

### **阶段二：Reward Model（偏好模型）**

**目标：学习人类偏好排序**

**输入：偏好对** $(x, y_w, y_l)$，其中 $y_w \succ y_l$（$y_w$ 优于 $y_l$）

**Loss（Pairwise Ranking）**:
$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]$$

**数据规模**：50K-100K 偏好对（可部分由 SFT 模型生成）
**模型**：在 SFT 模型基础上 + 价值头（scalar head）
**训练配置**：
- lr=1e-5
- batch=64
- epochs=1

**核心洞察**：
- 奖励模型只需捕捉**相对偏好**，无需绝对评分
- 奖励分布经过标准化（mean=0, std=1）
- **KL 惩罚准备**：$r_\phi$ 需平滑，避免过度自信

### **阶段三：PPO（策略优化）**

**目标**：最大化奖励 + 保持与 SFT 模型的接近度

**Loss（完整版）**:
$$\mathcal{L}_{PPO} = \underbrace{\mathbb{E} \left[ \min(r_t(\theta) \hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t) \right]}_{\text{策略损失}} - \underbrace{\beta \mathbb{E} [D_{KL}[\pi_{\theta_{old}} \| \pi_\theta]]}_{\text{KL 约束}} $$

**实现细节**：
```python
# 1. 旧策略生成数据（不更新）
old_policy = SFTModel()
old_policy.eval()

# 2. 采样轨迹
for prompt in prompts:
    response = old_policy.generate(prompt, max_length=512)
    reward = reward_model(prompt, response)
    log_prob = old_policy.log_prob(response)

# 3. 优势函数（GAE）
advantages = compute_gae(rewards, values, gamma=0.99, lam=0.95)

# 4. 新策略更新（多次 epoch）
for epoch in range(4):
    for batch in dataloader:
        # PPO Clip Loss
        loss = ppo_clip_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2)

        # KL 惩罚（监控）
        kl_div = kl_divergence(old_policy, new_policy)
        if kl_div > 0.2:  # 提前终止
            break

        optimizer.step()
```

**关键超参**：
- `clip_eps=0.2`：允许策略 20% 偏离
- `β=0.01`：KL 惩罚系数
- `lr=1e-6`：极低学习率（PPO 敏感）
- `batch=256`：大批量稳定训练

**训练曲线特征**：
- 奖励单调上升（需平滑）
- KL 散度控制在 0.01-0.1
- 策略熵缓慢下降（收敛标志）

## 工业映射

### OpenAI InstructGPT（原始 RLHF）

```yaml
SFT:
  data: 13K prompt-response pairs (human written)
  model: GPT-3 1.3B (base)
  epochs: 16

Reward Model:
  data: 33K prompt-comparison pairs (rm=1.3B)
  epochs: 1
  accuracy: 73% (vs 50% random)

PPO:
  data: 31K prompts (no new human)
  steps: 256K
  batch: 256
  clip_eps: 0.2
  beta: 0.01

Result: InstructGPT 1.3B > GPT-3 175B (in alignment)
```

### LLaMA-2-Chat 训练管线

```yaml
SFT:
  data: 27K high-quality examples (safety + helpfulness)
  epochs: 2
  lr: 2e-5

RM:
  data: 1M+ preference pairs (Meta AI + human annotators)
  model: LLaMA-2-7B (parallel head)
  epochs: 1

PPO:
  iterations: 4
  mini_batch: 64
  gradient_accumulation: 16
  effective_batch: 1024
  kl_reward: 0.01

Result: LLaMA-2-7B-Chat ≈ GPT-3.5 turbo (Chatbot Arena 1040 分)
```

### 字节跳动豆包（DPO 简化版）

```python
# 跳过显式 RM，直接用偏好数据优化
class DPOTrainer:
    def dpo_loss(self, prompt, chosen, rejected, beta=0.1):
        # π_θ(y_w|x) / π_θ(y_l|x)
        logp_chosen = policy.log_prob(chosen, prompt)
        logp_rejected = policy.log_prob(rejected, prompt)

        # π_ref(y|x)
        with torch.no_grad():
            logp_chosen_ref = ref_policy.log_prob(chosen, prompt)
            logp_rejected_ref = ref_policy.log_prob(rejected, prompt)

        # Loss
        loss = -F.logsigmoid(beta * (logp_chosen - logp_rejected) - beta * (logp_chosen_ref - logp_rejected_ref))
        return loss.mean()

# 优势: 训练速度快 2.5x，内存节省 50%
# 劣势: 对偏好数据质量敏感，需要更多样本
```

## 面试高频追问

**Q1: RLHF 为什么需要 SFT 预热？**

A: SFT 提供**策略初始化**：
- 冷启动 PPO 会探索空间过大，收敛慢且不稳定
- SFT 模型已具备基础指令遵循能力，PPO 只需微调
- 数据效率: SFT 用 10K 样本，PPO 用 100K+ 样本（采样效率低）

**Q2: Reward Model 的 scaling law？**

A:
- **数据**: 1M 偏好对 vs 1T 预训练 token（占比 0.1%）
- **模型**: 通常 RM 大小 = Policy 大小的 1/10 - 1/1（相同或更小）
- **能力**: RM 在分布内准确率可达 85%，分布外降至 60%
- **优化**: 迭代式更新 RM（每 100K PPO steps）

**Q3: KL 惩罚的作用？β 如何调？**

A: β 控制**对齐强度 vs 生成多样性**:
- β=0.01: 较弱约束，策略可偏离 SFT 20-30%（用于创造性任务）
- β=0.1: 中等约束，偏离 10%（通用场景）
- β=0.5: 强约束，偏离 5%（安全敏感场景，如医疗）

监控 KL 散度：
```python
kl = F.kl_div(new_log_probs, old_log_probs, reduction='batchmean')
if kl > 0.2: early_stop()  # 策略偏离过大
```

**Q4: RLHF vs SFT 的数据效率对比？**

A:
- SFT: 10K-100K 样本直达收敛（监督信号强）
- RLHF: 需要 100K-1M 偏好对 + 1M+ PPO 采样（稀疏信号）
- **成本**: RLHF 计算成本是 SFT 的 5-10x
- **质量**: RLHF 对齐精度更高（捕捉细粒度偏好）

---

**难度评级**: ⭐⭐⭐  
**出现频率**: 98%（所有 RLHF/后训练岗位）  
**掌握要求**: 三阶段流程 + 核心公式 + PPO 调参经验
