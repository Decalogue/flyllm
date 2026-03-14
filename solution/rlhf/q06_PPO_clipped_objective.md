# 006: PPO Clipped Objective 手写与 β 调优

## 核心定性
PPO（Proximal Policy Optimization）的 Clipped Objective 本质上是一个**在策略更新幅度与价值增益之间做鲁棒权衡**的 surrogate loss，通过**硬截断（Clip）机制**将策略比率 $r_t(θ) = π_θ(a_t|s_t)/π_{θ_{old}}(a_t|s_t)$ 限制在 $[1-ε, 1+ε]$ 区间内，防止 RLHF 训练中的**策略崩溃（Policy Collapse）**和**奖励黑客（Reward Hacking）**。

## Loss 函数手写

### PPO-Clip 核心公式

$$\mathcal{L}^{CLIP}(θ) = \mathbb{E}_t \left[ \min \left( r_t(θ) \hat{A}_t, \ \operatorname{clip}(r_t(θ), 1-ε, 1+ε) \hat{A}_t \right) \right]$$

其中：
- $r_t(θ) = \frac{π_θ(a_t|s_t)}{π_{θ_{old}}(a_t|s_t)}$：新策略 vs 旧策略的概率比
- $\hat{A}_t$：优势函数估计（GAE 或蒙特卡洛）
- $ε$：截断超参（通常 0.1-0.3）
- $\operatorname{clip}(x, a, b) = \min(\max(x, a), b)$

### PyTorch 实现

```python
class PPOClipLoss(nn.Module):
    def __init__(self, clip_eps=0.2):
        super().__init__()
        self.clip_eps = clip_eps

    def forward(self, log_probs, old_log_probs, advantages, mask=None):
        """
        Args:
            log_probs: 新策略的对数概率 [batch_size, seq_len]
            old_log_probs: 旧策略的对数概率（detach）
            advantages: GAE 估计的优势函数 [batch_size, seq_len]
            mask: 有效 token 掩码
        """
        # 计算概率比 r_t(θ)
        ratio = torch.exp(log_probs - old_log_probs)  # [batch, seq]

        # 无限制的损失
        surr1 = ratio * advantages

        # 截断后的损失
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surr2 = clipped_ratio * advantages

        # 取最小（悲观选择，防止过大更新）
        policy_loss = -torch.min(surr1, surr2)

        if mask is not None:
            policy_loss = policy_loss * mask

        return policy_loss.mean()

# 使用示例
ppo_loss = PPOClipLoss(clip_eps=0.2)

# 前向
new_log_probs = model(new_states).log_softmax(dim=-1)
old_log_probs = old_model(old_states).log_softmax(dim=-1).detach()

# GAE 计算
advantages = compute_gae(rewards, values, gamma=0.99, lam=0.95)

# Loss
loss = ppo_loss(new_log_probs, old_log_probs, advantages)
loss.backward()
```

### 完整训练循环

```python
def ppo_update(policy_model, value_model, old_policy, optimizer, dataloader, epochs=4):
    """
    PPO 多次 epoch 更新（重用数据）
    """
    for epoch in range(epochs):
        for batch in dataloader:
            states = batch['states']
            actions = batch['actions']
            old_log_probs = batch['log_probs']
            rewards = batch['rewards']
            dones = batch['dones']

            # 1. 计算新策略概率
            new_logits = policy_model(states)
            new_log_probs = F.log_softmax(new_logits, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            # 2. 价值估计
            values = value_model(states).squeeze(-1)

            # 3. GAE 优势函数（外部计算）
            advantages = compute_gae(rewards, values, gamma=0.99, lam=0.95)
            returns = advantages + values.detach()

            # 4. PPO Clip Loss
            ppo_loss = PPOClipLoss(clip_eps=0.2)
            policy_loss = ppo_loss(new_log_probs, old_log_probs, advantages)

            # 5. 价值 Loss（MSE）
            value_loss = F.mse_loss(values, returns)

            # 6. 总 Loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=0.5)
            optimizer.step()
