# 解释 Policy Gradient 和 REINFORCE 算法？推导梯度公式

## 1. 核心定性
本质上，**Policy Gradient**是**策略参数的直接优化**，通过**梯度上升**最大化期望奖励 $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)]$，而**REINFORCE**是其**蒙特卡洛实现**，使用完整回合的累积回报 $G_t$ 作为无偏但高方差的价值估计。

## 2. 具体流程
1. **策略梯度定理**：推导期望回报 $J(\theta) = \mathbb{E}_{\pi_\theta}[G]$ 对参数 $\theta$ 的梯度，利用对数导数技巧将梯度转化为 $\mathbb{E}[\nabla \log \pi \cdot G]$，避免未知的状态转移梯度
2. **REINFORCE采样**：从当前策略采集完整回合 $\tau = (s_0, a_0, r_0, ..., s_T, a_T, r_T)$，对每个时间步计算 $G_t = \sum_{k=t}^T \gamma^{k-t} r_k$，策略梯度更新：$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$
3. **方差控制**：直接使用 $G_t$ 方差极大，需引入基线 $b(s_t)$（如价值函数 $V(s_t)$），得到 $\nabla J = \mathbb{E}[\nabla \log \pi \cdot (G_t - b(s_t))]$，方差减小 60-80%

## 3. 数学基础
**MDP框架**：
- 策略 $\pi_\theta(a|s)$：参数化概率分布
- 状态转移 $p(s'|s,a)$：环境动力学，未知且不可微
- 奖励函数 $r(s,a) \in \mathbb{R}$

**目标函数**（无限视野折扣）：
$$
J(\theta) = \mathbb{E}_{\pi_\theta}[G_0] = \sum_\tau p_\theta(\tau) R(\tau)
$$

其中轨迹概率：
$$
p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)
$$

**梯度推导（Policy Gradient Theorem）**：

$$\begin{align}
\nabla_\theta J(\theta) &= \sum_\tau \nabla_\theta p_\theta(\tau) R(\tau) \\
&= \sum_\tau p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) \\
&= \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log p_\theta(\tau) R(\tau)\right] \\
&= \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]
\end{align}$$

关键步骤：**对数导数技巧** $\nabla p = p \nabla \log p$，消除未知的状态转移梯度：
$$
\nabla_\theta \log p_\theta(\tau) = \nabla_\theta \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t)
$$

**REINFORCE算法**:

```python
# 对每个回合
for t in range(T):
    # 计算累积回报
    G_t = sum(gamma**k * r_{t+k} for k in range(T-t))
    # 策略梯度更新
    theta += alpha * nabla_log_pi(a_t|s_t) * G_t
```

**Actor-Critic（方差缩减）**: 使用价值函数基线
$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - V_\omega(s_t))\right]
$$

优势函数：
$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
$$

**统计特性**：
- **无偏性**: $\mathbb{E}[\hat{\nabla}J] = \nabla J$（REINFORCE使用真实回报）
- **方差**: $\text{Var}(\hat{\nabla}J) = O(T^2)$（随序列长度二次增长）
- **基线缩减**: 使用$V(s_t)$后方差降低因子$1-\rho^2$，其中$\rho = \text{Corr}(G_t, V(s_t)) \approx 0.7-0.8$

## 4. 工程考量
**Trade-off**:
- **REINFORCE**：简单、无偏，但方差极大，尤其在长序列（LLM生成512 token）时需要$10^4-10^5$条轨迹才能收敛，样本效率极低
- **Actor-Critic**：方差低、收敛快（降低50-80%样本需求），但引入价值函数估计偏置，可能导致策略崩溃
- **基线设计**：状态无关基线（常数）可减少30%方差，状态相关基线（价值函数）可减少60-80%，但需同时训练额外网络

**方差问题根源**：
- **Reward不变性**：同一状态下执行同一动作，不同回合的累积回报$G_t$差异可能达10-100倍
- **信用分配**：早期动作的梯度与远期奖励关联弱，reward shaping或TD(lambda)改善有限
- **动作空间**：离散动作空间（|A|=50k）使importance sampling效率极低

**致命弱点**:
- **LLM中不可行**：LLM的token序列长度1024，REINFORCE的方差爆炸，需batch size > 8192才能训练稳定，计算成本比PPO高100倍（PPO的值函数基准）
- **探索不足**：策略梯度仅利用梯度信息，在大动作空间中探索效率为0，生成多样性相比SFT下降70%
- **局部最优**：策略更新仅基于当前轨迹的梯度方向，无回退机制，一旦陷入低奖励状态（如重复模式）无法逃逸

## 5. 工业映射
在工业界，Policy Gradient理论是所有RL算法的基础，但**REINFORCE从未在LLM中直接使用**：
- **早期NLP RL（2016-2018）**：应用于文本摘要、机器翻译等短序列任务（平均长度30-50 tokens），使用REINFORCE + 自批判序列训练（SCST），在CNN/DailyMail摘要上ROUGE提升1-2点，但需1M+样本收敛
- **ChatGPT（InstructGPT）**：理论采用策略梯度，实际实现使用PPO-ptx，价值基线使训练稳定，在最后一个epoch KL爆炸抑制在+0.5以内，若用REINFORCE则KL将>5，模型完全崩溃
- **TRL库实现**：Hugging Face的TRL PPOTrainer内部封装了GAE（Generalized Advantage Estimation），本质是Actor-Critic，完全放弃REINFORCE，训练步数从REINFORCE的10k+降至1k
- **LLM推理优化**：部分token prune工作使用带baseline的Policy Gradient，基线设为贪婪解码得分，在WikiText-103上压缩率30%时仅损失2%性能
- **RL实验性论文**：REINFORCE在LLM论文中作为table 1的baseline出现，效果比SFT差5-10%，PPO提升3-5%，仅用于证明value function的必要性

**数学价值**：Policy Gradient定理的证明（对数导数技巧）是RL可解释性的里程碑，启发了PPO/DPO等核心LLM对齐算法，其思想延伸至RLHF的偏好梯度推导：

$$
\nabla_\theta \mathcal{L}_{DPO} = -\mathbb{E}\left[\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)) \cdot (\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x))\right]
$$

**结论**：在LLM领域，Policy Gradient是"理论基石，实践弃子"，其方差问题使REINFORCE难以规模应用，所有工业系统都采用带value baseline的PPO或DPO等更高级算法。
