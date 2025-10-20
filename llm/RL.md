# 🎓 大模型强化学习方法对比

> 系统梳理8种主流LLM强化学习算法
DPO | PPO | GRPO | DAPO | GTPO | GRPO-S | GSPO | Tree-GRPO

## 📑 快速导航

| 方法 | 核心创新 | 性能特点 | 推荐场景 |
|------|---------|---------|---------|
| [**DPO**](#1-dpo-direct-preference-optimization) | 无需奖励模型 | 🚀 极快，显存极低 | 偏好对齐 |
| [**PPO**](#2-ppo-proximal-policy-optimization) | 裁剪策略梯度 | ✅ 通用稳定 | 经典RL基线 |
| [**GRPO**](#3-grpo-group-relative-policy-optimization) | 去Critic群体对比 | 📉 显存-40% | 资源受限 |
| [**DAPO**](#4-dapo-decoupled-clip-and-dynamic-sampling-policy-optimization) | Token级+Clip-Higher | 🔥 AIME +66% | 长链推理 |
| [**GTPO**](#5-gtpo-group-token-policy-optimization) | Token级熵加权 | 🎯 最精细信号 | 极致性能 |
| [**GRPO-S**](#6-grpo-s-sequence-level-group-relative-policy-optimization) | 序列级熵加权 | ⚡ 效率折中 | 工程实用 |
| [**GSPO**](#7-gspo-group-sentence-policy-optimization) | 整句级优化 | 🎯 MoE友好 | MoE模型 |
| [**Tree-GRPO**](#8-tree-grpo-tree-guided-reinforcement-policy-optimization) | Step级树搜索 | 🌲 语义完整 | Agent任务 |

**核心演进：** <span style="color:#DC2F02">粗粒度（GRPO）</span> → <span style="color:#F77F00">Token级（DAPO/GTPO）</span> → <span style="color:#06A77D">整句级（GSPO/GRPO-S）</span>

---

## 1. DPO (Direct Preference Optimization)
### 直接偏好优化

**一句话回答**  
<span style="color:#2E86AB">**无需奖励模型**</span>，直接利用人类偏好数据（如"回答A优于B"）调整策略的<span style="color:#A23B72">**监督学习方法**</span>。

**核心机制**
- **偏好对比:** 最大化优选样本概率，降低劣选样本概率
- **隐式正则:** 通过参考模型约束策略偏离幅度

**✅ 优势**
训练简单高效（单阶段），显存占用低。

**❌ 劣势**
依赖高质量偏好数据，复杂推理任务表现弱。

**适用场景**
偏好数据明确的小规模任务（如对话优化、文本润色）。

---

## 2. PPO (Proximal Policy Optimization)
### 近端策略优化

**一句话回答**  
通过<span style="color:#E63946">**裁剪策略更新幅度**</span>（限制新旧策略差异）实现稳定训练的经典强化学习算法，需同时训练策略模型（Actor）和<span style="color:#F77F00">**价值模型（Critic）**</span>。

**核心机制**
- **约束优化:** 用KL散度惩罚或概率比裁剪防止策略突变
- **优势估计:** 依赖价值模型（Critic）计算长期收益

**✅ 优势**
通用性强，适合复杂动作空间（如游戏AI、机器人控制）。

**❌ 劣势**
显存占用高（需额外Critic模型），训练速度慢。

**适用场景**
需精确价值估计的任务（如对话系统、多轮交互）。

---

## 3. GRPO (Group Relative Policy Optimization)
### 群体相对策略优化

**一句话回答**  
<span style="color:#06A77D">**DeepSeek提出**</span>的去价值模型算法，通过<span style="color:#E63946">**组内输出对比**</span>（如5个答案排序）估计相对优势。

**核心机制**
- **群体归一化:** 以组内平均奖励为基线计算优势（替代Critic的价值估计）
- **相对优势估计:** 标准化组内奖励差异

**✅ 优势**
<span style="color:#06A77D">**显存降低40%+**</span>，数学/代码推理任务表现突出。

**❌ 劣势**
需生成多个候选答案，初期训练依赖多阶段优化。

**适用场景**
资源受限的大模型训练，高难度推理任务（如数学解题）。

**⚠️ 核心局限**  
<span style="color:#DC2F02">**粗粒度信用分配**</span>——整个序列共享统一优势值，无法精确归因到关键token/步骤。

---

## 4. DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
### 动态采样与解耦裁剪策略优化

**一句话回答**  
<span style="color:#06A77D">**ByteDance提出**</span>的大规模LLM强化学习系统，通过<span style="color:#E63946">**Token级损失、解耦裁剪和动态采样**</span>解决长推理链训练难题。

**核心机制**
- **Token级策略梯度:** 长序列按token贡献计算loss，避免长回答被稀释
- **Clip-Higher策略:** 解耦ε_low/ε_high，提升低概率token探索能力，防止熵崩溃
- **动态采样:** 过滤acc=0或1的样本，保留有效梯度样本
- **软超长惩罚:** 长度感知的奖励计算，减少截断样本噪声
- **移除KL散度:** 针对长CoT推理，允许策略自然偏离初始分布

**✅ 优势**
- <span style="color:#06A77D">**AIME性能从GRPO的30提升至50（+66%↑）**</span>，仅用<span style="color:#F77F00">**50%训练步数**</span>
- 有效控制生成熵和回答长度（防止冗长/重复）
- 适配可验证任务的规则奖励（数学、代码等）

**❌ 劣势**
需动态调整采样预算，超参数调优复杂（ε_low, ε_high, α等）。

**适用场景**
长链推理任务（数学证明、代码生成、多步推理），需要过程级优化的场景。

**与GRPO对比**
| 维度 | GRPO | DAPO |
|------|------|------|
| **Loss粒度** | 样本级（序列平均） | Token级（按token贡献） |
| **裁剪策略** | 对称裁剪(1±ε) | 非对称裁剪(1-ε_low, 1+ε_high) |
| **KL约束** | 含KL惩罚项 | 显式移除 |
| **采样策略** | 固定采样 | 动态过滤无效样本 |
| **长回答处理** | 易产生冗长/重复 | 软惩罚+长度控制 |

---

## 5. GTPO (Group Token Policy Optimization)
### Token级群体策略优化

**一句话回答**  
<span style="color:#E63946">**动态熵加权框架（DEW）**</span>中粒度最细的实现，在token级引入<span style="color:#F77F00">**熵加权信用分配机制**</span>。

**核心机制**
- **熵加权奖励塑造:** 对成功序列，<span style="color:#06A77D">**高熵token获得更高奖励**</span>（鼓励有价值探索）
  - 公式：奖励 = 原始奖励 + 熵加权项 × 动态缩放
  - $H_{i,t}$：token生成熵（不确定性度量）
- **惩罚自信型错误:** 对失败序列，<span style="color:#DC2F02">**高置信度错误受更重惩罚**</span>（基于熵倒数）
- **归一化优势:** 在批次所有token上归一化，确保尺度一致

**✅ 优势**
- <span style="color:#06A77D">**最精细的过程信号**</span>，真正实现"过程导向推理改进"
- 在多个推理基准上持续优于DAPO/GRPO
- Token级估计有效降低梯度方差

**❌ 劣势**
计算开销最大（需逐token计算熵和优势）。

**适用场景**
需要极致细粒度反馈的复杂推理任务，长链CoT优化。

---

## 6. GRPO-S (Sequence-Level Group Relative Policy Optimization)
### 序列级群体策略优化

**一句话回答**  
GTPO的<span style="color:#F77F00">**序列级变体**</span>，在序列级应用熵加权原则，平衡<span style="color:#06A77D">**计算效率与信号粒度**</span>。

**核心机制**
- **序列级熵加权:** 基于序列平均token熵调整整体奖励
  - <span style="color:#06A77D">成功序列</span>：高平均熵获得更高奖励（奖励探索）
  - <span style="color:#DC2F02">失败序列</span>：低平均熵受更重惩罚（惩罚过度自信）
- **群体归一化:** 在G个序列上归一化优势函数

**✅ 优势**
- 相比GTPO<span style="color:#06A77D">**计算量显著降低**</span>
- 保留熵加权核心优势，性能与GTPO相当（某些任务更优）
- 更适合结果导向任务

**❌ 劣势**
粒度粗于GTPO，无法精确到token级归因。

**适用场景**
资源受限但需要比GRPO更细信号的场景，结果导向的推理任务。

---

## 7. GSPO (Group Sentence Policy Optimization)
### 整句级群体策略优化

**一句话回答**  
从<span style="color:#DC2F02">**Token级转向整句级**</span>优化的方法，专门解决<span style="color:#F77F00">**MoE模型收敛难**</span>与token级噪声问题。

**核心机制**
- **整句级策略梯度:** 以完整句子为单位计算loss，避免token级噪声干扰
- **MoE路由稳定性:** 解决新旧策略下expert组合变化导致的importance ratio失真
- **长度归一化:** $s_i(\theta) = (\pi_\theta(y_i|x)/\pi_{old}(y_i|x))^{1/|y_i|}$ 减少方差
- **序列级裁剪:** 对整个句子而非单个token进行裁剪，与序列级奖励机制对齐

**✅ 优势**
- 显著提升<span style="color:#06A77D">**MoE模型训练稳定性**</span>（解决expert路由震荡）
- 避免importance ratio失真导致的梯度崩溃
- 计算效率高于GTPO（无需逐token计算）
- 无需复杂的"Routing Replay"机制

**❌ 劣势**
- 粒度粗于Token级方法（DAPO/GTPO），信号精度有损失
- 对句子边界敏感，不适合流式生成场景
- 不同长度序列需要适配不同裁剪范围

**适用场景**
MoE架构大模型训练，token级噪声严重的场景，对训练稳定性要求高的项目。

**与GRPO对比**
| 维度 | GRPO (Token级) | GSPO (整句级) |
|------|---------------|---------------|
| **优化粒度** | Token级重要性采样 | 序列级重要性采样 |
| **MoE适配** | Expert路由震荡严重 | 路由稳定性好 |
| **Importance Ratio** | 易失真（expert组合变化） | 稳定（整句归一化） |
| **裁剪策略** | Token级统一裁剪 | 序列级自适应裁剪 |
| **工程复杂度** | 需Routing Replay | 无需额外机制 |

---

## 8. Tree-GRPO (Tree-Guided Reinforcement Policy Optimization)
### 树引导强化策略优化

**一句话回答**  
以<span style="color:#E63946">**"Agent Step"（Thought+Action+Observation）**</span>为节点进行树搜索的RL方法，适配具有明确步骤语义的Agent任务。

**核心机制**
- **Step级树搜索:** 每个节点是<span style="color:#F77F00">**完整的Agent交互步骤**</span>（非token/句子级）
- **Initialize-then-Expand策略:** 
  1. 初始化M条独立轨迹
  2. 每条轨迹随机采样N个节点，从根节点扩展到采样节点
  3. 重复2L次生成M棵树的反应轨迹
- **共享前缀:** 复用公共轨迹部分，<span style="color:#06A77D">**减少rollout预算**</span>

**✅ 优势**
- <span style="color:#06A77D">**Rollout预算更少**</span>（共享前缀减少重复计算）
- Step级信号更精细（相比token级更有<span style="color:#F77F00">**语义完整性**</span>）
- 支持LLM并行推理优化

**❌ 劣势**
仅适用于可分解为明确Step的Agent任务，通用性受限。

**适用场景**
工具调用Agent、多步交互任务（如HotpotQA、复杂查询）、需要中间验证的推理链。

---

## 📐 核心数学公式

### GSPO 目标函数

**GSPO Loss:**
$$J_{GSPO}(\theta) = \mathbb{E}_{x \sim D, \{y_i\} \sim \pi_{old}} \left[ \frac{1}{G} \sum_{i=1}^G \min(s_i(\theta) \hat{A}_i, \text{clip}(s_i(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_i) \right]$$

**序列级重要性权重:**
$$s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}\right)^{1/|y_i|} = \exp\left(\frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{old}(y_{i,t}|x,y_{i,<t})}\right)$$

**群体优势估计:**
$$\hat{A}_i = \frac{r(x,y_i) - \text{mean}(\{r(x,y_j)\}_{j=1}^G)}{\text{std}(\{r(x,y_j)\}_{j=1}^G)}$$

**关键创新:** 长度归一化 $1/|y_i|$ 确保不同长度序列的importance ratio在同一数值范围，避免方差过大。

---

## 📊 方法横向对比

| 方法 | 信用分配粒度 | Critic | 显存占用 | 探索能力 | 训练效率 | 最佳应用 |
|------|------------|---------------|---------|---------|---------|---------|
| **DPO** | 序列级 | ❌ | ⭐ 极低 | ⚠️ 弱 | 🚀 极快 | 偏好对齐 |
| **PPO** | Token级 | ✅ 需要 | ⭐⭐⭐⭐ 高 | ✅ 强 | 🐌 慢 | 通用RL任务 |
| **GRPO** | 序列级（统一） | ❌ | ⭐⭐ 中 | ⚠️ 中 | 🚀 快 | 资源受限推理 |
| **DAPO** | Token级 | ❌ | ⭐⭐ 中 | ✅ 强 | ⚡ 中快 | 长链CoT |
| **GTPO** | Token级（熵加权） | ❌ | ⭐⭐⭐ 中高 | 🔥 极强 | 🐌 慢 | 复杂推理优化 |
| **GRPO-S** | 序列级（熵加权） | ❌ | ⭐⭐ 中 | ✅ 强 | ⚡ 中 | 结果导向推理 |
| **GSPO** | 整句级 | ❌ | ⭐⭐ 中 | ✅ 强 | ⚡ 中快 | MoE模型 |
| **Tree-GRPO** | Step级 | ❌ | ⭐⭐⭐ 中高 | ✅ 强 | ⚡ 中 | Agent多步任务 |

---

## 🎯 核心演进脉络

```
DPO (偏好学习，监督范式)
  ↓
PPO (经典RL，需Critic) ← 显存瓶颈
  ↓
GRPO (去Critic，群体对比) → ⚠️ 问题：粗粒度信用分配
  ↓
DAPO (Token级loss + Clip-Higher + 动态采样) ← 🔥 长CoT突破
  ↓
粒度优化分支：
  ├─ 动态熵加权框架 (DEW) ← 🎯 过程导向优化
  │   ├─ GTPO (Token级熵加权，最细粒度) ← 性能极致
  │   └─ GRPO-S (序列级熵加权，效率优化) ← 工程折中
  │
  └─ GSPO (整句级优化，MoE专用) ← 🎯 稳定性优先
  
特化分支：
  Tree-GRPO (Step级树搜索，Agent场景) ← 语义完整性
```

### 🔍 关键洞察

**1. 粗→细→粗的演进路径**  
<span style="color:#E63946">序列级统一（GRPO）</span> → <span style="color:#F77F00">Token级精细（DAPO/GTPO）</span> → <span style="color:#06A77D">整句级平衡（GSPO/GRPO-S）</span>

**2. 探索与稳定的平衡**  
<span style="color:#2E86AB">Clip-Higher解耦参数</span>让低概率token有成长空间，同时防止熵崩溃。

**3. 粒度选择的权衡**  
- **Token级:** 信号最精细，但噪声大、MoE不友好（DAPO/GTPO）
- **整句级:** 噪声自然过滤，MoE稳定性好，长度归一化（GSPO）
- **序列级:** 效率最高，但粒度粗（GRPO/GRPO-S）

**4. MoE训练的关键问题**  
- **Expert路由震荡:** 新旧策略下expert组合变化导致importance ratio失真
- **GSPO解决方案:** 序列级重要性采样 + 长度归一化，避免Routing Replay

**5. 任务适配矩阵**  
- **通用对话/偏好对齐:** DPO/PPO
- **数学推理/代码生成:** DAPO/GTPO
- **MoE架构模型:** GSPO
- **Agent工具调用:** Tree-GRPO
- **资源受限场景:** GRPO/GRPO-S

---

## 💡 实践建议

### 🎯 选择指南

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| **快速对齐小模型** | <span style="color:#2E86AB">**DPO**</span> | 单阶段训练，需高质量偏好数据 |
| **资源充足+通用任务** | <span style="color:#E63946">**PPO**</span> | 经典可靠，生态成熟 |
| **资源受限+推理任务** | <span style="color:#06A77D">**GRPO → DAPO**</span> | GRPO快速起步，DAPO性能优先 |
| **极致推理性能** | <span style="color:#DC2F02">**GTPO**</span> 或 <span style="color:#F77F00">**GRPO-S**</span> | 熵加权精细优化 |
| **MoE模型训练** | <span style="color:#F77F00">**GSPO**</span> | 整句级抑制噪声，稳定expert路由 |
| **Agent/工具调用** | <span style="color:#A23B72">**Tree-GRPO**</span> | Step级语义完整 |
| **长CoT易失控** | <span style="color:#06A77D">**DAPO**</span> 必备 | 动态采样+软超长惩罚 |

### ⚙️ 关键超参

- **GRPO:** 组大小G=4~8，ε=0.2
- **DAPO:** ε_low=0.2，<span style="color:#F77F00">**ε_high=0.5~1.0**</span>（核心创新）
- **GTPO/GRPO-S:** α₁=1.0，α₂=0.5（熵权重）
- **GSPO:** 句子边界检测阈值，聚合窗口大小
- **Tree-GRPO:** M=4（树数），L=3（轮数），N=2（节点数）

### 🚨 常见陷阱

1. **GRPO熵崩溃:** 训练后期所有输出趋同 → <span style="color:#DC2F02">改用DAPO的Clip-Higher</span>
2. **长回答失控:** 生成4000+token冗余内容 → <span style="color:#DC2F02">启用DAPO软超长惩罚</span>
3. **梯度衰减:** 高acc样本占比过高 → <span style="color:#DC2F02">使用动态采样</span>
4. **自信型错误:** 模型高置信度犯错 → <span style="color:#DC2F02">GTPO/GRPO-S熵倒数惩罚</span>
5. **MoE路由震荡:** Token级方法导致expert频繁切换 → <span style="color:#DC2F02">使用GSPO整句级优化</span>

---

## 关注我，AI不再难 🚀
