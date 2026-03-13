# Agent 轨迹学习与技能沉淀

---

## 1. 核心定性

**本质上，Agent 技能学习是一个将执行轨迹通过逆强化学习/模式挖掘提炼为可复用策略（Skill Policy）的过程，核心挑战在于轨迹噪声过滤、意图对齐与跨任务泛化。**

---

## 2. 具体流程

1. **轨迹采集与结构化**：记录 Agent 每一步的 `(Observation, Thought, Action, Result)` 四元组，构建带标注的执行链；
2. **子目标分割与意图识别**：通过行为边界检测（如执行成功/失败信号、人类反馈）将长轨迹切分为原子技能片段，并用 LLM 提取其高层意图描述；
3. **技能泛化与入库**：对原子技能进行参数抽象（slot filling）和上下文无关化，存入向量数据库建立检索索引。

---

## 3. 数学基础

### 3.1 技能抽取的逆强化学习建模

将轨迹学习建模为马尔可夫决策过程（MDP）的逆问题：

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \gamma)
$$

其中：
- $\mathcal{S}$：Agent 状态空间（环境观测 + 上下文）
- $\mathcal{A}$：动作空间（API 调用、工具执行）
- $\mathcal{T}(s'|s,a)$：状态转移概率
- $\gamma$：折扣因子

**核心目标**：从专家轨迹 $\tau = \{(s_t, a_t)\}_{t=0}^{T}$ 中恢复奖励函数 $R(s,a)$ 并学习策略 $\pi(a|s)$：

$$
\max_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right] - \lambda H(\pi_{\theta})
$$

### 3.2 技能泛化的语义嵌入

技能 $K_i$ 的向量化表示通过对比学习获得：

$$
\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(K_i, K_j^+) / \tau)}{\sum_{k} \exp(\text{sim}(K_i, K_k) / \tau)}
$$

其中：
- $K_i = \text{Encoder}(I_i, P_i)$，$I_i$ 为意图描述，$P_i$ 为参数模板
- $K_j^+$ 为同一技能的另一实例（正样本）
- $\text{sim}(\cdot, \cdot)$：余弦相似度

### 3.3 伪代码：技能抽取流水线

```python
def extract_skill(trajectory: List[Step]) -> Skill:
    # Step 1: 子目标分割（基于成功信号或人类标注）
    segments = segment_by_success_signals(trajectory)
    
    # Step 2: 意图抽象（LLM 提取高层语义）
    intent = llm_extract_intent(segments)
    
    # Step 3: 参数槽位化（泛化关键变量）
    template, slots = parameter_abstraction(segments)
    
    # Step 4: 向量化入库
    embedding = encoder.encode(f"{intent}: {template}")
    skill_db.upsert(Skill(intent, template, slots, embedding))
    
    return skill
```

---

## 4. 工程考量

| 维度 | 核心取舍与风险 |
|------|---------------|
| **Trade-off** | 抽取粒度 vs. 复用性——太粗难以适配新场景，太细导致组合爆炸；通常采用**分层技能树**（primitive → composite → task-level）平衡 |
| **致命弱点 1** | **轨迹噪声陷阱**：失败轨迹占比高时，简单模仿学习（BC）会引入偏差，必须引入 **Filtered BC** 或 **Preference-based RLHF** 清洗数据 |
| **致命弱点 2** | **分布外崩溃（OOD）**：训练任务的观测分布稍有偏移，技能策略可能完全失效；需通过 **Domain Randomization** 或 **Meta-Learning**（MAML）增强泛化 |
| **致命弱点 3** | **组合爆炸**：技能数量增长导致检索/规划复杂度指数上升；需引入 **技能图（Skill Graph）** 或 **层级选项框架（Options Framework）** |

---

## 5. 工业映射

| 机制 | 工业应用 |
|------|---------|
| **轨迹学习 + 技能抽取** | **Voyager**（Minecraft Agent）— 通过代码生成沉淀可复用技能库，支持终身学习 |
| **技能向量检索** | **LangChain / LlamaIndex** 的 Tool Retrieval — 基于任务描述 embedding 匹配最适工具 |
| **分层技能框架** | **MetaGPT / AutoGPT** — 将复杂任务分解为可复用的原子 Action，支持跨任务组装 |
| **Human-in-the-loop RLHF** | **Cognosys / MultiOn** — 通过人工标注偏好对优化技能策略，解决噪声问题 |

---

**一句话总结**：Agent 技能沉淀的核心是将带噪声的执行轨迹逆推为结构化、可检索、可泛化的策略单元，关键在于**意图抽象、参数槽位化与分布外鲁棒性设计**。
