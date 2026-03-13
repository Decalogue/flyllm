# 多技能 Agent 技能选择与调度 vs Tool Calling 决策

## 1. 核心定性 (The 10-Second Hook)

**本质上，技能选择是一个基于语义相似度的"描述→匹配→路由"三层决策结构，而 Tool Calling 是基于 LLM 生成能力直接在 token 空间进行函数签名生成与参数填充。**

---

## 2. 具体流程 (Specific Process)

| 维度 | 多技能 Agent 技能选择 | Tool Calling 决策 |
|------|----------------------|-------------------|
| **触发机制** | ① 解析用户输入 → ② 语义向量化 → ③ 与技能描述库做相似度匹配 | ① LLM 解析上下文 → ② 自回归生成工具调用 JSON |
| **决策主体** | 外部路由层（Retriever / Router） | LLM 内部（端到端生成） |
| **输出形式** | 返回技能标识符（skill_id） | 返回结构化工具调用（`<tool>` 标签或 JSON） |

---

## 3. 数学基础 (The Hardcore Logic)

### 3.1 技能选择的相似度匹配

$$\text{skill\_id} = \arg\max_{s \in S} \text{cosine}(\text{Embed}(q), \text{Embed}(d_s))$$

其中：
- $q$: 用户查询文本
- $S$: 技能库集合 $\{s_1, s_2, ..., s_n\}$
- $d_s$: 技能 $s$ 的描述文本（SKILL.md 中的 `<description>`）
- $\text{Embed}(\cdot)$: 文本嵌入模型（如 text-embedding-3-small）

**路由决策阈值**：
$$
\text{select}(q) = 
\begin{cases} 
s^* & \text{if } \max_{s} \text{cosine}(q, s) \geq \tau \\
\text{default\_agent} & \text{otherwise}
\end{cases}
$$

其中 $\tau$ 通常为 0.75~0.85（余弦相似度）。

### 3.2 Tool Calling 的生成概率

$$P(\text{tool\_call} | q, C) = \prod_{t=1}^{T} P(t_t | q, C, t_{<t})$$

其中：
- $C$: 系统提示中的工具定义（JSON Schema）
- $t_t$: 第 $t$ 个生成的 token
- 工具选择隐式编码在生成概率分布中

---

## 4. 工程考量 (Engineering Trade-offs)

| 考量维度 | 技能选择 | Tool Calling |
|---------|---------|--------------|
| **精度 vs 召回** | 依赖描述质量，**描述模糊时路由错误率↑** | LLM 直接推理，上下文充足时精度高 |
| **可解释性** | ✅ **高**——相似度分数可审计 | ⚠️ **黑盒**——注意力权重难解读 |
| **动态扩展** | ⚠️ 需重新索引技能描述向量库 | ✅ 仅需更新 system prompt 的工具列表 |
| **延迟** | 一次向量检索（~50-100ms）+ LLM 调用 | 单次 LLM 调用，但生成更多 token |
| **致命弱点** | 用户意图与技能描述**语义鸿沟**（如"画张图"vs"图像生成"） | 复杂参数填充时**幻觉参数名/类型** |

**关键区别**：
- **技能选择是"先选后调"**：路由层先决策，再加载 SKILL.md 指导后续执行
- **Tool Calling 是"边想边调"**：LLM 自主决定何时调用、调用什么、传什么参数

---

## 5. 工业映射 (Industry Mapping)

| 架构 | 机制应用 | 场景 |
|------|---------|------|
| **OpenAI Assistants API** | Tool Calling 原生支持 | Function Calling + Retrieval |
| **LangChain / LlamaIndex** | 两者结合——Tool Retriever + LLM Tool Calling | Agent 先检索工具子集，再让 LLM 选择具体工具 |
| **AutoGPT / BabyAGI** | 纯 Tool Calling 驱动 | 端到端任务分解与工具链执行 |
| **EasyClaw (本项目)** | **分层决策**：`description` 匹配选技能 → skill 内再决定具体工具调用 | 技能粒度粗、工具粒度细的两层架构 |

**工业最佳实践**：
> "在 Google DeepMind 的 AlphaCode 和 OpenAI 的 GPT-4 Plugins 系统中，采用**混合架构**：先用轻量级 encoder 做技能预筛选（缩小候选空间），再用 LLM 做精确的 Tool Calling 决策，兼顾速度与精度。"

---

**一句话总结**：
> **技能选择是"离线描述匹配"的外层路由，Tool Calling 是"在线生成决策"的内层执行——前者解决"用什么能力"，后者解决"怎么调用"。**
