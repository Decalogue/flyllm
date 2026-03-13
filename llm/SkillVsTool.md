# Agent 的 Skill 与 Tool 的区别及技能库架构定位

## 1. 核心定性

**Skill 是面向目标的声明式编排单元，Tool 是面向过程的原子式执行单元；技能库是 Agent 的"战术手册层"，位于意图理解层与原子工具层之间，承担语义聚合与复用策略的职责。**

---

## 2. 具体流程

1. **用户意图输入 → 意图理解层 (Intent Parser)**：将自然语言转化为结构化目标
2. **技能匹配 → 技能库层 (Skill Registry)**：根据目标检索并绑定对应 Skill（如 `leetcode-coach`、`feishu-doc`）
3. **技能拆解 → 战术编排层 (Skill Runtime)**：Skill 按预定义 SOP 拆解为多个 Tool 调用序列
4. **原子执行 → 工具层 (Tool Layer)**：每个 Tool 完成单一确定性操作（如 `read`、`web_search`、`exec`）
5. **结果聚合 → 输出封装层**：Skill 整合 Tool 返回结果，按预设格式输出

---

## 3. 数学基础

**Skill-Tool 映射关系**：

$$
\text{Skill}_i = \{ \text{SOP}_i, \text{ToolSet}_i, \text{CtxPolicy}_i \}
$$

$$
\text{ToolSet}_i = \{ T_j \mid T_j \in \mathbb{T}, \text{DepGraph}(T_j) \subseteq \text{Skill}_i \}
$$

其中：
- $\text{Skill}_i$：第 $i$ 个技能，包含标准作业程序、所需工具集、上下文策略
- $\text{ToolSet}_i$：该技能依赖的工具子集
- $\mathbb{T}$：全局工具池
- $\text{DepGraph}(T_j)$：工具 $T_j$ 的依赖图

**技能检索匹配度**：

$$
\text{MatchScore}(Q, \text{Skill}_i) = \alpha \cdot \text{Sim}_{\text{sem}}(Q, D_i) + \beta \cdot \text{Freq}_{\text{hist}}(\text{Skill}_i) + \gamma \cdot \text{Prec}_{\text{ctx}}(C_{\text{curr}}, C_i)
$$

- $Q$：用户查询，$D_i$：技能描述向量
- $\text{Sim}_{\text{sem}}$：语义相似度（Embedding 余弦相似）
- $\text{Freq}_{\text{hist}}$：历史调用频率权重
- $\text{Prec}_{\text{ctx}}$：上下文精度匹配（当前会话上下文与技能所需上下文的交集率）

---

## 4. 工程考量

| 维度 | Skill | Tool |
|------|-------|------|
| **粒度** | 粗粒度，端到端任务闭环 | 细粒度，单一原子操作 |
| **状态** | 可维护会话状态、上下文记忆 | 无状态，幂等执行 |
| **编排** | 内置 SOP、决策分支、重试策略 | 纯函数式调用 |
| **复用** | 跨场景复用（如 `interview-master` 可答多类面试题） | 跨 Skill 复用（如 `web_search` 被 N 个 Skill 共用）|

**Trade-off**：
- Skill 层引入**抽象泄漏风险**：过度封装导致调试困难，Tool 失败时难以定位是编排问题还是原子能力问题
- 技能库膨胀导致**匹配延迟**：当 $|\text{Skills}| > 100$ 时，线性检索退化为 $O(N)$，需引入向量索引或分层聚类

**致命弱点**：
- **Skill 间上下文隔离**：不同 Skill 若未设计统一上下文协议，会导致跨 Skill 调用时状态丢失
- **版本漂移**：Tool 接口升级后，所有依赖 Skill 需同步更新，缺乏契约测试时易雪崩

---

## 5. 工业映射

在工业界，该分层架构被直接应用于：

| 系统 | Skill 层映射 | Tool 层映射 |
|------|-------------|-------------|
| **OpenAI GPTs / Assistants API** | `Assistant`（预配置指令 + 工具绑定）| `Function Calling`（外部函数描述 + 执行）|
| **LangChain** | `Chain` / `Agent`（逻辑编排 + 记忆管理）| `Tool` 类（单功能封装）|
| **EasyClaw (本项目)** | `skills/*.md`（SOP + 约束定义）| `tools/*.ts`（原子能力实现）|
| **Dify / Coze** | 应用 / 工作流（可视化编排）| 插件 / 工具节点（HTTP/本地函数）|

**具体场景**：在 **EasyClaw** 中，当用户说"给我今天的 AI 新闻"时：
1. 意图层识别目标 → 检索到 `daily-news-briefing` Skill
2. Skill 层按 SOP 执行：先调用 `web_search` Tool 聚合多源新闻 → 再调用 LLM Tool 做摘要 → 最后按模板格式化输出
3. 用户无需关心调用了哪些 Tool，只需关注"获取 AI 新闻"这一意图是否被满足

---

> **一句话总结**：**Tool 是 Agent 的手脚，Skill 是 Agent 的战术手册；技能库是连接"想做什么"与"能做什么"的语义编排中枢。**
