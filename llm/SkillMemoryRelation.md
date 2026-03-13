# 技能与记忆的关系：LLM 系统中的认知架构设计

## 1. 核心定性

**本质上，技能是对外暴露的工具调用能力（API），而记忆是对话上下文的持久化状态（State）；技能通过记忆获取上下文连续性，记忆通过技能实现动态更新，二者构成认知闭环。**

---

## 2. 具体流程

1. **技能检索 → 记忆注入**：用户请求触发技能匹配后，系统自动召回 MEMORY.md 中的历史上下文注入 Prompt，确保技能具备"长程记忆"能力。
2. **技能执行 → 记忆更新**：技能完成任务后，将关键信息（用户偏好、任务结果、新知发现）回写至记忆模块，实现增量学习。
3. **周期性压缩 → 长期固化**：高频访问的记忆通过总结算法压缩存入长期记忆，低频记忆淘汰或归档。

---

## 3. 硬核推演

技能与记忆的联动可建模为 **马尔可夫决策过程 + 向量检索** 的混合架构：

$$
\mathbf{Skill}_t = f(\mathbf{Query}_t, \mathbf{Memory}_t, \mathbf{Context}_{t-1})
$$

$$
\mathbf{Memory}_{t+1} = \mathbf{Memory}_t \oplus \Delta(\mathbf{Skill}_t, \mathbf{Reward}_t)
$$

其中：
- $\mathbf{Skill}_t$：时刻 $t$ 被激活的技能集合
- $\mathbf{Memory}_t$：检索召回的上下文记忆向量 $\{\mathbf{m}_1, \mathbf{m}_2, ..., \mathbf{m}_k\}$
- $f(\cdot)$：Prompt 组装函数，融合 Query + Memory + 历史上下文
- $\Delta(\cdot)$：记忆更新算子，根据技能执行反馈 $\mathbf{Reward}_t$ 生成记忆增量
- $\oplus$：记忆融合操作（向量化追加或摘要压缩）

**检索核心**：
$$
\text{Recall} = \text{TopK}\left( \frac{\mathbf{Query} \cdot \mathbf{Memory}_i}{\|\mathbf{Query}\| \|\mathbf{Memory}_i\|} \right), \quad i \in [1, N]
$$

---

## 4. 工程考量

| 维度 | Trade-off | 致命弱点 |
|------|-----------|----------|
| **记忆粒度** | 全文记忆保真度高但 Token 爆炸；摘要记忆省 Token 但信息损失 | 超长对话 (>100K) 时，检索噪声导致关键记忆丢失 |
| **更新频率** | 实时写盘一致性高但 IO 压力大；批量写入性能好但有延迟 | 系统崩溃时，未刷盘的增量记忆全部丢失 |
| **技能边界** | 细粒度技能精准但组合复杂；粗粒度技能简单但不够灵活 | 跨技能依赖（如"先查天气再发邮件"）需要显式状态传递机制 |

---

## 5. 工业映射

- **OpenAI Assistants API**：Threads 作为短期记忆，Files 作为长期知识库，Tools（Function Calling）作为技能，通过 `run` 对象串联执行链。
- **LangChain**：`Memory` 模块（ConversationBufferMemory / VectorStoreRetrieverMemory）+ `Tools` + `Agent` 的架构直接映射上述 MDP 模型。
- **MemGPT**：显式区分 **Working Memory**（上下文窗口）与 **External Memory**（分层存储），通过 `agent.step()` 触发记忆分页（paging）与召回，解决 LLM 上下文长度限制。

**一句话总结**：技能是"能做什么"，记忆是"记得做过什么"，二者通过 **检索-执行-固化** 的闭环实现持续进化。
