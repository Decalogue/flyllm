# EverMind 面试备忘（社招 · 两轮 · Agent 方向）

## 个人上下文

- **类型**：社招  
- **轮次**：共两轮  
- **方向**：Agent（与长期记忆、检索、决策层强相关）

## 公司侧背景（公开信息）

- EverMind 侧重 **AI 长期记忆 / EverMemOS（记忆 OS）**：记忆分层、检索与决策、RAG/混合检索、评测与基准、长上下文与稀疏/线性扩展等叙事。招聘与 JD 见 [Join EverMind](https://evermind.ai/careers)。  
- **算法岗 JD 明文**：扎实 **数据结构与基础算法**、Python、Transformer 与训练框架、工程调试与性能优化；加分含长期记忆、RAG、知识图谱、RL 等。  
- **无可靠公开「固定真题/面经」**：以下按 **JD + Agent 岗位共性 + 与记忆产品线的贴合度** 整理，非内部口径。

---

## Agent 方向：面试高概率重点（按优先级）

### 1. 记忆 × Agent（与 EverOS / 产品线最贴）

- 短期上下文 vs 长期记忆：何时写入外部记忆、如何更新、冲突与遗忘。  
- 检索再消费：多轮里何时检索、检索什么、结果如何进 prompt（与「先召回后写入」闭环一致）。  
- 与 RAG 的边界：多步、可选、可失败重试，而非单次检索单次生成。

### 2. 规划与多步执行

- Plan-and-execute、ReAct 类循环：终止条件、避免死循环与重复工具调用。  
- 子目标分解、依赖、失败回退（工具错、检索空、格式不对）。  
- 「多步检索 / 决策」类问题：可验证中间步骤、与决策层叙事同源。

### 3. 工具调用（Function / Tool Calling）

- Schema 设计、参数校验、幻觉调用、并行 vs 串行、超时与重试。  
- 可观测性：trace / 日志下区分模型错误 vs 工具/数据错误。

### 4. 系统设计与工程落地

- 用户输入 → 记忆写入 / 检索 / 推理 / 工具 / 再推理 的流水线与状态。  
- 延迟与成本：缓存、截断、模型分级（小模型路由/摘要 + 大模型决策）。  
- 评测：任务完成率、工具正确率、检索命中率、安全（拒答、越权）。

### 5. 安全与对齐（垂直岗更重）

- 工具权限、敏感数据、提示注入对 Agent 的影响与缓解。  
- 若涉及心理/医疗叙事：**不越界诊断、危机话术、隐私与合规**（以具体 JD 为准）。

### 算法题与「大模型八股」

- **算法**：社招 Agent 岗仍常考 **基础代码题**（控制流、边界）；深度有时偏 **状态机、编排、检索排序** 等与流程相关的思维，而非纯竞赛。  
- **八股**：更可能是 **能落到记忆/RAG/Agent/训练推理/评测** 的原理与方案讨论，而非脱离业务的纯背诵。

### 建议自备素材

- **1 个 Agent 项目**：状态、记忆、检索、工具、终止条件、失败策略能说清。  
- **RAG + Agent vs 纯 Chat**、**何时写记忆** 能口头设计。  
- 本仓库 `llm/` 下 ReAct、Tool Calling、MemGPT/RAG 类笔记可作口语素材。

---

## 两轮面试：社招常见节奏（参考，以实际通知为准）


| 轮次  | 常见侧重                                                         | 准备抓手                        |
| --- | ------------------------------------------------------------ | --------------------------- |
| 第一轮 | 基础能力 + 广度：代码/数据结构、Python、Transformer 与训练推理常识、简单 Agent/RAG 概念 | LeetCode 基础 + 本备忘「高概率重点」过一遍 |
| 第二轮 | 深度 + 系统设计：记忆与检索架构、多步 Agent 流程、评测、安全、与你简历项目的深挖                | 画 1～2 张流程图（请求全链路、记忆读写时机）    |


若某轮明确「无手写代码」，则把精力挪到 **系统设计 + 项目复盘 + 失败案例**。

---

## 赛道与产品（内部/业务备忘）

- **Track 1: Agent + Memory**（Content Creator Copilot）：设定/世界观、章节与伏笔、角色与情节一致性的**长期记忆**。  
- **记忆分工**：**semantic mesh**（实体/关系，必选）+ **EverMemOS**（摘要与跨章叙事，可选）+ **UniMem**（可选统一记忆/图谱）。  
- **主流程**：构思 → 召回（人物、伏笔、长线设定）→ 续写 → 质检⇄重写 → 实体提取 → 入库。  
- **EverMemOS**：写 — `retain_plan/chapter/polish/chat` → `add_memory`；读 — 规划/续写/润色/对话前 `recall_from_evermemos` 注入 prompt；续写可关 `use_evermemos_context`，高章节**长程召回**。环境：`EVERMEMOS_API_KEY`、`EVERMEMOS_ENABLED=1`。  
- **路线**：Agentic RL — 动态选 Agent、记忆压缩/更新/重组、成本与质量权衡。

## 主路径 vs Agent 模块

- **主路径**：`api_flask` → `creator_handlers` / `memory_handlers` / `api_EverMemOS` → `**ReactNovelCreator`** + `**context`（mesh）**；可选 UniMem、EverMemOS。`**task.novel` 不依赖 `orchestrator/`**（支线实验，未接 `/api/creator`）。  
- **Agent 能力**：ReAct 推理–行动；Master/Sub（**委托** = 隔离上下文 + Schema；**同步** = 共享历史）；上下文卸载/Compaction/Summarization；**L1/L2/L3** 行动空间；**tools**（Index+Discovery 省 Token）+ **skills**（SOP）；多模型实体投票 → 结构化记忆原料。  
- **数据流**：用户 → 创作器 → **召回记忆** → LLM → 工具（可选）→ **写记忆** → 质检 → 返回。

## 记忆与 API 关键词


| 组件        | 要点                               |
| --------- | -------------------------------- |
| mesh      | 实体–关系，一致性、质检                     |
| EverMemOS | 全流程 retain/recall；续写三类检索合并       |
| UniMem    | `memory_handlers` 懒加载，与 api 配置解耦 |


`read_mesh`、`write_mesh`、`recall_for_mode`、`retain_plan`、`retain_chapter`、`retain_polish`、`retain_chat`。

## UniMem（一句话 + 要点）

**分层存储（FoA/DA/LTM）+ 多维检索（RRF+重排）+ 涟漪/睡眠更新 + Retain/Recall/Reflect**；功能导向适配器（Operation / LayeredStorage / MemoryType / Graph / AtomLink / Retrieval / Update）。

- **编排向 API**：`context_for_agent` → `recall_for_agent` / `retain_for_agent`（注入 session/task/role，重要性融合 + **会话优先 FoA/DA**）。  
- **重要性**：时间衰减、`retrieval_count`、会话/任务匹配（`importance_weight`、`importance_decay_days`）。  
- **后端**：存储/图/向量可插拔（Redis、PG、Neo4j、Qdrant 等）。

## 30 秒口径（业务向）

1. **Agent**：主路径是 **ReAct 创作闭环** + tools/skills；编排支线独立，不混为主路径。
2. **Memory**：**Mesh 结构 + 云叙事记忆 + 可选 UniMem 统一层**；步骤级「先召回后写入」。
3. **边界**：`orchestrator`、部分 context 能力在文档中为实验/演进；以 **ReactNovelCreator + handlers** 为准。

