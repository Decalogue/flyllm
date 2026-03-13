# 技能库（Skill Library）设计

## 1. 核心定性

本质上，**Skill Library 是一个为 LLM 提供可插拔工具能力的「操作系统驱动层」**，通过**结构化语义描述** + **向量化索引** + **图拓扑组合**实现技能的动态发现、调用与编排。

## 2. 具体流程

1. **技能表示**：每个技能封装为 `(元数据 + 接口契约 + 实现载体)` 三元组，元数据包含名称、描述、输入输出 Schema；接口契约定义参数约束；实现载体可以是 Python 函数、API 端点或 MCP Server。
2. **技能检索**：用户 query 经 Embedding 向量化后，在向量索引（如 FAISS）中进行相似度搜索，召回 Top-K 候选技能，再通过重排序（Rerank）或 LLM 意图分类器进行精排。
3. **技能组合**：将召回的技能按依赖关系构建 DAG，通过拓扑排序确定执行顺序，支持串行、并行、条件分支等编排模式，最终生成可执行计划。

## 3. 数学基础

### 技能检索相似度计算
$$\text{Score}(q, s_i) = \cos(\mathbf{E}(q), \mathbf{E}(s_i)) = \frac{\mathbf{E}(q) \cdot \mathbf{E}(s_i)}{\|\mathbf{E}(q)\| \|\mathbf{E}(s_i)\|}$$

其中：
- $q$: 用户查询文本
- $s_i$: 第 $i$ 个技能的描述文本
- $\mathbf{E}(\cdot)$: Embedding 编码器（如 text-embedding-3-small）
- $\cos(\cdot)$: 余弦相似度，范围 $[-1, 1]$

### Top-K 召回与阈值过滤
$$\mathcal{S}_{\text{candidate}} = \{ s_i \mid \text{Rank}(\text{Score}(q, s_i)) \leq K \land \text{Score}(q, s_i) > \theta \}$$

其中：
- $K$: 召回数量上限（通常 5-10）
- $\theta$: 相似度阈值（通常 0.7-0.85）
- $\text{Rank}(\cdot)$: 按相似度降序排名

### 技能组合 DAG 拓扑排序
$$G = (V, E), \quad V = \{ \text{skill}_1, \text{skill}_2, ..., \text{skill}_n \}$$

$$\forall (u, v) \in E: \text{skill}_u \text{ 的输出} \subseteq \text{skill}_v \text{ 的输入}$$

拓扑排序结果 $\pi$ 满足：
$$\forall (u, v) \in E: \pi(u) < \pi(v)$$

## 4. 工程考量

### Trade-off 矩阵

| 策略 | 优势 | 代价 |
|------|------|------|
| **稠密向量检索** | 语义理解强，容错性好 | 需要 Embedding 推理开销，冷启动需要语料 |
| **稀疏关键词检索** | 零延迟，可解释性强 | 语义漂移敏感，"搜索"搜不到"检索" |
| **混合检索** (Dense + Sparse) | 兼顾语义与精确匹配 | 系统复杂度翻倍，需要融合排序 |
| **静态技能注册** | 确定性强，易于审计 | 灵活性差，新技能需重启/热更新 |
| **动态技能发现** | 支持 AutoGPT 式自扩展 | 安全风险高，不可控技能可能作恶 |

### 致命弱点

1. **意图漂移（Intent Drift）**：当 query 语义模糊时，可能召回错误技能。例：用户问"怎么删除文件"，可能误触发 `rm -rf /` 的危险技能。**必须通过权限沙箱 + 人工确认拦截**。

2. **组合爆炸**：技能数量 $N$ 增加时，组合复杂度 $O(N!)$ 爆炸。**需要引入层次化分类（Taxonomy）+ 领域隔离**。

3. **上下文窗口溢出**：组合多个技能后，中间结果累计超出 LLM 上下文限制。**必须引入摘要压缩或分片执行**。

## 5. 工业映射

| 模块 | 开源/商业实现 | 应用场景 |
|------|--------------|----------|
| **技能表示** | [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) | Anthropic 推出的标准化技能接口，统一工具描述格式 |
| **技能检索** | [FAISS](https://github.com/facebookresearch/faiss)、[Milvus](https://milvus.io/) | 大规模向量相似度搜索，支撑千万级技能库 |
| **技能组合** | [LangChain](https://langchain.com/)、[LlamaIndex](https://llamaindex.ai/) | Chain/Agent 编排框架，实现 ReAct、Plan-and-Execute 模式 |
| **端到端** | [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)、[Google Vertex AI Tools](https://cloud.google.com/vertex-ai) | 闭源 LLM 内置技能调用机制，零样本工具使用 |
| **安全沙箱** | [E2B](https://e2b.dev/)、[Sandbox](https://github.com/sandbox) | 代码执行类技能的隔离运行环境 |
