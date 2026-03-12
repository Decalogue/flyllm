# 🔥AI烽火：大模型时代的征途

> 大模型实战【笔记+代码+高频面试题】

> LLM & Agent & Memory（大模型 & 智能体 & 记忆）+ [leetcode](leetcode.md)

---

## 📋 问题列表

### LLM

#### Tokenizer - LX

1. 什么是分词器？为什么 LLM 需要分词而不是直接处理字符？
2. BPE 算法的核心思想是什么？能详细说说它的训练和编码过程吗？
3. SentencePiece 和 BPE 有什么区别？为什么 GPT 系列用 BPE，而 T5 用 SentencePiece？
4. WordPiece 和 BPE 的区别在哪里？BERT 为什么选择 WordPiece？
5. Unigram Language Model Tokenizer 的工作原理是什么？它有什么优势？
6. 为什么子词分词比词级分词更适合大模型？遇到稀有词怎么处理？
7. 词汇表大小如何确定？太大会怎样，太小会怎样？
8. 特殊标记 [CLS]、[SEP]、[PAD]、[MASK] 分别起什么作用？为什么需要它们？

#### Transformer 前12-LX 后12-XY

1. Transformer 的核心架构是什么？为什么它成为所有 LLM 的基础？Encoder 和 Decoder 的区别在哪里？
2. 自注意力机制的数学公式是什么？Q、K、V 分别代表什么？为什么要除以 sqrt(d_k)？
3. 为什么需要多头注意力？多头比单头好在哪里？头数怎么选择？
4. Cross-Attention 和 Self-Attention 有什么区别？分别用在什么场景？
5. 注意力机制的计算复杂度是 O(n²)，如何优化？有哪些降低复杂度的方法？
6. 因果掩码（Causal Mask）是什么？为什么自回归模型需要它？怎么实现？
7. 稀疏注意力有哪些实现方式？Longformer 和 BigBird 的区别是什么？
8. 局部注意力适用于什么场景？窗口大小怎么设计？
9. 线性注意力如何降低复杂度？Performer 和 Linformer 的区别是什么？
10. GQA（分组查询注意力）是什么？为什么 LLaMA 2 用它？它如何平衡性能和效率？
11. Longformer 的滑动窗口注意力是怎么实现的？窗口大小怎么选？
12. Flash Attention 解决了什么问题？它的核心思想是什么？Flash Attention 2 改进了什么？
13. [注意力权重可视化能告诉我们什么？如何分析模型的注意力模式？](llm/AttentionPatternAnalysis.md)
14. [注意力机制会出现梯度消失吗？如何缓解？](llm/AttentionVanishingGradient.md)
15. [位置偏置（Position Bias）是什么？如何消除？](llm/PositionBias.md)
16. [位置编码有哪些方式？绝对位置编码和相对位置编码的区别是什么？](llm/PositionalEncoding.md)
17. [RoPE 的原理是什么？为什么它能外推到更长序列？怎么实现的？](llm/RoPE.md)
18. [ALiBi 是什么？它如何实现位置编码？和 RoPE 比有什么优势？](llm/ALiBiVsRoPE.md)
19. [LayerNorm 和 BatchNorm 的区别是什么？为什么 Transformer 用 LayerNorm？](llm/LayerNormVsBatchNorm.md)
20. [RMSNorm 是什么？它和 LayerNorm 的区别？为什么 LLaMA 用 RMSNorm？](llm/RMSNorm.md)
21. [预训练、微调、指令微调的区别是什么？各自的目标是什么？](llm/PretrainingFinetuningInstructionTuning.md)
22. [模型并行、数据并行、流水线并行有什么区别？如何选择？能混合使用吗？](llm/Parallelism.md)
23. [ZeRO 优化器如何减少显存？ZeRO-1、ZeRO-2、ZeRO-3 的区别是什么？](llm/ZeRO.md)
24. [SFT 和 RLHF 的区别是什么？各自的优缺点？](llm/SFT_VS_RLHF.md)
25. [大模型的上下文长度限制是怎么产生的？如何扩展上下文窗口？有哪些方法？](llm/ContextWindowLimitAndExtension.md)
26. [BF16 FP16 之间的区别，适用的情况](llm/Bf16VsFp16.md)
27. [7B 模型显存计算](llm/显存计算.md)

#### Finetuning - LX

1. 全量微调和参数高效微调的区别是什么？什么时候用全量微调？
2. LoRA 的原理是什么？rank 和 alpha 参数怎么选？为什么有效？
3. QLoRA 如何进一步降低显存？4-bit 量化怎么实现？精度损失大吗？
4. Adapter 和 LoRA 的区别是什么？各自的适用场景？
5. 指令微调的数据怎么构建？指令格式怎么设计？需要多少数据？
6. 数据合成（Data Synthesis）有哪些方法？如何生成高质量的SFT数据？合成数据的质量如何评估？
7. SFT 训练中的 Loss 如何设计？交叉熵损失的具体计算过程是什么？如何处理不平衡的数据分布？
8. SFT 之后出现灾难性遗忘（Catastrophic Forgetting）的原因是什么？如何解决？有哪些具体方法？
9. 多任务微调如何平衡不同任务？损失函数怎么设计？如何避免任务间的负迁移？
10. 微调的学习率怎么设置？初始学习率怎么选？需要 Warmup 吗？学习率调度策略如何选择？
11. 早停机制怎么设计？验证指标怎么选？patience 怎么设置？如何防止过拟合？
12. 微调的数据增强有哪些方法？如何设计增强策略？如何保证增强后的数据质量？

#### RL - LX

1. 强化学习的基本概念是什么？如何应用到 LLM 训练？
2. Policy Gradient 方法的核心思想是什么？REINFORCE 算法如何计算梯度？
3. Actor-Critic 方法是什么？价值函数和策略函数如何训练？
4. PPO 为什么比 REINFORCE 更稳定？它的核心改进是什么？
5. RLHF 的完整流程是什么？每一步具体怎么做？
6. 奖励模型如何训练？奖励函数怎么设计？需要多少数据？
7. DPO 相比 RLHF 的优势是什么？如何实现？为什么更简单？
8. KL 散度惩罚为什么需要？KL 系数怎么设置？太大太小会怎样？
9. 奖励函数怎么设计？奖励黑客是什么？如何防止？
10. 离线强化学习如何应用到 LLM？有哪些挑战？
11. 探索与利用如何平衡？在 LLM 训练中如何体现？
12. 价值函数如何用神经网络表示？如何训练？
13. 讲一下 GRPO 原理
14. DAPO 相比 GRPO 做了哪些改进？

#### Inference - XY

1. [LLM 模型推理的流程是什么？有哪些关键步骤？如何优化？](llm/ModelInference.md)
2. [KV Cache 如何加速推理？如何实现？内存怎么优化？](llm/KVCache-kimi.md)
3. [vLLM 的 PagedAttention 机制是什么？如何实现 KV Cache 的内存管理？](llm/PagedAttention.md)
4. [vLLM 的并行机制有哪些？Tensor Parallelism、Pipeline Parallelism、Data Parallelism 在 vLLM 中如何结合？](llm/vLLM-Parallelism.md)
5. [vLLM 的 Continuous Batching 如何实现？如何动态调度请求？相比静态批处理有什么优势？](llm/vLLMContinuousBatching.md)
6. [vLLM 和 SGLang 的区别](llm/VllmVsSglang.md)
7. [生成解码策略和 temperature 的影响](llm/DecodingAndTemperature.md)
8. [批量推理如何优化？批处理策略怎么设计？动态批处理怎么实现？](llm/BatchInferenceOptimization.md)
9. [流式推理如何实现？如何优化流式输出？用户体验如何提升？](llm/StreamInference.md)
10. [推理量化有哪些方法？INT8、INT4、INT1 的区别？如何选择？量化后的精度损失如何评估？](llm/InferenceQuantization.md)
11. [推理缓存如何设计？缓存策略有哪些？如何管理缓存？缓存命中率如何提升？](llm/InferenceCacheDesign.md)
12. [推理延迟和吞吐量如何优化？如何平衡延迟和准确性？性能瓶颈如何定位和优化？](llm/LatencyThroughputOptimization.md)
13. [大模型流式解析](llm/LLMStream.md)
14. [KVCache-工业架构图](llm/KVCache-工业架构图.md)

#### RAG - XY

1. [RAG 的核心思想是什么？为什么有效？解决了什么问题？](llm/[RAG.md](http://RAG.md))
2. [RAG 系统的检索模块怎么设计？有哪些检索方法？怎么选择？](llm/RAGRetrievalDesign.md)
3. [向量检索和关键词检索的区别？如何结合？混合检索怎么设计？](llm/VectorAndKeywordRetrieval.md)
4. [Dense Retrieval 和 Sparse Retrieval 的区别？BM25 和 DPR 的区别？](llm/RetrievalMethodsComparison.md)
5. [Hybrid Search 如何实现？权重怎么平衡？](llm/HybridSearch.md)
6. [向量数据库怎么选？Milvus 和 Qdrant 的区别？](llm/VectorDB_Milvus_vs_Qdrant.md)
7. [Rerank 为什么需要？如何选择 Rerank 模型？什么时候用？](llm/RerankExplanation.md)
8. [RAG 文档分块策略怎么设计？有哪些方法？块大小怎么选？](llm/RAGChunkingStrategy.md)
9. [RAG 中的上下文窗口限制怎么处理？长文档如何优化？](llm/RAGContextWindowOptimization.md)
10. [RAG 中 Query Expansion 如何优化查询？查询重写怎么实现？](llm/RAG_QueryExpansion.md)
11. [字节跳动 RAG 实践手册](https://docs.qq.com/doc/DSXJiaE5taUtaVGx6?_t=1768892540919&nlc=1)

### Agent - XY

#### Agent Framework

1. [Agent的核心技术模块有哪些？每个模块的功能、难点，以及它们之间怎么联动？](llm/AgentCoreModules.md)
2. [Tool Calling 如何让模型学会使用工具？工具描述怎么设计？](llm/ToolCalling.md)
3. [Agent 的规划能力如何设计？有哪些规划方法？如何实现？](llm/AgentPlanningDesign.md)
4. [ReAct 框架如何实现？提示怎么设计？](llm/ReActFramework.md)
5. [Agent 的记忆机制有哪些类型？如何实现？](llm/AgentMemoryMechanism.md)
6. [Agent 的反思能力如何实现？反思机制怎么设计？](llm/AgentReflectionMechanism.md)
7. [Agent 的决策流程如何设计？有哪些决策框架？如何选择？](llm/AgentDecisionFramework.md)
8. [Agent 的错误恢复机制如何设计？重试策略怎么设计？](llm/AgentErrorRecoveryAndRetry.md)
9. [Agent 的自主性如何控制？控制机制怎么设计？](llm/AgentAutonomyControl.md)
10. [Agent 的长期记忆和短期记忆如何实现？记忆系统怎么设计？](llm/AgentMemorySystemDesign.md)
11. [Multi-Agent 系统如何实现？Agent 之间如何协作？通信机制怎么设计？](llm/MultiAgentSystem.md)
12. [LangChain 的核心组件是什么？Agent 系统怎么设计？](llm/LangChainCoreComponentsAndAgentDesign.md)

#### Function Call

1. Function Call 的多轮对话如何处理？为什么这是最难的部分？
2. Function Calling 的格式是什么？JSON Schema 如何定义？Schema 怎么设计？
3. 如何让模型学会选择合适的函数？训练方法有哪些？训练数据怎么设计？
4. Function Call 的参数提取错误如何处理？错误处理机制怎么设计？
5. Function Call 的流式输出如何实现？流式调用如何优化？
6. Function Call 的并行调用如何实现？并行调用如何管理？
7. Function Call 的验证机制如何设计？参数如何校验？验证规则怎么设计？
8. Function Call 的多轮交互如何维护上下文？上下文管理怎么设计？
9. Function Call 的链式调用如何实现？依赖关系如何处理？依赖图怎么设计？
10. Function Call 的容错机制如何设计？调用失败如何处理？重试策略怎么设计？
11. Function Call 的权限控制如何实现？如何限制可调用函数？权限系统怎么设计？
12. Function Call 的延迟如何优化？如何平衡准确性和速度？

#### Skills

### Memory

---

## 💻 手撕代码清单

### LLM

#### Transformer

1. 实现 Multi-Head Attention（MHA），包括 Q、K、V 的线性变换、多头分割、注意力计算和拼接
2. 实现 Cross-Attention，包括编码器-解码器注意力机制
3. 实现分组查询注意力（GQA），包括 KV 头的分组和注意力计算
4. 实现 RoPE（Rotary Position Embedding），包括位置编码的旋转矩阵计算和应用
5. 实现 ALiBi（Attention with Linear Biases），包括位置偏置的计算
6. 实现因果掩码（Causal Mask），支持自回归模型的掩码生成
7. 实现 Layer Normalization，包括均值和方差计算、归一化和缩放
8. 实现 RMSNorm，包括均方根归一化的计算
9. 实现位置编码（Positional Encoding），包括正弦余弦位置编码的生成
10. 实现 Flash Attention 的核心算法，包括分块计算和在线 softmax
11. 实现 MoE 的专家路由算法，包括 Top-K 选择和负载均衡
12. 实现梯度裁剪（Gradient Clipping），包括按范数裁剪和按值裁剪
13. 实现学习率调度器，包括 Warmup、Cosine、Step 等调度策略
14. 实现权重初始化，包括 Xavier 和 He 初始化方法

#### Finetuning

1. 实现 LoRA 的前向和反向传播，包括低秩矩阵分解和参数更新
2. 实现 Adapter 层，包括下投影、上投影和残差连接
3. 实现 Prefix Tuning，包括可训练前缀的初始化和前向传播
4. 实现梯度累积逻辑，包括梯度累加和参数更新时机控制
5. 实现早停（Early Stopping）机制，包括验证指标监控和最佳模型保存
6. 实现学习率查找器（Learning Rate Finder），用于自动选择学习率

#### RL

1. 实现 REINFORCE 算法，包括策略梯度计算和参数更新
2. 实现 PPO 算法，包括 clipped objective 和重要性采样
3. 实现 GAE（Generalized Advantage Estimation），包括优势函数和回报计算
4. 实现奖励模型的训练逻辑，包括对比损失和偏好学习
5. 实现 KL 散度惩罚项，用于 RLHF 中的分布约束
6. 实现 DPO 损失函数，包括直接偏好优化的目标函数

#### Inference

1. 实现 KV Cache 的数据结构和管理逻辑，包括缓存更新和内存优化
2. 实现批量推理的批处理逻辑，包括动态批处理和填充处理
3. 实现流式推理的生成器，包括 token 流式输出和缓冲管理
4. 实现模型量化的核心算法，包括 INT8、INT4 量化和反量化
5. 实现模型剪枝算法，包括结构化剪枝和非结构化剪枝
6. 实现 Speculative Decoding，包括草稿模型和验证逻辑
7. 实现连续批处理（Continuous Batching），包括请求队列管理和动态调度
8. 实现 PagedAttention 的内存管理，包括分页存储和内存分配

#### RAG

1. 实现 BM25 检索算法，包括词频、逆文档频率和得分计算
2. 实现向量检索功能，包括余弦相似度计算和 Top-K 检索
3. 实现混合检索（Hybrid Search），包括向量检索和关键词检索的融合和权重平衡
4. 实现文档分块（Chunking）算法，包括固定窗口、滑动窗口和语义分块
5. 实现 Rerank 功能，包括交叉编码器的推理和重排序
6. 实现 Query Expansion，包括查询重写和扩展词生成
7. 实现增量更新逻辑，包括新文档的索引更新和旧文档的删除

### Agent

#### Agent Framework

1. 实现 ReAct 框架的核心逻辑，包括推理和行动的循环
2. 实现 Agent 的状态机，包括状态转换和动作执行
3. 实现 Agent 的记忆系统，包括短期记忆和长期记忆的管理
4. 实现多 Agent 系统的通信机制，包括消息传递和协调逻辑
5. 实现 Agent 的规划算法，包括任务分解和步骤生成

#### Function Call

1. 实现 Function Call 的参数解析器，包括 JSON Schema 验证和类型转换
2. 实现 Function Call 的上下文管理器，支持多轮对话的上下文维护
3. 实现 Function Call 的链式调用逻辑，包括依赖关系解析和执行顺序
4. 实现 Function Call 的容错机制，包括错误捕获和重试策略
5. 实现 Function Call 的异步调用框架，包括并发控制和结果聚合

#### Skills

### Memory

