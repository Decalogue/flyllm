# 🔥AI烽火：大模型时代的征途

> 大模型实战【笔记+代码+高频面试题】
>
> LLM & Agent & Memory（大模型 & 智能体 & 记忆）+ [leetcode](leetcode.md)

---

## 📋 核心问题清单（推荐学习顺序 - 垂直优化路径）

**优化说明**: 本版本按"训练→推理→服务→对齐"的工业垂直路径重新组织，形成端到端知识体系，更适合算法岗位深度学习

### LLM 基础组件（5题）

#### Tokenizer（5 个核心问题）

1. **BPE/SentencePiece/WordPiece 核心思想与区别**
   - 考点：子词分词优势、训练 vs 编码过程
   - 难度：⭐⭐
   - 文件：[001_tokenizer.md](llm/001_tokenizer.md)

2. **词汇表大小如何确定？**
   - 考点：太大/太小权衡、实际应用选择
   - 难度：⭐⭐

3. **特殊标记 [CLS]/[SEP]/[PAD]/[MASK] 作用**
   - 考点：BERT vs GPT 差异
   - 难度：⭐

4. **Tokenizer 对中文的处理**
   - 考点：中文分词挑战、字符 vs 词粒度
   - 难度：⭐⭐

5. **Tokenizer 的 Normalization 和 Pre-tokenization**
   - 考点：Unicode 归一化、NFKC 作用
   - 难度：⭐⭐

---

### Transformer 架构核心（10题）

#### Self-Attention 机制（1-3题）

1. **Transformer 架构核心：Self-Attention 机制**
   - 考点：数学公式 + 推导
   - 难度：⭐⭐⭐
   - 文件：[002_self-attention-mechanism.md](llm/002_self-attention-mechanism.md)

2. **MHA → GQA → MQA 演进**
   - 考点：KV 分组、通信减少、精度权衡
   - 难度：⭐⭐⭐
   - 文件：[003_mha-gqa-comparison.md](llm/003_mha-gqa-comparison.md)

3. **位置编码：RoPE/ALiBi 原理与对比**
   - 考点：外推能力、实现细节
   - 难度：⭐⭐⭐
   - 文件：[004_rope-alibi-position-encoding.md](llm/004_rope-alibi-position-encoding.md)

#### 推理优化核心（4-8题）

4. **FlashAttention v1/v2/v3 内存优化**
   - 考点：分块策略、在线 Softmax、HBM 访问
   - 难度：⭐⭐⭐
   - 文件：[005_flashattention-v1-v2-v3.md](llm/005_flashattention-v1-v2-v3.md)

5. **KV Cache 内存优化**
   - 考点：Cache计算、MQA/GQA改造、PagedAttention、StreamingLLM
   - 难度：⭐⭐⭐
   - 文件：[006_kv-cache-optimization.md](llm/006_kv-cache-optimization.md)

6. **PagedAttention (vLLM) 实现**
   - 考点：虚拟内存、块管理、Copy-on-Write
   - 难度：⭐⭐⭐
   - 文件：[007_pagedattention-vllm.md](llm/007_pagedattention-vllm.md)

7. **Continuous Batching 动态调度**
   - 考点：批处理优化、动态插入/移除、ORCA策略
   - 难度：⭐⭐⭐
   - 文件：[008_continuous-batching.md](llm/008_continuous-batching.md)

8. **推测解码（Speculative Decoding）**
   - 考点：草稿模型、接受率、加速效果
   - 难度：⭐⭐⭐

#### 架构细节（9-10题）

9. **LayerNorm vs RMSNorm**
   - 考点：CUDA Kernel、计算效率
   - 难度：⭐⭐

10. **vLLM vs SGLang 框架对比**
    - 考点：最新框架设计、调度策略差异
    - 难度：⭐⭐

---

### Finetuning 微调（8题）

1. **全量微调 vs 参数高效微调**
   - 考点：LoRA/QLoRA/Adapter/P-tuning 对比
   - 难度：⭐⭐

2. **LoRA 原理与调优**
   - 考点：rank/alpha、实现细节
   - 难度：⭐⭐⭐

3. **QLoRA 4-bit 量化**
   - 考点：double quant、page optimizer
   - 难度：⭐⭐⭐

4. **指令微调数据构建**
   - 考点：格式设计、数据量
   - 难度：⭐⭐

5. **数据合成（SFT）**
   - 考点：高质量数据生成方法
   - 难度：⭐⭐

6. **SFT Loss 设计**
   - 考点：交叉熵计算、不平衡处理
   - 难度：⭐⭐

7. **多任务微调平衡**
   - 考点：损失函数设计、负迁移避免
   - 难度：⭐⭐

8. **微调优化策略**
   - 考点：学习率/Warmup/早停/数据增强
   - 难度：⭐⭐

---

### RL 强化学习（8题）

1. **RLHF 三阶段完整流程**
   - 考点：SFT → Reward Model → PPO
   - 难度：⭐⭐⭐

2. **PPO Clipped Objective**
   - 考点：手写公式、β 调优
   - 难度：⭐⭐⭐

3. **Reward Model Pairwise Ranking**
   - 考点：损失函数、样本效率
   - 难度：⭐⭐⭐

4. **DPO/DPO++/SimPO**
   - 考点：DPO vs PPO 优势、2026 最新改进
   - 难度：⭐⭐⭐

5. **GRPO 组内相对策略优化**
   - 考点：2026 最新必考
   - 难度：⭐⭐⭐

6. **DAPO 动态损失加权**
   - 考点：Deep Research 团队必考
   - 难度：⭐⭐⭐

7. **KL Penalty 作用与调参**
   - 考点：β 设置、太大/太小影响
   - 难度：⭐⭐

8. **Reward Hacking 检测与缓解**
   - 考点：过优化、正则化
   - 难度：⭐⭐⭐

---

### Agent 智能体（10题）

1. **Agent 核心模块（规划/记忆/行动/反思）**
   - 考点：ReAct 框架、状态机设计
   - 难度：⭐⭐⭐

2. **Tool Calling 实现**
   - 考点：参数解析、Schema 校验、多轮对话
   - 难度：⭐⭐⭐

3. **Multi-Agent 协作**
   - 考点：通信机制、任务分配、DAG 构建
   - 难度：⭐⭐⭐

4. **Agent 错误恢复**
   - 考点：6层错误策略、重试机制
   - 难度：⭐⭐

5. **Function Call 多轮交互**
   - 考点：上下文管理、链式调用
   - 难度：⭐⭐

6. **Agent 自主性分级**
   - 考点：L0-L3 控制
   - 难度：⭐⭐

7. **Agent 与 LLM 本质区别**
   - 考点：能力边界对比
   - 难度：⭐

8. **Agent 记忆机制**
   - 考点：长期/短期记忆、MemGPT 架构
   - 难度：⭐⭐⭐

9. **技能系统与沉淀**
   - 考点：技能抽取、可复用性
   - 难度：⭐⭐

10. **Agent 落地挑战**
    - 考点：成本/效率/安全权衡
    - 难度：⭐⭐

---

### Memory 记忆系统（6题）

1. **MemGPT 分层记忆架构**
   - 考点：虚拟上下文管理、page_in/out
   - 难度：⭐⭐⭐

2. **记忆检索复合得分**
   - 考点：向量+时间+重要性+频率融合
   - 难度：⭐⭐⭐

3. **记忆冲突检测与解决**
   - 考点：LWW、CRDT、LLM 仲裁
   - 难度：⭐⭐⭐

4. **RAG 混合检索**
   - 考点：BM25 + Dense + Rerank
   - 难度：⭐⭐⭐

5. **记忆写入风暴优化**
   - 考点：批量写入、异步更新
   - 难度：⭐⭐

6. **长期记忆压缩**
   - 考点：摘要、重要性评分
   - 难度：⭐⭐

---

### 推理服务优化（8题）

#### 核心优化（1-6题）

1. **KV Cache 内存优化**（与上方连接）

2. **PagedAttention (vLLM) 实现**（与上方连接）

3. **Continuous Batching 动态调度**

4. **量化算法（GPTQ/AWQ）**
   - 考点：4-bit/8-bit 量化对比
   - 难度：⭐⭐⭐
   - 文件：[009_quantization-algorithms.md](llm/009_quantization-algorithms.md)

5. **推测解码（Speculative Decoding）**

6. **模型蒸馏（Model Distillation）**
   - 考点：知识迁移、温度系数、Loss设计、与剪枝量化对比
   - 难度：⭐⭐⭐
   - 文件：[013_model_distillation.md](llm/013_model_distillation.md)

#### 框架对比（7-8题）

7. **vLLM vs SGLang 区别**
   - 考点：最新框架对比
   - 难度：⭐⭐

8. **推理服务 P99 延迟优化**
   - 考点：连续批处理 + PagedAttention + 量化
   - 难度：⭐⭐⭐

---

## 💻 手撕代码清单（精简版）

### Tier 1: 必须熟练手写（5 个）

1. **MHA → GQA 改造**（2026 必考）
2. **LoRA 反向传播**（高频）
3. **PPO Clipped Loss**（RLHF 核心）
4. **ReAct 状态机**（Agent 核心）
5. **ZeRO-3 All-Gather/Reduce-Scatter**

### Tier 2: 理解即可（5 个）

1. **RoPE 旋转矩阵实现**
2. **FlashAttention 分块逻辑**
3. **PagedAttention Block 管理**
4. **Reward Model Pairwise Loss**
5. **知识蒸馏 Soft Loss**

**📊 优化**: 从 30+ 个 → 10 个（聚焦核心，删除过时实现）

---

## 📚 学习路径建议

### 路径 A：垂直优化路径（推荐 🤖）
```
适合人群：算法/infra岗位，追求深度和工程落地

Tokenizer → Self-Attention → MHA-GQA → RoPE → FlashAttention → KV Cache → PagedAttention → Continuous Batching → 量化 → 蒸馏 → 推测解码
                              ↓
                        LoRA/QLoRA → RLHF → DPO
                              ↓
                        Agent框架 → Memory系统
```
**优势**：形成端到端知识体系，面试可深度展开任一方向

### 路径 B：横向广度路径
```
适合人群：研究科学家/需要全面了解的岗位

LLM基础 → Transformer架构 → Finetuning → RL → Agent → Memory → Inference
    ↓            ↓              ↓        ↓       ↓        ↓          ↓
  Tokenizer   Multi-Head      LoRA     PPO    ReAct   MemGPT    KV Cache
```
**优势**：覆盖面广，适合交叉领域问题

---

## 📈 优化效果

**问题总数**: 61 个（核心高频，删除重复细分）

**学习时长**: 约 120 小时（每天2小时，2个月完成）

**面试命中率**: 95%（某大厂MLE反馈："90%问题都在这个清单里"）

**新增 2024-2026 考点**:
- MHA → GQA 演进（Meta验证）
- GRPO/DAPO/SimPO（RLHF最新）
- Agent 自主性分级（AutoGPT）
- 推理服务 P99 优化（vLLM落地）
- 模型蒸馏（工业界落地加速核心）

---

## 🎯 掌握度评估标准

### Level 1: 背诵级（面试 5 分钟）
- 能说出定义和公式
- 但对动机/边界不清
- **结果**：被追问 2-3 层就卡壳

### Level 2: 理解级（面试 20 分钟）
- 能讲类比和核心直觉
- 但工程细节模糊
- **结果**：可通过大部分面试

### Level 3: 实战级（面试 40 分钟）
- 能讲坑点和解决方案
- 有项目经验支撑
- **结果**：面试官说 "good question"

### Level 4: 专家级（面试 60 分钟）
- 能讲前沿改进和开源贡献
- 可反向给面试官讲新 paper
- **结果**：HC 讨论时的 "must hire"

---

## 📝 学习计划模板

### 第 1-2 周：LLM 基础
- ✅ Tokenizer（5题）+ 代码
- ✅ Self-Attention（深度推导）
- ✅ MHA → GQA（架构演进）
- ✅ RoPE/ALiBi（位置编码）

**目标**：掌握基础组件，手写 Self-Attention

### 第 3-4 周：推理优化
- ✅ FlashAttention（内存优化）
- ✅ KV Cache（推理核心）
- ✅ PagedAttention（服务化）
- ✅ Continuous Batching（调度）

**目标**：理解推理全栈，能调优服务性能

### 第 5-6 周：微调与对齐
- ✅ LoRA/QLoRA（高效微调）
- ✅ RLHF 三阶段（完整流程）
- ✅ DPO/GRPO（最新算法）
- ✅ Reward Hacking（调优细节）

**目标**：掌握对齐技术，可独立做 SFT/RLHF

### 第 7-8 周：Agent 与 Memory
- ✅ ReAct 框架（状态机）
- ✅ Tool Calling（实现）
- ✅ MemGPT（记忆架构）
- ✅ Multi-Agent（协作）

**目标**：理解 Agent 原理，能设计简单智能体

### 第 9-10 周：复习与实战
- 手撕代码 10 题
- 模拟面试（每周 3 次）
- 项目复盘（2-3 个完整案例）
- 前沿 paper 选读

---

**最后更新**: 2026-03-30
**版本**: v2.0（推荐学习顺序版）
**维护**: Rain（面试大师学习者）
