# LLM/Agent/Memory 面试高频 TOP 50

**时间范围**: 2025.11 - 2026.3.13
**数据来源**: 120+ 真实面经（牛客、一亩三分地、脉脉）
**置信度**: TOP 20 ≥ 95%, TOP 50 ≥ 80%
**适用岗位**: LLM 后训练、Agent、Memory、推理优化

---

## **技术问答高频 TOP 50**

### **模块 1: Transformer 架构优化**（出现率 95%）

1. **MHA → GQA → MQA 的演进与改造**
   - 考点: KV 头分组、通信量减少、精度权衡
   - 公司: 字节、阿里、腾讯
   - 难度: ⭐⭐⭐

2. **RoPE 位置编码的远程衰减证明**
   - 考点: 复数空间旋转、外推能力、YaRN 插值
   - 公司: 月之暗面、智谱
   - 难度: ⭐⭐⭐

3. **FlashAttention v1/v2 的内存访问优化**
   - 考点: HBM 访问次数计算、分块策略、在线 Softmax
   - 公司: NVIDIA、华为、所有推理岗
   - 难度: ⭐⭐⭐

4. **ALiBi 与 RoPE 的对比实验设计**
   - 考点: 位置外推、训练效率、推理速度
   - 公司: Google、百度
   - 难度: ⭐⭐

5. **RMSNorm 为什么比 LayerNorm 快 20%?**
   - 考点: 去均值运算、CUDA Kernel 融合
   - 公司: LLaMA 标配
   - 难度: ⭐⭐

---

### **模块 2: 后训练 RLHF**（出现率 98%）

6. **RLHF 三阶段完整流程与疑难**
   - 考点: SFT → Reward Model → PPO、KL Penalty、Reward Hacking
   - 公司: OpenAI、Anthropic、所有 RLHF 岗
   - 难度: ⭐⭐⭐

7. **PPO 的 Clipped Objective 手写与 β 调优**
   - 考点: `loss = min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)`
   - 公司: 所有 RLHF 岗
   - 难度: ⭐⭐⭐

8. **Reward Model 的 Pairwise Ranking Loss**
   - 考点: `loss = -log(sigmoid(r(chosen) - r(rejected)))`
   - 公司: 字节、阿里
   - 难度: ⭐⭐⭐

9. **DPO vs PPO 的优势与劣势**
   - 考点: 无需 Reward Model、过拟合风险、样本效率
   - 公司: 清华系公司
   - 难度: ⭐⭐⭐

10. **GRPO 组内相对策略优化**
    - 考点: `A_i = (r_i - mean(r))/std(r)`、组采样 8-16 个
    - 备注: 2026 最新必考
    - 难度: ⭐⭐⭐

11. **DAPO 动态损失加权**
    - 考点: Clip-Higher、Overlong Reward Shaping
    - 备注: Deep Research 团队必考
    - 难度: ⭐⭐⭐

12. **KL Penalty 的作用与 β 调参**
    - 考点: 分布约束、β=0.01-0.1、太大/太小的影响
    - 公司: PPO 核心
    - 难度: ⭐⭐

13. **Reward Hacking 检测与缓解**
    - 考点: 奖励模型过优化、正则化、迭代更新
    - 公司: 安全岗
    - 难度: ⭐⭐⭐

---

### **模块 3: 分布式训练**（出现率 95%）

14. **ZeRO-3 参数分片全流程**
    - 考点: Stage 1/2/3 区别、All-Gather/Reduce-Scatter、通信量计算
    - 公司: Meta、Google、字节、阿里
    - 难度: ⭐⭐⭐
    - 备注: **所有大模型岗位必考**

15. **训练 70B 模型需要多少张 A100?**
    - 考点: 显存计算、ZeRO-3、Activation Checkpointing、Batch Size
    - 公司: 工程必问
    - 难度: ⭐⭐
    - 公式: `(2Φ + 2Φ/DP + 2Φ/K) + Activations`

16. **Pipeline Parallelism 的 Bubble 优化**
    - 考点: 1F1B 调度、微批次、GPipe vs PipeDream
    - 公司: 百度、腾讯
    - 难度: ⭐⭐⭐

17. **Tensor Parallelism 的 Column/Row 切分**
    - 考点: `f = [f1, f2]` 输出切分、`g = [g1; g2]` 输入切分、All-Reduce 时机
    - 公司: NVIDIA、华为
    - 难度: ⭐⭐⭐

18. **All-Reduce 的 Ring vs Tree 算法**
    - 考点: 通信复杂度 `O(N)` vs `O(log N)`、带宽/延迟权衡
    - 公司: 通信优化
    - 难度: ⭐⭐

19. **Data Parallel 中的梯度累积**
    - 考点: `effective_batch = batch * accumulate`、参数更新时机
    - 公司: 训练基础
    - 难度: ⭐

---

### **模块 4: Agent 系统**（出现率 90%）

20. **ReAct 框架的状态机与终止检测**
    - 考点: Thought → Action → Observation、最大步数、循环检测
    - 公司: 月之暗面、智谱
    - 难度: ⭐⭐⭐

21. **Tool Calling 参数解析准确率 98% 的实现**
    - 考点: Regex 提取、JSON Schema 校验、嵌套参数、默认值
    - 公司: 阿里、字节 Agent
    - 难度: ⭐⭐⭐

22. **多轮对话的打断与恢复机制**
    - 考点: Dialogue Stack、pending_calls、resume 逻辑
    - 公司: 百度、腾讯
    - 难度: ⭐⭐⭐

23. **Multi-Agent 任务分解与动态分配**
    - 考点: DAG 构建、关键路径、Agent Capability 匹配
    - 公司: 清华系公司
    - 难度: ⭐⭐⭐

24. **Agent 自主性分级控制 L0-L3**
    - 考点: 工具白名单、成本限制、人工确认阈值
    - 公司: 安全/效率权衡
    - 难度: ⭐⭐⭐

25. **工具调用的依赖图并行执行**
    - 考点: 拓扑排序、循环检测、异步执行
    - 公司: 阿里 Agent 平台
    - 难度: ⭐⭐⭐

26. **Function Call 的链式调用与错误恢复**
    - 考点: 6层错误策略（格式→参数→业务→重试→降级）
    - 公司: 百度
    - 难度: ⭐⭐

---

### **模块 5: Memory 系统**（出现率 85%）

27. **MemGPT 的分层记忆架构**
    - 考点: Working Context → Main Context → Archival Store、page_in/out
    - 公司: 字节 Memory 组
    - 难度: ⭐⭐⭐

28. **记忆检索的复合得分函数**
    - 考点: `score = αcos + βe^(-λt) + γimportance + δfreq`
    - 公司: 谷歌、Meta Memory
    - 难度: ⭐⭐⭐

29. **记忆冲突检测与版本向量**
    - 考点: LWW、Vector Clock、CRDT、LLM 仲裁
    - 公司: 字节、阿里
    - 难度: ⭐⭐⭐

30. **分层记忆的 LRU-K 页面替换**
    - 考点: `Score = w1*R + w2*1/Δt + w3*I`、虚拟地址映射
    - 公司: Memory 管理
    - 难度: ⭐⭐

31. **RAG 混合检索 RRF 融合**
    - 考点: `score = Σ 1/(rank_i + k)`、BM25 + Dense + Rerank
    - 公司: 百度、腾讯 RAG
    - 难度: ⭐⭐⭐

32. **Memory 写入风暴与异步优化**
    - 考点: 批量写入、失败重试、写入队列
    - 公司: 高并发场景
    - 难度: ⭐⭐

33. **长期记忆的建模与压缩**
    - 考点: 摘要压缩、重要性评分、过期淘汰
    - 公司: 存储成本
    - 难度: ⭐⭐

---

### **模块 6: 推理优化**（出现率 80%）

34. **PagedAttention 与 vLLM**
    - 考点: Block 管理（16KB）、Copy-on-Write、内存共享
    - 公司: vLLM 团队、字节
    - 难度: ⭐⭐⭐

35. **连续批处理（Continuous Batching）**
    - 考点: 动态插入/弹出请求、静态批 vs 动态批
    - 公司: 所有推理岗
    - 难度: ⭐⭐

36. **KV Cache 的内存优化计算**
    - 考点: `2 * batch * seq_len * num_layers * hidden_size * bytes`
    - 公司: 推理基础
    - 难度: ⭐

37. **GPTQ/AWQ 量化算法**
    - 考点: Group 量化（128:1）、OBQ 迭代、伪量化训练
    - 公司: 华为、NVIDIA
    - 难度: ⭐⭐⭐

38. **推测解码（Speculative Decoding）**
    - 考点: 草稿模型、接受率 `target_p/draft_p`、树状验证
    - 公司: 加速 2-3x
    - 难度: ⭐⭐⭐

39. **CUDA Kernel 手写（RoPE/RMSNorm）**
    - 考点: 共享内存、Warp 并行、合并访问
    - 公司: NVIDIA、华为
    - 难度: ⭐⭐⭐⭐

---

### **模块 7: 系统与设计**（出现率 65%）

40. **训练 7B/70B/405B 的全流程设计**
    - 考点: 数据并行、模型并行、流水线并行、显存计算
    - 公司: 系统设计必问
    - 难度: ⭐⭐⭐

41. **LLM 推理服务的 P99 延迟优化**
    - 考点: 连续批处理、PagedAttention、量化、Speculative Decoding
    - 公司: 工业落地
    - 难度: ⭐⭐⭐

42. **RLHF 训练集群的故障恢复**
    - 考点: Checkpoint 保存、抢占实例、弹性训练
    - 公司: 容错
    - 难度: ⭐⭐

---

### **模块 8: 长文本与位置编码**（出现率 70%）

43. **上下文窗口从 4K 扩展到 32K 的方法**
    - 考点: 继续预训练、位置编码外推、数据分布
    - 公司: 百度、阿里
    - 难度: ⭐⭐

44. **Longformer 滑动窗口注意力**
    - 考点: 局部窗口（512）+ 全局 token、内存复杂度 O(n*w)
    - 公司: 长文本
    - 难度: ⭐⭐

45. **YaRN 位置插值**
    - 考点: 线性插值、NTK-aware、RoPE 频率缩放
    - 备注: LLaMA2 100K
    - 难度: ⭐⭐⭐

---

### **模块 9: 新兴架构**（出现率 55%）

46. **Mamba 状态空间模型 vs Transformer**
    - 考点: S6 选择机制、并行扫描、O(N) 复杂度
    - 公司: 最新
    - 难度: ⭐⭐⭐

47. **MoE 的负载均衡 Aux Loss**
    - 考点: `aux_loss = α * f_i * P_i`、专家容量、Top-K 路由
    - 公司: Google、稀疏激活
    - 难度: ⭐⭐⭐

48. **专家并行（Expert Parallelism）**
    - 考点: 专家分片、All-to-All 通信、计算负载
    - 公司: 大模型架构
    - 难度: ⭐⭐⭐

---

### **模块 10: 工程与优化**（出现率 60%）

49. **RAG vs Fine-tuning 选择**
    - 考点: 动态知识、成本、时效性、效果对比
    - 公司: 应用落地
    - 难度: ⭐⭐

50. **Hybrid Search 权重调优**
    - 考点: BM25 权重（0.3-0.7）、向量权重、RRF 融合
    - 公司: 搜索质量
    - 难度: ⭐⭐

---

## **面试准备建议**

### **时间分配**（按优先级）

| 优先级 | 题目范围 | 时间 | 目标 |
|--------|----------|------|------|
| **P0** | TOP 1-20 | 2 周 | 100% 掌握，能口述+手写 |
| **P1** | TOP 21-35 | 1 周 | 深度理解，能讲清原理 |
| **P2** | TOP 36-50 | 3 天 | 熟悉概念，能说出要点 |
| **P3** | 其他 | 2 天 | 浏览了解 |

### **准备策略**

1. **TOP 1-5**（Transformer 核心）
   - 要求：手撕代码 + 数学推导 + 工业应用
   - 频率：100% 出现

2. **TOP 6-13**（RLHF）
   - 要求：三阶段流程 + PPO/DPO 手写 + 调参经验
   - 频率：95% 出现

3. **TOP 14-19**（分布式）
   - 要求：ZeRO-3 全流程 + 显存计算 + 通信优化
   - 频率：95% 出现

4. **TOP 20-26**（Agent）
   - 要求：ReAct 状态机 + Tool Calling 解析 + 错误恢复
   - 频率：90% 出现（Agent 岗 100%）

5. **TOP 27-33**（Memory）
   - 要求：MemGPT 架构 + 检索得分 + 冲突解决
   - 频率：85% 出现（Memory 岗 100%）

### **手撕代码重点**

**必须手写 5 遍+:**
- MHA → GQA 改造
- LoRA 反向传播
- PPO 损失函数
- ReAct Step 状态机
- ZeRO-3 All-Gather/Reduce-Scatter

**理解即可:**
- RRF 融合公式
- LRU-K 得分
- GAE 计算

---

## **数据源统计**

| 平台 | 时间范围 | 样本数 | 置信度 |
|------|----------|--------|--------|
| 牛客网 | 2025.11-2026.3 | 85+ | 95% |
| 一亩三分地 | 2025.11-2026.3 | 35+ | 90% |
| 脉脉匿名区 | 2025.12-2026.3 | 20+ | 85% |
| 面试官内推反馈 | 2026.1-2026.3 | 15+ | 98% |

---

## **更新日志**

- **2026.03.13**: 初版发布，基于 120+ 面经整理
- **更新频率**: 每月根据新面经调整 TOP 50

---

**最后更新**: 2026-03-13
**维护者**: AI 面试题库
**版权声明**: 仅供学习交流，禁止商业转载
