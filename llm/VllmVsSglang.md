# vLLM 和 SGLang 的区别

### 1. 核心定性
本质上，**vLLM** 是基于 PagedAttention 解决单次请求显存碎片的**通用高吞吐推理引擎**；而 **SGLang** 是基于 RadixAttention 实现跨请求前缀缓存的**结构化/Agent 专用加速栈**。

### 2. 具体流程
1. **vLLM** 借鉴操作系统虚拟内存，将 KV Cache 切分为固定大小的物理块（Block），通过分页表按需分配，消除显存外部碎片。
2. **SGLang** 后端将历史请求的 KV Cache 组织为 Radix Tree（基数树），自动在全局并发请求间匹配并复用最长公共前缀。
3. 面对系统提示词共享、多轮对话或 JSON 约束生成时，SGLang 调度器直接在基数树上执行状态转移并结合 LRU 驱逐，而纯 vLLM 架构（非 Prefix Caching 模式下）需重复进行沉重的 Prefill 计算。

### 3. 数学基础
**vLLM 核心机制：分页注意力 (PagedAttention)**
$PhysicalAddr = PageTable \left\lfloor \frac{LogicalAddr}{B} \right\rfloor \cdot B + (LogicalAddr \bmod B)$
其中：
- $B$: Block Size（每个物理块包含的 Token 数，通常为 16 或 32）
- $PageTable$: 请求专属的逻辑到物理块映射表
- 空间浪费被严格限制在最后一个不完整的 Block（内部碎片 $< B$）。

**SGLang 核心机制：基数树注意力 (RadixAttention)**
前缀命中长度计算：
$CacheHit(Q) = \max_{p \in RadixTree} length(LCP(Q, p))$
其中：
- $Q$: 当前新请求的 Token 序列
- $p$: Radix Tree 中节点维护的历史 Token 序列
- $LCP$: 最长公共前缀 (Longest Common Prefix) 函数
- $CacheHit(Q)$: 可以免于前向传播（Prefill）直接复用的 KV Cache 长度。

### 4. 工程考量
- **Trade-off**: vLLM 牺牲了多请求间细粒度的前缀感知与状态重用，换取了极低开销的调度器与稳定的批处理性能；SGLang 牺牲了系统复杂度（需维护全局树状锁、引用计数与 LRU 驱逐机制），换取了多轮交互和 Few-shot 场景下的极限加速。
- **致命弱点**: 
  - **vLLM**: 在 Multi-Agent 或长篇 Few-shot 场景并发下，相同的长前缀会被重复计算和存储，导致计算资源与显存严重浪费。
  - **SGLang**: 在请求载荷呈现**绝对零交集**（完全随机的独立长文本、无共同 Prompt）且高并发时，Radix Tree 的匹配开销与频繁的 LRU 驱逐会导致调度器开销飙升，退化至不如 vLLM。

### 5. 工业映射
在工业界，**vLLM** 被广泛作为基础大模型 CaaS（Model-as-a-Service）的**通用底层基座**支撑海量并发；而 **SGLang** 被直接应用于 **AutoGPT、复杂 RAG 链路、批量结构化提取（JSON 模式）** 等极度依赖 Prompt 复用与低 TTFT（首字延迟）的进阶架构中。