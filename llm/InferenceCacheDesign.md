# LLM推理缓存设计

## 1. 核心定性

本质上，**推理缓存**是一个为了解决大模型自回归生成中重复计算冗余问题，通过 KV-Cache 机制实现的内存换计算优化结构。

---

## 2. 具体流程

1. **缓存初始化**：首次解码时计算并存储每一层的 Key/Value 矩阵，形状为 `[batch_size, num_heads, seq_len, head_dim]`

2. **增量复用**：后续每个 token 生成时，仅计算当前 token 的 QKV，将新的 KV 拼接到缓存，避免重新计算历史上下文

3. **内存管理**：当序列长度超过预设阈值或显存不足时，按策略（LRU/滑动窗口）淘汰过期缓存页

---

## 3. 数学基础

### KV-Cache 内存计算公式

$$\text{Memory}_{\text{KV}} = 2 \times b \times l \times h \times d \times p$$

其中：
- $b$: batch size（批次大小）
- $l$: layer num（模型层数）
- $h$: num heads（注意力头数）
- $d$: head dim（每个头的维度，$d = d_{\text{model}} / h$）
- $p$: precision bytes（精度字节数，FP16=2, FP32=4）
- 系数 $2$: Key 和 Value 两份缓存

**示例**：LLaMA-7B（$l=32, h=32, d=128$），batch=1，FP16，序列长度 $s=4096$：

$$\text{Memory} = 2 \times 1 \times 32 \times 32 \times 128 \times 2 = 512\,\text{MB}$$

### 计算复杂度对比

| 模式 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 无缓存 | $O(s^3 \cdot d)$ | $O(1)$ |
| **KV-Cache** | $O(s \cdot d^2)$ | $O(s \cdot d \cdot l)$ |

---

## 4. 工程考量

### Trade-off
| 维度 | 牺牲 | 换取 |
|------|------|------|
| **空间 vs 时间** | 显存占用随序列长度线性增长 | 计算量从 $O(s^2)$ 降至 $O(s)$ |
| **精度 vs 效率** | 低精度量化（INT8/INT4）引入误差 | 显存减半，吞吐提升 |

### 致命弱点
1. **长文本爆炸**：当 $s > 100K$ 时，KV-Cache 显存占用超过模型权重本身，成为瓶颈
2. **动态序列惩罚**：多 batch 推理时，短序列需 padding 等待长序列，造成显存浪费和计算空洞
3. **分页碎片**：非连续内存分配导致显存碎片化，OOM 风险陡增

---

## 5. 工业映射

| 技术方案 | 工业实现 | 核心机制 |
|----------|----------|----------|
| **PagedAttention** | vLLM | 将 KV-Cache 分页管理，支持动态扩缩容和非连续存储，解决碎片化 |
| **Continuous Batching** | TensorRT-LLM, TGI | 动态调度新请求加入当前 batch，最大化 GPU 利用率 |
| **Multi-Query Attention (MQA)** | PaLM, Falcon | 所有头共享同一套 KV-Cache，显存降至 $1/h$ |
| **Group-Query Attention (GQA)** | LLaMA-2/3 | 折中方案：$h$ 个头分 $g$ 组共享 KV，平衡质量和效率 |
| **KV-Cache 量化** | AutoGPTQ, AWQ | INT8/INT4 压缩缓存，配合反量化动态计算 |
| **Prefix Caching** | SGLang, vLLM | 对共享前缀（如 system prompt）缓存复用，避免重复计算 |

---

## 缓存命中率提升策略

1. **语义级去重**：维护 Prompt 哈希表，相同/相似输入直接复用完整 KV-Cache（如 RAG 场景的固定 system prompt）

2. **前缀树匹配**：将历史序列组织为 Trie 树，新请求按最长公共前缀匹配缓存（Prefix Caching）

3. **滑动窗口 + 粗粒度淘汰**：限制单序列最大缓存长度，LRU 淘汰低频请求，保留热点上下文

4. **投机解码（Speculative Decoding）**：用小模型草稿 + 大模型验证，验证命中时复用缓存，加速 2-3x

5. **Early Exit 缓存分层**：浅层注意力输出缓存用于相似度快速过滤，深层缓存按需加载

**一句话总结**：推理缓存的本质是**用可控的显存膨胀换取指数级计算削减**，工业界通过分页管理、注意力结构改造和请求级调度将这一 trade-off 推向极致。
