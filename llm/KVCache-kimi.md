# KV Cache：大模型推理加速的核心杠杆

## 1. 核心定性

**本质上，KV Cache 是为了解决自回归生成中重复计算历史 token 的注意力键值对问题，通过空间换时间策略，将已计算的前缀 KV 张量缓存复用的内存驻留结构。**

---

## 2. 具体流程

1. **Prefill 阶段**：对输入 prompt 做完整前向传播，计算每层的 Q/K/V，**仅保留每层的 K、V 矩阵**（丢弃 Q 和输出），存入缓存张量。
2. **Decode 阶段**：每生成一个新 token，只计算当前 token 的 q，从缓存中读取历史 K/V，拼接后执行注意力：`Attention(q, [K_cache; k_new], [V_cache; v_new])`。
3. **增量更新**：将当前步计算出的新 k、v 追加写入缓存，缓存长度随生成 token 数线性增长。

---

## 3. 数学基础

### 3.1 标准自注意力 vs KV Cache 注意力

**标准自注意力（无缓存）—— 每步复杂度 $O(n^2)$：**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q, K, V \in \mathbb{R}^{n \times d_k}$，$n$ 为序列长度，每步都重新计算全部 KV。

**KV Cache 增量注意力 —— 每步复杂度 $O(n)$：**
$$\text{Attention}(q_t, K_{\leq t}, V_{\leq t}) = \text{softmax}\left(\frac{q_t [K_{\leq t}]^T}{\sqrt{d_k}}\right)[V_{\leq t}]$$

其中：
- $q_t \in \mathbb{R}^{1 \times d_k}$：当前 token 的 query 向量
- $K_{\leq t} \in \mathbb{R}^{t \times d_k}$：缓存的历史 keys（含当前步新生成的 $k_t$）
- $V_{\leq t} \in \mathbb{R}^{t \times d_k}$：缓存的历史 values
- 计算复杂度从 $O(n^2 \cdot d)$ 降至 $O(n \cdot d)$

### 3.2 显存占用公式

$$\text{KV Cache Size} = 2 \times \text{num\_layers} \times \text{num\_heads} \times \text{head\_dim} \times \text{seq\_len} \times \text{batch\_size} \times \text{bytes\_per\_param}$$

以 LLaMA-2-7B 为例（bf16）：
- 32 层 × 32 头 × 128 维 × 4096 长度 × 2 字节 ≈ **2GB / sequence**

---

## 4. 工程考量

| 维度 | 收益 | 代价 | 致命弱点 |
|------|------|------|----------|
| **时间** | 解码阶段从 $O(n^2)$ 降至 $O(n)$，长文本生成加速 **10-100x** | Prefill 阶段仍需 $O(n^2)$，首 token 延迟不变 | **前缀长度爆炸**：超长上下文（100K+）导致 cache 内存占用超过 GPU 显存 |
| **空间** | 单次计算终身复用 | 显存占用与序列长度线性增长 | **batch 规模受限**：大 batch + 长序列时，显存成为瓶颈，需 paging/quantization |
| **精度** | 支持 fp16/bf16 正常推理 | 无 | **量化副作用**：INT8/INT4 KV cache 虽省显存，但累积误差可能导致长文本生成质量漂移 |

**核心 Trade-off 公式：**
$$\text{加速比} \approx \frac{n}{\text{memory\_bandwidth\_pressure}}$$

当 sequence length $n$ 增长时，显存带宽成为新瓶颈（memory-bound），而非计算。

---

## 5. 工业映射

- **vLLM**：提出 **PagedAttention**，将 KV cache 分页管理（类似 OS 虚拟内存），实现动态分配、零碎片、大 batch 并发。
- **TensorRT-LLM / HuggingFace**：原生支持 KV cache 量化（INT8/INT4）、**Multi-Query Attention (MQA)** / **Grouped-Query Attention (GQA)** 减少 KV 头数，直接降低显存占用。
- **FlashAttention**：融合 kernel 减少 HBM 访问，与 KV cache 协同解决 memory wall 问题。
- **StreamingLLM / H2O**：引入 **KV eviction 策略**（保留 sink tokens + 局部窗口），在无限长流式场景下控制显存上界。
