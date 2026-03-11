# 大模型上下文长度限制与扩展方法

---

## 1. 核心定性

本质上，上下文长度限制是 **Self-Attention $O(n^2)$ 计算/内存复杂度** 与 **位置编码外推能力** 双重约束的产物；扩展方法通过 **稀疏注意力、位置编码外推、上下文压缩** 三大技术路线突破物理显存瓶颈。

---

## 2. 具体流程

1. **瓶颈定位**：标准 Transformer 的 Attention 计算 $QK^T$ 产生 $O(n^2)$ 的内存与计算量，长文本导致显存指数级爆炸。
2. **扩展路径**：通过**稀疏化注意力矩阵**（降低计算量）、**改进位置编码**（支持外推更长序列）、**KV Cache 压缩/分页**（降低显存占用）三类方法突破限制。
3. **工业实践**：现代 LLM 采用 **RoPE + SwiGLU + GQA** 基础架构，结合 **Ring Attention、Sparse Attention、上下文压缩** 实现百万级 token 长窗口。

---

## 3. 数学基础

### 3.1 Self-Attention 复杂度推导

标准 Self-Attention 计算：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q, K, V \in \mathbb{R}^{n \times d}$（$n$: 序列长度, $d$: 隐藏维度）
- $QK^T$ 运算产生 **$O(n^2 \cdot d)$** 计算复杂度
- Attention Score 矩阵占用 **$O(n^2)$** 显存

**显存公式**（推理阶段 KV Cache）：

$$\text{Memory}_{KV} = 2 \times n \times d_{model} \times l \times \text{batch\_size} \times \text{bytes\_per\_param}$$

其中：
- $l$: 层数
- 系数 2 对应 K 和 V 两个矩阵

---

### 3.2 位置编码外推公式

**RoPE (Rotary Position Embedding)**：

$$f(q, m) = q \cdot e^{i \cdot m \cdot \theta_j}, \quad \theta_j = b^{-2j/d}$$

其中：
- $m$: 位置索引
- $\theta_j$: 旋转角度（频率）
- $b = 10000$（基数）

**长度外推难题**：训练长度 $L_{train}$ 外的位置 $m > L_{train}$ 会产生 **分布外 (OOD)** 旋转角，导致注意力熵增、模型崩溃。

**NTK-Aware 扩展**（动态调整基数）：

$$\theta_j' = (b \cdot \lambda)^{-2j/d}, \quad \lambda = \left(\frac{L_{target}}{L_{train}}\right)^{d/(d-2)}$$

---

### 3.3 稀疏注意力复杂度

**Sliding Window Attention**（局部窗口 $w$）：

$$\text{Complexity} = O(n \cdot w \cdot d) \ll O(n^2 \cdot d)$$

**Sparse Transformer / Longformer**（固定稀疏模式）：

$$\text{Complexity} = O(n \cdot \sqrt{n} \cdot d)$$

---

## 4. 工程考量

| 扩展方法 | Trade-off | 致命弱点 |
|---------|-----------|----------|
| **RoPE + NTK 扩展** | 零训练成本支持 2-8x 外推 | 超过 8x 后 ppl 激增，长程依赖捕捉失效 |
| **ALiBi 位置编码** | 天然外推性，推理长度不受限 | 短序列性能略低于 RoPE，斜率参数敏感 |
| **Sliding Window Attention** | 线性复杂度，适合长文档 | 丢失全局依赖，"隔山打牛"类任务失败 |
| **Ring Attention / 序列并行** | 分布式切分序列，理论无限长 | 通信开销大，GPU 数量成为新瓶颈 |
| **KV Cache 压缩 (H2O, SnapKV)** | 显存占用下降 50-80% | 信息丢失，对需要精确 recall 的任务有害 |
| **RAG / 上下文压缩** | 突破物理限制，按需检索 | 检索质量决定效果，"Garbage In Garbage Out" |

**核心矛盾**：
- **计算复杂度** vs **全局依赖捕捉**
- **显存占用** vs **信息完整性**
- **零成本外推** vs **微调后性能上限**

---

## 5. 工业映射

| 技术 | 工业落地 |
|------|----------|
| **RoPE + NTK/RoPE Scaling** | **LLaMA 1/2/3**、**Qwen**、**Baichuan** 等主流模型标配，支持 4K→32K→128K 动态扩展 |
| **Sliding Window + Global Attention** | **Mistral 7B**（sliding window=4K, global=128K）、**Longformer**（Hugging Face 原生支持） |
| **Ring Attention / Context Parallelism** | **DeepSpeed Ulysses**、**Megatron-LM** 序列并行，支持百万级 token 训练 |
| **KV Cache 分页管理** | **vLLM**（PagedAttention）、**SGLang**，实现动态显存分配，支持 batch 内不同长度序列 |
| **上下文压缩 + RAG** | **LangChain**、**LlamaIndex** 生态，**Claude 3**、**Gemini 1.5** 原生支持 1M+ token（实际为压缩+检索混合架构） |
| **Mamba / State Space Model** | **Mamba-2.8B/7B**、**Jamba**（Mamba+Transformer 混合），理论线性复杂度 $O(n \cdot d)$，但长程依赖质量仍在验证 |

---

## 一句话总结

上下文限制源于 Attention 的 $O(n^2)$ 复杂度与位置编码的分布外失效；扩展路径在 **"稀疏化计算"**、**"外推位置编码"**、**"压缩上下文"** 三个维度上找平衡，工业界通过 **RoPE-NTK + GQA + PagedAttention + Ring Attention** 的组合拳实现百万 token 级长窗口。
