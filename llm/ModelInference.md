# 大模型推理流程

---

## 1. 核心定性

**本质上，模型推理是一个自回归条件概率生成过程，通过已训练参数将输入token序列映射为输出token序列，每一步最大化条件概率 $P(x_t | x_{<t})$。**

---

## 2. 具体流程

1. **Tokenization**: 输入文本经分词器(BPE/SentencePiece)编码为整数ID序列，添加特殊标记(BOS/EOS/PAD)。
2. **前向传播**: 输入嵌入+位置编码 → N层Transformer块(自注意力+FFN) → 输出Logits。
3. **解码生成**: 基于采样策略(贪婪/Beam Search/Top-p)从输出分布中迭代采样，直至生成终止符。

---

## 3. 数学基础

**自注意力核心计算:**
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**自回归生成目标:**
$$P(y|x) = \prod_{t=1}^{T} P(y_t | y_{<t}, x)$$

其中：
- $Q = XW_Q$, $K = XW_K$, $V = XW_V$：查询/键/值投影矩阵
- $d_k$：键向量的维度（缩放因子防止softmax梯度消失）
- $y_{<t}$：已生成的token历史序列
- $T$：最大生成长度

**KV-Cache优化后的推理:**
$$\text{Attention}_t(Q_t, K_{\leq t}, V_{\leq t}) = \text{softmax}\left(\frac{Q_t K_{\leq t}^T}{\sqrt{d_k}}\right)V_{\leq t}$$

---

## 4. 工程考量

| 维度 | 权衡 (Trade-off) | 致命弱点 |
|------|------------------|----------|
| **延迟 vs 吞吐** | 低延迟需小batch，高吞吐需大batch | 长序列($>$8k)时KV-Cache显存爆炸，OOM |
| **质量 vs 速度** | Beam Search提升质量但延迟×k倍 | 贪心解码陷入局部最优，重复生成 |
| **精度 vs 性能** | FP16/INT8量化提速但引入误差 | 异常输入激活值溢出，输出乱码 |
| **内存 vs 模型** | 模型并行切分跨卡通信开销大 | PP并行气泡(bubble)导致GPU空转 |

---

## 5. 工业映射

| 技术 | 开源实现 | 应用场景 |
|------|----------|----------|
| **PagedAttention** | vLLM | 高并发服务场景，通过Block Table管理非连续KV-Cache，将GPU利用率从40%提升至90%+ |
| **Continuous Batching** | TensorRT-LLM / TGI | 动态批处理，避免padding浪费，LLM-as-a-Service核心优化 |
| **Speculative Decoding** | Medusa / Lookahead | 小模型草稿+大模型验证，解码速度提升2-3倍 |
| **量化推理** | GPTQ / AWQ / GGML | 边缘部署，7B模型压缩至4GB显存运行 |

---
