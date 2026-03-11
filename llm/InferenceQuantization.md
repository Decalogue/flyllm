# 推理量化方法深度解析

---

## 1. 核心定性

**本质上，模型量化是一个为了在推理阶段用低比特数值（INT8/INT4/INT1）近似浮点权重/激活，通过缩小内存占用与计算位宽来换取吞吐提升与功耗降低的压缩技术。**

---

## 2. 具体流程

1. **量化映射**：确定浮点数值 $x$ 到整型数值 $x_q$ 的线性/非线性映射关系（$x_q = round(x/S) + Z$）
2. **校准统计**：用少量校准数据统计权重/激活的动态范围，求解最优缩放因子 $S$ 与零点 $Z$
3. **INT Kernel 替换**：将浮点 GEMM 替换为 INT 定点运算，配合反量化恢复输出精度

---

## 3. 数学基础

### 对称线性量化（Symmetric Linear Quantization）

$$x_q = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right), -2^{b-1}, 2^{b-1}-1\right)$$

**反量化：**
$$\hat{x} = x_q \cdot S$$

**其中：**
- $x$: 原始浮点张量（FP32/FP16）
- $x_q$: 量化后的整型值
- $S = \frac{\alpha}{2^{b-1}-1}$: 缩放因子（$\alpha = \max(|x_{\min}|, |x_{\max}|)$ 为截断阈值）
- $b$: 量化位宽（INT8时$b=8$，INT4时$b=4$）
- $\text{clamp}$: 截断函数，限制在目标整型范围内

### 非对称量化（Asymmetric，带 Zero-Point）

$$x_q = \text{round}\left(\frac{x}{S}\right) + Z, \quad S = \frac{x_{\max} - x_{\min}}{2^b - 1}, \quad Z = -\text{round}\left(\frac{x_{\min}}{S}\right)$$

---

## 4. 工程考量

### 量化方法分类与 Trade-off

| 方法 | 核心思想 | 精度-速度 Trade-off |
|------|----------|---------------------|
| **PTQ (Post-Training Quantization)** | 训练后直接量化，无需重训 | 牺牲精度换零开销，INT8友好，INT4以下精度崩 |
| **QAT (Quantization-Aware Training)** | 训练时模拟低比特前向，反向用 STE 近似梯度 | 精度损失小但训练成本高，适合 INT4/INT1 |
| **GPTQ / AWQ** | 逐层/逐组优化，用 Hessian 矩阵指导裁剪 | PTQ 的精度天花板，4bit 模型可用 |
| **SmoothQuant / LLM.int8()** | 分离激活异常值（Outlier），部分通道保持 FP16 | 解决 LLM 激活值分布长尾问题 |

### INT8 vs INT4 vs INT1 本质区别

| 位宽 | 表示范围 | 存储压缩比 | 精度损失 | 适用场景 |
|------|----------|------------|----------|----------|
| **INT8** | $[-128, 127]$ | 4× (vs FP32) | < 1% | 通用首选，CNN/Transformer 均可 |
| **INT4** | $[-8, 7]$ | 8× | 2-5% | 大模型权重-only，需 GPTQ/AWQ |
| **INT1** | $\{-1, +1\}$ 或 $\{0, 1\}$ | 32× | 10%+ | 极限压缩，BNN（二值神经网络），仅特定嵌入式场景 |

### 致命弱点与选型决策树

**选型逻辑：**

```
模型参数量 > 7B 且 显存吃紧？ → 尝试 INT4 (GPTQ/AWQ)
    ↓ No
部署在移动端/NPU？ → INT8 (PTQ 即可)
    ↓ No
学术实验/理论极限？ → INT1 (BNN，需 QAT)
```

**致命弱点：**
- **INT4 以下**：梯度消失敏感，LayerNorm 与 Softmax 层必须保留 FP16，否则数值爆炸
- **INT1**：信息瓶颈严重，仅适用于简单视觉任务，LLM 上几乎不可用
- **异常值通道**：LLM 中 0.1% 的激活通道占据 99% 的动态范围， naive INT8 会导致灾难性精度损失（需 SmoothQuant 解决）

---

## 5. 工业映射

在工业界，该机制被直接应用于：
- **TensorRT (NVIDIA)**：INT8 PTQ + QAT 流水线，配合 `IQuantizeLayer` 与 `IDequantizeLayer` 自动融合
- **OpenVINO (Intel)**：支持 INT8 的 POT (Post-Training Optimization Tool)，针对 CPU 的 AVX-512 VNNI 指令集优化
- **vLLM / llama.cpp**：INT4 (GPTQ/AWQ) 权重压缩，PagedAttention 结合 INT8 KV Cache，实现单卡 70B 模型推理
- **Transformer Engine (NVIDIA Hopper)**：FP8 训练与推理原生支持，作为 INT8 的精度升级替代

---

## 精度损失评估方法

| 指标 | 定义 | 阈值参考 |
|------|------|----------|
| **perplexity (PPL)** | 量化前后模型在验证集上的困惑度变化 | $\Delta \text{PPL} < 5\%$ 可接受 |
| **任务精度衰减** | 下游任务（如 MMLU、C-Eval）得分下降幅度 | $< 2\%$ 为优秀，$< 5\%$ 可用 |
| **SQNR (Signal-to-Quantization-Noise Ratio)** | $10 \cdot \log_{10}\left(\frac{\mathbb{E}[x^2]}{\mathbb{E}[(x-\hat{x})^2]}\right)$ | > 40dB 表示量化误差可控 |
| **余弦相似度** | 量化前后层输出的特征向量夹角 | > 0.99 表示语义一致 |

**评估 Pipeline：**
```
1. 量化模型 → 2. 跑验证集得 PPL → 3. 跑下游 Benchmark → 4. 对比 FP32 baseline
```
