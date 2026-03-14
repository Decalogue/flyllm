# 线性注意力如何降低复杂度？Performer 和 Linformer 的区别是什么？

## 1. 核心定性
本质上，**线性注意力**通过核技巧将 softmax 注意力分解为特征映射的内积，利用矩阵乘法的结合律改变计算顺序，将复杂度从 $O(n^2 d)$ 降至 $O(n d^2)$，在长序列场景下实现线性增长而非二次增长。

## 2. 具体流程
1. **核函数重构**：将 softmax $\exp(QK^T)$ 近似为特征映射内积 $\phi(Q)\phi(K)^T$，其中 $\phi: \mathbb{R}^d \rightarrow \mathbb{R}^r$ 且 $r \ll n$
2. **顺序重排**：利用 $(\phi(K)^TV)$ 先计算，避免显式存储 $n \times n$ 注意力矩阵。新计算流：$O = \phi(Q)[\phi(K)^T V]$，复杂度 $O(n d^2 + d^2 r)$
3. **特征选择**：Performer 使用正交随机特征（ORF）逼近 softmax，Linformer 通过低秩投影压缩 $K,V$ 到 $r$ 维度

## 3. 数学基础
**标准注意力**:
$$\text{Attn}(Q,K,V) = \text{softmax}(QK^T)V$$

**线性化目标**:
$$\text{softmax}(QK^T) \approx \phi(Q)\phi(K)^T$$

**Performer (FAVOR+) 使用特征映射**:
$$\phi(x) = \frac{1}{\sqrt{r}}[\exp(-\|x\|^2/2), \exp(-\|x\|^2/2)\cos(\omega_1^Tx), \ldots]$$

其中 $\omega_i \sim N(0, I_d)$ 是随机频率，通过随机特征逼近高斯核。

**计算重排**:
$$O = \phi(Q)[\phi(K)^T V]$$

复杂度分析：
- $\phi(K)^T V$: $O(r d n)$
- $\phi(Q)[\cdot]$: $O(r d n)$
- **总计**: $O(r d n)$，当 $r \approx d$ 时为 $O(n d^2)$

**Linformer 的低秩投影**:
$$K' = EK, \quad V' = EV, \quad E \in \mathbb{R}^{r \times n}$$

$$\text{Attn}(Q,K,V) \approx \text{softmax}(QK'^T)V'$$

其中 $E$ 是学习到的投影矩阵，$r = O(\log n)$。

**关键区别**:
- **Performer**: 特征映射逼近 softmax，不修改 $K,V$ 维度，保持表达能力
- **Linformer**: 显式压缩 $K,V$ 到 $r$ 维度，牺牲表达能力换取速度

**复杂度对比**:
| 方法 | 时间 | 空间 | 近似误差 |
|------|------|------|----------|
| Dense | $O(n^2 d)$ | $O(n^2)$ | 0 |
| Performer | $O(n d^2)$ | $O(nd)$ | 0.5-2% |
| Linformer | $O(n d r)$ | $O(n r)$ | 2-5% |

## 4. 工程考量
**Performer vs Linformer 的核心差异**:

| 维度 | Performer | Linformer |
|------|-----------|-----------|
| **近似对象** | Softmax 核函数 | 键值投影 |
| **随机性** | ORF 随机特征 | 可学习投影 |
| **表达能力** | 保持 $d$ 维度 | 压缩到 $r$ |
| **训练稳定性** | 高（特征固定） | 中（投影需学习） |
| **硬件适配** | 通用矩阵乘 | 需自定义 kernel |

**Performer 的优势**:
- **理论保证**：ORF 在 $r = O(d \log d)$ 时可 $\epsilon$-逼近 softmax，误差可控
- **训练稳定**：随机特征不引入额外可训练参数，不影响优化动态
- **快速推理**：特征映射可预计算，推理时仅需 $O(n d)$ 前向传播

**Linformer 的优势**:
- **内存极致**：$r = O(\log n)$ 时内存压缩 10-20 倍，适合极长序列
- **可学习性**：投影矩阵 $E$ 可针对任务优化，领域适应性好

**Trade-off**:
- **速度与精度**：特征维度 $r$ 越大，Performer 精度越高，但速度越慢。$r=256$ 时比 $r=128$ 误差下降 30%，但计算增加 2 倍
- **内存与质量**：Linformer 的投影压缩导致在需要细粒度对齐任务（如机器翻译）上 BLEU 下降 2-4 点，适合粗粒度任务（分类）

**致命弱点**:
- **核近似误差**：Performer 在 high temperature（$>2$）下逼近误差指数级增长，生成分布与 softmax 差异显著
- **低秩假设失效**：Linformer 在 $QK^T$ 满秩任务（如复杂推理）上表达能力崩溃，准确率下降 10-15%
- **因果注意力困难**：线性化在自回归生成中需累积状态，难以实现流式解码，增量计算成本与 dense 相当

## 5. 工业映射
在工业界，线性注意力是 **长序列任务的备选方案**：
- **Performer 在 Google**：用于蛋白质序列建模（ProteinBert），长度 4096，$r=256$，在 TAPE 基准上比 dense 慢 15%，但内存从 40GB 降至 16GB，可在 TPUv3-8 运行。关键优化：使用 Jax 的 lax.scan 实现循环累积，避免 materialize 大矩阵
- **Linformer 在 MSRA**：应用于超长文档检索，$n=8192$，$r=128$，内存压缩 64 倍，但 Recall@10 下降 3.2%。工程妥协：r=256 时仅下降 1.2，可接受
- **RWKV 的结合**：结合 RNN 式线性注意力和 token shift，实现 O(n) 训练和 O(1) 推理，14B 模型在 Pile 上训练，内存比 dense 少 50%，适合边缘部署
- **FlashAttention 淘汰论**：FlashAttention-2 在 A100 上优化后，16k 序列训练仅需 24GB，速度比 Performer 快 20%，导致线性注意力在 GPU 上失去优势。当前工业界首选 FlashAttention 而非线性注意力
- **标记**：在 Transformer 不支持的硬件（如 CPU 或旧 GPU），线性注意力仍有价值。Hugging Face 的 `performer-pytorch` 库在 CPU 上比 dense 快 5 倍，适合离线批处理

**趋势**：线性注意力在学术上有价值，但工程上被 FlashAttention 取代。新研究转向自适应线性注意力（根据序列动态选择近似方法），在重要位置用 dense，次要位置用 linear
