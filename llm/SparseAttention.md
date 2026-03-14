# 稀疏注意力有哪些实现方式？Longformer 和 BigBird 的区别是什么？

## 1. 核心定性
本质上，**稀疏注意力**通过将全局的 $O(n^2)$ 密集注意力矩阵约束为预定义或动态学习的稀疏模式，用计算图的局部连通性换取内存和复杂度的指数级下降，在长序列建模中实现次二次复杂度。

## 2. 具体流程
1. **模式设计**：定义稀疏模式（滑动窗口、随机块、全局块），将注意力计算限制在预定义的非零位置，如 Longformer 的局部窗口 $w=256$ 仅计算每个 token 的邻近注意力
2. **动态路由**：通过可学习的路由机制（如 Reformer 的 LSH 哈希）将查询分组到桶中，仅在同桶内计算注意力，复杂度降至 $O(n \log n)$
3. **层次聚合**：BigBird 结合局部窗口、随机块和全局块三种模式，局部捕获短距离依赖，随机连接远程信息，全局块确保关键位置（如 [CLS]）可访问全序列

## 3. 数学基础
设注意力矩阵 $A \in \mathbb{R}^{n \times n}$，稀疏模式定义掩码 $M \in \{0, 1\}^{n \times n}$：

$$M_{ij} = \begin{cases}
1 & \text{如果位置 } i \text{ 可以关注 } j \\
0 & \text{否则}
\end{cases}$$

**滑动窗口（Longformer）**:
$$M_{ij} = 1 \iff |i - j| \leq w$$

计算复杂度：$O(n w)$，当 $w \ll n$ 时接近线性。

**随机块（BigBird）**:
$$M_{ij} = 1 \iff |i - j| \leq w \text{ 或 } j \in \text{Random}(i, r) \text{ 或 } i, j \in G$$

其中 $G$ 是全局块位置集合，通常包含特殊标记或随机采样的全局 token。

**LSH 路由（Reformer）**:
将查询 $Q$ 通过局部敏感哈希映射到 $b$ 个桶：
$$h(q_i) = \arg\max_{k} \frac{q_i \cdot r_k}{||r_k||}$$

仅在同桶内计算注意力：$M_{ij} = 1 \iff h(q_i) = h(q_j)$。

**复杂度对比**：
| 方法 | 时间复杂度 | 空间复杂度 | 稀疏模式 |
|------|----------|----------|---------|
| 密集注意力 | $O(n^2 d)$ | $O(n^2)$ | 无 |
| Longformer | $O(n w d)$ | $O(n w)$ | 窗口 $w=256$ |
| BigBird | $O(n (w+r+g) d)$ | $O(n (w+r+g))$ | 窗口+随机+全局 |
| Reformer (LSH) | $O(n b d)$ | $O(n b)$ | 哈希桶 $b \ll n$ |

## 4. 工程考量
**Longformer vs BigBird 核心区别**：

| 特性 | Longformer | BigBird |
|------|-----------|---------|
| **稀疏模式** | 仅滑动窗口 | 窗口+随机+全局 |
| **计算量** | $O(n w)$ | $O(n (w+r+g))$ |
| **表达能力** | 理论上是 universal approximator | 已证明可逼近全注意力 |
| **全局依赖** | 依赖堆叠层传播 | 显式全局块 |
| **实现复杂度** | 简单，仅 mask | 复杂，三种模式融合 |

**Trade-off**:
- **稀疏度 vs 表达能力**：稀疏度越高（模式越简单），计算越快，但可能丢失长距离依赖。Longformer 在 4096 长度上 RoBERTa 性能下降 1-2%，BigBird 仅 0.5%
- **预训练成本**：稀疏注意力需从零设计预训练任务，直接微调密集模型效果差 5-8%。BigBird 需 2.2B token 预训练才能匹配 BERT 性能

**致命弱点**:
- **模式刚性**：预定义模式无法适应动态任务需求。在需要全局推理的 QA 任务上，纯滑动窗口准确率下降 8-12%，需手动插入全局 token
- **层级传播误差**：局部注意力需堆叠多层才能捕获远距离依赖，深层梯度消失更严重。48 层 Longformer 比 12 层训练不稳定度增加 3 倍
- **硬件适配**：稀疏模式导致矩阵不规则，GPU 矩阵单元利用率仅 30-40%，实际加速比理论值低 50%

## 5. 工业映射
在工业界，稀疏注意力是 **长上下文任务的标配**：
- **Longformer 在 AllenAI**：用于 SciBert 的科学文献理解，处理 4096 token 的论文摘要，在 SciERC 上 F1 78.3，比 BERT 的 512 长度提升 4.2 点。实现中自定义 CUDA kernel 实现 banded matrix multiplication，速度比 dense 快 3 倍
- **BigBird 在 Google Research**：应用于 Pegasus 摘要，$w=64, r=32, g=16$ 配置，在 CNN/DailyMail 上 ROUGE-2 21.8（vs dense 22.1），但训练时间从 3 天降至 1 天，内存占用 32GB→12GB，可在 TPUv3-8 上运行
- **Reformer 在 Hugging Face**：`ReformerModel` 使用 LSH 注意力，$b=8$ 桶，在 16k 文本生成上 OOM 风险从 100% 降至 0%，但困惑度上升 15-20%，质量损失显著
- **FlashAttention 的启示**：不通过稀疏化，而是通过 IO 优化实现 O(n^2) 计算但 O(n) 内存，在 A100 上比稀疏注意力更快，证明工程优化比算法近似更高效。现代 LLM 普遍采用 FlashAttention 而非稀疏模式
- **BigBird 在医疗**：处理电子病历的 8k token 长文本，全局块指定 ICD 编码位置和药物名称，在医疗 NER 上 F1 提升 2.3 点，证明领域相关的全局块设计是关键

**选择建议**：长文本理解用 Longformer（简单高效），需要全局依赖的生成任务用 BigBird（表达能力好），极端长度（16k+）尝试 Reformer（LSH），但现代首选 FlashAttention
</content>
