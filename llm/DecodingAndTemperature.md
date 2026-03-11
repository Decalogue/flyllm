# 生成解码策略和 temperature 的影响

### 1. 核心定性
本质上，**解码策略**是在自回归生成的概率空间中寻找最优序列的启发式搜索算法，而 **Temperature** 是通过缩放 Logits 来动态重塑概率分布平滑度的热力学控制系数。

### 2. 具体流程
1. 模型语言头（LM Head）输出词表中每个 Token 的未归一化打分（Logits）。
2. 将 Logits 除以 Temperature 参数后，通过 Softmax 函数映射为标准概率分布。
3. 采样器根据截断策略（Top-K / Top-p）或确切搜索（Greedy / Beam Search）从重塑后的分布中选取下一个 Token。

### 3. 数学基础
Temperature 调节下的 Softmax 分布方程：
$$p_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$
其中：
- $p_i$: 词表中第 $i$ 个 Token 被采样的最终概率。
- $z_i$: 模型最后一层输出的原始未归一化特征值 (Logit)。
- $T$: Temperature 参数 ($T > 0$)。

**边界推演：**
- **$T \to 0$ (极限降温)**: $\exp(z_{max}/T)$ 占据绝对统治地位，分布坍缩为独热编码 (One-hot)，等价于贪婪搜索 (Greedy Decoding)。
- **$T = 1$**: 标准 Softmax 分布，完全遵循模型训练时的困惑度 (Perplexity) 期望。
- **$T \to \infty$ (极限升温)**: 所有 $\exp(z_i/T) \to 1$，分布退化为绝对均匀分布 (Uniform Distribution)，输出完全随机。

### 4. 工程考量
- **Trade-off**: 创造性 (Diversity) 与 事实连贯性 (Coherence) 的零和博弈。高 $T$ 引入熵增换取多样性，但会拉高低质量 Token 的权重；低 $T$ 确保逻辑严密，但极易陷入局部最优。
- **致命弱点**: 
  1. **高 $T$ 崩溃**: 在结构化输出（如 JSON/SQL）或长逻辑链推导（CoT）中，微小的高频长尾噪声被放大，极易导致语法树崩溃或幻觉（Hallucination）。
  2. **低 $T$ 退化**: 纯贪婪解码在长文本生成时面临“重复惩罚失效”，极易陷入无限循环（Repetition Loop）陷阱。

### 5. 工业映射
在工业界，该机制被直接沉淀在 **vLLM / TGI 等推理引擎的采样算子（Sampling Logits Processor）** 中，用于应对不同业务架构：在 Copilot 代码/SQL 生成场景强制 $T=0$ 且关闭采样以保证确定性；在 Character.AI 等角色扮演架构中常配 $T=0.85$ 并硬绑定 Top-p=0.9，在截断长尾毒性 Token 的同时维持人格多样性。