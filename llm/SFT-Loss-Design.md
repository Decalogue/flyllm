# SFT 损失函数如何设计？与预训练损失有什么区别？

## 1. 核心定性
本质上，**SFT损失**是**指令响应的条件概率最大化**，仅对回答部分（Response）计算交叉熵；而**预训练损失**是**无条件自回归语言建模**，对全序列（包括用户指令）计算均匀交叉熵，两者差异在于**注意力掩码和计算域**。

## 2. 具体流程
1. **预训练损失**：对完整序列 $x_{1:n}$ 计算 $\mathcal{L}_{\text{PT}} = -\sum_{t=1}^n \log p_\theta(x_t|x_{<t})$，包括系统、用户、助手所有token
2. **SFT损失**：仅对助手回复部分 $y_{1:m}$ 计算 $\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^m \log p_\theta(y_t|x, y_{<t})$，前缀（指令+历史）使用因果掩码但损失为零
3. **掩码实现**：在PyTorch中通过`ignore_index=-100`实现，前缀token目标设为-100，反向传播时梯度为0，仅更新响应参数

## 3. 数学基础
**预训练损失（Causal LM）**：

$$\mathcal{L}_{\text{PT}}(\theta) = -\frac{1}{N}\sum_{i=1}^N \sum_{t=1}^{|x^{(i)}|} \log p_\theta(x_t^{(i)} \mid x_{<t}^{(i)})$$

**SFT损失（Response-only）**：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^N \sum_{t=s_i}^{|x^{(i)}|} \log p_\theta(x_t^{(i)} \mid x_{<t}^{(i)}) \cdot \mathbb{I}(t \geq s_i)$$

其中：
- $s_i$：第$i$个样本中助手回复的起始位置索引
- $\mathbb{I}(\cdot)$：指示函数，仅对回复位置取值为1

**带权SFT（Weighted SFT）**：

$$\mathcal{L}_{\text{Weighted}} = -\sum_{i=1}^N w_i \sum_{t=s_i}^{|x^{(i)}|} \log p_\theta(x_t^{(i)} \mid x_{<t}^{(i)})$$

权重设计：
- $w_i = 1$：标准SFT
- $w_i = \text{Quality}(y_i)$：基于回答质量加权（RLHF数据）
- $w_i = \lambda^{|y_i|}$：长度惩罚/奖励（$\lambda<1$惩罚长答案）

**梯度差异**：

$$\nabla_\theta \mathcal{L}_{\text{SFT}} = -\sum_{i=1}^N \sum_{t=s_i}^{|x^{(i)}|} (1 - p_\theta(x_t^{(i)})) \cdot \nabla_\theta \log p_\theta(x_t^{(i)})$$

前缀位置无梯度（$\mathbb{I}=0$），仅更新响应生成能力。

## 4. 工程考量
**核心区别**:
| 特性 | 预训练损失 | SFT损失 |
|------|-----------|---------|
| **计算范围** | 全序列 | 仅响应 |
| **参数更新** | 所有位置 | 仅响应位置 |
| **优化目标** | 语言建模 | 指令遵循 |
| **数据利用率** | 100% | 20-40%（响应占比） |

**Trade-off**:
- **预训练**：计算效率低但学到通用语言能力，在512 token序列中仅20%为有效参数更新
- **SFT**：计算聚焦但可能过拟合，在100k样本上SFT后通用能力下降5-10%，需混入5-10%预训练数据防止灾难性遗忘

**损失设计变体**:
- **Token-level权重**：对关键token（事实、代码关键指令）加权2-5倍，提升准确率3-5%
- **长度归一化**：按回复长度$|y_i|$归一化，防止偏好生成短答案，在摘要任务上ROUGE提升1-2点
- **Focal Loss**：对预测困难的token加权，公式：

$$\mathcal{L}_{\text{focal}} = -\alpha_t(1-p_t)^\gamma \log(p_t)$$

**致命弱点**:
- **负样本缺失**：SFT仅学习正例（正确回答），无法区分好坏回答，RLHF后效果好坏验证集上差距仅15%（无判别能力），必须用偏好学习补充
- **分布不匹配**：预训练分布$p_{\text{PT}}$与SFT分布$p_{\text{SFT}}$差异大，导致SFT后模型困惑度上升30-50%，需学习率极小（1e-6 vs 1e-5）防止震荡
- **计算浪费**：在batch中，每个样本仅部分token计算损失，GPU利用率低30-40%，梯度聚合时大量位置为零，需使用flash-attention优化内存访问模式

## 5. 工业映射
在工业界，损失设计是**对齐训练的第一道闸**，直接决定下游能力**：
- **ChatGPT（InstructGPT）**：SFT阶段使用标准response-only损失，在30k人工标注对话上训练，KL散度从基线控制0.5以内，学习率5e-6，batch size 128，3 epoch，验证损失不用于早停（过拟合反而好）
- **Claude**：使用加权SFT，对harmful请求的损失权重设为0.1（弱监督），对helpful请求权重设为1.0，在HHH（Helpful, Harmless, Honest）指标上超越ChatGPT 5个百分点
- **LLaMA 2**：SFT阶段混入5%预训练数据，比例通过消融实验确定，防止通用能力下降，在MMLU上仅降1.2% vs 不降预训练数据降4.5%
- **CodeLLaMA**：使用focal loss（$\gamma=2.0$），对语法关键点（如括号匹配、缩进）提升权重，在HumanEval上比标准SFT提升3-4%
- **SFT vs Pretrain计算开销**：在175B模型上，预训练消耗99%总算力（$10^{24}$ FLOPs），SFT仅0.1%，但SFT的损失设计决定最后10%的能力上限，是