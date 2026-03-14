# Reward Model 怎么训练？损失函数是什么？

## 1. 核心定性
本质上，**Reward Model训练**是学习一个**标量评分函数** $R_\phi(x,y)$ 拟合人类偏好，通过**Bradley-Terry排序损失**将成对比较转化为概率预测，目标是**最大化偏好对的一致性**，而非预测绝对奖励值。

## 2. 具体流程
1. **数据标注**：收集偏好样本$(x, y_w, y_l)$，同一prompt（x）生成2个回答（$y_w$胜出，$y_l$失败），由标注员评分，同一prompt需5-9个不同回答保证多样性
2. **模型初始化**：用SFT模型初始化，移除语言建模头（LM Head），替换为**线性回归头**（hidden_size → 1），输出标量分数
3. **对比训练**：将成对回答输入RM，计算差值，通过sigmoid转换为胜出的概率，最大化标注对的一致性，只用1 epoch防止过拟合

## 3. 数学基础
**Bradley-Terry排序模型**：

$$p(y_w \succ y_l | x) = \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))$$

其中：
- $x$：输入prompt（instruction）
- $y_w$：胜出的回答（chosen response）
- $y_l$：失败的回答（rejected response）
- $R_\phi(x,y) \in \mathbb{R}$：奖励模型输出的标量分数
- $\sigma(z) = \frac{1}{1 + e^{-z}}$：sigmoid函数

**损失函数**：

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)\sim D}\left[\log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))\right]$$

等价于二元交叉熵：

$$\mathcal{L} = -\log p_\phi(y_w \text{ 优于 } y_l)$$

**梯度推导**：

$$\begin{align}
\nabla_\phi \mathcal{L} &= -\nabla_\phi \log \sigma(\Delta) \\
&= -(1 - \sigma(\Delta)) \cdot \nabla_\phi \Delta \\
&= -(1 - \sigma(\Delta)) \cdot (\nabla_\phi R_\phi(x, y_w) - \nabla_\phi R_\phi(x, y_l))
\end{align}$$

其中$\Delta = R_\phi(x, y_w) - R_\phi(x, y_l)$。

**奖励差的直观**：
- 当$R_\phi(x, y_w) - R_\phi(x, y_l) \gg 0$：标注一致，损失小
- 当$R_\phi(x, y_w) - R_\phi(x, y_l) \ll 0$：预测错误，损失大
- 当$R_\phi(x, y_w) = R_\phi(x, y_l)$：模型无法区分，损失为$\log 2$

**多回答扩展（InstructGPT）**：

每个prompt生成$k$个回答（k=4-9），人类对这些回答排序：

$$\mathcal{L} = -\mathbb{E}\left[\sum_{i=1}^{k-1} \sum_{j=i+1}^k \log \sigma(R_\phi(y_i) - R_\phi(y_j))\right]$$

等价于所有**成对比较**的对数似然。

**标注一致性检验**：

Cohen's Kappa：

$$\kappa = \frac{p_o - p_e}{1 - p_e} \ge 0.65$$

其中：
- $p_o$：观察一致性（70-80%）
- $p_e$：随机一致性（50%）

要求$\kappa \ge 0.65$（substantial agreement）。

**统计检验**：

对比对数概率：

$$\text{LPP} = \log \frac{p(y_w \succ y_l)}{p(y_l \succ y_w)} = \log \frac{\sigma(\Delta)}{1 - \sigma(\Delta)} = \Delta = R_\phi(y_w) - R_\phi(y_l)$$

InstructGPT的LPP分布：
- 中位数：2.5（胜出的比失败的分数高2.5）
- 70%置信度：LPP \in [0.5, 6.0]
- 说明人类偏好是相对的，非绝对

## 4. 工程考量
**训练超参**：

| 参数 | InstructGPT | LLaMA 2 | 说明 |
|------|-------------|---------|------|
| 学习率 | 1e-5 (cosine) | 5e-6 | 比SFT高，快速拟合偏好 |
| Batch size | 32-64 | 512 | 对比对需要较多样本 |
| Epochs | 1 | 1 | **1 epoch最佳**，多epoch过拟合 |
| 最大长度 | 512 | 4096 | RLHF中长度影响奖励质量 |
| Dropout | 0.1 | 0.0 | BT模型本身有正则化 |

**过拟合问题**：

测试集准确率 vs 训练集准确率：

- **1 epoch**：训练72%，测试68%（差距4%）
- **2 epochs**：训练85%，**测试62%**（差距23%，严重过拟合）

**原因**：Bradley-Terry模型参数仅用于学习相对排序，训练集重复采样导致模型记忆prompt特征而非偏好模式。

**值域标准化**：

奖励值无绝对意义，需标准化：

- **Zero-mean**：$R_\phi(x,y) \leftarrow R_\phi(x,y) - \mathbb{E}_{(x,y)\sim D}[R_\phi(x,y)]$
- **Unit variance**：缩放到std=1

**批内配对策略**：

- **全配对**：每个batch内所有回答两两比较，batch size=64时产生$O(64^2)=4096$对比，计算量大但效率高
- **相邻配对**：仅比较相邻分数，计算量小但信息少

**Trade-off**:
- **准确率 vs 鲁棒性**：Finetune 1个epoch准确率68%，3个epoch训练准确率85%但鲁棒性差（对prompt扰动敏感），推荐**1-1.5 epoch**
- **容量 vs 泛化**：RM模型容量（6B vs 1B）提升测试准确率5-8%，但过拟合风险增加，需要更多正则化
- **标注一致性 vs 模型性能**：只使用高一致性样本（$\kappa \ge 0.75$）训练，准确率提升3%但数据量减少40%，性价比低

**致命弱点**:
- **奖励黑客（Reward Hacking）**：RM过度拟合表面特征（长度、表情符号），PPO后模型学会生成长回答、过度emoji、过度道歉，在GPT-4训练中早期出现"过度道歉"模式，reward +0.2但真实满意度-0.3
- **标注偏差（Annotation Bias）**：人类偏好"更长、更详细"的回答，导致RL后模型verbosity增加40%，需长度惩罚$R_{\text{final}} = R_\phi - \lambda \cdot |y|$
- **分布漂移**：训练时回答来自SFT策略，推理时来自PPO优化策略，RM在OOD数据上准确率下降15-20%，需在PPO过程中在线更新RM（每10k步）

**学习曲线震荡**：

RM训练中损失波动大：

$$\frac{\text{Std}(\mathcal{L}_{\text{batch}})}{\text{Mean}(\mathcal{L}_{\text{batch}})} \approx 0.3-0.5$$

原因是对比对的难度差异大。

## 5. 工业映射
在工业界，RM训练是**RLHF中最昂贵的阶段**（标注成本）：
- **InstructGPT（OpenAI）**：40标注员，33k对比对（来自30k prompts，每个prompt生成4-9个回答），标注一致性32%（不同人），同一标注员70%。RM准确率65%，使用6B模型，训练2个epoch测试准确率67%（1个epoch 68%，2个66%），验证1 epoch最佳。标注成本$100k（人力）+ \$50k（计算）
- **ChatGPT扩展**：标注团队增至400人，500k对比对，引入多模态（文本+图像），多人标注同一对（用majority vote），使用GPT-4 as judge自动标注30%简单样本，成本降至每对比$0.5。RM规模从6B扩至训练独立175B奖励模型（reward-only），准确率提升至75%，但推理成本增加100倍（专门硬件）
- **LLaMA 2（Meta开源）**：公开RM训练细节，使用SFT模型初始化，训练1.4M对比对（从350k prompts生成30个回答筛选），batch size=512，lr=5e-6，1 epoch测试准确率70%。使用MAB（Multi-armed bandit）自动采样困难样本，相比随机采样提升5%准确率
- **DeepSpeed-RLHF（微软）**：开源RM实现，支持7B-530B模型，用LoRA训练降低显存（减少30%），在OPT-30B上训练RM仅需2小时（8xA100），成本\$100，降低门槛。使用RankNet损失（与BT等效），但支持listwise ranking（多个回答）
- **Claude（Constitutional AI）**：RLAIF扩展，使用AI反馈而非人类标注，RM训练改用AI判断，成本降低90%。训练时加入Constitutional筛选（删除不道德回答），RM在有害性检测准确率从80%→95%，但有用性下降3%，需在损失中权重平衡
- **中文RM（文心一言）**：中文偏好数据标注困难（文化差异），使用"众包+专家"混合，众包500k对比对（一致性低50%），专家100k（一致性高85%），损失权重专家样本3倍。引入"文化适当性"维度，避免RL后模型西化，RM准确率68%

**工程最佳实践**：
1. **Early Stopping**：验证准确率不再提升时立即停止（通常1-1.5 epoch）
2. **Weight Decay**：系数0.01-0.1防止过拟合
3. **Dropout=0.1**：在RM中增加鲁棒性（InstructGPT发现）
4. **Loss Masking**：只对padding以外的token取平均（mean pooling而非last token）
5. **Margin Regularization**：在损失中增加$-\lambda ||R_\phi(y_w) - R_\phi(y_l)||^2$防止分数差异过大

**标注一致性提升方法**：
- **Test Questions**：在标注流中插入已知答案的测试对，过滤一致性<70%的标注员
- **Training**：标注员需完成2小时培训，标注100个测试样本
- **Multiple Annotation**：每个对比3人标注，majority vote（成本3x但准确性+8%）

**趋势**：
- **RLAIF**：用AI替代人工标注，Claude证明可行，成本降低90%，但需防止AI偏见循环
- **Process Supervision**：不只是奖励结果，也奖励过程（如数学推理步骤），OpenAI在2023年展示该方向，在Math任务上准确率+15%
- **多维度RM**：安全、有用、真实多目标融合，权重动态调整，是GPT-4对齐的关键