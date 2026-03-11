# 预训练 vs 微调 vs 指令微调

## 1. 核心定性 (The 10-Second Hook)

| 阶段 | 本质定义 |
|------|----------|
| **预训练** | 在海量无标注语料上通过自监督学习任务（MLM/CLM）构建通用语义表征空间，习得语言知识与世界常识 |
| **微调** | 在特定领域标注数据上，以预训练参数为初始化，通过监督学习将通用能力适配到下游任务 |
| **指令微调** | 通过 (指令, 输入, 输出) 三元组形式的数据，训练模型理解并遵循人类指令意图，实现跨任务泛化 |

---

## 2. 具体流程

**预训练**
1. 准备大规模无标注文本语料（TB 级）
2. 设计自监督目标（如 MLM: 随机 mask 15% token 预测；CLM: 因果语言建模）
3. 通过分布式训练优化参数，产出基础模型（如 GPT/Llama）

**微调 (Fine-tuning)**
1. 收集下游任务标注数据（如情感分类、NER）
2. 冻结或低学习率更新预训练参数
3. 在任务目标上收敛，产出专用模型

**指令微调 (Instruction Tuning)**
1. 构建指令格式数据集：$\{(instruction_i, input_i, output_i)\}_{i=1}^N$
2. 使用 next-token prediction 目标：$\mathcal{L} = -\sum_{t} \log P(x_t | x_{<t}; \theta)$
3. 使模型学会"按指令执行"的元能力

---

## 3. 数学基础

### 预训练目标（以 CLM 为例）

$$\mathcal{L}_{\text{pretrain}}(\theta) = -\sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P_\theta(x_t^{(i)} | x_{<t}^{(i)})$$

其中：
- $N$: 训练样本数
- $T_i$: 第 $i$ 个序列长度
- $x_t^{(i)}$: 第 $i$ 个序列的第 $t$ 个 token
- $\theta$: 模型参数

### 微调目标

$$\mathcal{L}_{\text{finetune}}(\theta) = -\sum_{(x,y) \in \mathcal{D}_{\text{task}}} \log P_\theta(y | x) + \lambda \|\theta - \theta_0\|^2$$

其中：
- $\mathcal{D}_{\text{task}}$: 下游任务标注数据集
- $\theta_0$: 预训练参数（L2 正则防止灾难性遗忘）

### 指令微调目标

$$\mathcal{L}_{\text{inst}}(\theta) = -\sum_{(I,X,Y) \in \mathcal{D}_{\text{inst}}} \sum_{t=1}^{|Y|} \log P_\theta(Y_t | I, X, Y_{<t})$$

其中：
- $I$: Instruction（指令）
- $X$: Input（输入上下文）
- $Y$: Output（期望输出）

---

## 4. 工程考量

| 维度 | 预训练 | 微调 | 指令微调 |
|------|--------|------|----------|
| **数据量级** | TB 级，万亿 token | GB 级，万~百万样本 | GB 级，十万~百万条指令 |
| **计算成本** | 极高（数千 GPU·天） | 低（单卡/数小时） | 中等（数十 GPU·天） |
| **核心风险** | 训练不稳定、loss spike | 灾难性遗忘、过拟合 | 指令分布偏移、幻觉增强 |
| **Trade-off** | 通用性 ↔ 计算资源 | 专用精度 ↔ 泛化能力 | 指令遵循 ↔ 原始能力保持 |

**致命弱点**：
- **预训练**：数据污染导致 benchmark 失效；long-tail knowledge 学习不足
- **微调**：全量微调破坏预训练知识；领域过拟合导致 zero-shot 能力下降
- **指令微调**：简单指令过度优化导致模式崩溃；复杂推理指令分布外泛化差

---

## 5. 工业映射

| 技术 | 工业实例 |
|------|----------|
| **预训练** | GPT-4、Claude、Llama-3 的 base 模型训练阶段，使用数万亿 token 在数千张 H100 上训练数月 |
| **微调** | BERT 在金融领域情感分析微调；医疗领域实体识别专用模型 |
| **指令微调** | ChatGPT（InstructGPT 的 SFT 阶段）、Llama-2-Chat、Alpaca、Vicuna 的指令对齐训练 |

**现代 LLM 训练流水线**：
```
Raw Data → 预训练 (Base Model) → 指令微调 (SFT Model) → RLHF/PPO/DPO (Aligned Model)
                    ↓
            可选：领域微调 (Domain-Adapted Model)
```
