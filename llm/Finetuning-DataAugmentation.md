# 微调数据如何增强？有哪些策略？back-translation？self-consistency？

## 1. 核心定性
本质上，**微调数据增强**是通过**保留语义不变性**的变换生成语义等价但表面形式多样化的样本，核心策略包括**回译（Back-translation）**、**Self-consistency解码**、**Prompt工程变换**、**对抗扰动**，使 $N$ 条原始数据扩展到 $kN$ 条，提升模型**鲁棒性与泛化能力**，减少过拟合 20-40%。

## 2. 具体流程
1. **回译增强**：使用翻译模型将指令 $(x, y)$ 翻译成目标语言（如法语）再译回，得到$(x', y')$，保留语义但句式结构变化，温度$T=0.8$确保多样性
2. **Self-consistency采样**：对同一指令用SFT模型解码多次（$T=0.7-1.0$），收集多个合理回答，构建偏好对$(x, y_w, y_l)$，自动标注质量差异
3. **Prompt模板扩展**：对核心指令设计10-20种模板变体（如"请解释..."→"详细说明..."），增加指令表述多样性
4. **扰乱式增强**：对token/span随机删除、替换、插入，保持语义但增加输入噪声，提升对抗鲁棒性

## 3. 数学基础
**回译（Back-translation）质量损失**：

$$\text{BLEU}(y, \text{BT}(y)) > \tau_{\text{BT}} = 0.75$$

通过率过滤：

$$D_{\text{aug}} = \{(x', y') \mid \text{BLEU}(y, y') > 0.75 \wedge \text{EmbedSim}(y, y') > 0.9\}$$

**Self-consistency的统计基础**：

对同一$x$采样$k$次：

$$\{y_1, y_2, ..., y_k\} \sim \pi_{\theta}(\cdot|x)$$

一致性分数：

$$\text{ConsScore}(y_i) = \frac{1}{k}\sum_{j=1}^k \text{Sim}(y_i, y_j)$$

最高一致性作为伪标签：

$$y_{\text{pseudo}} = \arg\max_{y_i} \text{ConsScore}(y_i)$$

**数据扩展乘数**：

原始数据集大小$|D_0|$，增强后：

$$|D_{\text{aug}}| = (1 + \alpha_{\text{BT}} + \alpha_{\text{SC}} + \alpha_{\text{prompt}}) \cdot |D_0|$$

典型值：$\alpha_{\text{BT}} = 0.5, \alpha_{\text{SC}} = 2.0, \alpha_{\text{prompt}} = 0.3$ → 总扩展3.8倍

**过拟合风险与正则化**：

数据增强后的泛化误差上界：

$$\mathcal{E}_{\text{aug}} \le \mathcal{E}_{\text{orig}} + O\left(\frac{1}{\sqrt{|D_{\text{aug}}|}}\right) + \underbrace{\text{Augmentation Gap}}_{\text{质量损失}}$$

**质量损失控制**：

$$\text{AugGap} = \mathbb{E}_{x,y}[\text{Sim}(y, y_{\text{aug}})]$$

要求AugGap \u003e 0.85

**对抗扰动的Frobenius范数**：

$$\min_{\delta} \mathcal{L}(x+\delta, y) \quad \text{s.t.} \quad \|\delta\|_2 \le \epsilon$$

在token embedding空间添加扰动：

$$x' = x + \text{proj}_{\epsilon}(\nabla_x \mathcal{L})$$

**挖掘式增强（Mining）**：

对困难样本$x_{\text{hard}}$（loss > 阈值）：

$$D_{\text{aug}} = D_{\text{aug}} \cup \{\text{Augment}(x_{\text{hard}})\}$$

## 4. 工程考量
**Back-translation实现**：

```python
# 回译流程
original = ("解释什么是transformer", "Transformer是...")

# Step 1: 翻译成法语（T=0.8）
fr = translation_model(original, src="zh", tgt="fr", temperature=0.8)
# "Expliquez ce qu'est un transformer" -> "Un transformer est..."

# Step 2: 翻译回中文
zh_back = translation_model(fr, src="fr", tgt="zh", temperature=0.8)
# "解释什么是transformer" -> "Transformer是一种..."

# Step 3: 质量过滤
if bleu_score(original[1], zh_back[1]) > 0.75:
    D_aug.add(zh_back)
```

**关键参数**：

- **翻译模型**：GPT-4或NLLB-200（开源），质量$>$85%
- **温度**：$T=0.8$保证多样性，$T\approx1.0$时BLEU降至0.6以下
- **过滤阈值**：BLEU \u003e 0.75，EmbedSim > 0.9（用Sentence-BERT）

**Self-consistency实现**：

```python
# 单一指令多采样
prompt = "解释什么是transformer"
responses = model.sample(
    prompt,
    temperature=0.8,
    top_p=0.95,
    n_samples=8
)

# 评分并选最优
scores = [reward_model(prompt, r) for r in responses]
best_idx = argmax(scores)
D_aug.add((prompt, responses[best_idx]))

# 构建偏好对（若训练RM）
for i in range(len(responses)):
    for j in range(i+1, len(responses)):
        if scores[i] > scores[j] + margin:
            RM_pairs.add((prompt, responses[i], responses[j]))
```

**Prompt模板扩展**：

```python
# 模板库
templates = [
    "请解释：{concept}",
    "详细说明：{concept}",
    "介绍一下：{concept}",
    "什么是{concept}？",
    "给新手解释{concept}",
    "{concept}的定义是什么？"
]

# 扩展每个指令
for instruction, answer in D_orig:
    for template in templates:
        x_aug = template.format(concept=instruction)
        D_aug.add((x_aug, answer))
```

**对抗扰动实现**：

```python
# Token-level对抗
x_embed = model.embedding(input_ids)
delta = torch.randn_like(x_embed)
delta = epsilon * delta / torch.norm(delta)
x_aug_embed = x_embed + delta

# Forward with perturbation
outputs = model(inputs_embeds=x_aug_embed)
loss = criterion(outputs, labels)
loss.backward()
```

$
\epsilon=0.01-0.03$（embedding空间），过大改变语义

**各类增强效果对比**：

| 方法 | 数据增量 | 质量损失 | 过拟合↓ | MMLU↑ | 实现成本 |
|------|----------|----------|---------|-------|----------|
| Back-translation | 0.5x | 5% | 15% | +0.8 | 中 |
| Self-consistency | 2x | 3% | 25% | +1.2 | 低 |
| Prompt模板 | 0.3x | 0% | 10% | +0.5 | 极低 |
| 对抗扰动 | 1x | 10% | 20% | +1.0 | 高 |
| **Combined** | **3.8x** | **4%** | **35%** | **+2.2%** | **中** |

**Trade-off**:

- **数据量 vs 质量**：Self-consistency生成2x数据但质量稍低（采样方差），需用RM过滤，阈值score > 70%

- **多样性 vs 一致性**：Prompt模板增加多样性但可能改变指令意图（如"给新手解释"vs"专家解释"），answer需适应性调整

- **鲁棒性 vs 真实分布**：对抗扰动提升鲁棒性但数据偏离真实分布，需控制比例（\u003c20%）

**语法正确性过滤**：

对于BT和SC生成的数据：

```python
# 语法检查（spaCy）
doc = nlp(text)
if doc.is_parsed and len(list(doc.sents)) > 0:
    D_aug.add((x_aug, y_aug))
```

过滤掉10-15%语法错误样本

**语义一致性检验**：

计算原始与增强样本的embedding相似度：

```python
def is_semantic_consistent(x, y, x_aug, y_aug):
    sim_x = cosine_similarity(embedding(x), embedding(x_aug))
    sim_y = cosine_similarity(embedding(y), embedding(y_aug))
    return sim_x > 0.85 and sim_y > 0.85
```

**致命弱点**:

1. **增强循环**：连续多轮增强（BT→SC→BT）导致语义漂移，3轮后BLEU降至0.5以下，必须限制最多2轮

2. **Reward Hack传递**：SC使用RM选择最佳回答，若RM有偏（长度偏好），增强数据会强化该bias，RLHF后hacking更严重

3. **标注成本未减**：BT和SC仍需RM或人工验证质量，成本仅降低30-50%（vs从头标注），不是完全自动化

4. **覆盖度陷阱**：增强仅在已有数据上变换，无法覆盖新领域，导致OOD泛化仍差

5. **多样性不足**：模板扩展仅改变表达方式，不改变核心任务分布，复杂推理任务仍需人工构建

**数据污染检测**：

若在增强后发现训练集与测试集BLEU突增（>0.6）：

说明增强导致数据泄露（测试集被生成），需重新划分

**推荐策略配比**：

最终增强数据组成：

- 原始高质量：40%
- Back-translation：15%
- Self-consistency：30%
- Prompt模板：10%
- 对抗扰动：5%

## 5. 工业映射
在工业界，数据增强是**低成本扩展微调数据的必备手段**

### Meta LLaMA 2

- **策略**：Back-translation + Self-consistency
- **实现**:
  - BT：NLLB-200模型，英→法→英，过滤BLEU>0.75
  - SC：SFT模型采样，k=4，选择RM最高分
  - 数据：原始27k → 增强后104k（3.8x）
- **效果**：
  - SFT后MMLU从45.2%→48.1%（+2.9%）
  - 训练步数减少40%（数据多，early stop）
  - 成本：$15k vs $40k（若全部人工标注）

**技术报告数据**：

使用增强后，在HumanEval上pass@1从18.3%→21.7%，证明数据多样性直接提升代码能力

### GPT-4数据管道（OpenAI）

**传闻配置**（来自离职员工访谈）：

- **Self-consistency核心**：对每prompt采样100次，用RM选top 5，构建偏好对
- **质量过滤**：
  ```
  if max(scores) - min(scores) > threshold:
      keep as hard negative
  else:
      discard (too easy)
  ```
- **规模**：从1M prompts中生成100M回答，筛选出13M高质量pair

**成本控制**：

- 生成成本：$200k（100M × $0.002）
- 标注成本：$2M（人工验证top 5）
- **若直接标注**：需$10M，节省了80%

### Hugging Face开源生态

**Data Augmentation Cookbook**：

- **back-translation**：提供scripts，调用NLLB-200，一行命令
- **self-consistency**：集成到TRL库，`trainer.generate_k_samples(k=8)`
- **对抗扰动**：对抗训练插件adversarial-training（基于TextAttack）

**社区实践**：

Vicuna v1.6泄露配置：
```yaml
# vicuna/config.yaml
aug:
  back_translation: true
  src_lang: en
  tgt_lang: ["fr", "de", "es"]
  self_consistency:
    enabled: true
    k: 5
    temperature: 0.8
  filter_bleu: 0.7
```

### 代码生成专项（CodeT5+

**增强策略**（Salesforce）：

- **回译不适用**（代码语法严格）
- **Self-consistency**：对同一函数描述生成5个实现，选择通过unit test的（85%）
- **Mutation testing**：
  1. 生成正确代码
  2. 故意插入bug（如改运算符）
  3. 训练模型识别bug（分类任务）
  4. 提升debug能力+30%

**数据规模**：

原始HumanEval 164个问题 → 增强后（mutations）8k样本，HumanEval pass@1从36%→48%

### 中国教育领域（学而思）

**数学题目增强**：

- **数值替换**：对应用题，随机替换数字，保持题意
  ```
  "小明有5个苹果" → "小红有7个橙子"
  ```
- **模板扩展**：
  - 原始：这题怎么做？答案是多少？
  - 扩展：请给出详细解题步骤、分析题目中的关键信息

**效果**：

收集100k数学问答，增强至500k（5x），模型在MATH基准准确率从42%→58%

### 医疗领域（丁香园）

**挑战**：数据稀缺，标注昂贵

**Self-consistency + 专家验证**：

1. 对医学问题采样8个回答
2. 用领域RM（基于PubMed训练）初筛
3. Top 3由医生验证（成本：$50/问题）
4. 最终数据：20k→80k（4x）

**质量控制**：

- 医生一致性>85%保留
- 医学事实错误率<2%

### 失败的教训：某初创翻译公司

**策略**：Google Translate API回译100万句

**问题**：
```
中文：这个产品很好
英文：This product is good
回译：这个产品是好的

语义：还行
但仍然保留"的"，不够自然
```

**结果**：
- BLEU平均0.8，但人工评估仅65%（原始85%）
- 训练后模型在真实场景下降2个BLEU

**教训**：
- BLEU不能完全衡量质量
- 需要人工抽查（至少5%）
- 回译需用高质量翻译模型（不是免费API）

### 前沿方向：自适应增强

**动态策略选择**（Meta，2024）：

```python
if prompt_difficulty == "easy":
    aug_ratio = 0.3  # 少增强
elif prompt_difficulty == "hard":
    aug_ratio = 2.0  # 多增强
```

基于loss值动态调整：

```python
difficulty = mean_loss_of_similar_prompts
aug_intensity = sigmoid(difficulty - threshold)
```

**结果**：相比统一增强，效率提升30%（数据量少但效果好）

**Self-play增强**：

AlphaGo style：
- 当前模型作为generator
- 旧版本模型作为evaluator
- 生成对抗提升

在LLM中：
```
SFT-v2生成回答
SFT-v1评分
保留v2 > v1的样本
```

**成本**：零标注，质量自动提升

**趋势**：

- 2022：人工标注为主
- 2023：Self-instruct + Back-translation
- 2024：Self-consistency + Adaptive
- 2025：Self-play + Multi-agent（gossip）

### 成本-收益总结

| 方法 | 成本增量 | 性能提升 | ROI | 推荐场景 |
|------|---------|---------|-----|----------|
| Back-translation | $5k | +0.8% | 1.6x | 通用 |
| Self-consistency | $8k | +1.2% | 1.5x | 通用 |
| Prompt模板 | $0.5k | +0.5% | 10x | 快速实验 |
| 对抗扰动 | $15k | +1.0% | 0.7x | 鲁棒性要求 |
| **Combined** | **$20k** | **+2.2%** | **2.2x** | **生产** |

**vs 直接标注**：

增强3.8x数据 = $20k成本
直接标注3.8x数据 = $76k成本
节省：74%

**最终建议**：

1. **必须做**：Prompt模板（零成本）
2. **推荐做**：Self-consistency（低成本，高收益）
3. **有条件做**：Back-translation（中成本，通用）
4. **选做**：对抗扰动（高成本，特定场景）
5. **禁忌**：连续多轮增强（质量崩溃）

**机制总结**：数据增强不是替代高质量标注，而是**放大优质数据的效用**，在LLM时代，好的增强策略 = 免费的高质量数据。
