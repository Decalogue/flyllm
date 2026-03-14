# 特殊标记 [CLS]、[SEP]、[PAD]、[MASK] 分别起什么作用？为什么需要它们？

## 1. 核心定性
本质上，**特殊标记（Special Tokens）** 是预训练任务与模型架构之间的控制信号载体，通过在离散 token 空间中注入结构化语义指示符，使自监督任务（MLM、NSP）能在统一序列表示中实现任务路由和边界感知。

## 2. 具体流程
1. **序列包装**：在句子前后插入 `[CLS]`（分类标记）和 `[SEP]`（分隔标记），如 `"[CLS] Hello world [SEP] How are you [SEP]"`，将多个句子打包为单序列以适应 Transformer 的固定输入格式
2. **填充对齐**：在批处理时，短序列用 `[PAD]`（填充标记）扩展到统一长度 `max_len`，注意力掩码 `attention_mask` 标记有效位置，防止梯度计算涉及填充部分
3. **掩码预测**：MLM 任务随机选择 15% token 替换为 `[MASK]`，模型需从上下文中推理原始词，预训练目标为 $\mathcal{L}_{\text{MLM}} = -\log p(w_{\text{orig}} \mid \text{[MASK]}, \text{context})$

## 3. 数学基础
设输入序列 $X = [x_1, x_2, ..., x_L]$，特殊标记集合 $T = \{[CLS], [SEP], [PAD], [MASK]\}$

**扩展序列**:
$$\tilde{X} = \text{concat}([CLS], X_1, [SEP], ..., X_k, [SEP], [PAD]_{p})$$

**注意力掩码**:
$$\text{attention\_mask}_i = \begin{cases}
1 & \text{if } \tilde{x}_i \neq [PAD] \\
0 & \text{if } \tilde{x}_i = [PAD]
\end{cases}$$

**位置编码偏移**:
$$\text{pos}([CLS]) = 0, \quad \text{pos}([SEP]_j) = |X_{\leq j}| + j$$

**MLM 损失函数**:
$$\mathcal{L}_{\text{MLM}} = -\frac{1}{|M|} \sum_{i \in M} \log \frac{\exp(h_i^T e_{w_i})}{\sum_{v \in V} \exp(h_i^T e_v)}$$

其中 $M$ 为掩码位置集合，$h_i$ 是第 $i$ 层输出，$e_v$ 是词 $v$ 的嵌入。

**NSP 损失函数**（BERT-base 使用）：
$$\mathcal{L}_{\text{NSP}} = -\log p(\text{IsNext} \mid h_{[CLS]})$$

**标记概率**:
- `[CLS]` 最终表示 $h_{[CLS]}$ 聚合全句信息：$h_{[CLS]} = \text{Transformer}([CLS], X)$
- `[MASK]` 的替换策略：80% `[MASK]`，10% 随机词，10% 原词，避免预训练-推理不一致

## 4. 工程考量
**Trade-off**:
- **额外参数 vs 任务适应性**：特殊标记增加 4 个嵌入向量（768-d）仅 12k 参数，但使单模型能适配分类、生成、排序等多种任务，避免为每个任务微调独立模型
- **序列长度占用**："[CLS]" + "[SEP]" 占用 2 个有效位置，在 512 长度限制下减少 0.4% 容量，但换来结构化表示能力，收益远大于成本

**致命弱点**:
- **位置敏感性**：`[CLS]` 必须在首位，位置编码固定为 0，若误放中间会导致下游分类错误率飙升 30-40%
-   <unk> 陷阱  ：`[MASK]` 仅预训练使用，若推理时输入 `[MASK]`，模型表现极差（困惑度上升 2-3 倍），需显式过滤或替换为 `[UNK]`

- **任务耦合**：NSP 任务在 RoBERTa 中被证明无效，但 `[CLS]` 和 `[SEP]` 保留，造成预训练-下游任务不匹配，需手动屏蔽 NSP 损失

## 5. 工业映射
在工业界，特殊标记是 **预训练模型微调的必修课**：
- **BERT 系列**：Hugging Face 的 `BertTokenizer` 自动将 `[CLS]`、`[SEP]` 加入序列，`[PAD]` 通过 `tokenizer.pad()` 批量处理，`attention_mask` 自动对齐。在 GLUE 基准测试上，`h_{[CLS]}` 接线性层即可达到 SOTA，无需复杂架构
- **GPT 系列**：舍弃 `[CLS]` 和 `[SEP]`，仅用 `[PAD]` 填充和 `[MASK]`（部分变体），因自回归生成无需句子边界标记，序列通过位置编码自然区分。在 GPT-3 175B 中，特殊标记从 4 个减至 2 个，嵌入层参数量减少 2k
- **T5 / mT5**: Google 创新使用 `[CLS]` 之外的任务特定标记如 `[SENTIMENT]`、`[SUMMARIZE]` 作为前缀，单模型实现多任务，在 T5-11B 上通过 prompt token 避免微调，零样本性能提升显著
- **ALBERT**: 通过参数共享和句子顺序预测（SOP）替代 NSP，`[CLS]` 保留但嵌入层仅为 128-d 投影，内存占用压缩到 BERT-base 的 1/10，推理速度提升 4-5 倍
