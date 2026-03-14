# Unigram Language Model Tokenizer 的工作原理是什么？它有什么优势？

## 1. 核心定性
本质上，**Unigram Language Model Tokenizer** 是一种基于概率语言模型的反向迭代剪枝算法，通过从超大候选词表出发逐步删除对语料似然贡献最小的 token，实现全局最优而非贪心局部最优的子词划分。

## 2. 具体流程
1. **初始化大词表**：从语料中提取所有高频子串（通常 n-gram，n∈[1, 8]）构建初始候选词表 $V_0$，大小可达目标词表的 5-10 倍（如 1M→32k）
2. **概率化建模**：将每个句子 $x$ 的切分视为隐变量，使用 EM 算法最大化对数似然：$\mathcal{L}(V) = \sum_{x} \log p(x)$，其中 $p(x) = \sum_{s \in \mathcal{S}(x,V)} p(s)$ 是该句子所有可能切分的概率和
3. **迭代剪枝**：每轮计算每个 token $v \in V_i$ 的损失增量 $\Delta \mathcal{L}(v)$，删除损失最小的 $\alpha|V_i|$ 个 token（通常 $\alpha=0.1$），直到达到目标词表大小或似然收敛

## 3. 数学基础
设候选词表 $V$，句子 $x = x_1x_2...x_T$，所有合法切分定义为：
$$\mathcal{S}(x, V) = \{[s_1, s_2, ..., s_m] \mid s_i \in V, s_1 + s_2 + ... + s_m = x\}$$

**Token 概率**：每个 token $v \in V$ 有概率 $p(v)$ 满足 $\sum_{v \in V} p(v) = 1$，极大似然估计为：
$$p(v) = \frac{\text{count}(v)}{\sum_{t \in V} \text{count}(t)}$$

**句子概率**：
$$p(x) = \sum_{s \in \mathcal{S}(x,V)} \prod_{i=1}^{|s|} p(s_i)$$

**损失计算**：使用 Forward-Backward 算法计算 token $v$ 删除后的负对数似然变化：
$$\Delta \mathcal{L}(v) = -\sum_{x} \log p(x \mid V \setminus \{v\}) + \sum_{x} \log p(x \mid V)$$

**EM 更新**：
- E 步：计算期望计数 $E[\text{count}(v)] = \sum_{x} p(v \mid x)$
- M 步：更新概率 $p(v) \propto E[\text{count}(v)]$

**复杂度**:
- 训练：O(N·L·|V|) 每轮 EM，需要计算所有可能切分，在 N=10M 语料上需数小时
- 推理：使用 Viterbi 算法找最大概率切分，O(L·|V|) 使用 Trie 树优化

## 4. 工程考量
**Trade-off**:
- **全局最优 vs 训练成本**：Unigram 通过多次迭代剪枝逼近全局最优，比 BPE 贪心合并的 BLEU 高 0.5-1.0，但训练时间增加 5-10 倍
- **概率灵活性 vs 确定性**：Unigram 可输出多个切分的概率分布，适合概率化建模，但推理时需 Viterbi 解码，比 BPE 的贪心匹配慢 1.2-1.5 倍

**致命弱点**:
- **内存爆炸**：初始候选词表可达数百万，存储所有 n-gram 统计需要数十 GB 内存，在处理 TB 级语料时限制严重
- **超参数敏感**：剪枝步长 $\alpha$、初始词表大小、平滑参数 $\lambda$ 对最终结果影响大，不同语言需手动调优，自动化程度低

## 5. 工业映射
在工业界，Unigram 是 **Google 多语言翻译的基石**：
- **mBART / mT5**: Google 使用 SentencePiece Unigram 训练多语言模型，词汇表 250k，在 101 种语言上平均 BLEU 比 BPE 高 1.5，尤其在低资源语言（斯瓦希里语、高棉语）上提升显著达 3-4 BLEU
- **内部实现**：sentencepiece 库的 `spm_train --model_type=unigram` 使用预采样策略，通过 `--input_sentence_size=10000000` 限制内存占用，在 100M 句子上 8 小时完成训练并自动优化 Trie 结构
- **推理优化**：为弥补解码慢的问题，sentencepiece 将 Viterbi 算法用 C++ 重写并引入 SIMD 指令，在 x86 上达到 50k token/s，接近 BPE 的 65k token/s
- **取舍决策**：BERT 系列未采用 Unigram 主要因 2018 年计算资源限制，若在今天重训，Unigram 会是更优选择，尤其在处理多义切分（"un + happy" vs "unhapp + y"）时更灵活
