# SentencePiece 和 BPE 有什么区别？为什么 GPT 系列用 BPE，而 T5 用 SentencePiece？

## 1. 核心定性
本质上，**SentencePiece** 是 BPE/Unigram 算法的工业化封装层，通过将分词建模为字符串到 token ID 的序列标注问题并引入可微调的似然目标，解决了多语言字符集不兼容和词汇表可扩展性问题。

## 2. 具体流程
1. **句子级建模**：SentencePiece 将原始文本视为 Unicode 字符串，不依赖预分词（whitespace 或 WordPiece），通过 SentencePiece 模型（SPM）直接在字节流上操作
2. **训练目标差异**：BPE 使用频数驱动的贪婪合并，而 Unigram SentencePiece 通过 EM 算法最大化语料似然：$\mathcal{L} = \sum_{s \in \text{CORPUS}} \log P(s)$，其中 $P(s) = \prod_{i} P(t_i|s)$ 是 token 概率
3. **工业级优化**：SentencePiece 采用 SentencePiece 原生格式存储词表，支持反向索引和 trie 树加速编码，在 NMT 中通过 `--character_coverage` 参数控制稀有字符覆盖率

## 3. 数学基础
设语料 $S = \{s_1, s_2, ..., s_N\}$，每个句子 $s_i$ 可划分为token序列：
$$s_i = t_{i,1} + t_{i,2} + ... + t_{i,|s_i|}$$

**BPE SentencePiece**:
$$\text{合并对} = \arg\max_{x,y} \sum_i \sum_j \delta(t_{i,j}=x \land t_{i,j+1}=y)$$

**Unigram SentencePiece**:
$$V = \arg\max_{V' \subseteq V_{\text{candidate}}} \sum_{i} \log \sum_{\pi \in \Pi(s_i, V')} p(\pi)$$

其中每一步采样的概率：
$$p(\pi) = \prod_{j} p(t_j \mid \theta), \quad \sum_{t \in V'} p(t \mid \theta) = 1$$

**词表剪枝**: 每轮迭代通过损失函数评估 token 重要性：
$$L(t) = -\sum_{n:v_n^{\text{rpn}}\,\forall t} \log p(t \mid \theta)$$

**编码复杂度**:
- 训练：O(N·L) L 为平均句长，与 BPE 相似
- 推理：Viterbi 解码搜索最优路径，复杂度 O(L·|V|) 使用动态规划优化

## 4. 工程考量
**Trade-off**:
- **语言独立性 vs 计算成本**：SentencePiece 的字节级建模支持任意 UTF-8 文本，但每个 token 前向传递需遍历 trie 树，比 BPE 的直接匹配慢 1.2-1.5 倍
- **词汇表可扩展性**：Unigram 可通过调整先验概率$p(t)$灵活增删 token，而 BPE 合并操作不可逆（除非重新训练），适应新语言需从头开始

**致命弱点**:
- **训练不稳定**：Unigram 的 EM 算法对初始化敏感，不同随机种子可能收敛到局部最优，词表质量差异可达 ±5% BLEU 分数
- **资源开销**：SentencePiece 模型（spm.model）需存储词表、分数和反向索引，体积比纯 BPE 词表（txt 格式）大 2-3 倍，边缘部署受限

## 5. 工业映射
在工业界，SentencePiece 是 **Google 多语言 NLP 的标配**而 BPE 统治 **英文预训练**：
- **T5 / mT5**: Google 在多语言任务中采用 SentencePiece，词汇表大小 250k，通过 `--character_coverage=0.9995` 确保低资源语言字符覆盖，在 XNLI 上实现 85.9% zero-shot 准确率
- **GPT-4**: OpenAI 坚持使用 BPE（tiktoken）因其英文语料占比超 95%，BPE 的贪婪合并更符合英语构词规律，且词表格式简洁便于跨平台部署
- **LLaMA 2**: 采用 Byte-level BPE 而非 SentencePiece，平衡了字节级通用性和 BPE 效率，在 15T token 上训练出 32k 词表，词表压缩率达到 3.8
- **工程妥协**: 实际部署中，Byte-level BPE 被视为"轻量级 SentencePiece"，既保留字节级回退又避免 Unigram 的复杂训练，成为英文 LLM 默认选择
