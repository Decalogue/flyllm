# WordPiece 和 BPE 的区别在哪里？BERT 为什么选择 WordPiece？

## 1. 核心定性
本质上，**WordPiece** 是融合语言模型似然的 BPE 变体，通过最大化语言模型似然而非简单频数统计来合并字符对，确保每个合并操作最大化词序列的联合概率从而捕获更优的语义边界。

## 2. 具体流程
1. **初始化与概率建模**：从字符集开始，WordPiece 统计共现对并计算合并后的语言模型似然增量：$\Delta = P(\text{merged}) - P(x)P(y|x)$，其中 $P$ 基于短语的 n-gram 概率
2. **似然最大化合并**：每轮选择使语料似然提升最大的对 $(x, y)$ 合并，而非频数最高的对，确保合并的 token 在上下文中具有语义连贯性
3. **特殊处理机制**：WordPiece 引入 `##` 前缀标记子词（如 "playing" -> "play + ##ing"），前缀匹配保证解码时唯一且可逆，而 BPE 可能导致歧义切分（如 "ear" + "ly" 或 "e" + "arly"）

## 3. 数学基础
设语料 $C = \{w_1, w_2, ..., w_N\}$，每个词 $w$ 可划分为子词序列 $\pi = [s_1, s_2, ..., s_k]$。

**合并评分函数**：对于候选对 $(x,y)$，计算似然增益：
$$\text{Score}(x,y) = \frac{P(xy)}{P(x)P(y)} = \frac{\text{count}(xy)}{\text{count}(x)\text{count}(y)}$$

其中条件概率使用最大似然估计：
$$P(y|x) = \frac{\text{count}(xy)}{\text{count}(x)}, \quad P(x) = \frac{\text{count}(x)}{|C|}$$

**训练目标**：最大化词序列的对数似然：
$$\mathcal{L}(V) = \sum_{w \in C} \log P(w) = \sum_{w \in C} \log \sum_{\pi \in \text{cuts}(w,V)} \prod_{i} P(s_i)$$

**Viterbi 解码**：编码时使用动态规划找最优切分：
$$\text{best}(i) = \max_{j<i} \{\text{best}(j) + \log P(w[j:i])\}$$
其中 $P(w[j:i])$ 是子串的词表概率（若不在词表则为 $-\infty$）

**复杂度**:
- 训练：O(N·|V|·L) 因需计算每对似然增益，比 BPE 的 O(N·L) 慢 3-5 倍
- 编码：动态规划 O(L^2) 或优化后 O(L) 使用 Trie 树和前向匹配

## 4. 工程考量
**Trade-off**:
- **语义连贯性 vs 计算成本**：WordPiece 的似然最大化捕获更好语义边界（如 "don't" 保持完整而非 "don" + "'t"），但训练时间比 BPE 多 2-3 倍，且似然计算需在内存中维护 n-gram 统计
- **确定性与灵活性**：WordPiece 的 `##` 前缀保证解码唯一性，但限制了子词组合自由度，无法处理 "un + happy -> unhappy" 这类派生

**致命弱点**:
- **领域适应**：在 STEM 文本（化学式"H2O"、代码"if("）中，WordPiece 的似然估计可能偏好低频完整 token，导致词表膨胀和稀有技术术语切分错误
- **跨语言迁移**：WordPiece 对语言模型似然的依赖使其在资源匮乏语言上易过拟合，词表可能过度捕获训练语料的统计巧合

## 5. 工业映射
在工业界，WordPiece 是 **Google 双向编码模型的专属选择**：
- **BERT / BERT-wwm**：Google 使用 WordPiece 构建 21k 中文词表（L-12_H-768_A-12），`##` 前缀标记让 MLM 任务能够精确预测掩码位置，在 MLM 损失上比 BPE 低 0.5-1.0 个点
- **T5 / mT5**：虽然 mT5 用 SentencePiece，但英文 T5 仍采用 WordPiece，因其编码器-解码器结构需要精确边界控制，`##` 标记在 Span Corruption 中减少对 partial token 的误掩码
- **ALBERT 优化**：为减少参数量，ALBERT 在 WordPiece 基础上使用参数共享，词表压缩到 5k 仍保持性能，证明 WordPiece 的子词语义密度高于 BPE
- **工程妥协**：实际部署中，Hugging Face 通过 `BertTokenizer` 封装 WordPiece，将 `##` 前缀逻辑隐藏在 `wordpiece_tokenizer.py` 中，对外提供统一接口但内部保持兼容性
