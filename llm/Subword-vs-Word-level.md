# 为什么子词分词比词级分词更适合大模型？遇到稀有词怎么处理？

## 1. 核心定性
本质上，**子词分词**通过将词拆分为可复用的子词单元，以固定的有限词汇表实现了对无限词汇的压缩表示，结合参数共享机制在避免词级分词的稀疏性诅咒的同时捕获了跨词的形态学规律。

## 2. 具体流程
1. **词汇表膨胀控制**：词级分词需维护 $|V| \approx |\text{unique words}|$ 的词表（可达 1000 万+），而子词分词通过 BPE/UniGram 将词表压缩至 3-50k，每个稀有词被拆解为高频子词（如 "finitesimal" → "fin" + "ites" + "imal"）
2. **参数效率优化**：LLM 的嵌入矩阵规模为 $|V| \times d_{model}$，子词分词的嵌入参数减少 $\frac{|V_{word}|}{|V_{subword}|}$ 倍，使模型在相同参数量下可分配到更深的层或更大的注意力维度
3. **稀有词处理机制**：OOV（词表外）词通过最小字符回退（<unk> → 字符级别）或字节级 BPE（任何 Unicode 映射到 256 字节），保证 100% 覆盖率和在线处理能力

## 3. 数学基础
设语料 $C$ 的词汇集合 $W$，词级分词的嵌入矩阵参数：
$$P_{word} = |W| \times d$$

子词分词词汇表 $V \subset W$（通常为高频子串），稀有词 $r \in W \setminus V$ 的切分函数：
$$\text{tokenize}(r) = [s_1, s_2, ..., s_k] \text{ where } k \geq 1, s_i \in V$$

**参数压缩比**：
$$\text{Compression Ratio} = \frac{|W|}{|V|} \approx \frac{10^7}{3 \times 10^4} \approx 300$$

**稀疏性问题**：词级分词下稀有词嵌入更新次数 $N_r \ll |C|$，导致过拟合。子词拆分为 $k$ 个子词后，每个子词更新次数：
$$N_{s_i} = \sum_{r: s_i \in \text{tokenize}(r)} N_r \approx \frac{N_{s_i}}{k}$$

通过参数共享，稀有词通过其子词嵌入的组合获得泛化能力：
$$\text{emb}(r) \approx \sum_{i=1}^{k} \text{emb}(s_i)$$

**字节级回退**：
对于任何 U+XXXX 格式的 Unicode 字符 $c$，映射到字节序列：
$$\phi(c) = [b_1, b_2, ..., b_m] \text{ where } b_i \in [0, 255]$$
确保输入字符集大小恒为 256，100% 覆盖且可逆。

## 4. 工程考量
**Trade-off**:
- **紧凑性 vs 序列长度**：子词分词增加序列长度 $L_{sub} = k \cdot L_{word}$，Transformer 注意力复杂度 $O(L^2d)$，导致推理延迟增加 10-20%。但通过 GQA 和 FlashAttention 可部分抵消
- **通用性 vs 语言特定性**：子词虽支持多语言，但表意文字（中日韩）和字母文字（英法德）在相同参数下压缩率差异大，需调 `-character_coverage` 参数或使用 SentencePiece 的字节级模式

**致命弱点**:
- **冷启动语义漂移**：稀有实体（如 "Covid-19"）被过度切分为 "Cov" + "id" + "-" + "19"，破坏语义完整性，导致实体识别准确率下降 5-8 点
- **粒度失配**：子词边界与语义边界不总是对齐，如 "unhappy" 切分为 "un" + "happy" 是理想情况，但 "database" 切分为 "data" + "base" 可能错误关联两个无关概念

## 5. 工业映射
在工业界，子词分词是 **LLM 部署的强制性要求**：
- **GPT-4 的 Byte-level BPE**: tiktoken 库将 100k 词表与字节级回退结合，对中文使用 UTF-8 编码为 3-4 字节序列，平均压缩率在 3.2，确保中英文统一处理。在推理时，任何输入字符都能编码为已知 token 的组合，避免 <unk> 导致的信息丢失
- **T5 mT5 的 SentencePiece**: Google 在多语言模型中采用子词 + 字符覆盖混合策略，`--character_coverage=0.99995` 保留 5% 字符给稀有字符，在 XNLI 上比纯词级分词提升 2.3 个百分点
- **稀有词处理策略**: 在检索增强（RAG）场景中，面对专有名词和术语，采用严格最小切分：先在词表查完整词，未命中则选最粗子词切分（如 "retinopathy" → "retinopathy" 而非 "re" + "tin" + "opathy"），通过 `--split_by_whitespace=false` 控制，在医疗领域 QA 中准确率提升 4%
