---
concept: "Tokenizer核心机制"
template: "对比矩阵 + 工程型"
difficulty: ⭐⭐
importance: 🌟🌟🌟🌟
user_mastery: 0.0
prerequisites: ["字符串编码", "统计频率"]
related_concepts: ["embedding", "subword-tokenization", "unicode"]
category: "LLM"
module: "基础组件"
generated_at: "2026-03-16"
---

# Tokenizer详解

## 1. 核心问题清单（5题）

### 1.1 BPE/SentencePiece/WordPiece 核心思想与区别

**BPE (Byte-Pair Encoding)**:
- 从字符出发,迭代合并最频繁字符对
- 算法复杂度: O(N·V)
- 优点: 简单高效,适合英文
- 应用: GPT系列, RoBERTa

**WordPiece**:
- 类似BPE,但用概率而非频率
- 选择使词表似然度增加最大的合并
- 优点: 训练更稳定
- 应用: BERT, 原版Transformer

**SentencePiece**:
- 把句子当Unicode序列,不依赖语言
- 优势: 支持任意语言,无需预分词
- 特点: 直接训练原始文本
- 应用: 多语言模型(mT5, XLM-R)

**对比表**:
| 维度 | BPE | WordPiece | SentencePiece |
|------|-----|-----------|---------------|
| 合并规则 | 频率 | 似然度 | 频率 |
| 语言依赖 | 有 | 有 | **无** |
| 预分词 | 需要 | 需要 | **不需要** |
| 计算效率 | 高 | 中 | 高 |
| 典型应用 | GPT | BERT | 多语言 |

### 1.2 词汇表大小如何确定?

**权衡分析**:
```
太小(1k):
  ✅ 内存小
  ❌ OOV严重,表达能力差

太大(200k):
  ✅ OOV少
  ❌ Embedding参数爆炸,训练慢

最佳实践:
  - GPT-2: 50,257
  - BERT: 30,522
  - LLaMA-2: 32,000
  - 经验值: 30k-50k (sweet spot)
```

**工程考量**:
- 训练数据量: 数据少 → 小词表
- 模型大小: 大模型 → 可稍大词表
- 多语言: 需更大词表

**显存计算**:
```python
vocab_size = 50000
dim = 768
memory_mb = vocab_size * dim * 4 / 1024 / 1024  # 153.6 MB
print(f"Embedding显存: {memory_mb:.1f} MB")
# BERT 110M参数总显存 ≈ 440MB, Embedding占35%
```

### 1.3 特殊标记 [CLS]/[SEP]/[PAD]/[MASK] 作用

**BERT vs GPT 差异**:

| 标记 | BERT中的用途 | GPT中的处理 |
|------|--------------|-------------|
| **[CLS]** | 句子表示,用于分类 | 不需要(无分类) |
| **[SEP]** | 分隔句子对 | 自然文本换行 |  
| **[PAD]** | 批处理填充 | 同样需要 |
| **[MASK]** | MLM任务掩码 | **不使用**(自回归) |

**面试考点**:
- **为什么BERT需要[CLS]**? 聚合整个句子信息做分类
- **为什么GPT不需要[MASK]**? 因为GPT是predict next token,不是fill-in-the-blank

### 1.4 Tokenizer 对中文的处理

**中文分词挑战**:
```
英文: "ChatGPT is great" → 天然空格分词
中文: "ChatGPT真棒" → 问题
  字符级: ["C","h","a","t","G","P","T","真","棒"] (冗余)
  词级: ["ChatGPT","真棒"] (需要分词器,OOV严重)
```

**字符 vs 词粒度**:

| 粒度 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **字符** | 无OOV,简单 | 序列太长 | 中文BERT |
| **词** | 语义完整,序列短 | 需要分词器,OOV | 传统NLP |
| **子词** | **平衡!** | 复杂度稍高 | **现代LLM** |

**SentencePiece优势**:
```python
# 不依赖任何分词器
text = "ChatGPT真棒，我爱北京天安门"
tokens = spm.encode(text)
# 自动学习: ["Chat","G","PT","真棒",",","我爱","北京","天安门"]
```

### 1.5 Tokenizer 的 Normalization 和 Pre-tokenization

**Normalization（Unicode归一化）**:
```python
# 问题: 同一个字符的不同Unicode表示
s1 = "café"  # e + 重音符(U+0065 U+0301)
s2 = "café"  # é字符(U+00E9)

# 解决方案: NFKC归一化
from unicodedata import normalize
normalize('NFKC', s1) == normalize('NFKC', s2)  # True
```

**Pre-tokenization（预分词）**:
```python
# 英文: 按空格和标点预分词
"Hello, world!" → ["Hello", ",", "world", "!"]

# 中文: 不需要预分词（无空格）
# 直接从字符开始学习
```

**工业界实践**:
- **T5**: 用空格预分词,然后SentencePiece
- **GPT**: 纯BPE,无需预分词
- **BERT**: WordPiece + 基础标点分割

## 2. 手撕BPE算法

```python
def bpe_train(corpus, vocab_size):
    """
    极简BPE训练
    corpus: 词列表
    vocab_size: 目标词表大小
    """
    # 1. 初始化词表: 所有字符
    vocab = set(''.join(corpus))
    
    # 2. 统计相邻字符对频率
    from collections import Counter
    pairs = Counter()
    for word in corpus:
        chars = list(word)
        for i in range(len(chars)-1):
            pairs[(chars[i], chars[i+1])] += 1
    
    # 3. 迭代合并
    while len(vocab) < vocab_size:
        # 找最频繁的pair
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merged = ''.join(best_pair)
        vocab.add(merged)
        
        # 更新语料和pairs
        corpus = [word.replace(merged, ' ' + merged + ' ') for word in corpus]
        corpus = [''.join(word.split()) for word in corpus]
        
        # 重新统计pairs
        pairs = Counter()
        for word in corpus:
            chars = list(word)
            for i in range(len(chars)-1):
                pairs[(chars[i], chars[i+1])] += 1
    
    return vocab

def bpe_encode(text, vocab):
    """用最长的词表项匹配"""
    tokens = []
    while text:
        # 找最长匹配
        match = None
        for token in sorted(vocab, key=len, reverse=True):
            if text.startswith(token):
                match = token
                break
        if match is None:
            # 未匹配，按字符分割
            match = text[0]
        tokens.append(match)
        text = text[len(match):]
    return tokens

# 示例
vocab = bpe_train(['chat', 'cat', 'car', 'bar'], 10)
print(vocab)
# 输出: {'c', 'h', 'a', 't', 'r', 'b', 'ch', 'ca', 'ba', 'chat'}

print(bpe_encode('chatbot', vocab))
# 输出: ['chat', 'b', 'o', 't']
```

**复杂度**: $O(N \cdot V)$, $N$是语料长度,$V$是词表大小。可用trie树优化到$O(N)$。

## 3. 工程优化要点

### 3.1 写入效率

```python
# 逐行写入
import codecs
with codecs.open('corpus.txt', 'w', 'utf-8') as f:
    for line in corpus:
        f.write(line + '\\n')

# SentencePiece训练命令
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='BPE',
    character_coverage=0.9995,  # 中文需要更高
)
```

### 3.2 内存优化

**问题**: 大语料在内存中统计pairs会OOM

**解决方案**:
1. **分批处理**: 每次处理100万行,合并统计结果
2. **磁盘存储**: 用sqlite存储pairs,避免内存爆炸
3. **采样训练**: 用20%数据训练,覆盖率99.9%

```python
# 分批统计
def count_pairs_batch(corpus_batch):
    pairs = Counter()
    for word in corpus_batch:
        chars = list(word)
        for i in range(len(chars)-1):
            pairs[(chars[i], chars[i+1])] += 1
    return pairs

# 合并结果
total_pairs = Counter()
for batch in read_corpus_in_batches('corpus.txt'):
    batch_pairs = count_pairs_batch(batch)
    total_pairs.update(batch_pairs)
```

## 4. 面试追问防御

### 追问1: "手写BPE合并算法"

```python
def bpe_train(corpus, vocab_size):
    vocab = set(''.join(corpus))
    pairs = Counter()
    
    # 统计pairs
    for word in corpus:
        chars = list(word)
        for i in range(len(chars)-1):
            pairs[(chars[i], chars[i+1])] += 1
    
    # 迭代合并
    while len(vocab) < vocab_size:
        best_pair = max(pairs, key=pairs.get)
        merged = ''.join(best_pair)
        vocab.add(merged)
        
        # 更新语料
        corpus = [w.replace(merged, ' ' + merged + ' ') for w in corpus]
        corpus = [''.join(w.split()) for w in corpus]
        
        # 重新统计
        pairs = Counter()
        for word in corpus:
            chars = list(word)
            for i in range(len(chars)-1):
                pairs[(chars[i], chars[i+1])] += 1
    
    return vocab

# 时间复杂度: O(N·V)
# 空间复杂度: O(V^2)
```

**加分项**:trie树优化到O(N)，工业级实现用C++。

### 追问2: "词汇表50k×768×4字节=?"

```python
vocab_size = 50000
dim = 768
memory_mb = vocab_size * dim * 4 / 1024 / 1024
# 153.6 MB

# BERT总显存 ≈ 440MB
# Embedding占35%（大头！）
```

**经验**:每增加10k词表，显存+30MB，训练速度-2%。

### 追问3: "设计多语言Tokenizer"

**Level 1**:用SentencePiece
```python
spm.SentencePieceTrainer.train(
    input='multilingual.txt',
    vocab_size=64000,
    character_coverage=0.9995,  # 覆盖多语言字符
)
```

**Level 2**:分层词表
```python
shared_vocab = 32k  # 跨语言共享
lang_specific = {
    'zh': 8k, 'ja': 8k, 'ar': 8k, 'ko': 8k
}
# 总计: 32k + 4×8k = 64k
```

**Level 3**:动态裁剪
```python
# 训练时用大词表,推理时裁剪
def prune_vocab(model, keep_ratio=0.8):
    # 保留高频词
    sorted_vocab = sorted(model.vocab.items(), key=lambda x: x[1])
    cutoff = int(len(sorted_vocab) * keep_ratio)
    return dict(sorted_vocab[:cutoff])
# mT5发现64k是sweet spot
```

## 5. 面试总结

### 5.1 电梯演讲

"Tokenizer将文本转为模型可处理的token IDs。现代用子词分词（BPE/WordPiece/SentencePiece），平衡表达能力和效率。BERT用WordPiece+特殊标记,GPT用BPE,多语言用SentencePiece。"

### 5.2 必背速查表

| 知识点 | 答案 |
|--------|------|
| BPE vs WordPiece | 频率 vs 似然度 |
| 词表大小 | 30k-50k sweet spot |
| [CLS]作用 | BERT分类,非GPT |
| 中文处理 | 字符/词/子词平衡 |
| SentencePiece优势 | 无语言依赖 |

### 5.3 工业经验

"我们训练多语言模型时:SentencePiece用64k词表覆盖50种语言,character_coverage设0.9995对中日韩友好。Embedding占显存35%,是优化大头。"

---

*掌握度: 待评估*
*推荐: Transformer核心-Attention机制*
