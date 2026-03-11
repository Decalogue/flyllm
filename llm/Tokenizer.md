# Tokenizer 完全指南

## 一、前置知识

### 1. 什么是Tokenizer？

Tokenizer（分词器）是NLP和LLM中的核心组件，负责将原始文本转换为模型可以理解的数字序列。

**核心作用：**
- 将文本分割成更小的单元（tokens）
- 将tokens映射为数字ID
- 处理未知词（OOV, Out-of-Vocabulary）

**示例：**
```
输入文本: "I love AI"
Tokenizer处理: ["I", "love", "AI"]
转换为ID: [101, 2293, 3958]
```

### 2. 为什么需要Tokenizer？

**问题1：词汇表爆炸**
- 如果把每个词都作为一个token，词汇表会非常大
- 英语有17万+个词，中文更多
- 大词汇表 → 模型参数多 → 训练慢

**问题2：未知词（OOV）**
- 训练时没见过的词无法处理
- 例如：新词、拼写错误、罕见词

**问题3：词形变化**
- "run", "running", "runs" 是同一个词的不同形式
- 如果都当作独立token，浪费资源

**解决方案：子词分词（Subword Tokenization）**
- 将词分解为更小的单元
- 平衡词汇表大小和表达能力
- 处理OOV问题

### 3. Tokenizer的演进历史

**阶段1：基于词的分词（Word-based）**
```python
# 简单的按空格分词
text = "I love AI"
tokens = text.split()  # ['I', 'love', 'AI']
```
**缺点：** 词汇表大、OOV问题严重

**阶段2：基于字符的分词（Character-based）**
```python
# 按字符分词
text = "love"
tokens = list(text)  # ['l', 'o', 'v', 'e']
```
**缺点：** 序列太长、丢失词义信息

**阶段3：子词分词（Subword-based）**
```python
# BPE、WordPiece、Unigram等
text = "tokenization"
tokens = ["token", "##ization"]  # 平衡词汇表和表达能力
```
**优点：** 平衡了词汇表大小和表达能力

### 4. 主流Tokenizer算法对比

| 算法 | 代表模型 | 核心思想 | 优点 | 缺点 |
|------|----------|----------|------|------|
| **BPE** | GPT系列 | 字节对编码，频繁合并 | 简单高效 | 贪心算法 |
| **WordPiece** | BERT | 最大似然估计 | 考虑概率 | 训练慢 |
| **Unigram** | T5 | 概率模型 | 理论完善 | 实现复杂 |
| **SentencePiece** | LLaMA | 直接处理原始文本 | 语言无关 | 输出不保留空格/词界，分词结果不直观 |

---

## 二、BPE算法详解

### 1. BPE核心思想

**Byte Pair Encoding（字节对编码）**
- 最初用于数据压缩
- 2016年引入NLP领域
- **核心：迭代合并最频繁的字符对**

### 2. BPE算法流程

#### 训练阶段

**输入：** 语料库
**输出：** 词汇表 + 合并规则

**步骤：**

```python
# 初始状态
词汇表 = 所有字符
语料库 = ["low", "lower", "newest", "widest"]

# 初始化为字符级别
初始tokens = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3
}
# </w>表示词尾

# 迭代1: 统计字符对频率
最频繁的对 = ("e", "s") 出现9次
合并 → "es"
更新词汇表 += ["es"]

# 迭代2: 继续统计
最频繁的对 = ("es", "t") 出现9次
合并 → "est"
更新词汇表 += ["est"]

# 迭代3...
# 重复直到达到预设词汇表大小
```

#### 推理阶段

**输入：** 新文本
**输出：** Token序列

**步骤：**
```python
# 1. 初始化为字符序列
输入 = "lowest"
初始 = ["l", "o", "w", "e", "s", "t"]

# 2. 应用学到的合并规则（按训练顺序）
规则1: "e" + "s" → "es"
结果 = ["l", "o", "w", "es", "t"]

规则2: "es" + "t" → "est"
结果 = ["l", "o", "w", "est"]

规则3: "l" + "o" → "lo"
结果 = ["lo", "w", "est"]

规则4: "lo" + "w" → "low"
结果 = ["low", "est"]

# 最终输出
tokens = ["low", "est"]
```

### 3. BPE代码实现

```python
import re
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.vocab = set()
        self.merges = {}
        
    def get_stats(self, vocab):
        """统计所有相邻token对的频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        """合并词汇表中的token对"""
        vocab_out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in vocab:
            w_out = word.replace(bigram, replacement)
            vocab_out[w_out] = vocab[word]
        return vocab_out
    
    def train(self, corpus):
        """训练BPE模型"""
        # 1. 初始化词汇表（字符级别）
        vocab = defaultdict(int)
        for word in corpus:
            # 在词尾添加</w>标记
            word = ' '.join(list(word)) + ' </w>'
            vocab[word] += 1
        
        # 2. 迭代合并最频繁的token对
        for i in range(self.num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            
            # 找出频率最高的token对
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges[best_pair] = i
            
            print(f"合并 {i+1}: {best_pair} -> {''.join(best_pair)}")
        
        # 3. 构建最终词汇表
        self.vocab = set()
        for word in vocab:
            self.vocab.update(word.split())
    
    def tokenize(self, text):
        """对文本进行分词"""
        # 1. 初始化为字符序列
        word = ' '.join(list(text)) + ' </w>'
        
        # 2. 应用学到的合并规则
        while True:
            pairs = [(word[i:i+2], i) for i in range(len(word.split())-1)]
            if not pairs:
                break
            
            # 找出可以合并的token对（按训练顺序）
            bigram = min(pairs, key=lambda x: self.merges.get(
                tuple(word.split()[x[1]:x[1]+2]), float('inf')
            ))
            
            if tuple(word.split()[bigram[1]:bigram[1]+2]) not in self.merges:
                break
            
            first, second = word.split()[bigram[1]:bigram[1]+2]
            word = word.replace(f"{first} {second}", f"{first}{second}")
        
        return word.split()

# 使用示例
if __name__ == "__main__":
    # 训练语料
    corpus = ["low", "low", "low", "low", "low",
              "lower", "lower",
              "newest", "newest", "newest", "newest", "newest", "newest",
              "widest", "widest", "widest"]
    
    # 训练BPE
    tokenizer = BPETokenizer(num_merges=10)
    tokenizer.train(corpus)
    
    # 测试分词
    print("\n分词结果:")
    print(tokenizer.tokenize("lowest"))
    print(tokenizer.tokenize("newest"))
```

### 4. BPE的优缺点

**优点：**
- ✅ 简单高效，易于实现
- ✅ 能处理OOV问题
- ✅ 平衡词汇表大小和表达能力
- ✅ 适用于多种语言

**缺点：**
- ❌ 贪心算法，可能不是最优解
- ❌ 依赖于训练语料的质量
- ❌ 对不同语言需要不同处理

### 5. BPE的变体

**Byte-Level BPE（GPT-2）：**
- 直接在字节级别操作
- 避免Unicode字符的复杂性
- GPT-2、GPT-3使用

**SentencePiece BPE：**
- 不依赖预分词
- 直接处理原始文本
- 适用于所有语言

---

## 三、WordPiece算法详解

### 1. WordPiece核心思想

**与BPE的区别：**
- BPE：选择**频率最高**的token对合并
- WordPiece：选择**对数似然增加最大**的token对合并

**公式：**
```
score(x, y) = log P(xy) / (log P(x) + log P(y))
```

### 2. WordPiece算法流程

#### 训练阶段

```python
# 1. 初始化为字符级别
词汇表 = 所有字符

# 2. 迭代合并
while 词汇表大小 < 目标大小:
    # 统计所有可能的token对
    for each pair (x, y):
        # 计算合并后的对数似然增益
        score = log P(xy) / (log P(x) + log P(y))
    
    # 选择score最大的token对合并
    best_pair = argmax(score)
    词汇表 += [best_pair]
```

#### 推理阶段

```python
# 贪心最长匹配算法
def tokenize(word):
    tokens = []
    start = 0
    
    while start < len(word):
        end = len(word)
        found = False
        
        # 从最长开始尝试匹配
        while start < end:
            substr = word[start:end]
            if start > 0:
                substr = "##" + substr  # 非首token加##前缀
            
            if substr in vocab:
                tokens.append(substr)
                start = end
                found = True
                break
            end -= 1
        
        if not found:
            return [UNK]  # 未知token
    
    return tokens
```

### 3. 手撕 WordpieceTokenizer

```python
def whitespace_tokenize(text):
    """简单的空格分词"""
    return text.strip().split()

class WordpieceTokenizer:
    def __init__(self, vocab, unk_token, max_chars=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_chars = max_chars

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_chars:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                # 从后往前遍历，找到最长的子串
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果找不到子串，则认为是一个坏的token
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens
```

### 4. WordPiece vs BPE对比

| 特性 | BPE | WordPiece |
|------|-----|-----------|
| **选择标准** | 频率最高 | 对数似然最大 |
| **理论基础** | 数据压缩 | 语言模型 |
| **训练速度** | 快 | 慢 |
| **代表模型** | GPT系列 | BERT |
| **子词标记** | 无特殊标记 | ##前缀 |

---

## 四、面试高频问题

### 1. 为什么需要Tokenizer？

**答案要点：**
- 词汇表爆炸问题
- OOV（未知词）问题
- 词形变化问题
- 子词分词是最佳平衡方案

### 2. BPE和WordPiece的区别？

**答案：**
- **BPE**：选择频率最高的字符对合并（贪心算法）
- **WordPiece**：选择对数似然增加最大的字符对合并（基于概率）
- **实际效果**：差异不大，但WordPiece理论上更优

### 3. 如何处理未知词（OOV）？

**答案：**
- 子词分词可以将未知词分解为已知子词
- 最坏情况下分解为字符级别
- 例如："tokenization" → ["token", "##ization"]

### 4. Tokenizer的词汇表大小如何选择？

**答案：**
- 太小：序列太长，训练慢
- 太大：模型参数多，内存消耗大
- 经验值：
  - 英文：30k-50k
  - 中文：20k-40k
  - 多语言：100k+

### 5. ##前缀的作用是什么？

**答案：**
- 标记子词的边界
- 区分词首和非词首
- 例如：["token", "##ization"] → "tokenization"

### 6. 如何处理中文分词？

**答案：**
- 方法1：预分词（jieba等） + BPE/WordPiece
- 方法2：字符级别 + BPE（适合大模型）
- 方法3：SentencePiece（直接处理原始文本）

### 7. 为什么GPT使用BPE，BERT使用WordPiece？

**答案：**
- **历史原因**：BERT借鉴了Google的经验
- **实际效果**：两者差异不大
- **实现难度**：BPE更简单
- **理论基础**：WordPiece更严谨

### 8. Tokenizer如何影响模型性能？

**答案：**
- 词汇表大小影响模型参数量
- 分词粒度影响序列长度
- 分词质量影响模型理解能力
- 需要在效率和效果之间平衡

---

## 五、实战技巧

### 1. 使用HuggingFace Tokenizers

```python
from transformers import BertTokenizer, GPT2Tokenizer

# BERT的WordPiece
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = bert_tokenizer.tokenize("tokenization")
print(tokens)  # ['token', '##ization']

# GPT-2的BPE
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = gpt2_tokenizer.tokenize("tokenization")
print(tokens)  # ['token', 'ization']
```

### 2. 训练自定义Tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 创建BPE模型
tokenizer = Tokenizer(models.BPE())

# 设置预分词器
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 设置训练器
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# 训练
files = ["corpus.txt"]
tokenizer.train(files, trainer)

# 保存
tokenizer.save("my-tokenizer.json")
```

### 3. Tokenizer性能优化

```python
# 1. 批量处理
texts = ["text1", "text2", "text3"]
tokens = tokenizer(texts, padding=True, truncation=True)

# 2. 使用fast tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# 3. 多进程处理
from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(tokenizer.tokenize, texts)
```

---

## 六、总结

### 核心要点

1. **Tokenizer是NLP的基础**
   - 将文本转换为模型可理解的数字
   - 平衡词汇表大小和表达能力

2. **子词分词是主流方案**
   - BPE：频率导向，简单高效
   - WordPiece：概率导向，理论完善
   - Unigram：概率模型，灵活强大

3. **面试重点**
   - 理解BPE和WordPiece的原理和区别
   - 能手写基本的Tokenizer代码
   - 了解OOV处理和词汇表大小选择

### 学习路径

```
1. 理解基本概念 → 为什么需要Tokenizer
2. 学习BPE算法 → 核心原理和代码实现
3. 学习WordPiece → 与BPE的区别
4. 实践应用 → 使用HuggingFace Tokenizers
5. 深入研究 → Unigram、SentencePiece等
```

### 推荐资源

**代码库：**
- HuggingFace Tokenizers: https://github.com/huggingface/tokenizers
- SentencePiece: https://github.com/google/sentencepiece

---

## 关注我，AI不再难 🚀