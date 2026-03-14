# 早停机制怎么设计？验证指标怎么选？patience 怎么设置？如何防止过拟合？

## 1. 核心定性
本质上，**早停（Early Stopping）** 是一种自适应训练终止策略，通过在验证集上监控模型性能，在训练损失继续下降但泛化性能开始恶化时提前终止，利用验证集作为代理测试集实现隐式正则化，防止模型过度拟合训练数据。

## 2. 具体流程
1. **验证监控**: 每个 epoch 或固定 step 在验证集上计算指标
2. **性能比较**: 与历史最佳性能比较，如果提升则保存 checkpoint
3. **耐心计数**: 如果连续 N 个周期无提升，触发早停
4. **模型恢复**: 加载验证集性能最佳的 checkpoint 作为最终模型

## 3. 数学基础

**过拟合度量**:
$$\text{GeneralizationGap} = \mathcal{L}_\text{val} - \mathcal{L}_\text{train}$$

早停发生在 GeneralizationGap 开始增大时。

**Patience 形式化**:
设 $t$ 是训练步数，$t_{best}$ 是最佳验证性能步数：
$$t_{best} = \arg\min_t \mathcal{L}_\text{val}(t)$$

早停条件：
$$t - t_{best} \ge P$$

其中 $P$ 是 patience。

**验证集选择**:
$$\mathcal{D}_\text{val} \sim P_\text{data}, \quad |\mathcal{D}_\text{val}| = 0.1 \times |\mathcal{D}_\text{train}|$$

**性能指标**:

**损失函数**:
$$\mathcal{L} = \begin{cases}
-\log P(y|x) & \text{NLL} \\
\text{CrossEntropy} & \text{分类} \\
\text{MSE} & \text{回归}
\end{cases}$$

**任务特定指标**:
- 分类: Accuracy, F1, AUC
- 生成: BLEU, ROUGE, Perplexity
- 信息抽取: F1, Precision, Recall

**平滑处理**:
为了避免验证噪声导致的误判：
$$\mathcal{L}_\text{val}^{\text{smooth}}(t) = 0.9 \mathcal{L}_\text{val}(t) + 0.1 \mathcal{L}_\text{val}(t-1)$$

**Patience 动态调整**:
$$P(t) = P_0 \cdot \exp(-\alpha (t - t_{best}))$$

但在实践中通常使用固定 patience。

## 4. 工程考量

**验证指标选择**:

**推荐**: Main metric + Auxiliary metrics

1. **主要指标**（决定早停）:
   - **Perplexity**: 通用语言能力最敏感
   - **Task-specific loss**: 直接监控训练目标
   - **F1 / Accuracy**: 任务性能

2. **辅助指标**（验证鲁棒性）:
   - **训练/验证 gap**: 监控过拟合程度
   - **不同子集性能**: 如不同长度、不同领域
   - **梯度范数**: 监控优化稳定性

**Patience 设置指南**:

| 数据集大小 | 训练步数 | 推荐 Patience | 说明 |
|------------|----------|---------------|------|
| <1K | 100-500 | 20-50 steps | 早停敏感 |
| 1K-10K | 500-2000 | 50-100 steps | 标准设置 |
| 10K-100K | 2000-5000 | 100-200 steps | 鲁棒设置 |
| >100K | 5000+ | 200-500 steps | 避免噪声 |

**经验法则**: Patience = 5-10% 总训练步数

**不同设置的影响**:

- **Patience 太小** (10-20 steps):
  - 训练不充分，欠拟合
  - 验证噪声敏感，模型不稳定

- **Patience 太大** (500+ steps):
  - 浪费计算
  - 可能轻微过拟合（但在 LoRA 时代可接受）

- **Sweet spot** (50-200 steps):
  - 平衡泛化和性能
  - 对验证噪声鲁棒

**早停策略**:

**策略 1: 监控主要指标**
```python
best_loss = inf
patience = 0
for epoch in range(max_epochs):
    val_loss = validate()
    if val_loss < best_loss:
        best_loss = val_loss
        save_checkpoint()
        patience = 0
    else:
        patience += 1
        if patience >= PATIENCE:
            break
```

**策略 2: 综合多个指标**
```python
best_score = -inf
for epoch in range(max_epochs):
    # 主要指标
    val_loss = validate_loss()
    # 辅助指标
    val_perplexity = validate_ppl()
    val_f1 = validate_f1()
    
    # 综合评分
    score = -val_loss - 0.5 * val_perplexity + val_f1
    
    if score > best_score:
        best_score = score
        save_checkpoint()
        patience = 0
    else:
        patience += 1
        if patience >= PATIENCE:
            break
```

**策略 3: 监控过拟合程度**
```python
best_gap = inf
for epoch in range(max_epochs):
    train_loss = train()
    val_loss = validate()
    gap = val_loss - train_loss
    
    if gap < best_gap:
        best_gap = gap
        save_checkpoint()
        patience = 0
    else:
        patience += 1
        if patience >= PATIENCE:
            break
```

**防止过拟合的其他方法**:

1. **权重衰减（L2 正则化）**:
   $$\mathcal{L}_\text{total} = \mathcal{L} + \lambda \|W\|^2$$
   推荐 $\lambda=0.01$，对 LoRA 可增大到 0.1

2. **Dropout**:
   - Attention dropout: 0.1
   - Hidden dropout: 0.1
   - Embedding dropout: 0.05

3. **数据增强**:
   - 随机删除/替换 token
   - 随机打乱句子顺序
   - 多任务混合

4. **Label Smoothing**:
   $$\mathcal{L}_\text{smooth} = -\frac{1}{K} \sum_{k=1}^K (1-\epsilon) \log p_k + \frac{\epsilon}{K-1} \log (1-p_k)$$
   通常 $\epsilon=0.1$

**EarlyStopping vs 其他正则化**:

| 方法 | 效果 | 成本 | 推荐度 |
|------|------|------|--------|
| EarlyStop | ★★★★★ | 无 | ★★★★★ |
| Weight Decay | ★★★★☆ | 无 | ★★★★★ |
| Dropout | ★★★★☆ | 微增推理时间 | ★★★★☆ |
| Data Aug | ★★★☆☆ | 数据准备时间 | ★★★★☆ |
| Label Smooth | ★★★☆☆ | 无 | ★★★☆☆ |

**Patience 调优技巧**:

1. **学习曲线法**:
   - 先跑 10% 数据，看验证 loss 何时平稳
   - Patience = 2-3 × 平稳后步数

2. **交叉验证**:
   - 在小数据集上调最优 Patience
   - 迁移到大数据集

3. **Rule of Thumb**:
   - 如果验证 loss 每 100 step 下降 0.01 → Patience=500
   - 如果验证 loss 每 100 step 下降 0.001 → Patience=2000

**EarlyStopping 在 LLM 时代的演变**:

**传统 ML**:
- Patience 小（20-50），避免过拟合
- 主要目的：正则化

**LLM + LoRA**:
- Patience 大（100-500），充分训练
- 主要目的：节省计算（而非防止过拟合）
- LoRA 本身正则化强，过拟合风险低

**验证集大小**:

| 训练集 | 验证集 | 说明 |
|--------|--------|------|
| < 1K | 10-20% | 小验证集方差大 |
| 1K-10K | 5-10% | 标准 |
| 10K-100K | 1-5% | 足够统计量 |
| > 100K | 0.5-1% | 降低验证成本 |

**批量验证 vs 在线验证**:

**批量验证**:
- 每个 epoch 结束时验证
- 优点：准确，方差小
- 缺点：发现过拟合慢

**在线验证**:
- 每 N step（如 100）验证
- 优点：快速响应
- 缺点：方差大，可能误判

**推荐**: 每 10% epoch 验证

## 5. 工业映射

**HuggingFace Trainer**:
```python
training_args = TrainingArguments(
    evaluation_strategy='steps',
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    patience=50
)
```
- 内置早停，监控任意指标
- 自动加载最佳 checkpoint

**PyTorch 实现**:
```python
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience
```

**Google BERT 微调**:
- Patience=500 (4% total steps)
- 监控 MLM loss
- 通常 2-3 epoch 早停

**GPT-3 微调**:
- 不早停，固定训练 1 epoch
- 因为数据量大，过拟合风险低
- 验证集仅用于超参选择

**LLaMA 微调实践**:
- LoRA: Patience=100 (充分训练)
- Full fine-tune: Patience=50（防止过拟合）
- 监控 ppl + task F1

**经验法则总结**:

1. **Patience**: 100-200 steps（5-10% 总训练）
2. **Min delta**: 0.001（相对改善 0.1%）
3. **监控**: Val loss + 辅助指标
4. **保存**: 始终保存最佳，而非最后
5. **恢复**: 训练结束后加载最佳 checkpoint

**防止过拟合的综合策略**:

**推荐组合**（LoRA 微调）:
- EarlyStopping: Patience=100
- Weight Decay: λ=0.01
- Dropout: 0.05
- Replay: 5% 预训练数据

**推荐组合**（全量微调）:
- EarlyStopping: Patience=50
- Weight Decay: λ=0.1
- Dropout: 0.1
- Small lr: 1e-5
- Data Aug: 必须

**常见误区**:

- ❌ Patience 太小: 训练不充分
- ❌ 只看训练 loss: 严重过拟合
- ❌ 不看验证指标: 无法发现退化
- ❌ 保存最后模型: 不是最佳状态
- ❌ 不调 min_delta: 噪声敏感

**结论**: EarlyStopping 是微调最关键的正则化手段。合理设置 patience（100-200），监控验证损失，配合权重衰减和 LoRA，可有效防止大模型微调崩溃或过拟合
