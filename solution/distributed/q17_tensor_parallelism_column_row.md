# Tensor Parallelism 的 Column-Row 切分

## 1. 核心消解

本质上，Tensor Parallelism通过Column-Row切分将大矩阵乘法分解到多个GPU，Column切分处理权重矩阵，Row切分处理输入输出，使用All-Reduce同步结果，避免单卡显存瓶颈。

## 2. 具体流程

1. **Column切Linear**: 权重W按输出维度切分为[W₁, W₂]，每卡计算部分输出
2. **Row切Linear**: 权重W按输入维度切分为[W₁; W₂]，每卡需要All-Reduce聚合
3. **All-Reduce同步**: 在Row切后使用All-Reduce合并各卡结果

## 3. 数学基础

**Column Parallel Linear**:
```python
# 输入x (batch, input_dim)
# 权重W (output_dim, input_dim)切分为[W₁, W₂]

# GPU 0
z₀ = x · W₁ᵀ  # (batch, output_dim/2)

# GPU 1
z₁ = x · W₂ᵀ  # (batch, output_dim/2)

# 输出拼接
z = concat([z₀, z₁], dim=-1)  # (batch, output_dim)

# 通信: 无（各卡输出不同部分）
```

**Row Parallel Linear**:
```python
# 输入x切分为[x₁, x₂]
# 权重W (output_dim, input_dim)切分为[W₁; W₂]

# GPU 0
z₀ = x₁ · W₁ᵀ  # (batch, output_dim)

# GPU 1
z₁ = x₂ · W₂ᵀ  # (batch, output_dim)

# All-Reduce聚合
z = AllReduce(z₀ + z₁)  # (batch, output_dim)

# 通信: All-Reduce (2·output_dim·bytes)
```

**MLP的TP设计**:
```python
# MLP = Linear₁ → Gelu → Linear₂
# Linear₁: Column切分（增大隐藏维度）
# Linear₂: Row切分（减少回原始维度）

# GPU 0
h₀ = GeLU(x · W₁₀ᵀ)  # (batch, hidden/2)
z₀ = h₀ · W₂₀       # (batch, output_dim)

# GPU 1
h₁ = GeLU(x · W₁₁ᵀ)  # (batch, hidden/2)
z₁ = h₁ · W₂₁       # (batch, output_dim)

# All-Reduce聚合
z = AllReduce(z₀ + z₁)

# 通信: 1次All-Reduce
```

**注意力机制的TP**:
```python
# QKV Linear: Column切分（3份）
Q = x · W_Qᵀ  # GPU i计算Q_i
K = x · W_Kᵀ  # GPU i计算K_i
V = x · W_Vᵀ  # GPU i计算V_i

# 注意力计算（每卡独立）
attn_i = softmax(Q_i · K_iᵀ / √d) · V_i

# 输出Linear: Row切分
output_i = attn_i · W_Oᵢ
output = AllReduce(output_i)
```

**通信分析**:
- Column Parallel: 0通信（输出天然分散）
- Row Parallel: 1次All-Reduce（每TP组）
- 每层Transformer: 2次All-Reduce（MLP+Attention输出）

## 4. 工程考量

**Trade-off**:
- 牺牲：卡间通信（每层2次All-Reduce）
- 换取：权重显存从O(H²)降到O(H²/TP)
- 降低：单卡计算量（完美线性扩展）

**致命弱点**:
- **负载不均**: Attention的softmax和Layernorm无法并行
- **通信瓶颈**: TP组内All-Reduce延迟影响整体吞吐
- **实现限制**: LayerNorm和Dropout需要同步随机种子
- **扩展性差**: TP度通常≤8（NVLink限制）

**实现细节**:
```python
# All-Reduce策略选择
- Ring All-Reduce: 带宽最优，延迟O(2·(TP-1))
- Tree All-Reduce: 延迟最优，O(log₂TP)

# 典型配置
TP=4: 单节点内，PCIe/NVLink
TP=8: 单节点内，完整NVLink
TP>8: 跨节点，网络带宽成为瓶颈
```

**精度注意**:
- All-Reduce应该先合后输出（避免累加误差）
- 使用NCCL的fp16 All-Reduce提升速度
- LayerNorm必须在All-Reduce后执行

## 5. 工业映射

在工业界，该机制被直接应用于Megatron-LM，NVIDIA使用8路TP训练GPT-3 175B，将每层Linear层的权重分布到8张A100。Hugging Face的transformers通过`tensor_parallel`参数支持TP，在BLOOM 176B推理中使用4路TP降低单卡显存需求。DeepSpeed的ZeRO-3与TP结合使用，在训练MT-NLG 530B时实现8×64卡配置。在推理场景中，TensorRT-LLM的TP实现通过融合Column和Row切分，将通信量从4次All-Reduce降到2次，提升30%吞吐。最新的NVIDIA GH200通过900GB/s NVLink C2C，支持单机256路TP，为万亿模型训练奠定基础。
