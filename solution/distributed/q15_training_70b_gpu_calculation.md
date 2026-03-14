# 训练 70B 模型需要多少张 A100

## 1. 核心定性

本质上，计算训练70B模型所需A100数量是显存预算问题，需综合考虑模型参数、梯度、优化器状态、激活值和通信开销，通过公式精确计算最小GPU数量。

## 2. 具体流程

1. **参数计算**: 估算模型参数、梯度、优化器状态显存
2. **激活分析**: 根据序列长度和batch size计算激活值
3. **策略选择**: 确定并行策略（ZeRO-3 + Activation Checkpointing）
4. **GPU数量**: 求解不等式：单卡显存需求 < 80GB

## 3. 数学基础

**显存计算公式**:
```python
M_total = M_params + M_grads + M_optim + M_activations
```

**参数显存**:
```python
M_params = 2Φ  # fp16
M_grads = 2Φ   # fp16
M_optim = 4Φ + 2Φ/K  # AdamW (fp32+momentum)

M_params_70B = 140 GB
M_grads_70B = 140 GB
M_optim_70B = 280 GB + 140 GB/K
```

其中：
- $Φ = 70×10^9$: 参数量
- $K$: Tensor Parallel度

**激活值计算** (无checkpointing):
```python
M_activations = B·L·S·H·4  # 每层激活

B = 4   # batch size per GPU
L = 80  # LLaMA-2 70B层数
S = 4096  # 序列长度
H = 8192  # 隐层维度

M_activations = 4·80·4096·8192·4 / 10^9 = 42 GB
```

**ZeRO-3分片后** (N张GPU):
```python
M_per_gpu = (M_params + M_grads + M_optim)/N + M_activations
```

## 4. 工程考量

**不同配置计算**:

**配置1: ZeRO-3 + Checkpointing**:
```python
M_per_gpu = (140 + 140 + 420)/N + 42/√N

N = 8:  M = 87.5 + 14.8 = 102.3 GB  > 80GB ✗
N = 16: M = 43.8 + 10.5 = 54.3 GB   < 80GB ✓
```

**配置2: ZeRO-3 + TP=4**:
```python
M_per_gpu = (140 + 140 + 280 + 70)/N + 42/√N

N = 8:  M = 78.8 + 14.8 = 93.6 GB  > 80GB ✗
N = 16: M = 39.4 + 10.5 = 49.9 GB  < 80GB ✓
```

**Trade-off**:
- 增加GPU数量：线性降低参数/优化器显存，但增加通信开销
- Activation Checkpointing：激活值从$O(L)$降到$O(√L)$，但增加20-30%计算量
- Tensor Parallelism：降低优化器显存，但增加卡间通信

**致命弱点**:
- **内存碎片**: PyTorch分配器导致实际显存 > 理论计算
- **临时张量**: Attention计算中的中间结果未计入
- **CUDA Kernel**: 某些算子需要额外workspace

**实际经验**:
- **保守估计**: 理论值 + 20% overhead
- **最小配置**: 16×A100 80GB (ZeRO-3 + checkpointing)
- **推荐配置**: 32×A100 80GB (考虑训练速度和稳定性)
- **极限优化**: 8×A100 + CPU Offload (但速度极慢)

**Batch Size优化**:
```python
M_activations ∝ B·S

# 降低batch size节省显存
B = 2: M_activations = 21 GB → 单卡需求降至70GB (N=8)
```

## 5. 工业映射

在工业界，LLaMA-2 70B的训练使用32×A100 80GB，配合ZeRO-3和Activation Checkpointing。Falcon 180B在AWS上训练需要512×A100，通过ZeRO-3和pipeline并行实现。Bloom 176B在JeanZay超算使用384×A100 80GB。对于资源受限场景，QLoRA技术允许在单张A100上微调70B模型，通过4-bit量化和分页优化器实现。最新的PyTorch FSDP（Fully Sharded Data Parallel）作为ZeRO-3的原生实现，在Llama-3训练中成功将70B模型的GPU需求降到16×A100。
