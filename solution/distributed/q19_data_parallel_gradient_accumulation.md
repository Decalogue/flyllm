# Data Parallel 梯度累积

## 1. 核心消解

本质上，梯度累积是在显存受限情况下模拟大batch训练的技巧，通过多次前向/反向传播累积梯度，在固定step后统一更新参数，实现effective batch size = micro_batch × accumulate_steps。

## 2. 具体流程

1. **前向传播**: 计算micro-batch（小batch）的输出和损失
2. **反向传播**: 计算梯度但不更新参数，而是累积到grad_buffer
3. **参数更新**: 累积N次后，梯度取平均并更新参数，清空buffer

## 3. 数学基础

**标准小batch更新**:
```python
θ_{t+1} = θ_t - η·∇L(θ_t; B_micro)
```

**梯度累积更新**:
```python
# 累积阶段（不更新）
g_acc = 0
for i in range(K):
    g_i = ∇L(θ_t; B_i)  # B_i是第i个micro-batch
    g_acc += g_i

# 更新阶段（每K步）
θ_{t+1} = θ_t - η·(g_acc / K)
```

其中：
- $K$: 梯度累积步数（accumulate_steps）
- $B_i$: 第i个micro-batch的数据
- $B_{effective} = \sum_i |B_i|$: 等效batch size

**等效性证明**:
```python
# 当loss是均值时
L(θ; B_{effective}) = (1/K)·∑_{i=1}^K L(θ; B_i)

∇L(θ; B_{effective}) = (1/K)·∑_{i=1}^K ∇L(θ; B_i)

# 因此
θ_{t+1} = θ_t - η·∇L(θ_t; B_{effective})
```

**显存分析**:
```python
# 标准batch=B的内存
M_total = M_activations(B) + M_params + M_grads + M_optim

# 梯度累积的内存（batch=B/K）
M_total = M_activations(B/K) + M_params + M_grads + M_optim + M_buffer

内存降低:
M_activations(B/K) ≈ M_activations(B)/K  # 激活与batch线性相关
M_buffer ≈ M_grads  # 额外梯度存储

当K=4时，内存降低约60-70%
```

## 4. 工程考量

**Trade-off**:
- 牺牲：训练速度（K倍更多前向/反向计算）
- 换取：更大等效batch size
- 降低：单步计算速度换取训练稳定性

**致命弱点**:
- **BatchNorm/LayerNorm**: 统计量基于micro-batch，与full-batch有偏差
  ```python
  # 解决方案：同步BN stats
  sync_batch_norm(gather(stats_from_all_gpus))
  ```
- **Dropout**: 不同micro-batch使用不同mask，影响梯度精度
  ```python
  # 解决方案：固定随机种子或使用deterministic模式
  ```
- **学习率**: effective batch size变化需要调整LR
  ```python
  # Linear scaling rule
  η_effective = η_base × K
  ```
- **Warmup**: warmup steps也需要乘以K

**实现细节**:
```python
optimizer.zero_grad()
for i, batch in enumerate(data_loader):
    loss = model(batch) / accumulate_steps  # loss缩放
    loss.backward()

    if (i + 1) % accumulate_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

**性能优化**:
```python
# Gradient checkpointing + 累积
for i in range(K):
    with torch.no_grad():
        # 第1到K-1步只计算激活
        activations = forward(x_i)

    # 第K步计算梯度
    loss = forward_with_activations(x_K, activations)
    loss.backward()

# 内存再降50%，计算增加25%
```

## 5. 工业映射

在工业界，该机制被直接应用于Hugging Face的transformers库，通过`gradient_accumulation_steps`参数支持。BERT-large在单卡16GB显存下训练，使用batch_size=8 + accumulate=8模拟batch_size=64。Stable Diffusion训练使用accumulate=4在24GB显存上模拟batch_size=64。在RLHF场景中，Anthropic的PPO实现使用accumulate_steps=16来处理每个生成样本的multiple epochs，提升样本利用率。最新的DeepSpeed实现了ZeRO-Offload + Gradient Accumulation，允许在单张消费级GPU（RTX 3090）上微调LLaMA-2 70B。FSDP（Fully Sharded Data Parallel）原生支持梯度累积，在PyTorch 2.0中与activation checkpointing结合，实现极限显存优化。
