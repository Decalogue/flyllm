# ZeRO-3 参数分片全流程

## 1. 核心定性

本质上，ZeRO-3是DeepSpeed提出的激进参数分片策略，通过将模型参数、梯度和优化器状态分片到所有GPU，实现线性显存扩展，支持在有限GPU上训练超大规模模型。

## 2. 具体流程

1. **参数分片**: 将模型参数按层分片，每个GPU只保存1/N参数
2. **前向传播**: 需要时通过All-Gather收集完整参数计算激活值
3. **反向传播**: 计算梯度后通过Reduce-Scatter聚合梯度并释放内存

## 3. 数学基础

**显存计算公式**:
```python
M_{ZeRO-3} = M_{params} + M_{grads} + M_{optim} + M_{activations}
```

具体分片：
```python
M_{params_per_gpu} = Φ / N
M_{grads_per_gpu} = Φ / N
M_{optim_per_gpu} = 2Φ / N + 2Φ/(K·N)
```

其中：
- $Φ$: 模型参数量
- $N$: GPU数量（Data Parallel度）
- $K$: Tensor Parallel度

**All-Gather通信**:
```python
# 前向：收集完整参数
params_all = all_gather(params_shard)
output = linear(input, params_all)

delete params_all  # 立即释放
```

**Reduce-Scatter通信**:
```python
# 反向：梯度聚合
dloss_dparams_local = backward(output_grad)
dloss_dparams_shard = reduce_scatter(dloss_dparams_local)
```

**通信量分析**:
- All-Gather: $2Φ·(1 - 1/N)$ bytes
- Reduce-Scatter: $2Φ·(1 - 1/N)$ bytes
- 每层的通信量: $4Φ·(1 - 1/N)$ bytes

## 4. 工程考量

**Trade-off**:
- 牺牲：大量通信开销（每层两次集合通信）
- 换取：显存从O(Φ)降到O(Φ/N)
- 平衡：通信与计算重叠来隐藏延迟

**致命弱点**:
- **通信瓶颈**: 在小规模集群（N<8）时通信比重过高
- **实现复杂**: 需要重写所有算子支持动态收集参数
- **内存碎片**: 频繁的All-Gather/释放导致内存碎片
- **负载不均**: 某些层（如embedding）参数巨大难以均匀分片

**优化技巧**:
- **通信压缩**: 使用fp16压缩通信数据
- **计算重叠**: All-Gather与前向计算流水线并行
- **CPU Offload**: 将优化器状态卸载到CPU内存
- **NVMe Offload**: 将不常用参数卸载到SSD

## 5. 工业映射

在工业界，该机制被直接应用于Meta的LLaMA-2 70B训练中，在512个A100上实现全参数训练。Alpaca和Vicuna的微调也使用ZeRO-3，允许在8×V100上微调65B模型。Hugging Face的Accelerate库封装了ZeRO-3，使得在transformers中只需配置`deepspeed_config.json`即可使用。在推理场景中，vLLM借鉴了ZeRO-3的思想，通过分页加载权重来支持超大模型的多卡推理。最新的ZeRO-Infinity更进一步，支持将参数、激活值全部卸载到NVMe，使得在单机上训练万亿模型成为可能。
