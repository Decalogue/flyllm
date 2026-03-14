# 004: ZeRO-3 全流程与工程实现

## 核心定性
本质上，ZeRO-3（Zero Redundancy Optimizer Stage 3）是为解决**大模型训练的显存墙瓶颈**，通过**参数/梯度/优化器状态分片 + 动态通信**，在 DP（数据并行）的基础上实现**跨 GPU 的极致显存节省**（8 卡训练 70B 模型仅需 40GB/卡），将模型规模扩展能力提升 10 倍以上。

## 具体流程

### ZeRO 三阶段演进

```python
# 模型状态 = 参数（Param）+ 梯度（Grad）+ 优化器状态（Opt State）
model_states = {
    "params": 2Φ,          # Φ = 参数数量（FP16）
    "grads": 2Φ,
    "optimizer": 4Φ + 4Φ,   # Adam: momentum + variance（FP32）
    "total": 12Φ
}

# Phi-70B = 70B 参数
# 显存需求 = 12 * 70B * 2 bytes = 1680GB（不可行）
```

**Stage 1: 优化器状态分片**
```python
# 每卡只保存 1/DP 的 optimizer state
# 显存: 2Φ + 2Φ + (4Φ + 4Φ)/DP = 4Φ + 8Φ/DP
# DP=8: 4Φ + Φ = 5Φ = 350GB（仍不可行）
```

**Stage 2: 梯度分片**
```python
# 每卡只保存 1/DP 的 grad + optimizer state
# 显存: 2Φ + 2Φ/DP + (4Φ + 4Φ)/DP = 2Φ + 10Φ/DP
# DP=8: 2Φ + 1.25Φ = 3.25Φ = 227.5GB（仍不可行）
```

**Stage 3: 参数分片**
```python
# 每卡只保存 1/DP 的 param + grad + optimizer state
# 显存: 2Φ/DP + 2Φ/DP + (4Φ + 4Φ)/DP = 12Φ/DP
# DP=8: 1.5Φ = 105GB（可运行，但需 Activation Checkpointing）
```

### ZeRO-3 训练时序图

**前向传播（Forward Pass）**:
```python
# 每层的 All-Gather 流程
def zero3_forward_layer(layer_id, input_activations, rank):
    """
    1. All-Gather 参数: 从所有卡收集完整参数
    2. 计算前向: 得到输出激活
    3. 丢弃参数: 释放显存（只保留分片）
    """
    # 步骤 1: All-Gather（通信）
    # 输入: param_shard = 1/8 参数 [Φ/8, d]
    # 输出: param_full = 完整参数 [Φ, d]
    param_full = all_gather(param_shard, dp_group)

    # 步骤 2: 计算（计算）
    output = linear_forward(input_activations, param_full)

    # 步骤 3: 释放（内存管理）
    del param_full  # 释放显存
    torch.cuda.empty_cache()

    return output
```

**通信量计算**:
- 每层 All-Gather: 2Φ bytes（FP16 参数）
- L 层模型: L × 2Φ bytes
- 前向总通信: 2LΦ = 2 × 80 × 70B = 11.2TB（单次传播）

**反向传播（Backward Pass）**:
```python
def zero3_backward_layer(layer_id, grad_output, rank):
    """
    1. All-Gather 参数: 再次收集（前向已丢弃）
    2. 计算梯度: 得到 d_input（对前一层）和 d_param（对权重）
    3. Reduce-Scatter 梯度: 将 d_param 汇总并分片
    4. 释放参数: 清理显存
    """
    # 步骤 1: All-Gather 参数（通信）
    param_full = all_gather(param_shard, dp_group)

    # 步骤 2: 反向计算（计算）
    d_input, d_param = linear_backward(grad_output, param_full)

    # 步骤 3: Reduce-Scatter 梯度（通信）
    # 输入: d_param = 每个卡的局部梯度 [Φ, d]
    # 输出: d_param_shard = 平均后的分片 [Φ/8, d]
    d_param_shard = reduce_scatter(d_param, dp_group, op='mean')

    # 步骤 4: 保存本地梯度（内存）
    grad_buffer[layer_id] = d_param_shard

    # 步骤 5: 释放
    del param_full

    return d_input
```

**通信量计算**:
- All-Gather 参数: 2Φ bytes
- Reduce-Scatter 梯度: 2Φ bytes（总和后分片）
- 每层: 4Φ bytes
- 反向总通信: 4LΦ = 22.4TB（单次反向）

**优化器步骤（Optimizer Step）**:
```python
def zero3_optimizer_step(optimizer, rank):
    """
    1. 所有卡有相同的 param_shard 和 grad_shard
    2. 本地更新 optimizer state（momentum, variance）
    3. 更新 param_shard
    4. 同步参数（确保一致性）
    """
    # 每个卡的 optimizer 只更新自己分片的参数
    # 无需通信（AllReduce 已在 Reduce-Scatter 时完成）

    for layer_id in range(num_layers):
        # 加载梯度
        grad_shard = grad_buffer[layer_id]

        # 加载参数
        param_shard = param_buffer[layer_id]

        # Adam 更新
        m = beta1 * m + (1-beta1) * grad_shard
        v = beta2 * v + (1-beta2) * grad_shard**2

        # 参数更新
        param_shard -= lr * m / (sqrt(v) + eps)

        # 清理梯度
        grad_buffer[layer_id] = None
```

## 数学基础

### 显存公式推导

**ZeRO-3 总显存**: $M_{total} = M_{model} + M_{grads} + M_{opt} + M_{act} + M_{comm}$

其中：
$$M_{model} = \frac{2Φ}{DP} \quad \text{（FP16 参数分片）}$$
$$M_{grads} = \frac{2Φ}{DP} \quad \text{（FP16 梯度分片）}$$
$$M_{opt} = \frac{4Φ + 4Φ}{DP} = \frac{8Φ}{DP} \quad \text{（Adam FP32 状态）}$$
$$M_{act} = \frac{2Nbd}{TP} \times \frac{1}{recompute_depth} \quad \text{（激活检查点）}$$
$$M_{comm} = 2Φ \quad \text{（All-Gather 临时缓冲区）}$$

**总计**:
$$M_{total} = \frac{12Φ}{DP} + \frac{2Nbd}{TP \times recompute} + 2Φ$$

### 训练 70B 模型实例

```python
Φ = 70B = 70e9
DP = 8
d = 4096
N = 4096（序列长度）
batch = 4
TP = 2
recompute_depth = 2

M_model = 12 * 70e9 / 8 * 2 bytes = 210GB
M_act = 2 * 4096 * 4 * 4096 / 2 / 2 * 2 bytes = 67MB（可忽略）
M_comm = 2 * 70e9 * 2 bytes = 280GB（可释放）

# 实际占用（非计算时）
M_total ≈ 210GB / 8 + 2GB = 28GB（每卡）
# 计算峰值（AllGather 时）
M_peak ≈ 28GB + 280GB = 308GB（临时，瞬时）

# 真实场景（A100-80G，4卡）
实际分批计算：
- 参数分片: 210GB / 4 = 52.5GB/卡
- 通信缓冲区: 280GB / 4 = 70GB/卡（可复用）
- 激活: ~50GB
- 总计: ~150GB < 4 * 80GB = 320GB ✓
```

### 通信复杂度分析

**前向传播**:
- **All-Gather**: $(DP-1) \cdot \frac{2Φ}{DP}$ bytes（每卡接收）
- **计算**: $O(N^2d)$ FLOPs
- **并行效率**: $\eta = \frac{1}{1 + \frac{L \cdot t_{comm}}{t_{comp}}}$

当 $t_{comm} = 2Φ/B \approx 10ms$, $t_{comp} = 100ms$ 时：
$$\eta \approx 90\%$$

**反向传播**:
- **All-Gather**: 同前向
- **Reduce-Scatter**: $(DP-1) \cdot \frac{2Φ}{DP}$ bytes
- **通信量**: 2× 前向

## 关键代码实现

### PyTorch 模拟（简化版）

```python
import torch
import torch.distributed as dist

class Zero3LinearLayer:
    def __init__(self, in_features, out_features, dp_rank, dp_size):
        """
        dp_rank: 当前卡在数据并行中的 rank
        dp_size: 数据并行总卡数
        """
        self.in_features = in_features
        self.out_features = out_features
        self.dp_rank = dp_rank
        self.dp_size = dp_size

        # 参数分片：只保存 1/dp_size
        self.weight_shard = nn.Parameter(
            torch.randn(out_features // dp_size, in_features)
        )
        self.bias_shard = nn.Parameter(
            torch.randn(out_features // dp_size)
        )

    def forward(self, x):
        # 步骤 1: All-Gather 参数
        full_weight = self._all_gather(self.weight_shard)
        full_bias = self._all_gather(self.bias_shard)

        # 步骤 2: 计算
        output = F.linear(x, full_weight, full_bias)

        # 步骤 3: 释放（可选，立即释放）
        del full_weight, full_bias

        return output

    def _all_gather(self, tensor):
        """模拟 All-Gather 操作"""
        world_size = self.dp_size
        rank = self.dp_rank

        # 准备接收缓冲区
        output_list = [torch.zeros_like(tensor) for _ in range(world_size)]

        # All-Gather（真实使用 NCCL）
        dist.all_gather(output_list, tensor)

        # 拼接
        return torch.cat(output_list, dim=0)

    def _reduce_scatter(self, tensor):
        """模拟 Reduce-Scatter 操作"""
        world_size = self.dp_size
        rank = self.dp_rank

        # 分割
        chunks = tensor.chunk(world_size, dim=0)

        # Reduce（求和）
        reduced = torch.zeros_like(chunks[rank])
        for c in chunks:
            reduced += c

        # 平均（除以 world_size）
        return reduced / world_size
```

### DeepSpeed ZeRO-3 源码参考

```python
# https://github.com/microsoft/DeepSpeed/deepspeed/runtime/zero/stage3.py
class DeepSpeedZeroOptimizerStage3:
    def __init__(self, module, dp_process_group, config):
        """
        module: 模型（参数已注册）
        dp_process_group: 数据并行通信组
        config: ZeRO 配置
        """
        self.module = module
        self.dp_group = dp_process_group
        self.world_size = dist.get_world_size(dp_process_group)

        # 注册参数分片
        self._create_param_shards()

        # 激活检查点配置
        self.contiguous_gradients = config.contiguous_gradients
        self.overlap_comm = config.overlap_comm

    def _create_param_shards(self):
        """
        核心：将所有参数按 dp_group 分片
        """
        for name, param in self.module.named_parameters():
            # 每个卡只保留 1/world_size
            shard_size = param.numel() // self.world_size
            start_idx = shard_size * self.local_rank
            end_idx = start_idx + shard_size

            # 创建参数分片
            param_shard = param.data.flatten()[start_idx:end_idx]
            param.ds_tensor = param_shard
            param.ds_numel = shard_size

    def step(self):
        """
        优化器步骤（在 Reduce-Scatter 之后）
        """
        for param in self.module.parameters():
            if param.grad is None:
                continue

            # 梯度 Reduce-Scatter（已在外层完成）
            grad_shard = param.grad.ds_tensor

            # 加载优化器状态
            state = self.state[param]
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            # Adam 更新
            exp_avg.mul_(self.beta1).add_(grad_shard, alpha=1-self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad_shard, grad_shard, value=1-self.beta2)

            denom = exp_avg_sq.sqrt().add_(self.eps)
            param.ds_tensor.addcdiv_(exp_avg, denom, value=-self.lr)

            # 清理梯度
            param.grad = None
```

### 通信重叠优化

```python
class CommOverlapEngine:
    """
    关键技术：计算与通信重叠（Overlap）
    """
    def __init__(self):
        self.compute_stream = torch.cuda.Stream()
        self.comm_stream = torch.cuda.Stream()

    def forward_with_overlap(self, layer_id, x):
        """
        并行执行：
        - 当前层计算
        - 下一层 All-Gather
        """
        # 在当前流计算
        with torch.cuda.stream(self.compute_stream):
            output = layer(x)

        # 在通信流准备下一层参数
        with torch.cuda.stream(self.comm_stream):
            next_layer.gather_params()

        # 同步（确保下一层参数已准备好）
        torch.cuda.synchronize()

        return output
```

**加速效果**: 重叠后，通信时间隐藏 50-70%。

## 工业映射

### Meta 训练 LLaMA2-70B

```yaml
training_config:
  model_size: 70b
  hardware: A100-80GB × 128
  parallelism:
    dp: 8
    pp: 4
    tp: 2
  zero_stage: 3
  batch_size: 4
  seq_len: 4096

# 显存占用（每卡）
model_states: 105GB
activations: 40GB  # checkpointing 后
buffers: 10GB
total: 155GB < 80GB?  # 错误！

# 真实配置: 4D 并行 + CPU Offload
# Offload 优化器状态到 CPU 内存
cpu_offload: true
nvme_offload: false

# 优化后（每卡）
gpu_memory: 75GB  # 可运行
training_speed: 1800 tokens/s/gpu
```

**成果**: 70B 模型训练 1.4T tokens，耗时 21 天（效率 90%）。

### 阿里云训练通义千问

```python
# 异构训练: GPU + CPU + NVMe
deep_speed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",  # 优化器状态 CPU 卸载
            "pin_memory": True
        },
        "offload_param": {
            "device": "nvme",  # 参数 NVMe 卸载（极端优化）
            "nvme_path": "/mnt/nvme/",
            "pin_memory": True
        },
    },
    "overlap_comm": True,
    "contiguous_gradients": True,
}

# 效果:
# - GPU 显存: 40GB/卡（从 80GB 降低）
# - CPU 内存: 512GB（优化器状态）
# - NVMe: 2TB（参数临时存储）
# - 速度损失: 仅 15%（通信重叠优化）
```

### 字节跳动豆包生产部署

**场景**: 在线训练（增量更新），不能停服

```python
class ElasticZero3Trainer:
    """
    弹性训练: 支持动态扩缩容
    """
    def __init__(self):
        self.deepspeed_engine = initialize_zero3()

    def on_node_failure(self, failed_rank):
        """
        节点故障处理
        """
        # 1. 保存 checkpoint（所有 GPU）
        self.save_checkpoint()

        # 2. 重新初始化进程组（减少 world_size）
        new_dp_group = dist.new_group(ranks=remaining_ranks)
        self.dp_group = new_dp_group

        # 3. 重新加载 checkpoint（自动处理分片）
        self.load_checkpoint()

        # 4. 继续训练
        resume_step = checkpoint['step']
        self.train(start_step=resume_step)

# 效果:
# - 故障恢复时间: 5 分钟内
# - 数据不丢失: 每 100 步 checkpoint
# - 成本节省: 支持 Spot 实例（便宜 70%）
```

## 面试高频追问

**Q1: ZeRO-3 为什么需要 All-Gather？**

A: 参数分片后，每张卡只有 1/DP 参数。计算需要完整参数，因此必须从其他卡收集。虽然通信有成本，但换取了：**单卡显存减少 DP 倍**，使大模型训练成为可能。

**Q2: 通信如何重叠到计算中？**

A: 使用 CUDA Stream 并行：
```python
# Stream 1: 计算当前层
# Stream 2: All-Gather 下一层参数
# Barrier: 确保下一层参数在计算前到达
```
重叠率可达 50-70%，实际训练效率 85-90%。

**Q3: CPU/NVMe Offload 的优劣？**

A:
- **CPU Offload**: 延迟较低（PCIe 带宽 32GB/s），适合优化器状态
- **NVMe Offload**: 容量大，延迟高（3-5μs），适合不频繁访问参数
- **组合使用**: 优化器状态 → CPU，参数 → GPU（计算时），冷数据 → NVMe

**Q4: 70B 模型在 8×A100 上如何配置？**

A:
```yaml
DP=8, PP=1, TP=1, ZeRO-3=True
Batch=4, Seq=4096
CPU_Offload=True
Activation_Checkpointing=True

# 每卡显存: 75GB（可运行）
# 吞吐: 1800 tokens/s/gpu
# 训练 1T tokens: ~15 天
```

---

**难度评级**: ⭐⭐⭐
**出现频率**: 100%（所有大模型训练岗位）
**掌握要求**: 三阶段对比 + 时序图 + 显存公式推导
