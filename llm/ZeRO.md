# ZeRO 优化器显存优化与阶段对比

---

## 1. 核心定性

本质上，**ZeRO（Zero Redundancy Optimizer）是一种通过分片（Partitioning）策略消除数据并行中的冗余参数副本，将显存占用随 GPU 数量线性降低的分布式优化器。**

---

## 2. 具体流程

1. **显存分解**：将模型训练所需显存划分为 **Model States**（参数/梯度/优化器状态）和 **Residual States**（激活值/临时缓存），仅对 Model States 进行分片优化。

2. **渐进式分片**：ZeRO-1 仅分片优化器状态，ZeRO-2 增加梯度分片，ZeRO-3 将参数也纳入分片，实现全量 Model States 的跨 GPU 分布式存储。

3. **通信重放**：通过 `all-gather` 按需聚合当前层参数，前向/反向传播后立即释放，用通信带宽换取显存容量。

---

## 3. 数学基础

设：
- $N$：GPU 数量
- $\Psi$：模型参数量（单位：bytes，假设 FP16/BF16）
- $K$：优化器状态膨胀系数（Adam 中 $K=12$，因需存储 FP32 动量 $m$、二阶矩 $v$ 及 FP32 参数副本）

单卡显存占用公式：

| 阶段 | 显存公式 | 相对于单卡数据并行 |
|------|----------|-------------------|
| **Baseline (DP)** | $4\Psi + K\Psi$ | $1\times$ |
| **ZeRO-1** | $\frac{K\Psi}{N} + 4\Psi$ | $\approx \frac{K}{K+4} \times$ |
| **ZeRO-2** | $\frac{(K+2)\Psi}{N} + 2\Psi$ | $\approx \frac{K+2}{K+4} \times$ |
| **ZeRO-3** | $\frac{(K+4)\Psi}{N}$ | $\frac{1}{N} \times$ |

其中：
- $4\Psi$：FP16 参数 ($2\Psi$) + FP16 梯度 ($2\Psi$)
- $\frac{K\Psi}{N}$：分片后的优化器状态（Adam: 4+4+4=12 bytes/param）
- $\frac{2\Psi}{N}$：ZeRO-2 分片梯度（FP16 梯度）
- $\frac{2\Psi}{N}$：ZeRO-3 分片参数（FP16 参数）

---

## 4. 工程考量

### Trade-off 矩阵

| 维度 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|--------|--------|--------|
| **显存节省** | 中（~75%） | 高（~85%） | 极高（随 $N$ 线性） |
| **通信开销** | $1.5\times$ Baseline | $2\times$ Baseline | $3\times$ Baseline |
| **实现复杂度** | 低 | 中 | 高（需 hook 所有 forward） |

### 致命弱点

1. **ZeRO-3 的通信墙**：每层的 `all-gather` 操作成为硬瓶颈，在小 batch size 场景下 GPU 利用率极低，通信延迟掩盖计算。

2. **参数一致性风险**：若出现网络分区或节点故障，分片的参数片段可能不一致，恢复需从 checkpoint 重放全部梯度历史。

3. **激活值未优化**：ZeRO 仅解决 Model States，大模型训练中的激活值（Residual States）仍需靠 Checkpointing 或序列并行处理。

---

## 5. 工业映射

在工业界，ZeRO 机制被直接集成于 **DeepSpeed（Microsoft）** 和 **PyTorch FSDP（Fully Sharded Data Parallel）** 中：
- **DeepSpeed ZeRO-Infinity**：将 ZeRO-3 拓展至 NVMe 卸载，支持训练万亿参数模型
- **Hugging Face Transformers**：`TrainingArguments(deepspeed="ds_config_zero3.json")` 已成为大模型微调的标准范式
- **GPT-3/LLaMA 预训练**：均采用 ZeRO-3 + Gradient Checkpointing 组合，在 1024 张 A100 上实现 175B 参数模型训练

---
