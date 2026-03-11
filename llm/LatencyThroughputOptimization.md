# 大模型推理延迟优化与吞吐量优化

## 1. 核心定性

**推理延迟优化**本质上是通过**减少每个 token 的浮点运算量（FLOPs）** 或**提高硬件利用率（TFLOPS）** 来降低生成时间；**吞吐量优化**则是通过**批量聚合（Batching）和请求调度**将硬件算力"填满"以提升整体利用率。

## 2. 具体流程

1. **延迟优化路径**：KV Cache 复用 → 算子融合 → 量化压缩 → 投机采样（Speculative Decoding） → 算子内核调优。
2. **吞吐量优化路径**：Continuous Batching（In-flight Batching） → PagedAttention 显存管理 → 请求优先级调度 → 多卡并行（TP/PP/DP）。
3. **延迟-准确性平衡**：依据任务容忍度选择量化精度（FP16/INT8/INT4），用**提前退出（Early Exit）** 或**级联推理（Cascade Inference）** 在精度和速度间动态折中。

## 3. 数学基础

### 3.1 推理延迟公式

$$T_{latency} = \sum_{i=1}^{L} \left( T_{prefill} \cdot \mathbb{1}_{[i=1]} + T_{decode} \cdot \mathbb{1}_{[i>1]} \right)$$

其中：
- $L$: 生成长度（sequence length）
- $T_{prefill}$: 首 token 预填充时间，计算量 $\mathcal{O}(n^2 \cdot d)$，$n$ 为 prompt 长度，$d$ 为模型维度
- $T_{decode}$: 自回归解码步时间，每步 $\mathcal{O}(b \cdot 1 \cdot d^2)$，$b$ 为 batch size

### 3.2 吞吐量公式

$$\text{Throughput} = \frac{\text{Total Tokens Processed}}{T_{wall}} = \frac{\sum_{j=1}^{B} L_j \cdot b}{T_{compute} + T_{idle}}$$

PagedAttention 显存利用率：

$$U_{memory} = \frac{\text{Used KV Blocks}}{\text{Total GPU Memory}} \approx \frac{\sum_{i} \lceil \frac{L_i}{B_{block}} \rceil}{N_{blocks}}$$

其中：
- $B_{block}$: KV block 大小（如 16 tokens）
- $N_{blocks}$: 总 block 数量

### 3.3 量化误差边界

$$\| W_{quant} - W_{fp16} \|_{F} \leq \epsilon \cdot \sqrt{d_{out} \cdot d_{in}}$$

其中 $\epsilon$ 为量化步长，INT8 时 $\epsilon \approx 1/256$，INT4 时 $\epsilon \approx 1/16$。

## 4. 工程考量

| 维度 | Trade-off | 致命弱点 |
|------|-----------|----------|
| **量化** | 用精度换显存/速度 | INT4 以下模型能力断崖式下降，数学推理任务误差放大 |
| **Continuous Batching** | 用调度复杂度换吞吐 | 长尾延迟（tail latency）不可控，单个长序列阻塞整批 |
| **Speculative Decoding** | 用 draft model 开销换速度 | draft model 接受率低时反而负优化，增加系统复杂度 |
| **Pipeline Parallelism** | 用通信换显存扩展 | Bubble time 导致 GPU 空转，短序列场景效率极低 |
| **Early Exit** | 用动态计算图换平均延迟 | 出口策略设计不当导致漏判，需要人工设计退出阈值 |

## 5. 工业映射

在工业界，上述机制被直接应用于：

- **vLLM / SGLang**: 通过 **PagedAttention** 实现 Continuous Batching，KV Cache 显存利用率从 20-40% 提升至 80%+，支持单卡千级并发。
- **TensorRT-LLM / DeepSpeed-Inference**: 实现 **算子融合（FlashAttention/MLA）** 与 **FP8/INT4 量化**，在 A100/H100 上达到 90%+ 硬件利用率。
- **OpenAI Triton / CUDA kernel 定制**: 针对特定硬件架构（Hopper Tensor Core）手写 fused kernel，消除 kernel launch 开销。
- **Speculative Decoding**: Google Gemini / Anthropic Claude 内部使用小型 draft model（如 7B 辅助 70B）提升 2-3x 解码速度。

## 6. 性能瓶颈定位方法论

### 6.1 诊断工具链

```bash
# 1. Roofline Model 分析
# 确定是 compute-bound 还是 memory-bound
Operational Intensity = FLOPs / Bytes (AI)

# 2. Nsight Systems / PyTorch Profiler
- 查看 kernel launch gap（>10μs 为异常）
- 定位 cudaStreamSynchronize 阻塞点

# 3. 显存带宽分析
Actual Bandwidth = (2 * params_size + 4 * KV_cache_size) / Time
H100 理论: 3.35 TB/s，实际 >80% 为达标
```

### 6.2 瓶颈层级判定

| 瓶颈特征 | 根因 | 优化手段 |
|----------|------|----------|
| GPU 利用率 <50%，CPU 100% | DataLoader 或 tokenizer 瓶颈 | 预取（prefetch）、多进程 tokenization |
| Attention kernel 耗时 >60% | Memory-bound，KV Cache 访问零散 | FlashAttention-2/3、GQA/MQA |
| 通信耗时 >30%（多卡） | TP/PP 通信瓶颈 | 降低 TP degree，改用 PP 或 ZeRO-3 |
| Batch size 上不去 | 显存碎片或 OOM | PagedAttention、Gradient Checkpointing |

### 6.3 延迟-准确性平衡策略

```
级联推理（Cascade Inference）:
┌──────────────┬──────────────┬──────────────┐
│  快速模型     │   中等模型    │   大模型     │
│  (INT4/7B)   │  (INT8/70B)  │  (FP16/405B) │
└──────────────┴──────────────┴──────────────┘
     ↓               ↓               ↓
  置信度检查      置信度检查       最终 fallback
  (Threshold)    (Threshold)
```

**核心原则**：用 90% 的 query 在轻量级模型上解决，仅 10% 的 hard case 触发大模型，整体吞吐量提升 5-10x 而平均精度损失 <2%。

---

💡 **一句话总结**：延迟优化的本质是"算得更快"，吞吐量优化的本质是"算得更满"，二者平衡的关键在于**量化容忍度**和**动态资源调度**的精准把控。
