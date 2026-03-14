# GQA（分组查询注意力）是什么？为什么 LLaMA 2 用它？它如何平衡性能和效率？

## 1. 核心定性
本质上，**GQA (Grouped Query Attention)** 是 Multi-Query Attention (MQA) 和 Multi-Head Attention (MHA) 的折中方案，通过将查询头分组并让每组共享一个键/值头，在保持 MHA 表达能力的同时将 KV-cache 内存减少为原来的 $1/g$（$g$ 为分组数），推理速度提升 30-40%。

## 2. 具体流程
1. **头部分组**: 将 $h$ 个查询头分为 $g$ 组，每组 $h/g$ 个查询头共享一个键头和值头
2. **KV-cache 压缩**: KV-cache 从 $h$ 个键/值头压缩到 $g$ 个，内存占用降至 $1/g$
3. **路由计算**: 每个查询组 $i$ 使用共享的键/值头 $K_i, V_i$ 计算注意力，通过分组内广播实现并行化

## 3. 数学基础

**传统 MHA**:
$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right)V_i$$

KV-cache 大小: $2 \times h \times n \times d_k$

**MQA (极端)**:
$$K_1 = K_2 = ... = K_h$$
$$V_1 = V_2 = ... = V_h$$

KV-cache 大小: $2 \times 1 \times n \times d_k$ (减少 $h$ 倍)

**GQA (折中)**:
将 $h$ 个头分为 $g$ 组，每组 $h/g$ 个头:
$$\text{Group}_i = \{\text{head}_j | j \in [i \cdot \frac{h}{g}, (i+1) \cdot \frac{h}{g})\}$$

$$K_i = K_j, \quad V_i = V_j \quad \forall j \in \text{Group}_i$$

KV-cache 大小: $2 \times g \times n \times d_k$ (减少 $h/g$ 倍)

**复杂度对比**:
| 方法 | KV-cache | 参数量 | 质量损失 |
|------|----------|--------|----------|
| MHA | $2hnd_k$ | 0% | 0% (基准) |
| GQA | $2gnd_k$ | 0% | 1-2% |
| MQA | $2nd_k$ | 0% | 3-5% |

## 4. 工程考量

**为什么 LLaMA 2 使用 GQA**:

1. **推理效率**: 
   - LLaMA 2 70B: $h=64$, 在 $n=4096$ 时 KV-cache = 2×64×4096×128×4B = 2GB
   - 使用 GQA ($g=8$): KV-cache = 2×8×4096×128×4B = 250MB (减少 8x)
   - 推理批大小从 1 提升到 8，吞吐量增加 8x

2. **质量平衡**:
   - MQA 在 70B 上 SacreBLUE 下降 2.3 点
   - GQA 只下降 0.5 点
   - 证明分组保留了必要的表示多样性

3. **实现简单**:
   - 修改注意力计算，键/值张量重复 $h/g$ 次
   - 一行代码实现: `k = k.repeat_interleave(h//g, dim=0)`

**分组数 $g$ 选择**:

| 模型规模 | $h$ | 推荐 $g$ | KV-cache 减少 | 质量损失 |
|----------|-----|----------|---------------|----------|
| 7B | 32 | 4 | 8x | <0.5% |
| 30B | 48 | 6 | 8x | 0.5-1% |
| 70B | 64 | 8 | 8x | 1-2% |

**经验法则**: $g = \sqrt{h}$ 或 $h/8$

**Trade-off**:

- $g=1$ (MHA): 质量最好，推理慢，内存大
- $g=2-4$: 平衡，质量损失 <1%（推荐）
- $g=h$ (MQA): 推理最快，质量损失 3-5%

**实现细节**:

**HuggingFace 实现**:
```python
from transformers import LlamaConfig

config = LlamaConfig(
    num_attention_heads=32,
    num_key_value_heads=4,  # g=4
)
```

**手动实现**:
```python
# Q: [batch, h, seq, d_k]
# K, V: [batch, g, seq, d_k]

# Expand K, V to match Q
k = k.unsqueeze(2).expand(-1, -1, h//g, -1, -1).reshape(batch, h, seq, d_k)
v = v.unsqueeze(2).expand(-1, -1, h//g, -1, -1).reshape(batch, h, seq, d_k)

# Now standard attention
attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
attn = F.softmax(attn, dim=-1)
output = torch.matmul(attn, v)
```

**内存-速度优化顺序**:
MHA → GQA ($g=2$) → GQA ($g=4$) → GQA ($g=8$) → MQA

**关键发现**:

- **质量拐点**: 从 $g=2$ 到 $g=4$，质量几乎无损；从 $g=4$ 到 $g=8$，下降 0.5-1%
- **批处理友好**: GQA 在 batch=8 时比 MHA 快 2.3x，在 batch=1 时快 1.5x
- **精度敏感**: 4-bit 量化下 GQA 质量下降比 MHA 小 30%

**致命弱点**:

- **分组边界**: 当 $h$ 不能被 $g$ 整除时实现复杂，需额外处理
- **注意力模式**: GQA 假设同一组内查询头关注相似信息，在极端任务（如多模态）上可能失效
- **外推能力**: GQA 在训练长度外推时比 MHA 差 5-10%

## 5. 工业映射

**HuggingFace Transformers**:
- 从 4.32.0 支持 `num_key_value_heads` 参数
- Llama-2 70B: `num_key_value_heads=8`
- 一行配置启用 GQA

**vLLM 推理加速**:
- GQA + PagedAttention 组合
- 70B 模型在 A100-80G 上批大小从 2 → 16
- 吞吐量提升 8x

**LLaMA-2 70B 实测**:
- MHA: 推理 4096 长度，显存 72GB
- GQA ($g=8$): 显存 40GB（减少 44%）
- 速度: 40 tok/s（vs MHA 25 tok/s，提升 60%）
- 质量: HumanEval 35.2% → 34.8%（下降 0.4%，可忽略）

**Amazon Bedrock**:
- 托管 LLaMA-2 使用 GQA
- 每 token 成本降低 30%
- P99 延迟从 800ms 降至 500ms

**TGI (Text Generation Inference)**:
- 专门优化 GQA 的 KV-cache 管理
- Sharding 时按 group 分片而不是 head
-通信量减少 $g/h$

**模型量化后**:

- 4-bit 量化 + GQA:
  - 内存: 70B 模型从 140GB → 35GB
  - 可单卡推理（A100-40GB）
  - 质量: HumanEval 34.8% → 33.1%（额外下降 1.7%）

**与其他优化结合**:

- **FlashAttention + GQA**:
  - Speedup = FlashAttention ▽ + GQA ▽▽▽
  - 70B 速度提升 3x vs 单独 FA
  
- **MQA vs GQA 选择**:
  - Batch inference: MQA 更优（最大化吞吐）
  - Streaming inference: GQA 更优（平衡速度和质量）

**选择建议**:

| 场景 | 推荐 | 原因 |
|------|------|------|
| 70B+ 推理 | GQA ($g=8$) | 内存关键 |
| 30B 推理 | GQA ($g=4$) | 平衡 |
| 7B 推理 | MHA | g 小收益不大 |
| 多租户 | GQA | 批处理友好 |
| 边缘设备 | GQA + 量化 | 极致压缩 |

**实现建议**:
- **训练**: 从 MHA 开始，转换为 GQA 需重新训练或 distillation
- **推理**: 可直接使用 pre-trained GQA 模型
- **蒸馏**: 从 MHA teacher 到 GQA student，需添加 attention map 匹配 loss

**结论**: GQA 是 LLM 推理优化的关键技术，在 LLaMA 2 中验证了其有效性。在 30B+ 模型上应优先使用 GQA（g=4-8），在 7B 以下可使用 MHA。GQA + FlashAttention + 量化是当前推理的三驾马车
