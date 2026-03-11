# 流式推理实现与优化

## 1. 核心定性 (The 10-Second Hook)

**本质上，流式推理是为了解决首字节延迟（TTFB）问题，通过将自回归生成的序列逐 token 推送到客户端，实现边生成边输出的流水线结构。**

---

## 2. 具体流程

1. **分块生成**：模型在解码阶段每生成一个 token，立即将其从 GPU 显存拷贝到 CPU 内存，经 detokenize 后推入 SSE/WebSocket 通道
2. **传输流水线**：服务器维持 HTTP 长连接或 WebSocket，采用 chunked-transfer 或 Server-Sent Events 协议，逐帧推送 `{"token": "xxx", "finish_reason": null}`
3. **客户端渲染**：前端收到 token 流后，直接追加到 DOM 或执行打字机效果，直至接收到 `finish_reason: "stop"` 后关闭连接

---

## 3. 数学基础 (硬核推演)

### 3.1 流式生成的时序模型

设用户输入序列长度为 $L_{prompt}$，生成目标长度为 $L_{gen}$，单个 decode step 的延迟为 $t_{decode}$，网络往返延迟为 $t_{network}$：

**传统非流式总延迟**：
$$T_{blocking} = L_{gen} \cdot t_{decode} + t_{network}$$

**流式输出首字节延迟**：
$$T_{ttfb} = t_{prefill} + t_{decode} + t_{network}$$

**流式感知延迟**（用户主观体验）：
$$T_{perceived} \approx \frac{L_{gen} \cdot t_{decode}}{2} + t_{network}$$

其中：
- $t_{prefill}$: prompt 阶段的并行编码延迟（与 $L_{prompt}^2$ 或线性相关，取决于注意力实现）
- $t_{decode}$: 单个 token 的自回归解码延迟（受内存带宽瓶颈主导）

### 3.2 核心解码伪代码

```python
# KV-Cache 流式生成核心逻辑
past_kv = prefill(input_ids)           # (batch, seq, hidden)

for i in range(max_new_tokens):
    logits, past_kv = decode_step(
        input_ids=last_token,          # 仅输入最新 token
        past_key_values=past_kv        # 复用历史 KV，避免重复计算
    )
    
    next_token = sample(logits)        # greedy / temperature / top-p
    
    # 关键：立即推送，不等待完整序列
    yield detokenize(next_token)       # SSE 流式输出
    
    if next_token == EOS:
        break
```

### 3.3 解码阶段复杂度对比

| 阶段 | 计算复杂度 | 内存访问 | 瓶颈 |
|------|-----------|---------|------|
| Prefill | $O(L_{prompt}^2 \cdot d)$ | 高 | 计算密集型 |
| Decode | $O(L_{gen} \cdot d)$ | 极高 | **内存带宽密集型** |

> Decode 阶段每次只处理 1 个新 token，但需加载全部 KV-Cache，导致 **内存带宽 wall**。

---

## 4. 工程考量 (Engineering Trade-offs)

### 4.1 核心取舍

| 优化方向 | 收益 | 代价 |
|---------|------|------|
| **牺牲 token 级实时性 → 采用 micro-batch 聚合** | 减少网络包数量，提升吞吐 | 增加 50-200ms 的缓冲延迟 |
| **牺牲精确性 → 推测解码 (Speculative Decoding)** | 2-3x 速度提升 | 草稿模型占用额外显存，拒绝时回滚开销 |
| **牺牲通用性 → 连续批处理 (Continuous Batching)** | GPU 利用率最大化 | 实现复杂，需动态内存管理 |

### 4.2 致命弱点

1. **首 token 诅咒**：Prefill 阶段必须等待完整 prompt 编码完成，长上下文（32k+）时 $t_{prefill}$ 可能高达秒级，流式也无法缓解
2. **显存碎片雪崩**：Continuous Batching 场景下，不同请求的序列长度差异导致 KV-Cache 内存碎片，显存占用随时间膨胀，最终触发 OOM
3. **网络抖动放大**：每个 token 独立传输，高抖动网络下会出现 "卡顿-突刺" 现象，用户体验反而劣于阻塞式输出

---

## 5. 工业映射 (Industry Mapping)

| 机制 | 开源实现 | 应用场景 |
|------|---------|---------|
| **Continuous Batching + KV-Cache Paging** | [vLLM](https://github.com/vllm-project/vllm) 的 PagedAttention | 高并发在线推理服务，显存利用率提升 2-4x |
| **Speculative Decoding** | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)、[Medusa](https://github.com/FasterDecoding/Medusa) | 延迟敏感场景（对话机器人），解码吞吐量提升 2x+ |
| **Chunked Prefill** | [SGLang](https://github.com/sgl-project/sglang) | 将长 prompt 拆分为 chunks，与 decode 阶段 interleave，降低 TTFT |
| **Streaming SSE/WebSocket** | [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference) | 标准化流式 API，支持生成进度回调、断线重连 |

**工业级优化 checklist**：
- [ ] KV-Cache 分页管理（vLLM PagedAttention）
- [ ] 动态连续批处理（inflight batching）
- [ ] FP8/INT8 量化 + Tensor Parallelism
- [ ] 输出 token 聚合（聚合 4-8 tokens 再 flush）
- [ ] 前端打字机 debounce（避免 DOM 频繁重绘）

---

**一句话总结**：流式推理的本质是**用网络带宽换感知延迟**，配合 KV-Cache 复用和 Continuous Batching，可将用户体验延迟从秒级降至毫秒级；但长上下文 Prefill 瓶颈和显存管理仍是工业部署的核心挑战。
