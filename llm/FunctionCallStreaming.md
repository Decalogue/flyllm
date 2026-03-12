# Function Call 流式输出实现与优化

---

## 1. 核心定性

本质上，Function Call 流式输出是为了**解决首Token延迟问题**，通过**增量解析 Partial JSON**机制，在**模型生成过程中实时提取工具调用参数**的边计算边消费架构。

---

## 2. 具体流程

1. **流式接收**：SSE 通道逐 Chunk 接收模型输出，每个 Chunk 包含增量文本（`delta.content` 或 `delta.tool_calls`）
2. **增量累积**：维护缓冲区聚合 Partial JSON，每收到新 Chunk 即尝试解析工具名和参数片段
3. **提前触发**：当累积内容满足工具调用触发条件（如函数名确认、参数完整），立即发起下游调用，无需等待流结束

---

## 3. 硬核推演

**增量解析状态机**：

$$
S_{t} = f(S_{t-1}, \delta_t) = 
\begin{cases} 
\text{IDLE} & \text{if } \nexists \text{tool\_calls} \\
\text{NAME\_PARTIAL} & \text{if name 不完整} \\
\text{ARGS\_STREAMING} & \text{if args 流式累积} \\
\text{EXECUTABLE} & \text{if JSON 可解析}
\end{cases}
$$

**首Token延迟收益**：

$$
\Delta T = T_{\text{full\_stream}} - T_{\text{first\_executable}} = \sum_{i=k}^{N} \frac{L_i}{R} \approx \frac{(N-k) \cdot \bar{L}}{R}
$$

其中：
- $N$: 总 Chunk 数
- $k$: 首次满足执行条件的 Chunk 索引
- $R$: 模型生成速率 (tokens/s)
- $\bar{L}$: 平均 Chunk 长度

**关键解析伪代码**：
```python
class StreamingToolParser:
    def __init__(self):
        self.buffer = {}
        self.state = IDLE
        
    def feed(self, chunk: ToolCallDelta) -> Optional[Call]:
        # 增量累积
        if chunk.index not in self.buffer:
            self.buffer[chunk.index] = {"name": "", "arguments": ""}
        
        self.buffer[chunk.index]["name"] += chunk.function.name or ""
        self.buffer[chunk.index]["arguments"] += chunk.function.arguments or ""
        
        # 尝试解析
        try:
            args = json.loads(self.buffer[chunk.index]["arguments"])
            return ExecutableCall(
                name=self.buffer[chunk.index]["name"],
                arguments=args
            )
        except json.JSONDecodeError:
            return None  # 继续累积
```

---

## 4. 工程考量

| Trade-off |  sacrificed | gained |
|-----------|-------------|--------|
| **解析复杂度** | 状态机维护成本 | 首Token延迟 ↓ 30-50% |
| **容错性** | 错误JSON的恢复难度 | 并行调用能力 |
| **带宽占用** | SSE长连接开销 | 渐进式参数填充 |

**致命弱点**：
1. **JSON截断陷阱**：模型在参数边界处切分 Chunk，导致 `{"key": "val` 悬停，需设置**强制刷新阈值**（如 50ms 无新数据则尝试截断修复）
2. **多函数竞态**：并行工具调用时，索引错乱会导致参数错配（`index` 字段校验不可省略）
3. **幻觉累积**：Partial 阶段过早决策可能基于不完整参数，需**可逆性设计**（预检→确认→执行）

---

## 5. 工业映射

在工业界，该机制被直接应用于：
- **OpenAI GPT-4 Turbo**: `tool_calls` 的 `delta` 模式，配合 `stream_options: {"include_usage": true}`
- **LangChain Streaming**: `on_tool_start` 回调在参数累积完成时触发，而非流结束
- **Vercel AI SDK**: `streamText` 的 `toolCallStreaming` 选项，实现前端渐进式渲染工具调用卡片

---