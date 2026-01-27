# 大模型流式解析：面试题整理

## 一、出题方式

### 1. 概念题

| 题目 | 考查点 |
|------|--------|
| LLM 流式输出和一次性返回有什么区别？为什么要做流式？ | 流式 vs 非流式、TTFB、体验与内存 |
| SSE 和 WebSocket 在 LLM 场景下怎么选？各自适用什么情况？ | 传输协议、单向 vs 双向 |
| 流式解析具体指解析什么？是解析 token 流、HTTP 流，还是解析成 JSON？ | 概念边界、解析对象 |

### 2. 实现题

| 题目 | 考查点 |
|------|--------|
| 如何用 Python 调用 OpenAI 兼容 API 做流式输出？写出核心循环。 | `stream=True`、chunk 迭代、delta |
| FastAPI/Flask 如何把 LLM 的 generator 转成 HTTP 流式响应？用什么 Response 类型？ | `StreamingResponse`、`stream_with_context` |
| 如果模型同时返回 `reasoning_content` 和 `content`，流式时如何区分并处理？ | 思维链模型、delta 字段 |

### 3. 编码 / 解析题

| 题目 | 考查点 |
|------|--------|
| 流式时为什么要用 `TextDecoder` + `{ stream: true }`？不用会出什么问题？ | UTF-8 多字节、跨 chunk 乱码 |
| UTF-8 多字节字符被拆到两个 chunk 里怎么办？前端/后端各怎么处理？ | 流式解码、缓冲与拼接 |
| 流式返回的是纯文本，但要解析成 JSON，如何做增量/流式 JSON 解析？ | 协议设计、JSON Lines、SSE |

### 4. 场景题

| 题目 | 考查点 |
|------|--------|
| 打字机效果如何实现？要不要做缓冲？缓冲多大合适？ | buffer、flush 策略、体验 |
| 流式过程中连接中断或模型报错，如何做错误恢复和用户提示？ | 错误处理、AbortController |
| 前端用 `fetch` + `ReadableStream` 消费流式 API 时，核心代码怎么写？ | 前端流式消费 |

---

## 二、核心概念与原理

### 2.1 流式 vs 一次性返回

| 对比项 | 一次性返回 | 流式返回 |
|--------|------------|----------|
| 首字节时间（TTFB） | 等整段生成完才返回 | 边生成边返回，延迟低 |
| 用户体验 | 长时间白屏 | 打字机效果，减少焦虑 |
| 服务端内存 | 整段在内存 | 边生成边推送，可边释放 |
| 实现复杂度 | 简单 | 需处理 chunk、编码、错误等 |

**本质**：流式 = **按 chunk 推送 token/文本**，客户端**边收边解码、边展示**。

### 2.2 SSE vs WebSocket

- **SSE（Server-Sent Events）**
  - HTTP 长连接，**服务器 → 客户端** 单向推送
  - `Content-Type: text/event-stream`，数据形如 `data: xxx\n\n`
  - 适合：聊天、日志、状态推送等**只需下行**的场景

- **WebSocket**
  - 全双工，双向通信
  - 适合：实时协作、游戏等需要**频繁双向**的场景

- **LLM 场景**：通常只需**下行流**，用 **SSE 或 HTTP chunked**（如 `text/plain`）即可；浏览器原生支持 `EventSource` 消费 SSE。

### 2.3 为何 `TextDecoder` 要 `{ stream: true }`？

- UTF-8 多字节字符（如中文）可能**跨 chunk 切分**：前半在一个 chunk，后半在下一个。
- 若不用 `stream: true`，每个 chunk 单独解码，**不完整的多字节序列会乱码**。
- `stream: true` 时，decoder **保留未完成的字节**，与下一 chunk 拼接后再解码，保证正确性。

### 2.4 流式解析结构化数据（如 JSON）

- **纯文本流**（如 `text/plain` 按 chunk 推送）：只能**先拼接完整再 `JSON.parse`**，无法真正流式解析 JSON。
- **增量 JSON** 的常见做法：
  - **SSE**：每个 event 一个完整小 JSON（如按句/段）。
  - **JSON Lines**：每行一个 JSON，按行解析。
  - **自定义协议**：如「长度前缀 + JSON」，按块解析。

**面试区分**：「流式解析」在 LLM 里多指**对 token/文本流的解码与展示**；若问「流式解析 JSON」，需说明**协议如何分块**再解析。

### 2.5 缓冲策略（打字机效果）

- **要缓冲**：避免逐 token 推送，减少请求次数与 UI 刷新频率，更流畅。
- **缓冲大小**：通常 **10～50 字符** 或 **遇换行即 flush**；过小像抖动，过大首字延迟增加。
- **实现**：后端 `buffer` 累加，达到阈值或 `\n` 时 `yield`；前端同理可按需做小幅缓冲再更新 DOM。

---

## 三、标准回答与代码示例

### 3.1 后端：Python + OpenAI 兼容 API

```python
BUFFER_SIZE = 10  # 字符数，可按需调整

response = client.chat.completions.create(
    model="...",
    messages=messages,
    stream=True,
    max_tokens=max_new_tokens,
)

buffer = ''
for chunk in response:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    text = getattr(delta, 'reasoning_content', None) or delta.content or ''
    if text:
        buffer += text
        if len(buffer) >= BUFFER_SIZE or '\n' in text:
            yield buffer
            buffer = ''
if buffer:
    yield buffer
```

**要点**：
- `stream=True` 后返回 **chunk 迭代器**，不是一次性 `message.content`。
- 有 `reasoning_content` 的模型要在 `delta` 里区分，否则会混入“思考过程”。
- **buffer + 按长度/换行 flush**：平衡实时性与流畅度。

### 3.2 HTTP 层：FastAPI / Flask

**FastAPI**（`StreamingResponse`，常用 `text/plain` 作 chunked）：

```python
def stream_data():
    for t in model_stream(messages):
        yield t

return StreamingResponse(
    stream_data(),
    media_type="text/plain; charset=utf-8",
)
```

**Flask**（`stream_with_context` 保活请求上下文）：

```python
@stream_with_context
def stream_data():
    for t in model_stream(messages):
        yield t

return Response(stream_data(), mimetype="text/plain; charset=utf-8")
```

### 3.3 前端：fetch + ReadableStream

```javascript
const res = await fetch('/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ messages, stream: true }),
});
const reader = res.body.getReader();
const decoder = new TextDecoder('utf-8', { stream: true });

let fullText = '';
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  fullText += decoder.decode(value, { stream: true });
  updateUI(fullText);  // 累积更新，实现打字机效果
}
```

**说明**：`fullText` 累积完整已解码内容；`{ stream: true }` 保证跨 chunk 的 UTF-8 正确解码。

### 3.4 错误与中断

- **连接中断**：`fetch` 会 reject，前端 `try/catch` 提示「连接断开，请重试」。
- **服务端错误**：若已开始流式，可在协议里约定**错误行**（如 `data: {"error": "..."}`），前端解析到即停止并提示。
- **用户取消**：用 `AbortController`，将 `signal` 传入 `fetch`，取消后连接关闭，后端 generator 随之结束。

```javascript
const ac = new AbortController();
fetch(url, { signal: ac.signal, ... }).then(...).catch((e) => {
  if (e.name === 'AbortError') { /* 用户取消 */ }
});
ac.abort();  // 取消请求
```

---

## 四、简版背诵（口述用）

1. **流式的目的**：降低首字延迟、打字机体验、控制内存；本质是 **chunk 序列的生成与解析**。
2. **后端流程**：`stream=True` → `for chunk in response` → 取 `delta.content` / `reasoning_content` → **buffer 再 yield** → `StreamingResponse` / `Response(stream_with_context(...))`。
3. **前端流程**：`fetch` → `res.body.getReader()` → `TextDecoder('utf-8', { stream: true })` 逐块 `decode` → 累积更新 UI。
4. **UTF-8**：必须 `stream: true`，否则跨 chunk 的多字节字符会乱码。
5. **SSE vs WebSocket**：LLM 只推不下发，用 **SSE 或 HTTP chunked** 即可。
6. **JSON**：纯文本流无法真正流式解析 JSON，需靠**协议**（SSE、JSON Lines 等）分块后再解析。

---

## 五、速查表

| 概念 | 一句话 |
|------|--------|
| 流式解析 | 对 LLM 输出的 **token/文本流** 边收边解码、边展示；若涉及 JSON，则指按协议分块后的解析。 |
| `stream: true`（API） | 接口返回 chunk 迭代器，逐个 `delta` 推送。 |
| `TextDecoder` 的 `stream: true` | 跨 chunk 正确解码 UTF-8，避免多字节被切断导致乱码。 |
| buffer 缓冲 | 累积到一定长度或换行再 yield，平衡实时性与流畅度；常用 10～50 字或遇 `\n` flush。 |
| `reasoning_content` | 思维链模型的“思考”输出，流式时需与 `content` 区分处理。 |
