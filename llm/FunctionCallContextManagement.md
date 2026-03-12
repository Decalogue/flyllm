# Function Call 多轮交互上下文管理设计

## 1. 核心定性
本质上，Function Call 的上下文维护是一个**状态机驱动的有限历史窗口管理问题**，通过**对话级消息链 + 工具结果注入**机制实现多轮语义连续性。

---

## 2. 具体流程

1. **消息链累加**：每一轮将 `user message → assistant thinking → tool_calls → tool_results → final_response` 完整序列追加到上下文窗口。
2. **角色标记注入**：工具返回结果以 `role="tool"` + `tool_call_id` 回环绑定，确保模型能区分「用户输入」与「工具输出」。
3. **滑动窗口截断**：当 token 超限，按优先级丢弃早期 `tool`/`function` 消息对，保留 system 指令与最近 user-assistant 对。

---

## 3. 硬核推演：上下文结构体

```json
{
  "messages": [
    {"role": "system", "content": "你是一个助手..."},
    {"role": "user", "content": "查北京天气"},
    {"role": "assistant", "tool_calls": [
      {"id": "call_123", "type": "function", 
       "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_123", "content": "{\"temp\":25,\"humidity\":60}"},
    {"role": "assistant", "content": "北京今天25度..."},
    {"role": "user", "content": "那上海呢？"}  // ← 依赖上文，模型需理解"那"指天气查询
  ],
  "tools": [...],
  "tool_choice": "auto"
}
```

**上下文传播公式**：
$$Context_t = \{System\} \cup \{M_{t-k}, ..., M_{t-1}, M_t\}$$

其中：
- $k$: 最大历史轮数（受 `max_context_length` 约束）
- $M_i$: 第 $i$ 轮的消息四元组 $(role, content, tool\_calls, tool\_call\_id)$
- **关键约束**：包含 `tool_calls` 的 assistant 消息必须与对应的 `role="tool"` 消息成对出现，否则模型报错

---

## 4. 工程考量

| Trade-off | 说明 |
|-----------|------|
| **空间换连贯** | 保留完整工具调用链路（占 token）换取多轮指代消解能力 |
| **致命弱点** | **长链依赖断裂**：当 `k` 受限，早期工具结果被截断后，用户说"刚才那个结果再查一下"，模型丢失引用目标 |
| **并发陷阱** | 多工具并行调用（parallel tool calls）时，若某工具超时，上下文处于「半完成」状态，需原子性回滚或标记失败 |

---

## 5. 工业映射

在工业界，该机制被直接应用于：
- **OpenAI Chat Completions API**：`messages` 数组强制要求 `tool` 消息紧跟对应 `assistant` 的 `tool_calls`
- **LangChain ConversationBufferMemory**：通过 `buffer` 维护消息链，提供 `prune` 策略应对超长上下文
- **MCP (Model Context Protocol)**：标准化的工具调用上下文生命周期管理协议，确保跨模型/跨平台一致性
