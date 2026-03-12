# Function Call 格式与 JSON Schema 设计

## 1. 核心定性

本质上，Function Call 是 LLM 与外部工具世界的**标准化协议接口**，通过结构化 JSON Schema 描述函数签名，使模型能精确输出可执行的函数调用指令。

---

## 2. 具体流程

1. **声明阶段**：系统在系统提示中注入可用函数的 JSON Schema 定义，告知模型"你有这些工具可用"
2. **决策阶段**：用户提问后，LLM 判断是否需要调用工具，若需要则生成符合 Schema 的 JSON 对象（函数名+参数）
3. **执行阶段**：宿主程序解析模型输出的 JSON，执行对应函数，并将结果再次输入模型生成最终回复

---

## 3. 数学基础 / 核心结构

### 标准 Function Call Schema 结构

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的实时天气",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "城市名称，如'北京'"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "default": "celsius"
        }
      },
      "required": ["location"]
    }
  }
}
```

### 模型输出格式（标准化 JSON）

```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\":\"上海\",\"unit\":\"celsius\"}"
      }
    }
  ]
}
```

**关键变量定义：**
- `name`: 函数唯一标识符，必须与宿主注册名严格一致
- `parameters.properties`: 参数 Schema，遵循 JSON Schema Draft 7 规范
- `required`: 必填参数数组，缺失会导致调用失败
- `enum`: 枚举约束，限制参数取值范围

---

## 4. 工程考量

| Trade-off | 分析 |
|-----------|------|
| **灵活性 vs 稳定性** | Schema 越宽松（如大量 `anyOf`）模型理解成本越高，输出不确定性增加；过度严格则限制模型推理能力 |
| **描述精度 dilemma** | `description` 字段是模型理解函数意图的唯一线索，描述模糊会导致错误调用；但过长描述会挤占上下文窗口 |
| **嵌套深度限制** | 参数嵌套超过 3 层时，模型生成准确率显著下降（实测下降 15-30%） |

**致命弱点：**
- **Schema 漂移**：函数签名变更后，旧版本模型仍可能输出过时参数结构，导致运行时错误
- **幻觉调用**：模型可能"编造"不存在的参数值（尤其在 `additionalProperties: true` 时）
- **并发地狱**：多函数并行调用时，参数依赖关系难以在 Schema 层面表达

---

## 5. 工业映射

在工业界，该机制被直接应用于：
- **OpenAI GPT-4/GPT-3.5**: 原生 `tools`/`functions` API，已成为事实标准
- **Claude (Anthropic)**: `tool_use` 模块，支持嵌套工具调用
- **LangChain**: `StructuredTool` 抽象层，统一封装不同模型的 Function Call 差异
- **MCP (Model Context Protocol)**: Anthropic 2024 年推出的开放协议，标准化模型与外部数据源的连接方式

---

## Schema 设计最佳实践

```json
{
  "functions": [
    {
      "name": "search_database",
      "description": "通过SQL查询数据库。仅用于查询，禁止UPDATE/DELETE/INSERT。",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "完整的SQL查询语句，必须使用参数化查询防止注入"
          },
          "limit": {
            "type": "integer",
            "default": 100,
            "maximum": 1000,
            "description": "返回结果数量上限，防止数据过载"
          }
        },
        "required": ["query"]
      }
    }
  ]
}
```

**设计原则：**
1. **单一职责**：每个函数只做一件事，通过组合实现复杂逻辑
2. **防御性约束**：用 `enum`/`minimum`/`maximum`/`pattern` 限制输入范围
3. **显式优于隐式**：必填参数必须列入 `required`，禁止依赖默认值
4. **描述即契约**：`description` 需包含调用时机、参数格式、边界情况
