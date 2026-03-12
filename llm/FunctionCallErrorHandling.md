# Function Call 参数提取错误处理机制

## 1. 核心定性

本质上，Function Call 错误处理是一个**输入验证 → 错误分类 → 降级恢复**的三层防御体系，核心目标是在**用户意图保真**与**系统稳定性**之间建立隔离墙。

---

## 2. 具体流程

1. **输入层校验**：LLM 输出结构化参数后，先执行 Schema 校验（类型匹配、必填字段、枚举值范围），失败即触发 `ParseError`
2. **语义层校验**：参数格式合法但业务逻辑违规（如负数金额、越界索引），触发 `ValidationError` 并生成修复提示
3. **执行层保护**：调用外部 API 前使用熔断器模式，超时或异常时返回优雅降级结果，避免级联故障

---

## 3. 数学基础

**三层错误概率模型**：

$$P_{total} = P_{parse} \cdot P_{semantic|parse} \cdot P_{exec|semantic}$$

其中：
- $P_{parse}$：Schema 校验失败率，由 LLM 输出质量决定，通常控制在 $<10^{-2}$
- $P_{semantic|semantic}$：给定格式正确的条件下语义违规的概率，$P_{semantic} \approx \frac{N_{invalid}}{N_{total}}$
- $P_{exec|semantic}$：业务校验通过后执行失败的概率，包含网络超时、服务不可用等

**重试策略公式**（指数退避）：

$$T_{retry} = T_{base} \cdot 2^{attempt} \cdot (1 + jitter), \quad jitter \in [0, 1)$$

**伪代码逻辑**：
```python
def execute_function_call(llm_output, schema, max_retry=3):
    # Layer 1: Parse & Schema Validation
    try:
        params = json.loads(llm_output)
        validate(instance=params, schema=schema)  # JSON Schema
    except (JSONDecodeError, ValidationError) as e:
        return ErrorResponse(type="PARSE_ERROR", retryable=False, hint=fix_hint(e))
    
    # Layer 2: Semantic Validation
    if not semantic_check(params):  # 业务规则校验
        return ErrorResponse(type="SEMANTIC_ERROR", retryable=True, hint=correction_prompt())
    
    # Layer 3: Execution with Circuit Breaker
    for attempt in range(max_retry):
        try:
            with circuit_breaker():
                return external_api.call(params)
        except TransientError:
            sleep(exp_backoff(attempt))
        except FatalError:
            return ErrorResponse(type="EXEC_ERROR", retryable=False)
    
    return ErrorResponse(type="TIMEOUT", retryable=False)
```

---

## 4. 工程考量

| 维度 | Trade-off | 致命弱点 |
|------|-----------|----------|
| **严格性 vs 灵活性** | Schema 越严格，召回率下降但精确率上升 | 过度严格导致 LLM 轻微偏差即失败，用户体验断裂 |
| **同步 vs 异步** | 同步校验简单但阻塞；异步队列解耦但引入延迟 | 高并发场景下同步校验成为瓶颈，QPS 骤降 |
| **重试风暴** | 指数退避缓解压力但增加长尾延迟 | 熔断阈值设置不当，级联雪崩时仍不断重试打垮下游 |

**关键设计原则**：
- **区分可重试错误**：网络抖动 (`retryable=True`) vs 参数逻辑错误 (`retryable=False`)
- **错误信息分层**：内部记录完整堆栈，对外仅暴露脱敏的 `error_code` + `user_message`
- **兜底策略**：核心功能必须提供**硬编码默认值**或**缓存历史结果**作为最后防线

---

## 5. 工业映射

在工业界，该机制被直接应用于：
- **OpenAI Function Calling API**：采用 JSON Schema 强校验，失败时返回 `invalid_function_call` 错误码，并提供 `tool_choice` 强制模式降低幻觉
- **LangChain/LangGraph**：通过 `RetryParser` 和 `OutputFixingParser` 实现自动重试与错误修复闭环，核心代码位于 `langchain.output_parsers.retry` 模块
- **Uber Cadence / Temporal**：工作流引擎中使用 `Saga Pattern` 补偿事务处理长时间运行的 Function Call 失败，确保最终一致性
