# Function Call 验证机制设计

## 1. 核心定性
本质上，Function Call 验证机制是一套**「输入契约约束系统」**，通过 Schema 声明式定义参数类型/范围/依赖关系，在推理后、执行前进行多阶段校验，防止非法输入穿透到业务层。

## 2. 具体流程

1. **Schema 注册阶段**：函数以 JSON Schema 或 Pydantic Model 形式注册，声明参数类型、必填项、约束规则（range、enum、regex）。
2. **推理拦截阶段**：LLM 输出 Function Call 请求后，先经 **语法解析层**（JSON 合法性）→ **Schema 匹配层**（字段存在性、类型一致性）→ **业务规则层**（跨字段依赖、权限校验）。
3. **失败处置阶段**：校验失败时执行 **拒识策略**（Reject/Retry/Clarify），将错误信息反喂给 LLM 进行自修正。

## 3. 数学基础

### 3.1 验证状态机
验证过程可建模为确定性有限自动机（DFA）：

$$M = (Q, \Sigma, \delta, q_0, F)$$

其中：
- $Q = \{Idle, Parsing, TypeCheck, ConstraintCheck, BusinessCheck, Accept, Reject\}$：验证状态集合
- $\Sigma$：输入符号集（参数键值对）
- $\delta: Q \times \Sigma \rightarrow Q$：状态转移函数
- $q_0 = Idle$：初始状态
- $F = \{Accept, Reject\}$：终止状态集

### 3.2 核心验证伪代码

```python
def validate(function_call: dict, schema: Schema) -> Result:
    # Phase 1: 结构校验
    if not match_schema_structure(function_call, schema):
        return Reject(error="STRUCTURE_MISMATCH")
    
    # Phase 2: 类型校验（递归下降）
    for field, value in function_call.arguments.items():
        expected_type = schema.fields[field].type
        if not type_check(value, expected_type):
            return Reject(error=f"TYPE_ERROR: {field}")
    
    # Phase 3: 约束校验
    for constraint in schema.constraints:
        if not eval(constraint.predicate, function_call.arguments):
            return Reject(error=f"CONSTRAINT_VIOLATION: {constraint.name}")
    
    # Phase 4: 业务规则校验（可外部化）
    if schema.business_rule and not schema.business_rule(function_call):
        return Reject(error="BUSINESS_RULE_VIOLATION")
    
    return Accept()
```

### 3.3 校验复杂度
设参数个数为 $n$，嵌套深度为 $d$，约束条件数为 $m$：

$$T_{validate} = O(n \cdot d + m)$$

空间复杂度为递归栈深度 $O(d)$。

## 4. 工程考量

| Trade-off | 方案 A：严格前置校验 | 方案 B：宽松后置校验 |
|-----------|---------------------|---------------------|
| **优势** | 零污染穿透、安全性高 | 灵活性高、容错性强 |
| **牺牲** | 误杀率高（LLM 输出轻微偏差即被拒） | 需业务层兜底、风险后置 |
| **适用** | 金融交易、数据删除等高危操作 | 搜索、推荐等可降级场景 |

**致命弱点**：
- **Schema 漂移**：函数签名变更但 Schema 未同步 → 静默失败或误拒
- **语义鸿沟**：LLM 输出的 `"2024-01-01"` 是字符串，但业务需 `Date` 对象，类型系统无法捕获此转换失败
- **级联爆炸**：多轮 Function Call 场景下，A 函数的脏输出成为 B 函数的输入，错误传播链难以追踪

## 5. 工业映射

- **OpenAI Function Calling**：采用 JSON Schema 作为契约，运行时通过 `strict: true` 模式启用参数校验，失败时返回 `invalid_function_call` 错误码。
- **LangChain/LangGraph**：集成 Pydantic 进行声明式校验，支持 `ValidationError` 捕获与自动重试机制（`max_retries`）。
- **MCP (Model Context Protocol)**：工具注册时强制要求 JSON Schema，服务端执行前进行 **双层校验**（协议层 + 应用层），错误通过 `-32602: Invalid params` JSON-RPC 错误返回。
- **K8s Admission Webhook**：类比思想——在 Pod 创建（函数执行）前，由 Webhook（验证器）拦截并校验资源 Spec（参数），违反策略时拒绝准入。
