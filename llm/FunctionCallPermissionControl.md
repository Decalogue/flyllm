# Function Call 权限控制实现原理

---

## 1. 核心定性

**本质上，Function Call 权限控制是一个通过「声明式 Schema 白名单 + 运行时策略引擎」双层架构，解决 LLM 对工具访问越界问题的沙箱隔离机制。**

---

## 2. 具体流程

1. **Schema 注册阶段**：服务端将可调用的工具以 OpenAPI 规范格式注册到 Function Registry，每个工具绑定权限标签（`read-only` / `destructive` / `admin`）。

2. **运行时过滤阶段**：每次请求携带 `allowed_functions` 列表或权限级别，LLM 的 Function Calling 模块仅在白名单内生成工具调用。

3. **执行校验阶段**：实际执行前通过 Policy Engine 二次校验（用户身份 + 工具敏感度 + 调用参数），拒绝越权调用。

---

## 3. 数学基础

### 权限决策函数

$$
\text{Allow}(f, u, ctx) = 
\begin{cases}
1, & \text{if } f \in F_{allowed}(u) \land S(f) \leq L(u) \land C(f, params) = \text{true} \\
0, & \text{otherwise}
\end{cases}
$$

其中：
- $f$: 待调用的函数
- $u$: 当前用户/会话上下文
- $F_{allowed}(u)$: 用户 $u$ 被显式授权的工具集合
- $S(f) \in \{1, 2, 3\}$: 函数敏感度等级（1=只读，2=写入，3=危险操作）
- $L(u) \in \{1, 2, 3\}$: 用户权限等级
- $C(f, params)$: 参数级细粒度校验函数（如限制 `DELETE` 只能操作特定前缀路径）

### 核心数据结构

```python
@dataclass
class FunctionPerm:
    name: str
    sensitivity: int           # 1-3 敏感度
    allowed_roles: Set[str]    # 角色白名单
    param_constraints: Dict    # 参数约束规则
    
@dataclass  
class ExecutionContext:
    user_id: str
    session_id: str
    permission_level: int
    scope_boundary: List[str]  # 资源作用域边界
```

---

## 4. 工程考量

### Trade-off 矩阵

| 策略 | 优点 | 代价 |
|------|------|------|
| **Schema 白名单**（静态） | 零运行时开销、LLM 层直接拦截 | 灵活性差、无法动态调整 |
| **Policy Engine**（动态） | 细粒度控制、可审计 | 增加 RTT 延迟 ~5-20ms |
| **参数级校验** | 防注入、最小权限原则 | 配置复杂度高 |

### 致命弱点

1. **Prompt Injection 绕过**：攻击者通过构造恶意输入诱导 LLM 生成越权调用意图，**纯 Schema 层无法防御**。必须结合输入过滤 + 输出校验双层防护。

2. **权限扩散**：多轮对话中函数 A 的返回结果被用于构造函数 B 的参数，形成**间接调用链**，单一环节的权限校验可能失效。

3. **Tool Shadowing**：同名函数在不同上下文绑定不同实现，动态注册场景下可能发生**权限配置漂移**。

---

## 5. 工业映射

在工业界，该机制被直接应用于：

- **OpenAI GPTs / Assistants API**：通过 `tools` 字段显式声明可调用函数，结合 `tool_choice` 强制控制调用行为。
- **LangChain**：`bind_tools()` 方法实现运行时函数白名单注入，`Router Chain` 实现基于输入的智能权限路由。
- **Azure OpenAI**：通过 RBAC 将 Function Calling 权限与 Azure AD 身份体系打通，实现企业级细粒度控制。

---
