# Agent 自主性控制机制

## 1. 核心定性

本质上，Agent 自主性控制是一个**分层权限仲裁系统**，通过 **"人类指令层 → 策略约束层 → 执行沙箱层"** 的三级 gate 机制，将 LLM 的生成行为限制在预设的安全与意图边界内。

---

## 2. 具体流程

1. **意图拦截**：用户输入先经过 **Intent Classifier**，判定任务风险等级（Safe / Sensitive / Critical），高风险任务必须触发人类确认 gate。
2. **策略路由**：根据风险等级路由到不同的 **Policy Module**（允许列表 / 拒绝列表 / 动态沙箱），决定 Agent 可调用的工具集合与参数范围。
3. **执行审计**：每次工具调用进入 **Execution Sandbox**，实时比对行为轨迹与预设的约束策略，异常操作触发熔断并回滚状态。

---

## 3. 数学基础

**权限判定函数**：

$$\text{Permission}(u, t, c) = \min\left(\text{Risk}(u), \text{Policy}(t), \text{Context}(c)\right) \in \{0, 1, 2\}$$

$$\text{Execute}(a) = \begin{cases} 
\text{Direct} & \text{if } \text{Permission} = 0 \\
\text{Confirm}(u) & \text{if } \text{Permission} = 1 \\
\text{Reject} & \text{if } \text{Permission} = 2
\end{cases}$$

其中：
- $u$: 用户指令的 embedding 向量
- $t$: 目标工具的风险权重矩阵
- $c$: 当前会话上下文状态向量
- $\text{Risk}(u) = \sigma(W \cdot \text{Embed}(u) + b)$，$\sigma$ 为 Sigmoid 输出风险概率

**约束传播公式**（基于 Bellman 方程的权限衰减）：

$$V(s_t) = R(s_t) + \gamma \cdot \max_{a \in A_{allowed}} V(s_{t+1})$$

---

## 4. 工程考量

| Trade-off | 牺牲 | 换取 |
|-----------|------|------|
| **权限粒度** | 系统响应延迟 ↑（每层 gate 增加 50-200ms） | 安全性与可控性 |
| **策略静态化** | 灵活性 ↓（无法适应突发场景） | 可审计性与确定性 |
| **沙箱隔离** | 资源开销 ↑（容器/VM 启动成本） | 故障爆炸半径收敛 |

**致命弱点**：
- **意图漂移（Prompt Injection）**：攻击者通过上下文污染绕过 Intent Classifier，将 Critical 任务伪装为 Safe 任务。传统关键词过滤对语义级攻击无效。
- **工具链逃逸**：Agent 通过组合多个 Low-Risk 工具实现 High-Risk 效果（如 "读取文件 → 编码 → 发送邮件" 实现信息泄露），线性权限模型无法捕获组合风险。
- **熔断悖论**：过于敏感的熔断策略导致 **可用性雪崩**（false positive 率 > 5% 时用户直接关闭控制机制）。

---

## 5. 工业映射

在工业界，该机制被直接应用于：

| 项目 | 模块 | 应用场景 |
|------|------|----------|
| **OpenAI GPTs / Assistants API** | `tool_choice` + `function calling` 约束 | 强制 Agent 遵循预定义的 JSON Schema 调用外部工具，拒绝越权操作 |
| **LangChain** | `AgentExecutor` 的 `max_iterations` + `handle_parsing_errors` | 限制推理深度，防止无限循环；异常输出触发重试或降级 |
| **AutoGPT** | `Config` 系统的 `ALLOWLISTED_COMMANDS` | 白名单机制控制 Shell 命令执行权限 |
| **Kubernetes** | RBAC + Admission Webhook | 类比：User → Role → Resource 的权限链，与 Agent 的 "意图 → 策略 → 执行" 三层 gate 完全同构 |

**设计范式**：
> "将 LLM 视为一个 **不可信但高能力的内核**（类似浏览器中的 JavaScript），必须通过 **Capability-based Security** 模型进行沙箱化。"
