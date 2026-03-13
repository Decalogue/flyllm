# 技能可组合性与链式/条件分支技能的工程实现

## 1. 核心定性

本质上，**技能可组合性**是通过**统一接口契约（Contract）+ 上下文状态机（State Machine）**实现的能力编排系统，链式与条件分支分别对应**函数式管道（Pipeline）**与**策略路由（Router）**两种控制流模式。

---

## 2. 具体流程

1. **抽象层**：定义 `Skill` 接口（`execute(ctx) -> Result`），所有技能实现同一契约，保证可替换性与可测试性。
2. **编排层**：链式技能通过 `CompositeSkill` 顺序调用 `skills[i].execute(ctx)` 并将输出注入下游输入；条件分支通过 `RouterSkill` 根据上下文特征选择分支路径。
3. **执行层**：运行时维护共享 `Context` 对象承载中间状态，支持短路（Short-circuit）、重试（Retry）、回滚（Rollback）等控制语义。

---

## 3. 数学基础

### 链式技能（Sequential Composition）

```
Chain(ctx) = Sₙ ∘ Sₙ₋₁ ∘ ... ∘ S₁(ctx)
```

其中：
- **Sᵢ**: 第 i 个技能的执行函数 Sᵢ: Context → Context
- **∘**: 函数复合运算符，满足 Sⱼ ∘ Sᵢ (ctx) = Sⱼ(Sᵢ(ctx))

### 条件分支技能（Conditional Routing）

```
Route(ctx) = 
    Sₐ(ctx),  if f(ctx) ∈ Cₐ
    Sᵦ(ctx),  if f(ctx) ∈ Cᵦ
    ...
    S_default(ctx),  otherwise
```

其中：
- **f(ctx)**: 特征提取函数，f: Context → ℱ
- **Cᵢ**: 决策区域（Decision Region），∪ᵢ Cᵢ = ℱ

### 核心结构体（Python 伪代码）

```python
@dataclass
class Context:
    input: Any
    state: Dict[str, Any]
    metadata: ExecutionMeta

class Skill(ABC):
    @abstractmethod
    def execute(self, ctx: Context) -> Context: ...

class ChainSkill(Skill):
    def __init__(self, skills: List[Skill]):
        self.pipeline = skills
    
    def execute(self, ctx: Context) -> Context:
        for skill in self.pipeline:
            ctx = skill.execute(ctx)
            if ctx.metadata.should_stop:  # 短路机制
                break
        return ctx

class RouterSkill(Skill):
    def __init__(self, 
                 classifier: Callable[[Context], str],
                 routes: Dict[str, Skill]):
        self.classifier = classifier
        self.routes = routes
    
    def execute(self, ctx: Context) -> Context:
        route_key = self.classifier(ctx)
        target = self.routes.get(route_key, self.default)
        return target.execute(ctx)
```

---

## 4. 工程考量

| 维度 | Trade-off | 致命弱点 |
|------|-----------|----------|
| **耦合度** | 统一接口降低耦合，但过度抽象的 `Context` 会成为**上帝对象（God Object）** | 高并发下共享 `Context` 的读写竞争导致状态污染 |
| **可观测性** | 链式结构天然支持 AOP 拦截，但深度嵌套（>10层）时链路追踪成本指数上升 | 循环依赖（Circular Dependency）导致死循环或栈溢出 |
| **容错性** | 中间件模式支持熔断/降级，但级联故障（Cascading Failure）风险增加 | 分支条件判断逻辑复杂时，**组合爆炸**导致维护噩梦 |
| **延迟** | 异步执行（Async/Await）可并行化无依赖技能，但引入上下文切换开销 | 同步链路过长时，尾部延迟（Tail Latency）服从正态分布右偏 |

**关键取舍**：**声明式（DSL/YAML 编排）** vs **命令式（代码硬编码）**
- DSL 适合业务人员配置，但调试困难、版本管理复杂
- 代码编排类型安全、可测试，但灵活性差

---

## 5. 工业映射

在工业界，该机制被直接应用于：

| 项目 | 应用模块 | 具体实现 |
|------|----------|----------|
| **LangChain** | `RunnableSequence` / `RunnableBranch` | 通过 `pipe` 运算符实现链式组合，`RouterRunnable` 实现条件路由 |
| **OpenAI Assistants API** | `Run` 对象的 `tool_choice` 与 `required_action` | 服务端状态机驱动多轮工具调用，工具即技能 |
| **Kubernetes** | **Controller 的 Reconcile 循环** | 声明式 API + 控制器链（Admission Webhooks 链式校验） |
| **Netflix Conductor** | **Workflow DSL** | JSON 定义任务流，支持 `SWITCH` 条件分支与 `DO_WHILE` 循环 |
| **Uber Cadence/Temporal** | **Workflow Engine** | 事件驱动状态机，支持 Saga 模式实现分布式事务补偿 |

**高阶形态**：**DAG（有向无环图）编排**（如 Apache Airflow、Prefect）是链式+分支的超集，通过拓扑排序实现并行优化，但引入环路检测复杂度 O(V+E)。
