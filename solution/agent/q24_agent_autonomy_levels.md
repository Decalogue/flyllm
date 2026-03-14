# Agent 自主性分级控制 L0-L3

## 1. 核心定性

本质上，Agent自主性分级控制是通过L0-L3四级模型（完全人工→工具白名单→成本限制→完全自主）实现渐进式授权，平衡自动化效率与操作风险的精细化权限管理系统。

## 2. 具体流程

1. **L0级（人工确认）**: 所有工具调用需人工审核批准
2. **L1级（工具白名单）**: 只允许预定义安全的工具集合
3. **L2级（成本限制）**: 可调用任意工具，但受成本/时间/风险阈值约束
4. **L3级（完全自主）**: Agent自主决策，仅记录审计日志

## 3. 数学基础

**分级决策**:
```python
autonomy_level(task) =
    3  if risk_score(task) < 0.1 and cost_estimate(task) < C_limit
    2  if risk_score(task) < 0.3
    1  if tool in whitelist
    0  otherwise
```

**风险评估函数**:
```python
risk_score(task) =
    w₁·tool_risk + w₂·data_sensitivity + w₃·operation_irreversibility

tool_risk = {
    "read": 0.1,     # 查询类
    "write": 0.3,    # 写入类
    "delete": 0.9,   # 删除类
    "payment": 1.0   # 支付类
}

data_sensitivity = classify_data(task.parameters)  # 0-1
operation_irreversibility = 1 - reversibility_score(task)

w = [0.4, 0.4, 0.2]  # 权重
```

**成本约束（L2级）**:
```python
# 多维度成本模型
total_cost(task) =
    α·api_cost + β·time_cost + γ·compute_cost + δ·risk_cost

constraints = {
    "api_cost": {
        "max_per_call": $1.0,
        "max_per_session": $10.0,
        "budget_alert": 0.8  # 80%时警告
    },
    "time_cost": {
        "max_execution_time": 300,  # 秒
        "max_user_wait": 30
    },
    "compute_cost": {
        "max_gpu_hours": 0.1,
        "max_memory_gb": 16
    }
}

# 约束检查函数
def check_constraints(task, constraints, current_usage):
    projected = current_usage + estimate_cost(task)

    violations = []
    for dim, limit in constraints.items():
        if projected[dim] > limit["max"]:
            violations.append((dim, projected[dim], limit["max"]))

    return len(violations) == 0, violations
```

**阈值决策**:
```python
# 自适应阈值
threshold(task) =
    τ_base + Δ(context)

# 上下文调整
Δ_context =
    +0.1 if user_experience_level == "expert"
    -0.1 if user_experience_level == "beginner"
    +0.2 if past_success_rate > 0.95
    -0.2 if past_failure_rate > 0.2
    -0.3 if outside_working_hours

τ_base = 0.3  # 基础阈值
```

**L1白名单动态更新**:
```python
# 白名单更新策略
whitelist = {tool for tool in tools if tool.safety_score > 0.8}

# 根据使用统计更新
safety_score(tool) =
    λ·manual_approval_rate + (1-λ)·success_rate

# 时间衰减赋予最近行为更高权重
approval_rate(t) = Σ w(tᵢ)·approvalᵢ / Σ w(tᵢ)
w(t) = e^{-α·(now - t)}  # 指数衰减
```

## 4. 工程考量

**Trade-off**:
- 提升自主性：减少人工干预，提效50-80%
- 增加风险：需要精细的权限控制和审计机制
- 平衡：使用渐进式授权策略

**致命弱点**:
- **误判风险**: 风险评分可能不准确
  ```python
  # 低概率高风险事件（如删库）可能被低估
  solution: 关键操作必须L0人工确认
  irreversible_ops = ["delete_db", "transfer_funds"]
  if any(op in irreversible_ops for op in task.tools):
      force_level(0)
  ```

- **成本估算偏差**: API延迟失败、重试累积
  ```python
  # 解决方案：保守估计 + 熔断机制
  estimate_cost = expected_cost + 2·std_dev + buffer

  if failure_rate > 0.3:
      trigger_circuit_breaker()
  ```

- **白名单滞后**: 新工具评估周期长
  ```python
  # 解决方案：分阶段评估
  # 1. 沙箱测试（模拟调用）
  # 2. 影子模式（真实调用但结果不生效）
  # 3. L2级试用（有限制）
  # 4. 达到白名单标准
  ```

- **用户体验不一致**: 同类型任务有时需要确认有时不需要
  ```python
  # 解决方案：透明度 + 可解释性
  explain_decision():
      return f"这是读取操作(risk=0.1)，数据敏感度低，自动执行节省您{time}s"
  ```

**降级策略**:
```python
# 降级到L0
if agent_confidence < 0.6 or user_feedback == "unsatisfied":
    autonomy_level = max(0, autonomy_level - 1)

# 升级
if consecutive_tasks > 10 and success_rate > 0.95:
    autonomy_level = min(3, autonomy_level + 1)
```

## 5. 工业映射

在工业界，该机制被直接应用于微软Copilot的分级授权系统，普通查询L3自动执行，文件删除L0必须人工确认。Scale AI的测试自动化平台使用L2级Agent，每个call限制$0.1成本，超过需pm批准。IBM Watson在医疗诊断中使用L1+白名单，只允许调用FDA认证的医疗数据库。LlamaIndex的Agent Executor提供autonomy_level参数，开发者可根据场景设置。GitHub Copilot Workspace对代码修改采用L0级（人工确认），对文件读取L3级完全自主。在自动驾驶中，特斯拉的FSD根据车速和环境复杂度动态调整自主性（停车场L3，高速公路L1），确保安全边界。最新Microsoft Semantic Kernel增加了Risk Evaluation Service，在调用插件前实时计算风险分数，动态选择授权级别。
