# Function Call 的链式调用与错误恢复

## 1. 核心定性

本质上，Function Call链式调用与错误恢复是通过6层防御性策略（格式错误→参数缺失→业务异常→速率限制→超时→不可恢复错误）逐层降级处理，实现高可用函数调用链的容错系统。

## 2. 具体流程

1. **格式层**: 检测JSON格式错误，使用LLM修复或返回格式错误码
2. **参数层**: 捕获参数缺失/类型错误，补充默认值或请求用户澄清
3. **业务层**: 处理业务逻辑异常，进行重试或调用备选方案
4. **重试层**: 指数退避重试，处理瞬态故障（网络抖动）
5. **降级层**: 调用简化版本或mock数据，保证基本可用
6. **终止层**: 所有恢复失败，返回错误信息并终止执行

## 3. 数学基础

**错误分级模型**:
```python
E = E₁ ∪ E₂ ∪ E₃ ∪ E₄ ∪ E₅ ∪ E₆

E₁ = {format_error}                    # 格式错误（层级1）
E₂ = {param_missing, type_mismatch}    # 参数错误（层级2）
E₃ = {biz_exception, not_found}        # 业务异常（层级3）
E₄ = {rate_limit, service_unavailable} # 瞬态错误（层级4）
E₅ = {timeout, circuit_open}          # 超时熔断（层级5）
E₆ = {auth_failed, unrecoverable}     # 不可恢复（层级6）
```

**恢复策略**:
```python
recovery_rate(e) =
    0.95 if e ∈ E₁  # LLM修复格式
    0.85 if e ∈ E₂  # 默认值/用户澄清
    0.70 if e ∈ E₃  # 重试/备选方案
    0.90 if e ∈ E₄  # 指数退避
    0.50 if e ∈ E₅  # 降级策略
    0.00 if e ∈ E₆  # 直接失败

# 综合恢复率
P_recover = 1 - Π_{layer=1..6} (1 - recovery_rate(E_layer))^{n_layer}

# 典型配置下
P_recover ≈ 0.98  # 98%的调用可恢复
```

**指数退避**:
```python
retry_delay(attempt) = base_delay × 2^{min(attempt, max_exponent)}

# 加入随机抖动
jitter = random.uniform(0, 0.3)  # 0-30%抖动
actual_delay = retry_delay × (1 + jitter)

# 参数设置
base_delay = 0.1  # 100ms
max_exponent = 6   # 最大延迟: 0.1×2⁶ = 6.4s
max_attempts = 5
```

**熔断器模型**:
```python
# 状态机: CLOSED → OPEN → HALF_OPEN → CLOSED

failure_rate = failures / total_calls
if failure_rate > threshold and total_calls > min_calls:
    state = OPEN
    open_until = now + timeout

elif state == OPEN and now > open_until:
    state = HALF_OPEN
    allow_trial_call()

elif state == HALF_OPEN and trial_call_succeeded:
    state = CLOSED
    reset_counts()
```

**超时分级**:
```python
timeout(call) = adjust_base_timeout(call) × multiplier

# 基于历史性能调整
adjust_base_timeout(call) =
    p99_duration(call) if enough_history
    else default_timeout

multiplier = {
    "read": 1.0,   # 读操作较快
    "write": 2.0,  # 写操作较慢
    "batch": 3.0,  # 批处理更慢
    "search": 5.0  # 搜索可能很慢
}

# 总请求超时
request_timeout = Σ timeout(callᵢ) + overhead
```

其中：
- `p99_duration`: 历史99分位延迟
- `enough_history`: 调用次数 > 30

**降级质量评估**:
```python
# 降级后结果质量
quality_degradation = 1 - (quality_degraded / quality_original)

# 降级策略选择
fallback_strategy =
    "simplified" if quality_degradation >= 0.8  # 质量损失<20%
    "cached" if quality_degradation >= 0.6      # 质量损失40%
    "mock" if quality_degradation >= 0.4        # 质量损失60%
    "fail" otherwise                            # 质量损失太大
```

## 4. 工程考量

**Trade-off**:
- 增加：延迟（重试和降级策略）
- 换取：可用性从~95%提升到99.9%
- 增加：系统复杂度（需要维护状态机）

**致命弱点**:
- **级联失败**:
  ```python
  # 一个调用失败导致整个链路失败
  # a() → b(a.result) → c(b.result)
  # 如果b失败，c无法执行

  # 解决方案：断路器快速失败
  if dependency.failed:
      fail_fast(dependency.error)
      return  # 不浪费计算资源
  ```

- **状态一致性**:
  ```python
  # 部分调用成功，部分失败，如何回滚？
  a() ✓
  b() ✗
  c()？  # 是否执行？

  # Saga模式补偿事务
  @compensate
def undo_a():
      reverse_a()

  if b():
      try:
          c()
      except:
          undo_a()  # 补偿
  ```

- **重试风暴**:
  ```python
  # 多个调用同时重试压垮下游服务
  # 概率: P(all_retry) ~ (0.1)⁵ = 0.001% (小但可能发生)

  # 解决方案：退避对齐 + 请求合并
  if same_target_call_within_10ms:
      wait_and_reuse_result()
  ```

- **过度降级**:
  ```python
  # 系统进入不可逆的降级状态
  # 用mock数据用久了，忘记恢复真实调用

  # 解决方案：定期健康检查 + 自动恢复
  if health_check():
      exit_degraded_mode()
  ```

**错误注入测试**:
```python
# Chaos Engineering
chaos_config = {
    "error_rate": 0.05,      # 5%调用注入错误
    "latency_p99": 5000,     # 99%延迟5秒
    "circuit_break": True,   # 随机触发熔断
}

# 验证系统在各种错误下的行为
async def test_fault_tolerance():
    with chaos_injection(**chaos_config):
        success_rate = await execute_chain(n=100)
        assert success_rate >= 0.98  # 98%成功率
```

**监控指标**:
```python
# 关键监控指标
metrics = {
    "error_rate_by_type": Counter(),      # 各层级错误率
    "recovery_rate": Gauge(),             # 恢复成功率
    "retry_count": Histogram(),           # 重试次数分布
    "degraded_mode_duration": Timer(),    # 降级持续时间
    "circuit_breaker_trips": Counter(),   # 熔断触发次数
}
```

## 5. 工业映射

在工业界，该机制被直接应用于AWS的Fault Injection Simulator测试微服务容错能力。Netflix的Hystrix熔断器模式被广泛用于Spring Cloud，在秒杀系统中防止级联故障。Stripe的API库使用指数退避重试，在网络抖动场景下99.99%成功率。Google的gRPC实现自动重试和 Hedging（同时发多个请求，用最快结果），将P99延迟从2s降低到200ms。在LLM应用中，LangChain的FallbackChain自动在模型失败时切换到备选方案（GPT-4 → Claude → 本地模型），保证服务可用性99.5%。最新的Temporal.io在workflow层面提供Saga模式实现分布式事务补偿，在金融转账场景保证最终一致性。
