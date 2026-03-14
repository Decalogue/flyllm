# ReAct 框架的状态机与终止检测

## 1. 核心定性

本质上，ReAct框架的状态机是管理LLM推理循环的有限状态机，通过Thought → Action → Observation三状态循环执行，配合最大步数和循环检测机制实现可控终止。

## 2. 具体流程

1. **Thought状态**: LLM生成推理过程，决定下一步Action
2. **Action状态**: 执行工具调用或知识检索，返回Observation
3. **Observation状态**: 将结果加入prompt，判断是否满足终止条件
4. **终止检测**: 达到max_steps或answer confidence > threshold时结束

## 3. 数学基础

**状态机定义**:
```python
S = {INITIAL, THOUGHT, ACTION, OBSERVATION, TERMINAL}
Σ = {generate_thought, execute_action, observe_result, check_terminate}
δ: S × Σ → S
```

**状态转移函数**:
```python
delta(INITIAL, generate_thought) = THOUGHT
delta(THOUGHT, execute_action) = ACTION
delta(ACTION, observe_result) = OBSERVATION
delta(OBSERVATION, check_terminate) =
    TERMINAL  if stop_condition met
    THOUGHT   otherwise
```

**终止条件**:
```python
stop = (step ≥ max_steps) ∨ (confidence > τ) ∨ (is_answer_complete)

confidence = P(answer|context) = exp(log_prob) / Z

is_answer_complete = LM predicts "[FINISH]" token
```

其中：
- `max_steps`: 最大步数（通常10-30步）
- `τ`: 置信度阈值（通常0.8-0.9）
- `step`: 当前循环次数

**循环检测**:
```python
# 使用SimHash检测重复状态
hash_current = SimHash(thought + action + observation)

if hash_current in hash_history:
    # 检测到循环，强制终止
    return TERMINAL

hash_history.add(hash_current)

# SimHash相似度
similarity = (64 - bit_count(hash1 ^ hash2)) / 64
is_same_state = similarity > 0.95
```

**状态表示**:
```python
state_representation = {
    "history": [
        ("Q", question),
        ("T1", thought1),
        ("A1", action1),
        ("O1", observation1),
        ...
    ],
    "step": current_step,
    "confidence": answer_confidence,
    "pending_actions": queue_length
}
```

## 4. 工程考量

**Trade-off**:
- 牺牲：交互延迟（需等待工具调用）
- 换取：可解释性（Thought追踪）和准确率（工具增强）
- 增加：状态管理复杂性

**致命弱点**:
- **无限循环**: LLM陷入死循环（如反复搜索同一关键词）
- **状态爆炸**: 历史记录过长超出context window
  ```python
  # 解决方案：窗口滑动或重要度过滤
  keep_recent = 5  # 只保留最近5轮
  keep_important = similarity_with_question > 0.7
  ```
- **过早终止**: 复杂任务需要更多步，threshold设置过低
- **过晚终止**: 简单任务浪费时间，max_steps设置过高

**调优策略**:
```python
# 自适应max_steps
max_steps =
    base_steps (10) +
    complexity_score(question) × 5

# 动态confidence threshold
τ = 0.8 if step < 5 else 0.85 if step < 10 else 0.9

# 循环惩罚
if same_action_count > 3:
    confidence *= 0.5  # 强制降低置信度
```

## 5. 工业映射

在工业界，该机制被直接应用于LangChain的AgentExecutor，使用`max_iterations`和`early_stopping_method`控制终止。AutoGPT的BabyAGI实现包含循环检测机制，当任务列表不再更新时主动停止。Hugging Face的Transformers Agents使用`stop` token和启发式规则双重保险。在医学诊断场景中，IBM Watson的ReAct实现设置max_steps=15，防止过度检查。最新的LangGraph引入状态持久化，支持跨会话的ReAct执行，在金融问答系统中使用状态哈希缓存已计算路径，提升30%响应速度。Google的Bard使用变体的ReAct，在内部搜索工具失败时自动降级到直接回答模式，确保用户永远获得响应。
