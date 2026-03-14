# 多轮对话的打断与恢复机制

## 1. 核心定性

本质上，多轮对话打断与恢复是通过Dialogue Stack管理pending的工具调用和对话状态，当用户插入新意图时保存当前上下文，处理完成后使用resume逻辑恢复原任务的状态管理机制。

## 2. 具体流程

1. **状态保存**: 发生打断时，将当前对话状态（工具调用序列、已收集参数、中间结果）压入Dialogue Stack
2. **意图切换**: 处理新的用户查询，可能触发新的工具调用
3. **状态恢复**: 完成打断任务后，从Stack弹出原对话状态并恢复执行
4. **对话合并**: 将打断过程中获取的相关信息融合回原任务

## 3. 数学基础

**Dialogue Stack 结构**:
```python
Stack = [Frame₁, Frame₂, ..., Frameₙ]

Frame = {
    "dialogue_id": str,
    "intent": str,
    "state": "executing" | "waiting" | "completed",
    "tool_calls": [Call₁, Call₂, ...],
    "collected_params": dict,
    "intermediate_results": list,
    "resume_point": str
}
```

**栈操作**:
```python
# 打断时压栈
push(stack, current_frame) → stack'
stack_depth' = stack_depth + 1

# 恢复时弹栈
pop(stack) → (top_frame, stack')
stack_depth' = stack_depth - 1
```

**状态转移方程**:
```python
# 正常执行
state(t+1) = execute_action(state(t), tool_output)

# 打断发生
if detect_interrupt(user_input):
    state_saved = save_state(current_state)
    stack = push(stack, state_saved)
    state = reset_for_new_intent(user_input)

# 恢复执行
if stack and current_task_completed:
    prev_state, stack = pop(stack)
    state = merge_states(prev_state, new_knowledge)
```

**打断检测**:
```python
P(interrupt) = σ(w₁·intent_shift + w₂·context_overlap + w₃·explicit_signal)

intent_shift = JS_divergence(P_intent(current), P_intent(new))
context_overlap = cosine_similarity(embed(current), embed(new))
explicit_signal = 1 if "wait" in user_input else 0

if P(interrupt) > 0.7:
    trigger_interruption()
```

其中：
- `w₁, w₂, w₃`: 权重参数（通常0.4, 0.3, 0.3）
- `JS_divergence`: Jensen-Shannon散度
- `σ`: sigmoid函数

**Resume策略**:
```python
# 基于依赖图的恢复点选择
resume_point = min_{call in pending_calls} importance(call) × completeness(call)

importance(call) = 1 - depth(call) / max_depth  # 越靠后越重要
completeness(call) = |params_filled| / |params_required|

# 参数继承
if new_knowledge.related_to(prev_frame):
    inherited_params = intersect(new_params, old_params)
    prev_frame.params.update(inherited_params)
```

## 4. 工程考量

**Trade-off**:
- 增加：内存开销（维护Stack和多个Frame）
- 换取：用户体验（可随时切换任务）
- 牺牲：实现复杂度（需要管理嵌套对话）

**致命弱点**:
- **栈溢出**: 深度嵌套打断导致Stack过大
  ```python
  # 限制栈深度
  MAX_STACK_DEPTH = 5
  if len(stack) >= MAX_STACK_DEPTH:
      refuse_new_interrupt("请先完成当前任务")
  ```

- **状态丢失**: 长时间打断后context window截断
  ```python
  # 解决方案：关键信息摘要
  summary = LLM.summarize(intermediate_results)
  frame.intermediate_results = [summary]  # 压缩存储
  ```

- **恢复冲突**: 打断过程中修改了共享资源
  ```python
  # 示例：原任务预订会议室A，打断任务改为预订B
  resume时会议室状态已变
  ```
  解决方案：
  ```python
  # 保存资源快照
  frame.resource_snapshot = {
      "room": room.status,
      "booking": booking.info
  }
  # 恢复时检查一致性
  if frame.resource_snapshot != current_state:
      ask_user_confirmation()
  ```

- **用户体验**: 用户忘记被打断的原任务
  ```python
  # 提供任务回顾
  on_resume():
      show_reminder("您之前想查询天气，现在继续吗？")
      show_summary("已收集信息：北京，明天")
  ```

**实现模式**:
```python
class DialogueManager:
    def __init__(self):
        self.stack = []
        self.current_frame = None

    async def handle_input(self, user_input):
        # 检测打断
        if self.detect_interrupt(user_input):
            await self.handle_interrupt(user_input)
        else:
            await self.continue_dialogue(user_input)

    async def handle_interrupt(self, new_input):
        # 保存当前状态
        if self.current_frame:
            self.stack.append(self.current_frame)

        # 创建新frame
        self.current_frame = Frame(
            intent=extract_intent(new_input),
            state="executing"
        )

        # 处理新意图
        await self.process_intent()

    async def resume_previous(self):
        if not self.stack:
            return

        # 完成当前任务
        self.current_frame = None

        # 恢复栈顶
        self.current_frame = self.stack.pop()
        await self.process_intent(resume=True)
```

## 5. 工业映射

在工业界，该机制被直接应用于Anthropic的Claude的Artifacts功能，支持多任务并行编辑，每个任务保存在独立的对话栈中。Microsoft Copilot使用类似的Dialogue Stack管理Office操作序列，用户在创建文档时打断写邮件，完成后自动回到文档编辑。LangChain的ConversationalAgent使用BaseMemory栈实现嵌套对话，在代码生成任务中处理用户的临时查询。在客服场景中，LivePerson使用优先级队列管理多个客户意图，确保高优先级问题（如投诉）优先处理。最新的OpenAI Assistants API引入thread分叉机制，正式支持对话打断与恢复，在分析任务中用户可以随时了解数据详情，然后无缝回到主分析流程。Google的Bard在复杂任务中使用隐式对话栈，通过对话历史追踪恢复点，避免显式栈管理的复杂性。
