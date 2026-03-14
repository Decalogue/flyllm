# 008: ReAct 状态机与终止检测

## 核心定性
本质上，ReAct 是通过**有限状态机（FSM）**建模的**推理-行动-观察循环系统**，其状态由**当前上下文、任务进度、历史轨迹**共同定义，核心难点在于**动态决策边界判定**（何时继续推理、何时执行工具、何时终止任务），防止**无限循环与过早终止**。

## 状态机定义

### 状态空间 S

$$S = \{ \underbrace{G}_{\text{目标}}, \underbrace{H}_{\text{对话历史}}, \underbrace{T}_{\text{工具库}}, \underbrace{C}_{\text{当前上下文}}, \underbrace{n}_{\text{步数}} \}$$

**状态分类**:
- `$S_{initial}$`: 初始状态（接收用户任务）
- `$S_{thinking}$`: 推理状态（LLM 生成 Thought）
- `$S_{acting}$`: 行动状态（执行工具调用）
- `$S_{observing}$`: 观察状态（接收工具返回）
- `$S_{terminal}$`: 终止状态（输出最终答案）

### 状态转移函数 δ

$$\delta: S \times A \rightarrow S$$

```python
class ReActStateMachine:
    def __init__(self, max_steps=15):
        self.state = "INITIAL"
        self.max_steps = max_steps
        self.current_step = 0
        self.trajectory = []  # [(thought, action, observation), ...]

    def transition(self, action):
        if self.state == "INITIAL":
            # 初始 → 推理
            self.state = "THINKING"
            self.current_step = 0

        elif self.state == "THINKING":
            if "Action:" in action:
                # 推理 → 行动
                self.state = "ACTING"
            elif "Final Answer:" in action:
                # 推理 → 终止
                self.state = "TERMINAL"
            self.current_step += 1

        elif self.state == "ACTING":
            # 行动 → 观察（执行工具后）
            self.state = "OBSERVING"

        elif self.state == "OBSERVING":
            # 观察 → 推理（继续循环）
            self.state = "THINKING"

        # 步数限制
        if self.current_step >= self.max_steps:
            self.state = "TERMINAL"
            self.termination_reason = "MAX_STEPS_EXCEEDED"

        return self.state
