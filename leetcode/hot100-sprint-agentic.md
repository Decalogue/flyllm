# LeetCode 冲刺：Agentic 学习模式

**Version**: 1.1  
**Date**: 2026-03-22  
**Status**: 与 [hot100-sprint-plan.md](./hot100-sprint-plan.md) v3.0 对齐  

---

## 模式概述

**Agentic Learning** 指代理在理解**长期目标**的前提下，主动规划、持续执行并随反馈调整，而不是被动响应单条指令。

在本仓库中，该模式服务于 **LeetCode Hot 100 冲刺**：

| 项目 | 当前约定 |
|------|-----------|
| 题单权威 | [hot100.md](./hot100.md) 附录（**127 题去重**） |
| 日程 | [hot100-sprint-plan.md](./hot100-sprint-plan.md)（**5 天**，Day 5 含复盘） |
| 状态 | [session-state.json](../workspace/leetcode/coach/session-state.json) |
| 教练流程 | `.claude/skills/leetcode-coach`（与文件化 Q&A 配合） |

---

## 核心设计

### 1. 从单次任务到目标编排

**Skill / 单次问答**：一问一答，调用结束即无全局上下文。  

**Agentic**：先对齐「5 天 / 127 题 / 当前 Day 与题号」，再逐题推进，并写回进度。

### 2. 从无状态到持久状态

**原则**：对话可中断，学习进度不丢。

| 机制 | 作用 |
|------|------|
| `session-state.json` | `current_problem`、`current_day`、`total_days`、`progress_tracking.day_*` |
| `workspace/leetcode/coach/*-q&a.md` | 单题费曼式记录 |
| `hot100.md` / 冲刺计划 | 题单与日程锚点 |

### 3. 从被动响应到主动推进

代理可主动：读题单、对照计划表定位下一题、发现状态与计划不一致时提醒用户、根据「时间不够」等反馈建议压缩或延后模块（仍以 127 题为总目标）。

---

## 实现要点

### 状态文件（示例结构）

实际字段以仓库内 JSON 为准；以下为说明性摘录：

```json
{
  "session": {
    "total_problems": 127,
    "total_days": 5,
    "current_problem": 15,
    "current_day": 1,
    "status": "in_progress"
  },
  "progress_tracking": {
    "day_1": { "total": 32, "completed": [], "reviewed": [], "skipped": [] },
    "day_2": { "total": 29, "completed": [], "reviewed": [], "skipped": [] },
    "day_3": { "total": 28, "completed": [], "reviewed": [], "skipped": [] },
    "day_4": { "total": 26, "completed": [], "reviewed": [], "skipped": [] },
    "day_5": { "total": 12, "completed": [], "reviewed": [], "skipped": [] }
  }
}
```

### 恢复会话（与 CLAUDE.md 一致）

```bash
cat workspace/leetcode/coach/session-state.json | python3 -m json.tool
ls workspace/leetcode/coach/*.md
day=$(grep '"active_day"' workspace/leetcode/coach/session-state.json | cut -d: -f2 | tr -d ', ' | xargs)
grep -A 50 "### Day $day:" leetcode/hot100-sprint-plan.md | head -40
```

用户说「继续上次冲刺」时：读取 `session-state.json` → 对齐 [hot100-sprint-plan.md](./hot100-sprint-plan.md) 中对应 `Day` 的表格 → 从 `current_problem` 或下一题继续。

### 自适应交互（示例）

| 用户反馈 | 合理响应 |
|----------|----------|
| 希望用文件答题 | 延续 `*-q&a.md` 命名与目录 |
| 时间不够 | 当日减量、标记 🔄，或把部分题挪到周末；**总目标仍为 127 题** |
| 某题过难 | 标记难点，插入复习点，不强行跳过附录题号而不记录 |

---

## 与 Skill-only 的对比

| 维度 | 单次 Skill / 单题 | Agentic 冲刺 |
|------|-------------------|----------------|
| 目标粒度 | 单题 | 全局（127 题 + 5 日节奏） |
| 状态 | 一般无持久 | JSON + Markdown |
| 恢复 | 难 | 可从上次的题号/Day 继续 |
| 适用 | 概念题、单次改代码 | 长周期刷题与复盘 |

---

## 适用场景

- **适合 Agentic**：多日复盘型刷题、依赖进度文件的教练流程、需与 `session-state.json` 同步的冲刺。
- **适合单次 Skill**：单题讲解、与当前日程无关的算法概念问答。

---

## 后续可迭代方向（备忘）

**短期**：启动时检测未完成冲刺并提示是否继续；`session-state` 更新时尽量局部改写，避免整文件冲突。  

**中期**：进度统计、薄弱标签、按附录分类的复习列表。  

**长期**：题间依赖简图、个人错题画像（非本文件承诺范围，仅作方向）。

---

## 实践建议

1. **目标表述**：尽量对齐附录，例如「按 hot100 附录完成 127 题，跟 5 日计划」。
2. **定期对齐**：问「当前第几天、下一题题号、与计划表是否一致」。
3. **中断前**：确认 `session-state.json` 与当天完成的题号已更新。

---

## 成功度量（自评用）

- 计划完成率：周期内独立 AC（或等价掌握）题数 / 127。  
- 恢复成功率：新会话能否在 1～2 轮内接上题号与 Day。  
- 状态一致性：`session-state.json` 与 `*-q&a.md`、自述进度是否一致。

---

## 版本历史

- **v1.0** (2026-03-19)：初稿；对应当时 4 日 / 126 题表述。  
- **v1.1** (2026-03-22)：对齐 **5 日**、**127 题**、[hot100-sprint-plan.md](./hot100-sprint-plan.md)；修正链接与 `session-state` 示例；压缩重复修辞。

---

## 相关文件

| 文件 | 说明 |
|------|------|
| [hot100.md](./hot100.md) | 分类、考点、**127 题附录** |
| [hot100-sprint-plan.md](./hot100-sprint-plan.md) | **5 天**日程与每日题表 |
| [session-state.json](../workspace/leetcode/coach/session-state.json) | 冲刺状态 |
| [CLAUDE.md](../CLAUDE.md) | 仓库说明与恢复命令 |
| `workspace/leetcode/coach/*-q&a.md` | 单题问答归档 |

---

**最后更新**: 2026-03-22
