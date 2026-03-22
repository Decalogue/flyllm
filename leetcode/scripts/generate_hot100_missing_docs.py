#!/usr/bin/env python3
"""
从 hot100.md 附录解析 127 道去重题，与 leetcode/*.md 已有题号对比，
为「尚未存在」的题目生成 Markdown 题解文件。

用法（仓库根目录或 leetcode 目录均可）：
  python3 leetcode/scripts/generate_hot100_missing_docs.py
  python3 leetcode/scripts/generate_hot100_missing_docs.py --dry-run
  python3 leetcode/scripts/generate_hot100_missing_docs.py --force  # 覆盖已存在（慎用）

内容来源：同目录下 hot100_content_overrides.py 中的 OVERRIDES 字典；
未在 OVERRIDES 中的题号会生成带「待补全」提示的规范骨架，便于后续分批填写。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def repo_root() -> Path:
    p = Path(__file__).resolve()
    # leetcode/scripts/xxx.py -> parents[2] == repo root if flyllm layout
    if (p.parents[2] / "leetcode" / "hot100.md").exists():
        return p.parents[2]
    if (p.parents[1] / "hot100.md").exists():
        return p.parents[1]
    return Path.cwd()


def parse_appendix(hot100_path: Path) -> dict[int, tuple[str, str]]:
    """返回 {题号: (难度英文, 中文标题)}，仅解析「附录：去重题号一览」之后的表格。"""
    text = hot100_path.read_text(encoding="utf-8")
    start = text.find("## 附录：去重题号一览")
    if start == -1:
        raise ValueError("未找到附录标题")
    chunk = text[start:]
    rows = re.findall(r"^\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*([^|]+)\s*\|", chunk, re.MULTILINE)
    out: dict[int, tuple[str, str]] = {}
    for num_s, diff, title in rows:
        pid = int(num_s)
        title = title.strip()
        out[pid] = (diff, title)
    return out


def existing_problem_ids(leetcode_dir: Path) -> set[int]:
    ids: set[int] = set()
    for p in leetcode_dir.glob("*.md"):
        m = re.match(r"^(\d+)\.", p.name)
        if m:
            ids.add(int(m.group(1)))
    return ids


def diff_to_stars(diff_en: str) -> str:
    d = diff_en.lower()
    if d == "easy":
        return "⭐ 简单"
    if d == "medium":
        return "⭐⭐ 中等"
    if d == "hard":
        return "⭐⭐⭐ 困难"
    return diff_en


def safe_filename_title(title: str) -> str:
    # Windows / 跨平台：替换路径非法字符
    for c in '<>:"/\\|?*':
        title = title.replace(c, "／")
    return title.strip()


def skeleton_md(pid: int, title: str, diff_en: str) -> str:
    stars = diff_to_stars(diff_en)
    return f"""# {pid}. {title}

**标签**：`Hot 100` `待补充分类` `{diff_en}`  
**难度**：{stars}  
**频率**：🔥 见 hot100.md 分类

---

## 题目描述

本题见 [LeetCode 热题 HOT 100](https://leetcode.cn/problem-list/2cktkvj/) 附录题号 **{pid}**。请先在力扣阅读完整题面、样例与数据范围。

---

## 📋 面试要点速查

**一句话总结**：（待补全：本题核心算法与数据结构）

**关键词**：`待补充`

- **时间复杂度**：（待补全）
- **空间复杂度**：（待补全）

---

## 解题思路

### 核心思想

（待补全）

### 算法步骤

1. （待补全）

---

## 代码实现

```python
class Solution:
    def placeholder(self, *args, **kwargs):
        \"\"\"待补全：替换为力扣要求的方法签名与实现。\"\"\"
        raise NotImplementedError
```

---

## 复杂度分析

- **时间复杂度**：（待补全）
- **空间复杂度**：（待补全）

---

## 相关题目

- 见 `leetcode/hot100.md` 同章节题目。

---

> 本文件由 `scripts/generate_hot100_missing_docs.py` 自动生成骨架；完善请替换「待补全」并删除本说明行。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印将创建的文件，不写盘")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 md（慎用）")
    ap.add_argument(
        "--only-ids",
        type=str,
        default="",
        help="只处理给定题号，逗号分隔，例如 3,4,76（用于分批覆盖）",
    )
    args = ap.parse_args()

    root = repo_root()
    leetcode_dir = root / "leetcode"
    hot100 = leetcode_dir / "hot100.md"
    if not hot100.exists():
        print("找不到 hot100.md:", hot100, file=sys.stderr)
        return 1

    appendix = parse_appendix(hot100)
    have = existing_problem_ids(leetcode_dir)
    missing = sorted(set(appendix.keys()) - have)
    only: set[int] | None = None
    if args.only_ids.strip():
        only = {int(x.strip()) for x in args.only_ids.split(",") if x.strip()}

    if only is not None:
        targets = sorted(p for p in only if p in appendix)
    else:
        targets = missing

    # 可选：手写覆盖正文（完整或部分）：hot100_content_overrides.py 或 hot100_batch_*.py 中 OVERRIDES
    overrides: dict[int, str] = {}
    script_dir = Path(__file__).resolve().parent
    override_path = script_dir / "hot100_content_overrides.py"
    if override_path.exists():
        ns: dict = {}
        exec(override_path.read_text(encoding="utf-8"), ns, ns)
        overrides.update(ns.get("OVERRIDES", {}) or {})
    for bp in sorted(script_dir.glob("hot100_batch_*.py")):
        ns = {}
        exec(bp.read_text(encoding="utf-8"), ns, ns)
        overrides.update(ns.get("OVERRIDES", {}) or {})

    created = 0
    for pid in targets:
        diff_en, title = appendix[pid]
        name = safe_filename_title(title)
        path = leetcode_dir / f"{pid}.{name}.md"
        if path.exists() and not args.force:
            continue
        body = overrides.get(pid) or skeleton_md(pid, title, diff_en)
        if args.dry_run:
            print("would write:", path.relative_to(root))
        else:
            path.write_text(body, encoding="utf-8")
            created += 1
            print("write:", path.relative_to(root))

    print(
        f"附录题数: {len(appendix)} 已有题解编号: {len(have)} 缺失: {len(missing)} "
        f"本次处理题号数: {len(targets)} 本次写入: {created}"
    )
    if args.dry_run:
        print("(dry-run, 未写文件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
