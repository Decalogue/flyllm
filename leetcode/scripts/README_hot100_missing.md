# Hot 100 缺失题解批量生成说明

## 现状

- 题单来源：`leetcode/hot100.md` 附录「去重题号一览」（**127** 题）。
- 本地已有 `{题号}.{题名}.md` 的题号与附录对比后，缺失部分已由  
  `scripts/generate_hot100_missing_docs.py` 生成**规范骨架**（含待补全标记）。
- **完善正文**的推荐方式：在本目录增加 `hot100_batch_*.py`（或 `hot100_content_overrides.py`），其中定义：

```python
OVERRIDES = {
    题号: r"""完整的 Markdown 正文（建议第一行为 '# 题号. 题名'）""",
}
```

然后执行：

```bash
# 仅覆盖指定题号（需已有骨架或配合 --force）
python3 leetcode/scripts/generate_hot100_missing_docs.py --only-ids "3,4,76" --force

# 重新生成全部缺失（一般不用；首次已跑过）
python3 leetcode/scripts/generate_hot100_missing_docs.py
```

## 文件约定

- 生成器会**合并** `hot100_content_overrides.py` 与所有 `hot100_batch_*.py` 中的 `OVERRIDES`（后加载的键可覆盖先加载的）。
- 当前仓库已提供：`hot100_batch_w1.py` … `hot100_batch_w5.py`，覆盖附录中此前缺失题号的主体内容（思路 + 代码 + 复杂度等）。
- 文件名由附录中的中文题名规范化得到，与力扣中文名一致即可。

## 编号说明

- 力扣 **5（最长回文子串）** 在仓库中可能与 `12.最长回文子串.md` 混用；若同时存在 `5.*.md`，可在 `OVERRIDES[5]` 中写短说明指向另一文件，避免重复维护。
