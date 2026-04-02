# 单 token 改动与注意力路由：实验说明（`attention_token_flip_analysis`）

本文档与下列文件一一对应，建议同目录阅读：

| 文件 | 作用 |
|------|------|
| `attention_token_flip_analysis.py` | 主程序：双句对比、probe 注意力、next-token 统计、报告 |
| `run_attention_flip.sh` | 从任意工作目录调用；环境变量覆盖默认参数 |
| `outputs/attention_flip_result.json` | 结构化结果（可选精简体积） |
| `outputs/attention_flip_layer_head.csv` | 逐层逐头 margin / delta（便于表格分析） |
| `outputs/attention_flip_report.md` | 人类可读摘要 |
| `outputs/attention_*_heatmap_*.png` | `--plot` 时生成的层×候选热力图 |

---

## 1. 要解决什么问题

**最小对立句**（仅一字之差）：

- 句 A：`电脑放不进纸箱，因为它太大了`
- 句 B：`电脑放不进纸箱，因为它太小了`

希望在**同一因果语言模型**上定量回答：

1. 替换「大 / 小」是否引起**头级注意力权重**的系统性变化？
2. 变化能否传到**下一词分布**（first-token 概率、整词续写对数概率）？
3. 为什么在代词「它」所在位置往往**看不到** A/B 差异？

---

## 2. 核心概念

### 2.1 探针位置（probe）

| `probe-mode` | 含义 | 因果 LM 下典型现象 |
|--------------|------|-------------------|
| `pronoun` | 与 `--pronoun` 匹配的 token span 的**首 token** 位置 | 若该位置在「太大/太小」**之前**，则 A/B 左上下文相同 → 注意力与 margin 常**完全一致** |
| `last` | 序列中**最后一个有字符偏移的内容 token**（一般为句末如「了」） | 已看见「太大/太小」→ **允许**出现显著 `delta` |
| `both` | 上述两者都跑 | 推荐默认，便于对照 |

**注意**：`last` 不是必须在「太小」这一个 token 上；句末任意在「太小」**之后**的位置都已具备「可见后半句」的条件，用脚末做探针是为了固定、可复现。

### 2.2 因果掩码 vs 双向编码器

- **因果 LM（本脚本）**：位置 \(t\) 只能 attend \(j \le t\)。因此「它」若写在形容词前面，在该位点**物理上**读不到「大/小」，这不是 bug。
- **若**在双向模型（如 BERT 式）且掩码允许看全句，则「它」位点也可以直接受「大/小」调制——与因果 LM 结论**不可混用**。

### 2.3 头级「路由」在算什么

对层 \(l\)、头 \(h\)、探针位置 \(t\)、键位置 \(j\)：

\[
s_{t,j}^{(l,h)}=\frac{q_t^{(l,h)}\cdot k_j^{(l,h)}}{\sqrt{d_h}} + m_{t,j},\quad
\alpha_{t,j}^{(l,h)}=\mathrm{softmax}_j(s^{(l,h)}_{t,j})
\]

对候选词 \(c_1,c_2\)（如「电脑」「纸箱」），脚本在 token span 上对 \(\alpha\) **求和**得到该头的 `scores[c]`，并定义

\[
\mathrm{margin}_{l,h} = \mathrm{score}(c_1) - \mathrm{score}(c_2).
\]

A→B 的 **delta** 为 \(\Delta\mathrm{margin}_{l,h} = \mathrm{margin}^B - \mathrm{margin}^A\)。  
单字改动会通过 \(\Delta q\)、\(\Delta k\) 改变 \(s\)，再经 softmax **放大**成 \(\Delta\alpha\)，即常见的「头级路由重排」。

---

## 3. 程序行为概览

1. 加载模型，**强制 `attn_implementation=eager`**，否则许多实现下 `output_attentions=True` 不可用。
2. 用 **offset mapping** 把「它 / 候选词」映射到 token span（不要求单字独占一个 token）。
3. 对每个 `probe-mode` 跑一次前向，取 `outputs.attentions`，在探针行上对候选 span 聚合。
4. 统计：**整体 margin**、`top-k` 的 \(|\Delta\mathrm{margin}|\)、**正负头个数**与 `net_sum_delta`。
5. **next-token**：句尾 logits 上候选**首 token** 概率；**continuation logprob**：把「原句 + 候选词」整段编码后，对候选部分逐 token 条件对数概率求和。
6. 写出 JSON、CSV、可选 Markdown 报告与热力图。

---

## 4. 命令行参数（`attention_token_flip_analysis.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model-path` | （必填） | 本地模型目录 |
| `--sentence-a` / `--sentence-b` | 上面两句 | 最小对立对 |
| `--pronoun` | `它` | 用于 `pronoun` probe 的子串定位 |
| `--candidate-1` / `--candidate-2` | `电脑` / `纸箱` | 先行词 span 与 margin 定义 |
| `--output-dir` | `llm/code/outputs` | 相对 **当前工作目录** |
| `--device` | `auto` | `cpu` / `cuda` |
| `--dtype` | `auto` | `float32` / `float16` / `bfloat16` |
| `--probe-mode` | `both` | `pronoun` / `last` / `both` |
| `--top-k-heads` | `10` | 报告 \(|\Delta\mathrm{margin}|\) 最大的头 |
| `--csv-probe-mode` | `last` | 写入 CSV 时使用哪种 probe 的 A/B 对比 |
| `--plot` | 关 | 需要 `matplotlib` |
| `--minimal-json` | 关 | **不写** `result_A`/`result_B` 进 JSON，显著减小文件（细节见 CSV） |
| `--include-all-head-directions` | 关 | 在 JSON / 终端中附带**每个头**的 direction 列表（很大） |
| `--quiet` | 关 | 终端只打路径与精简统计 |

---

## 5. Shell：`run_attention_flip.sh`

- 使用脚本**所在目录**解析 `attention_token_flip_analysis.py` 与默认 `OUTPUT_DIR=${SCRIPT_DIR}/outputs`，可在任意目录执行。
- 常用环境变量：`MODEL_PATH`、`OUTPUT_DIR`、`PROBE_MODE`、`TOP_K_HEADS`、`CSV_PROBE_MODE`、`RUN_PLOT`（`0` 关闭画图）、`MINIMAL_JSON`、`QUIET`、`INCLUDE_ALL_HEAD_DIRS` 等。

示例：

```bash
# 仓库根目录
bash llm/code/run_attention_flip.sh

# 小 JSON + 安静模式
MINIMAL_JSON=1 QUIET=1 RUN_PLOT=0 bash llm/code/run_attention_flip.sh
```

---

## 6. 输出文件怎么读

- **`attention_flip_report.md`**：一页结论；适合贴笔记。
- **`attention_flip_result.json`**  
  - `meta`：本次运行配置；  
  - `analysis.<probe>.summary`：整体 margin 与 delta；  
  - `analysis.<probe>.top_k_head_deltas`：变化最大的头；  
  - `analysis.<probe>.direction_summary`：±0 头计数与 `net_sum_delta`；  
  - `next_token`：first-token 与续写 logprob 及 B−A。  
  若使用 `--minimal-json`，则**无** `result_A` / `result_B` 大块字段。
- **`attention_flip_layer_head.csv`**：由 `--csv-probe-mode` 选定的一种 probe，逐层逐头的 `a_*` / `b_*` / `delta_margin`，用于排序、筛选或画图。

---

## 7. 示例现象（数值以你本机一次运行为准，此处为说明性）

一次典型因果 LM 运行可能出现：

- **`pronoun`**：`delta_margin ≈ 0`，且几乎所有头的 \(\Delta\mathrm{margin}=0\)（左上下文相同）。
- **`last`**：大量头 \(\Delta\mathrm{margin}\neq 0\)，`top_k` 中出现层 10–18 附近的大 |\Delta|。
- **next-token**：句 B 相对句 A，某一候选续写 logprob 上升、另一下降，与「大/小」触发的常识解读方向一致。

具体数字请以上述 `outputs/` 下文件为准；不要在文档里硬编码结论替代复现。

---

## 8. 机制小结（面试可用表述）

1. **因果掩码**决定「它」位点是否能用到「大/小」；不能用时 A/B 行为一致是**结构必然**。  
2. **句末（或任意可见后半句的位置）**上，\(q,k\) 已携带不同语义，`softmax(\cdot)\)**对 logits 差敏感**，故头级路由可大幅重排。  
3. **注意力总和**只是可解释探针之一；**next-token / 续写 logprob** 更贴近「模型实际偏好是否翻转」。  
4. 正负 \(\Delta\mathrm{margin}\) 并存：不同头可能分工不同，最终由残差与 MLP **堆叠**成 logits 变化。

---

## 9. 后续可做实验（扩展）

- 对「太小」**所在 token** 单独设 probe（需在脚本中加 `phrase` 模式）。  
- 对 `top_k` 头做 **attention 屏蔽**或权重置换，看 next-token 是否回退。  
- 替换其它最小对立（长/短、轻/重），看是否复用相似层段的头。

---

## 10. 与本仓库路径的对应关系

默认文档中模型路径示例：`/root/data/AI/pretrain/Qwen2.5-7B-Instruct`。  
若你的模型在其它目录，请通过 `--model-path` 或 `MODEL_PATH` 覆盖。
