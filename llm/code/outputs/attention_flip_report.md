# 注意力翻转分析报告 / Attention Flip Analysis Report

## Probe 摘要 / Probe Summary
- Probe `pronoun`:
  - 边际变化 / delta_margin(c1-c2): 0.000000 (A=0.453814, B=0.453814)
  - 头方向统计 / head directions: +0 / -0 / 0=784 (net_sum_delta=0.000000)
  - 最强变化头 / strongest head: L0H0 (delta=0.000000, abs=0.000000)
- Probe `last`:
  - 边际变化 / delta_margin(c1-c2): -0.002706 (A=0.458895, B=0.456188)
  - 头方向统计 / head directions: +335 / -443 / 0=6 (net_sum_delta=-2.121752)
  - 最强变化头 / strongest head: L12H23 (delta=0.489380, abs=0.489380)

## Next-token 证据 / Next-token Evidence
- 首 token 概率 / first_token_probs A: 电脑=0.00009441, 纸箱=0.00002098; B: 电脑=0.00005889, 纸箱=0.00004578
- 首 token 概率变化 / first_token_prob_delta(B-A): 电脑=-0.00003552, 纸箱=0.00002480
- 续写对数概率 / continuation_logprobs A: 电脑=-9.250000, 纸箱=-10.756165; B: 电脑=-9.750000, 纸箱=-10.004059
- 续写对数概率变化 / continuation_logprob_delta(B-A): 电脑=-0.500000, 纸箱=0.752106

## 解读 / Interpretation
- 在因果掩码下，`pronoun` probe 通常接近 0；主要变化出现在能看到完整上下文的后续位置。 `pronoun` probe is usually near-zero under causal masking; changes mostly appear at later positions.
- 建议将 `last` probe 与 next-token delta 联合作为核心证据。 Use `last` probe plus next-token deltas as primary evidence.
