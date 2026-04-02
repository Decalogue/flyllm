#!/usr/bin/env python3
"""
Compare minimal-pair sentences: attention @ probe positions vs next-token odds.

Requirements:
- transformers, torch; matplotlib optional for --plot
- Causal LM must use eager attention so output_attentions works (script sets it).

Example (from repo root):
  python llm/code/attention_token_flip_analysis.py \\
    --model-path /path/to/Qwen2.5-7B-Instruct \\
    --output-dir llm/code/outputs --plot

Outputs:
  attention_flip_result.json, attention_flip_layer_head.csv,
  attention_flip_report.md, optional heatmap PNGs.
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SentenceConfig:
    text: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare attention differences on minimal pair sentences."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local model path, e.g. /root/data/AI/pretrain/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--sentence-a",
        type=str,
        default="电脑放不进纸箱，因为它太大了",
        help="Sentence version A (e.g. with '大').",
    )
    parser.add_argument(
        "--sentence-b",
        type=str,
        default="电脑放不进纸箱，因为它太小了",
        help="Sentence version B (e.g. with '小').",
    )
    parser.add_argument(
        "--pronoun",
        type=str,
        default="它",
        help="Pronoun token to probe.",
    )
    parser.add_argument(
        "--candidate-1",
        type=str,
        default="电脑",
        help="Candidate antecedent #1.",
    )
    parser.add_argument(
        "--candidate-2",
        type=str,
        default="纸箱",
        help="Candidate antecedent #2.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="llm/code/outputs",
        help="Directory for json/csv/(optional) figures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, save attention heatmaps (requires matplotlib).",
    )
    parser.add_argument(
        "--probe-mode",
        type=str,
        default="both",
        choices=["pronoun", "last", "both"],
        help="Probe attention at pronoun, sentence-last token, or both.",
    )
    parser.add_argument(
        "--top-k-heads",
        type=int,
        default=10,
        help="Report top-k heads by absolute delta margin.",
    )
    parser.add_argument(
        "--csv-probe-mode",
        type=str,
        default="last",
        choices=["pronoun", "last"],
        help="Which probe's layer-head table to write to CSV (default: last).",
    )
    parser.add_argument(
        "--include-all-head-directions",
        action="store_true",
        help="Include per-head direction list in JSON/terminal (large; CSV has full grid).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print output paths and one-line summary stats.",
    )
    parser.add_argument(
        "--minimal-json",
        action="store_true",
        help="Write JSON without result_A/result_B (saves space; use CSV for full grid).",
    )
    parser.add_argument(
        "--report-lang",
        type=str,
        default="bilingual",
        choices=["zh", "en", "bilingual"],
        help="Language for generated markdown report.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_dtype(dtype_arg: str):
    if dtype_arg == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float32


def safe_decode(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")
    except Exception:
        return f"<id:{token_id}>"


def locate_span_by_offsets(
    sentence_text: str,
    phrase: str,
    offsets: List[Tuple[int, int]],
) -> Tuple[int, int]:
    char_start = sentence_text.find(phrase)
    if char_start < 0:
        raise ValueError(f"Cannot find phrase `{phrase}` in sentence: {sentence_text}")
    char_end = char_start + len(phrase)

    token_indices = []
    for i, (s, e) in enumerate(offsets):
        # Skip special tokens, they usually have empty offsets.
        if e <= s:
            continue
        # Keep token if it overlaps [char_start, char_end).
        if not (e <= char_start or s >= char_end):
            token_indices.append(i)

    if not token_indices:
        raise ValueError(
            f"Cannot map phrase `{phrase}` to token span; char range=({char_start}, {char_end})"
        )

    return token_indices[0], token_indices[-1] + 1


def collect_attention(
    model,
    tokenizer,
    sentence: SentenceConfig,
    pronoun: str,
    candidates: List[str],
    device: str,
    probe_label: str,
    probe_pos: int,
    probe_span: Tuple[int, int],
) -> Dict:
    encoded = tokenizer(sentence.text, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True, use_cache=False)

    input_ids = encoded["input_ids"][0].tolist()
    tokens = [safe_decode(tokenizer, t) for t in input_ids]

    pronoun_start, pronoun_end = locate_span_by_offsets(
        sentence_text=sentence.text,
        phrase=pronoun,
        offsets=offsets,
    )

    candidate_spans: Dict[str, List[Tuple[int, int]]] = {}
    for c in candidates:
        span = locate_span_by_offsets(
            sentence_text=sentence.text,
            phrase=c,
            offsets=offsets,
        )
        candidate_spans[c] = [span]

    per_layer_head_scores = []
    # Each item: [batch, heads, q_len, k_len]
    for layer_idx, attn in enumerate(outputs.attentions):
        # Focus on one sample.
        attn_sample = attn[0]  # [heads, q_len, k_len]
        num_heads = attn_sample.shape[0]
        for head_idx in range(num_heads):
            row = attn_sample[head_idx, probe_pos, :]  # [k_len]
            cand_scores = {}
            for c, spans in candidate_spans.items():
                # If multiple spans, take the first one (can be extended).
                span_start, span_end = spans[0]
                score = row[span_start:span_end].sum().item()
                cand_scores[c] = float(score)
            per_layer_head_scores.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "scores": cand_scores,
                    "winner": max(cand_scores, key=cand_scores.get),
                }
            )

    layer_aggregates = {}
    num_layers = len(outputs.attentions)
    num_heads = outputs.attentions[0].shape[1]
    for layer_idx in range(num_layers):
        records = [r for r in per_layer_head_scores if r["layer"] == layer_idx]
        layer_aggregates[layer_idx] = {
            c: float(sum(r["scores"][c] for r in records) / num_heads) for c in candidates
        }

    overall = {
        c: float(sum(r["scores"][c] for r in per_layer_head_scores) / len(per_layer_head_scores))
        for c in candidates
    }

    return {
        "label": sentence.label,
        "text": sentence.text,
        "input_ids": input_ids,
        "tokens": tokens,
        "probe": {
            "label": probe_label,
            "span": [probe_span[0], probe_span[1]],
            "position_used": probe_pos,
            "token_ids": input_ids[probe_span[0] : probe_span[1]],
        },
        "pronoun": {
            "text": pronoun,
            "token_ids": input_ids[pronoun_start:pronoun_end],
            "span": [pronoun_start, pronoun_end],
        },
        "candidate_spans": {
            c: {
                "token_ids": tokenizer.encode(c, add_special_tokens=False),
                "span": list(candidate_spans[c][0]),
            }
            for c in candidates
        },
        "layer_head_scores": per_layer_head_scores,
        "layer_aggregates": layer_aggregates,
        "overall": overall,
    }


def compute_topk_head_deltas(result_a: Dict, result_b: Dict, c1: str, c2: str, top_k: int) -> List[Dict]:
    map_a = {(r["layer"], r["head"]): r for r in result_a["layer_head_scores"]}
    map_b = {(r["layer"], r["head"]): r for r in result_b["layer_head_scores"]}
    rows = []
    for key in sorted(set(map_a.keys()) & set(map_b.keys())):
        layer, head = key
        a1 = map_a[key]["scores"][c1]
        a2 = map_a[key]["scores"][c2]
        b1 = map_b[key]["scores"][c1]
        b2 = map_b[key]["scores"][c2]
        a_margin = a1 - a2
        b_margin = b1 - b2
        delta = b_margin - a_margin
        rows.append(
            {
                "layer": layer,
                "head": head,
                "a_margin_c1_minus_c2": a_margin,
                "b_margin_c1_minus_c2": b_margin,
                "delta_margin": delta,
                "abs_delta_margin": abs(delta),
            }
        )
    rows.sort(key=lambda x: x["abs_delta_margin"], reverse=True)
    return rows[: max(0, top_k)]


def summarize_head_directions(
    result_a: Dict,
    result_b: Dict,
    c1: str,
    c2: str,
    eps: float = 1e-8,
    include_per_head: bool = False,
) -> Dict:
    map_a = {(r["layer"], r["head"]): r for r in result_a["layer_head_scores"]}
    map_b = {(r["layer"], r["head"]): r for r in result_b["layer_head_scores"]}
    pos = 0
    neg = 0
    zero = 0
    sum_pos = 0.0
    sum_neg = 0.0
    details: List[Dict] = []
    for key in sorted(set(map_a.keys()) & set(map_b.keys())):
        layer, head = key
        a1 = map_a[key]["scores"][c1]
        a2 = map_a[key]["scores"][c2]
        b1 = map_b[key]["scores"][c1]
        b2 = map_b[key]["scores"][c2]
        delta = (b1 - b2) - (a1 - a2)
        if delta > eps:
            pos += 1
            sum_pos += delta
            sign = "positive"
        elif delta < -eps:
            neg += 1
            sum_neg += delta
            sign = "negative"
        else:
            zero += 1
            sign = "zero"
        if include_per_head:
            details.append(
                {
                    "layer": layer,
                    "head": head,
                    "delta_margin": delta,
                    "direction": sign,
                }
            )
    total = pos + neg + zero
    out: Dict = {
        "total_heads": total,
        "positive_count": pos,
        "negative_count": neg,
        "zero_count": zero,
        "positive_ratio": (pos / total) if total else 0.0,
        "negative_ratio": (neg / total) if total else 0.0,
        "zero_ratio": (zero / total) if total else 0.0,
        "positive_sum_delta": sum_pos,
        "negative_sum_delta": sum_neg,
        "net_sum_delta": sum_pos + sum_neg,
    }
    if include_per_head:
        out["all_head_directions"] = details
    return out


def analysis_for_json(
    analysis: Dict,
    include_per_head_directions: bool,
    minimal: bool,
) -> Dict:
    """Build JSON-serializable analysis; optionally drop heavy per-head tensors."""
    out: Dict = {}
    for mode, payload in analysis.items():
        ds = dict(payload["direction_summary"])
        if not include_per_head_directions:
            ds.pop("all_head_directions", None)
        block: Dict = {
            "summary": payload["summary"],
            "top_k_head_deltas": payload["top_k_head_deltas"],
            "direction_summary": ds,
        }
        if not minimal:
            block["result_A"] = payload["result_A"]
            block["result_B"] = payload["result_B"]
        out[mode] = block
    return out


def continuation_logprob(model, tokenizer, sentence: str, candidate: str, device: str) -> float:
    prompt_ids = tokenizer.encode(sentence, add_special_tokens=False)
    cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if not cand_ids:
        return float("-inf")

    all_ids = prompt_ids + cand_ids
    input_ids = torch.tensor([all_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[0]  # [seq_len, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    # Token all_ids[t] is predicted by position t-1.
    start = len(prompt_ids)
    total = 0.0
    for t in range(start, len(all_ids)):
        pred_pos = t - 1
        tok_id = all_ids[t]
        total += float(log_probs[pred_pos, tok_id].item())
    return total


def next_token_stats(model, tokenizer, sentence: str, candidates: List[str], device: str) -> Dict:
    encoded = tokenizer(sentence, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, use_cache=False)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    first_token_probs = {}
    continuation_logprobs = {}
    for c in candidates:
        c_ids = tokenizer.encode(c, add_special_tokens=False)
        if c_ids:
            first_token_probs[c] = float(probs[c_ids[0]].item())
        else:
            first_token_probs[c] = 0.0
        continuation_logprobs[c] = continuation_logprob(model, tokenizer, sentence, c, device)
    return {
        "first_token_probs": first_token_probs,
        "continuation_logprobs": continuation_logprobs,
    }


def build_text_report(
    analysis: Dict,
    next_token: Dict,
    c1: str,
    c2: str,
    report_lang: str = "bilingual",
) -> str:
    lines: List[str] = []
    if report_lang == "zh":
        lines.append("# 注意力翻转分析报告")
    elif report_lang == "en":
        lines.append("# Attention Flip Analysis Report")
    else:
        lines.append("# 注意力翻转分析报告 / Attention Flip Analysis Report")
    lines.append("")
    if report_lang == "zh":
        lines.append("## Probe 摘要")
    elif report_lang == "en":
        lines.append("## Probe Summary")
    else:
        lines.append("## Probe 摘要 / Probe Summary")
    for mode, payload in analysis.items():
        s = payload["summary"]
        d = payload["direction_summary"]
        lines.append(f"- Probe `{mode}`:")
        if report_lang == "zh":
            lines.append(
                f"  - 边际变化 delta_margin(c1-c2): {s['delta_margin']:.6f} "
                f"(A={s['A_margin_c1_minus_c2']:.6f}, B={s['B_margin_c1_minus_c2']:.6f})"
            )
            lines.append(
                f"  - 头方向统计: +{d['positive_count']} / -{d['negative_count']} / 0={d['zero_count']} "
                f"(net_sum_delta={d['net_sum_delta']:.6f})"
            )
        elif report_lang == "en":
            lines.append(
                f"  - delta_margin(c1-c2): {s['delta_margin']:.6f} "
                f"(A={s['A_margin_c1_minus_c2']:.6f}, B={s['B_margin_c1_minus_c2']:.6f})"
            )
            lines.append(
                f"  - head directions: +{d['positive_count']} / -{d['negative_count']} / 0={d['zero_count']} "
                f"(net_sum_delta={d['net_sum_delta']:.6f})"
            )
        else:
            lines.append(
                f"  - 边际变化 / delta_margin(c1-c2): {s['delta_margin']:.6f} "
                f"(A={s['A_margin_c1_minus_c2']:.6f}, B={s['B_margin_c1_minus_c2']:.6f})"
            )
            lines.append(
                f"  - 头方向统计 / head directions: +{d['positive_count']} / -{d['negative_count']} / 0={d['zero_count']} "
                f"(net_sum_delta={d['net_sum_delta']:.6f})"
            )
        top = payload.get("top_k_head_deltas", [])
        if top:
            h = top[0]
            if report_lang == "zh":
                lines.append(
                    f"  - 最强变化头: L{h['layer']}H{h['head']} "
                    f"(delta={h['delta_margin']:.6f}, abs={h['abs_delta_margin']:.6f})"
                )
            elif report_lang == "en":
                lines.append(
                    f"  - strongest head: L{h['layer']}H{h['head']} "
                    f"(delta={h['delta_margin']:.6f}, abs={h['abs_delta_margin']:.6f})"
                )
            else:
                lines.append(
                    f"  - 最强变化头 / strongest head: L{h['layer']}H{h['head']} "
                    f"(delta={h['delta_margin']:.6f}, abs={h['abs_delta_margin']:.6f})"
                )
    lines.append("")
    if report_lang == "zh":
        lines.append("## Next-token 证据")
    elif report_lang == "en":
        lines.append("## Next-token Evidence")
    else:
        lines.append("## Next-token 证据 / Next-token Evidence")
    pA = next_token["A"]["first_token_probs"]
    pB = next_token["B"]["first_token_probs"]
    lA = next_token["A"]["continuation_logprobs"]
    lB = next_token["B"]["continuation_logprobs"]
    dP = next_token["delta"]["first_token_prob_delta"]
    dL = next_token["delta"]["continuation_logprob_delta"]
    if report_lang == "zh":
        lines.append(
            f"- 首 token 概率 A: {c1}={pA[c1]:.8f}, {c2}={pA[c2]:.8f}; "
            f"B: {c1}={pB[c1]:.8f}, {c2}={pB[c2]:.8f}"
        )
        lines.append(
            f"- 首 token 概率变化 delta(B-A): {c1}={dP[c1]:.8f}, {c2}={dP[c2]:.8f}"
        )
        lines.append(
            f"- 续写对数概率 A: {c1}={lA[c1]:.6f}, {c2}={lA[c2]:.6f}; "
            f"B: {c1}={lB[c1]:.6f}, {c2}={lB[c2]:.6f}"
        )
        lines.append(
            f"- 续写对数概率变化 delta(B-A): {c1}={dL[c1]:.6f}, {c2}={dL[c2]:.6f}"
        )
    elif report_lang == "en":
        lines.append(
            f"- first_token_probs A: {c1}={pA[c1]:.8f}, {c2}={pA[c2]:.8f}; "
            f"B: {c1}={pB[c1]:.8f}, {c2}={pB[c2]:.8f}"
        )
        lines.append(
            f"- first_token_prob_delta(B-A): {c1}={dP[c1]:.8f}, {c2}={dP[c2]:.8f}"
        )
        lines.append(
            f"- continuation_logprobs A: {c1}={lA[c1]:.6f}, {c2}={lA[c2]:.6f}; "
            f"B: {c1}={lB[c1]:.6f}, {c2}={lB[c2]:.6f}"
        )
        lines.append(
            f"- continuation_logprob_delta(B-A): {c1}={dL[c1]:.6f}, {c2}={dL[c2]:.6f}"
        )
    else:
        lines.append(
            f"- 首 token 概率 / first_token_probs A: {c1}={pA[c1]:.8f}, {c2}={pA[c2]:.8f}; "
            f"B: {c1}={pB[c1]:.8f}, {c2}={pB[c2]:.8f}"
        )
        lines.append(
            f"- 首 token 概率变化 / first_token_prob_delta(B-A): {c1}={dP[c1]:.8f}, {c2}={dP[c2]:.8f}"
        )
        lines.append(
            f"- 续写对数概率 / continuation_logprobs A: {c1}={lA[c1]:.6f}, {c2}={lA[c2]:.6f}; "
            f"B: {c1}={lB[c1]:.6f}, {c2}={lB[c2]:.6f}"
        )
        lines.append(
            f"- 续写对数概率变化 / continuation_logprob_delta(B-A): {c1}={dL[c1]:.6f}, {c2}={dL[c2]:.6f}"
        )
    lines.append("")
    if report_lang == "zh":
        lines.append("## 解读")
        lines.append(
            "- 在因果掩码下，`pronoun` probe 通常接近 0；主要变化出现在能看到完整上下文的后续位置。"
        )
        lines.append(
            "- 建议将 `last` probe 与 next-token 的 delta 联合作为单 token 触发路由变化的主要证据。"
        )
    elif report_lang == "en":
        lines.append("## Interpretation")
        lines.append(
            "- `pronoun` probe should be near-zero under causal masking; "
            "changes mainly appear at later positions with full context."
        )
        lines.append(
            "- Use `last` probe + next-token deltas as primary evidence for token-triggered routing shifts."
        )
    else:
        lines.append("## 解读 / Interpretation")
        lines.append(
            "- 在因果掩码下，`pronoun` probe 通常接近 0；主要变化出现在能看到完整上下文的后续位置。 "
            "`pronoun` probe is usually near-zero under causal masking; changes mostly appear at later positions."
        )
        lines.append(
            "- 建议将 `last` probe 与 next-token delta 联合作为核心证据。 "
            "Use `last` probe plus next-token deltas as primary evidence."
        )
    return "\n".join(lines) + "\n"


def save_csv(result_a: Dict, result_b: Dict, c1: str, c2: str, out_csv: str) -> None:
    rows = []
    map_a = {(r["layer"], r["head"]): r for r in result_a["layer_head_scores"]}
    map_b = {(r["layer"], r["head"]): r for r in result_b["layer_head_scores"]}
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    for layer, head in keys:
        a1 = map_a[(layer, head)]["scores"][c1]
        a2 = map_a[(layer, head)]["scores"][c2]
        b1 = map_b[(layer, head)]["scores"][c1]
        b2 = map_b[(layer, head)]["scores"][c2]
        rows.append(
            {
                "layer": layer,
                "head": head,
                f"a_{c1}": a1,
                f"a_{c2}": a2,
                f"b_{c1}": b1,
                f"b_{c2}": b2,
                "a_margin_c1_minus_c2": a1 - a2,
                "b_margin_c1_minus_c2": b1 - b2,
                "delta_margin": (b1 - b2) - (a1 - a2),
            }
        )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def maybe_plot(result: Dict, candidates: List[str], out_png: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    num_layers = len(result["layer_aggregates"])
    num_candidates = len(candidates)
    mat = np.zeros((num_layers, num_candidates), dtype=float)
    for layer in range(num_layers):
        for j, c in enumerate(candidates):
            mat[layer, j] = result["layer_aggregates"][layer][c]

    plt.figure(figsize=(6, max(4, num_layers * 0.2)))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Attention weight")
    plt.yticks(range(num_layers), [f"L{l}" for l in range(num_layers)])
    plt.xticks(range(num_candidates), candidates)
    probe_lbl = result.get("probe", {}).get("label", "probe")
    plt.title(f"Layer-wise mean attention @ {probe_lbl} (sentence {result['label']})")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()
    return out_png


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        )
    model.to(device)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()

    sent_a = SentenceConfig(text=args.sentence_a, label="A")
    sent_b = SentenceConfig(text=args.sentence_b, label="B")
    candidates = [args.candidate_1, args.candidate_2]

    def get_probe(sentence_text: str, mode: str) -> Tuple[str, int, Tuple[int, int]]:
        encoded_local = tokenizer(sentence_text, return_tensors="pt", return_offsets_mapping=True)
        offsets_local = encoded_local["offset_mapping"][0].tolist()
        input_ids_local = encoded_local["input_ids"][0].tolist()
        if mode == "pronoun":
            s, e = locate_span_by_offsets(sentence_text, args.pronoun, offsets_local)
            return "pronoun", s, (s, e)
        # last content token
        idx = len(input_ids_local) - 1
        while idx >= 0 and offsets_local[idx][1] <= offsets_local[idx][0]:
            idx -= 1
        if idx < 0:
            raise ValueError("Cannot locate last content token.")
        return "last", idx, (idx, idx + 1)

    modes = ["pronoun", "last"] if args.probe_mode == "both" else [args.probe_mode]

    analysis = {}
    c1, c2 = candidates
    for mode in modes:
        probe_label_a, probe_pos_a, probe_span_a = get_probe(sent_a.text, mode)
        probe_label_b, probe_pos_b, probe_span_b = get_probe(sent_b.text, mode)
        result_a = collect_attention(
            model=model,
            tokenizer=tokenizer,
            sentence=sent_a,
            pronoun=args.pronoun,
            candidates=candidates,
            device=device,
            probe_label=probe_label_a,
            probe_pos=probe_pos_a,
            probe_span=probe_span_a,
        )
        result_b = collect_attention(
            model=model,
            tokenizer=tokenizer,
            sentence=sent_b,
            pronoun=args.pronoun,
            candidates=candidates,
            device=device,
            probe_label=probe_label_b,
            probe_pos=probe_pos_b,
            probe_span=probe_span_b,
        )

        summary = {
            "probe_mode": mode,
            "model_path": args.model_path,
            "device": device,
            "dtype": str(dtype),
            "candidates": candidates,
            "sentence_A_overall": result_a["overall"],
            "sentence_B_overall": result_b["overall"],
            "A_margin_c1_minus_c2": result_a["overall"][c1] - result_a["overall"][c2],
            "B_margin_c1_minus_c2": result_b["overall"][c1] - result_b["overall"][c2],
            "delta_margin": (
                (result_b["overall"][c1] - result_b["overall"][c2])
                - (result_a["overall"][c1] - result_a["overall"][c2])
            ),
        }
        top_heads = compute_topk_head_deltas(
            result_a=result_a,
            result_b=result_b,
            c1=c1,
            c2=c2,
            top_k=args.top_k_heads,
        )
        analysis[mode] = {
            "summary": summary,
            "top_k_head_deltas": top_heads,
            "direction_summary": summarize_head_directions(
                result_a=result_a,
                result_b=result_b,
                c1=c1,
                c2=c2,
                include_per_head=args.include_all_head_directions,
            ),
            "result_A": result_a,
            "result_B": result_b,
        }

    next_token_A = next_token_stats(model, tokenizer, sent_a.text, candidates, device)
    next_token_B = next_token_stats(model, tokenizer, sent_b.text, candidates, device)
    next_token = {
        "A": next_token_A,
        "B": next_token_B,
        "delta": {
            "first_token_prob_delta": {
                c: next_token_B["first_token_probs"][c] - next_token_A["first_token_probs"][c]
                for c in candidates
            },
            "continuation_logprob_delta": {
                c: next_token_B["continuation_logprobs"][c]
                - next_token_A["continuation_logprobs"][c]
                for c in candidates
            },
        },
    }

    meta = {
        "sentence_a": args.sentence_a,
        "sentence_b": args.sentence_b,
        "pronoun": args.pronoun,
        "candidates": candidates,
        "probe_modes": modes,
        "csv_probe_mode": args.csv_probe_mode,
        "minimal_json": args.minimal_json,
        "include_all_head_directions": args.include_all_head_directions,
    }
    full_json = {
        "meta": meta,
        "analysis": analysis_for_json(
            analysis,
            args.include_all_head_directions,
            args.minimal_json,
        ),
        "next_token": next_token,
    }

    json_path = os.path.join(args.output_dir, "attention_flip_result.json")
    csv_path = os.path.join(args.output_dir, "attention_flip_layer_head.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, ensure_ascii=False, indent=2)
    report_path = os.path.join(args.output_dir, "attention_flip_report.md")
    report_text = build_text_report(
        analysis=analysis,
        next_token=next_token,
        c1=c1,
        c2=c2,
        report_lang=args.report_lang,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    csv_mode = args.csv_probe_mode if args.csv_probe_mode in analysis else (
        "last" if "last" in analysis else modes[0]
    )
    save_csv(
        analysis[csv_mode]["result_A"],
        analysis[csv_mode]["result_B"],
        c1,
        c2,
        csv_path,
    )

    plot_paths = []
    if args.plot:
        for mode in modes:
            pa = maybe_plot(
                analysis[mode]["result_A"],
                candidates,
                os.path.join(args.output_dir, f"attention_A_heatmap_{mode}.png"),
            )
            pb = maybe_plot(
                analysis[mode]["result_B"],
                candidates,
                os.path.join(args.output_dir, f"attention_B_heatmap_{mode}.png"),
            )
            plot_paths.extend([p for p in [pa, pb] if p])

    print("=== Attention Flip Analysis Complete ===")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path} (probe={csv_mode})")
    print(f"MD  : {report_path}")
    if plot_paths:
        print("Plots:")
        for p in plot_paths:
            print(f"  - {p}")
    if args.quiet:
        one_line = {
            m: {
                "delta_margin": analysis[m]["summary"]["delta_margin"],
                "heads_+/-/0": (
                    f"{analysis[m]['direction_summary']['positive_count']}/"
                    f"{analysis[m]['direction_summary']['negative_count']}/"
                    f"{analysis[m]['direction_summary']['zero_count']}"
                ),
            }
            for m in modes
        }
        print("Summary (compact):", json.dumps(one_line, ensure_ascii=False))
        dnt = next_token["delta"]
        print(
            "Next-token delta(B-A) continuation_logprob:",
            json.dumps(dnt["continuation_logprob_delta"], ensure_ascii=False),
        )
        return

    print("Summary:")
    print(json.dumps({m: analysis[m]["summary"] for m in modes}, ensure_ascii=False, indent=2))
    print("Top-K head deltas:")
    print(
        json.dumps(
            {m: analysis[m]["top_k_head_deltas"] for m in modes},
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Next-token stats:")
    print(json.dumps(next_token, ensure_ascii=False, indent=2))
    dir_out = {}
    for m in modes:
        d = dict(analysis[m]["direction_summary"])
        if not args.include_all_head_directions:
            d.pop("all_head_directions", None)
        dir_out[m] = d
    print("Direction summary:")
    print(json.dumps(dir_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
