#!/usr/bin/env bash
# 一键跑：最小对立句 attention 对比 + next-token 统计
# 在任意 cwd 下均可执行；默认输出到本目录下的 outputs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认模型路径可按机器修改
MODEL_PATH="${MODEL_PATH:-/root/data/AI/pretrain/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
SENTENCE_A="${SENTENCE_A:-电脑放不进纸箱，因为它太大了}"
SENTENCE_B="${SENTENCE_B:-电脑放不进纸箱，因为它太小了}"
PRONOUN="${PRONOUN:-它}"
CANDIDATE_1="${CANDIDATE_1:-电脑}"
CANDIDATE_2="${CANDIDATE_2:-纸箱}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
PROBE_MODE="${PROBE_MODE:-both}"
TOP_K_HEADS="${TOP_K_HEADS:-10}"
CSV_PROBE_MODE="${CSV_PROBE_MODE:-last}"
# 设为 0 跳过 --plot（无需 matplotlib）；默认关闭
RUN_PLOT="${RUN_PLOT:-0}"
# 传 "1" 或 "true" 开启；默认开启精简 JSON + 安静输出
MINIMAL_JSON="${MINIMAL_JSON:-1}"
QUIET="${QUIET:-1}"
INCLUDE_ALL_HEAD_DIRS="${INCLUDE_ALL_HEAD_DIRS:-0}"

EXTRA=()
if [[ "${MINIMAL_JSON}" == "1" || "${MINIMAL_JSON}" == "true" ]]; then
  EXTRA+=(--minimal-json)
fi
if [[ "${QUIET}" == "1" || "${QUIET}" == "true" ]]; then
  EXTRA+=(--quiet)
fi
if [[ "${INCLUDE_ALL_HEAD_DIRS}" == "1" || "${INCLUDE_ALL_HEAD_DIRS}" == "true" ]]; then
  EXTRA+=(--include-all-head-directions)
fi
if [[ "${RUN_PLOT}" == "1" || "${RUN_PLOT}" == "true" ]]; then
  EXTRA+=(--plot)
fi

exec python "${SCRIPT_DIR}/attention_token_flip_analysis.py" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --sentence-a "${SENTENCE_A}" \
  --sentence-b "${SENTENCE_B}" \
  --pronoun "${PRONOUN}" \
  --candidate-1 "${CANDIDATE_1}" \
  --candidate-2 "${CANDIDATE_2}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --probe-mode "${PROBE_MODE}" \
  --top-k-heads "${TOP_K_HEADS}" \
  --csv-probe-mode "${CSV_PROBE_MODE}" \
  "${EXTRA[@]}"
