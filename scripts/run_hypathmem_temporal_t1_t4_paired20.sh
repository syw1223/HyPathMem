#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/qa/hypathmem_temporal_v0_1_paired20"
PYTHON_BIN="${PYTHON_BIN:-/home/sunyuwei/miniconda3/envs/python311/bin/python}"
mkdir -p "$OUT"

for variant in t1 t2 t3 t4; do
  "$PYTHON_BIN" "$ROOT/scripts/133_run_hypathmem_temporal_t1_t4_qa.py" \
    --variant "$variant" \
    --output "$OUT/${variant}_gpt41mini_judge_gpt4omini.json" \
    --log "$OUT/${variant}_gpt41mini_judge_gpt4omini.log" \
    --resume
done
