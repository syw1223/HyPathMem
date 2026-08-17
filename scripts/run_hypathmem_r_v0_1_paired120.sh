#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/sunyuwei/miniconda3/envs/python311/bin/python"
RUNNER="$ROOT/scripts/131_run_hypathmem_r_v0_1_paired_qa.py"
OUT_DIR="$ROOT/outputs/qa/hypathmem_r_v0_1_paired120"

mkdir -p "$OUT_DIR"

"$PYTHON" "$RUNNER" \
  --variant d1 \
  --output "$OUT_DIR/d1_structured_gpt41mini_judge_gpt4omini.json" \
  --log "$OUT_DIR/d1_structured_gpt41mini_judge_gpt4omini.log" \
  --resume

"$PYTHON" "$RUNNER" \
  --variant d2 \
  --output "$OUT_DIR/d2_raw_grounded_gpt41mini_judge_gpt4omini.json" \
  --log "$OUT_DIR/d2_raw_grounded_gpt41mini_judge_gpt4omini.log" \
  --resume
