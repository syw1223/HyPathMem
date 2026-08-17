#!/usr/bin/env bash
set -euo pipefail

cd /home/sunyuwei/HyTopoMem
mkdir -p outputs/qa/compiler_balanced80

run_one() {
  local name="$1"
  local mode="$2"
  local maxchars="$3"
  /home/sunyuwei/miniconda3/envs/python311/bin/python scripts/06_run_qa_eval.py \
    --graph outputs/graphs/locomo_graph_v3_6b_qwen_all.json \
    --paths outputs/paths/smoke_balanced_cat_qtype_80_v3_9_top20.json \
    --output "outputs/qa/compiler_balanced80/${name}.json" \
    --log "outputs/qa/compiler_balanced80/${name}.log" \
    --k 20 --categories 1,2,3,4 \
    --model gpt-4.1-mini --judge-model gpt-4o-mini \
    --context-mode "$mode" --answer-protocol v2_ops --verify-answer none \
    --max-answer-tokens 128 --max-context-chars "$maxchars" \
    --resume --save-every 1
}

run_one c0_hybrid_12k hybrid 12000
run_one c1_window_chunks_30k mnemis_c1_window_chunks 30000
run_one c4_window_full_30k mnemis_c4_window_full 30000
