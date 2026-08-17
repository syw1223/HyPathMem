#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sunyuwei/HyTopoMem"
PY="/home/sunyuwei/miniconda3/envs/python311/bin/python"
GRAPH="$ROOT/outputs/longmemeval_s/v3_10_fine/graph_sentence_semantic_episode_qwen_pathnodes_top50_v3_10.json"
CANDIDATES="$ROOT/outputs/longmemeval_s/v3_10_fine/cards/qwen3_card_guided_expand_top50summary120.json"
CARD_CACHE="$ROOT/outputs/longmemeval_s/v3_10_fine/cards/qwen3_cards_ctx50_top50summary.jsonl"
CE_MODEL="/home/sunyuwei/LightMem/LightMem/models/cross-encoder-ms-marco-MiniLM-L6-v2"
CARDCE_DIR="$ROOT/outputs/eval/longmemeval_v3_10_cardce_guided_ctx50_top50summary_top50_gpu6"
SELECTOR_DIR="$ROOT/outputs/eval/longmemeval_v3_10_expand120_24_feature_cardce_selector_top50summary_top50"
QA_OUT="$ROOT/outputs/qa/longmemeval_v3_10_cardquota_light_top50_c1_window_chunks_80k_gpt41mini_judge_gpt4omini.json"
QA_LOG="$ROOT/outputs/qa/longmemeval_v3_10_cardquota_light_top50_c1_window_chunks_80k_gpt41mini_judge_gpt4omini.log"

cd "$ROOT"
mkdir -p "$CARDCE_DIR" "$SELECTOR_DIR" "$ROOT/outputs/qa"

for required in "$GRAPH" "$CANDIDATES" "$CARD_CACHE"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required artifact: $required" >&2
    exit 1
  fi
done

# Qwen summaries/cards and GPU1 embedding routes are immutable upstream artifacts
# for this budget-only ablation. Verify that the requested GPU0 Qwen service is alive.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost,::1"
curl --fail --silent --show-error --max-time 10 \
  -H 'Authorization: Bearer EMPTY' \
  http://127.0.0.1:8000/v1/models >/dev/null

echo "[$(date -Is)] stage 1/3: CardCE rerank on physical GPU6"
CUDA_VISIBLE_DEVICES=6 "$PY" scripts/96_run_v3_9_cardce_guided_selection.py \
  --graph "$GRAPH" \
  --candidates "$CANDIDATES" \
  --card-cache "$CARD_CACHE" \
  --ce-model "$CE_MODEL" \
  --ce-device cuda:0 \
  --ce-batch-size 128 \
  --top-cards 3 \
  --topk 5 20 50 \
  --output-dir "$CARDCE_DIR"

echo "[$(date -Is)] stage 2/3: five-fold selector, final Top50"
"$PY" scripts/120_run_longmemeval_24_feature_card_selector.py \
  --graph "$GRAPH" \
  --candidates "$CANDIDATES" \
  --cardce-paths "$CARDCE_DIR/cardce_guided_topk_paths.json" \
  --candidate-topn 120 \
  --topk 50 \
  --output-dir "$SELECTOR_DIR"

echo "[$(date -Is)] stage 3/3: Top50 + 80,000-char QA"
"$PY" scripts/121_run_longmemeval_qa_eval.py \
  --graph "$GRAPH" \
  --paths "$SELECTOR_DIR/card_quota_light_top50_paths.json" \
  --processed data/longmemeval/processed/longmemeval_s_mvp.json \
  --output "$QA_OUT" \
  --log "$QA_LOG" \
  --k 50 \
  --model gpt-4.1-mini \
  --judge-model gpt-4o-mini \
  --max-context-chars 80000 \
  --compiler c1_window_chunks \
  --quant-mode private_ie \
  --max-answer-tokens 128 \
  --max-judge-tokens 160 \
  --save-every 1 \
  --resume

echo "[$(date -Is)] complete: $QA_OUT"
