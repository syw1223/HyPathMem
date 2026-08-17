#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sunyuwei/HyTopoMem"
PY="/home/sunyuwei/miniconda3/envs/python311/bin/python"
CE="/home/sunyuwei/LightMem/LightMem/models/cross-encoder-ms-marco-MiniLM-L6-v2"

cd "$ROOT"

GRAPH_BASE="outputs/longmemeval_s/v3_10_fine/graph_sentence_semantic_episode_v3_cap6_t035.json"
PRE_PATHS="outputs/longmemeval_s/v3_10_fine/paths/pre_summary_euhyp_all_four_top150_paths.json"
GRAPH_SUMMARY="outputs/longmemeval_s/v3_10_fine/graph_sentence_semantic_episode_qwen_pathnodes_top50_v3_10.json"
CACHE_TOP50="outputs/llm_annotations/longmemeval_v3_10_qwen_pathnode_summary_top50.jsonl"

echo "[v3.10] waiting for pre-summary union paths: ${PRE_PATHS}"
while [[ ! -s "$PRE_PATHS" ]]; do
  date
  sleep 60
done

echo "[v3.10] Qwen path-node semantic summary"
if [[ ! -s "$CACHE_TOP50" && -s outputs/llm_annotations/longmemeval_v3_10_qwen_pathnode_summary_top150.jsonl ]]; then
  cp outputs/llm_annotations/longmemeval_v3_10_qwen_pathnode_summary_top150.jsonl "$CACHE_TOP50"
fi
VLLM_API_KEY=EMPTY VLLM_BASE_URL=http://127.0.0.1:8006/v1 \
"$PY" scripts/124_annotate_longmemeval_event_topic_qwen.py \
  --graph "$GRAPH_BASE" \
  --output "$GRAPH_SUMMARY" \
  --cache "$CACHE_TOP50" \
  --paths "$PRE_PATHS" \
  --scope path_nodes \
  --path-topn 50 \
  --resume \
  --save-every 25 \
  --timeout-seconds 120

echo "[v3.10] train summary-aware Lorentz router"
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/50_train_graph_v2_lorentz_router.py \
  --graph "$GRAPH_SUMMARY" \
  --hierarchy-version v3_3 \
  --embedding-cache outputs/longmemeval_s/v3_10_fine/embeddings/qwen_pathnodes_top50_lorentz_sentence_fact_event_episode_topic_minilm.npz \
  --output outputs/longmemeval_s/v3_10_fine/models/lorentz_router_sentence_v3_3_qwen_pathnodes_top50.pt \
  --epochs 8 \
  --steps-per-epoch 260 \
  --batch-size 768 \
  --branch-batch-size 768 \
  --num-negatives 16 \
  --negative-strategy random \
  --embedding-batch-size 256 \
  --device cuda:0

echo "[v3.10] final TD/BU Eu on summary graph"
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/115_run_longmemeval_topdown_eu_ee.py \
  --graph "$GRAPH_SUMMARY" \
  --hierarchy-version v3_3 \
  --topk 5,20,50,100,150 \
  --max-candidates 150 \
  --embedding-device cuda:0 \
  --embedding-batch-size 256 \
  --embedding-cache outputs/longmemeval_s/v3_10_fine/embeddings/final_qwen_top50_td_eu_event_topic_minilm.npz \
  --output-json outputs/eval/longmemeval_v3_10_pre_summary/final_qwen_top50_topdown_eu_ee.json \
  --output-md outputs/eval/longmemeval_v3_10_pre_summary/final_qwen_top50_topdown_eu_ee.md \
  --output-predictions outputs/longmemeval_s/v3_10_fine/predictions/final_qwen_top50_topdown_eu_ee_candidates.json

echo "[v3.10] final TD/BU Hyp on summary graph"
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/116_run_longmemeval_hyp_routes.py \
  --graph "$GRAPH_SUMMARY" \
  --hierarchy-version v3_3 \
  --checkpoint outputs/longmemeval_s/v3_10_fine/models/lorentz_router_sentence_v3_3_qwen_pathnodes_top50.pt \
  --topk 5,20,50,100,150 \
  --max-candidates 150 \
  --embedding-device cuda:0 \
  --embedding-batch-size 256 \
  --event-topic-cache outputs/longmemeval_s/v3_10_fine/embeddings/final_qwen_top50_hyp_event_topic_minilm.npz \
  --fact-node-cache outputs/longmemeval_s/v3_10_fine/embeddings/qwen_pathnodes_top50_lorentz_sentence_fact_event_episode_topic_minilm.npz \
  --router-device cuda:0 \
  --router-batch-size 1024 \
  --output-json outputs/eval/longmemeval_v3_10_pre_summary/final_qwen_top50_hyp_routes.json \
  --output-md outputs/eval/longmemeval_v3_10_pre_summary/final_qwen_top50_hyp_routes.md \
  --output-predictions outputs/longmemeval_s/v3_10_fine/predictions/final_qwen_top50_hyp_routes_candidates.json

echo "[v3.10] final EuHyp top150 union with CE"
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/118_build_longmemeval_dual_geometry_candidate_paths.py \
  --graph "$GRAPH_SUMMARY" \
  --eu-predictions outputs/longmemeval_s/v3_10_fine/predictions/final_qwen_top50_topdown_eu_ee_candidates.json \
  --hyp-predictions outputs/longmemeval_s/v3_10_fine/predictions/final_qwen_top50_hyp_routes_candidates.json \
  --candidate-topn 150 \
  --topk 5,20,50,100,150 \
  --ce-model "$CE" \
  --ce-device cuda:0 \
  --ce-batch-size 128 \
  --output outputs/longmemeval_s/v3_10_fine/paths/final_qwen_top50_euhyp_all_four_top150_ce_paths.json \
  --summary-json outputs/eval/longmemeval_v3_10_pre_summary/final_qwen_top50_euhyp_all_four_top150_ce_summary.json

echo "[v3.10] query-conditioned relation cards"
mkdir -p outputs/longmemeval_s/v3_10_fine/cards
VLLM_API_KEY=EMPTY VLLM_BASE_URL=http://127.0.0.1:8006/v1 \
"$PY" scripts/94_build_v3_9_query_conditioned_relation_cards.py \
  --graph "$GRAPH_SUMMARY" \
  --base-paths outputs/longmemeval_s/v3_10_fine/paths/final_qwen_top50_euhyp_all_four_top150_ce_paths.json \
  --base-topn 150 \
  --context-topn 50 \
  --workers 4 \
  --resume \
  --cache outputs/longmemeval_s/v3_10_fine/cards/qwen3_cards_ctx50_top50summary.jsonl \
  --output outputs/longmemeval_s/v3_10_fine/cards/qwen3_card_annotated_top150_paths_top50summary.json \
  --summary-json outputs/eval/longmemeval_v3_10_pre_summary/qwen3_cards_ctx50_top50summary_summary.json \
  --summary-md outputs/eval/longmemeval_v3_10_pre_summary/qwen3_cards_ctx50_top50summary_summary.md

echo "[v3.10] CardCE"
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/96_run_v3_9_cardce_guided_selection.py \
  --graph "$GRAPH_SUMMARY" \
  --candidates outputs/longmemeval_s/v3_10_fine/cards/qwen3_card_annotated_top150_paths_top50summary.json \
  --card-cache outputs/longmemeval_s/v3_10_fine/cards/qwen3_cards_ctx50_top50summary.jsonl \
  --ce-model "$CE" \
  --ce-device cuda:0 \
  --ce-batch-size 128 \
  --top-cards 3 \
  --topk 5 20 \
  --output-dir outputs/eval/longmemeval_v3_10_cardce_guided_ctx50_top50summary

echo "[v3.10] card-guided expand120"
"$PY" scripts/109_build_v3_9_card_guided_local_expansion.py \
  --input outputs/longmemeval_s/v3_10_fine/cards/qwen3_card_annotated_top150_paths_top50summary.json \
  --base-topn 100 \
  --extra 20 \
  --output-prefix outputs/longmemeval_s/v3_10_fine/cards/qwen3_card_guided_expand_top50summary

echo "[v3.10] 24-feature LightGBM selector"
"$PY" scripts/120_run_longmemeval_24_feature_card_selector.py \
  --graph "$GRAPH_SUMMARY" \
  --candidates outputs/longmemeval_s/v3_10_fine/cards/qwen3_card_guided_expand_top50summary120.json \
  --cardce-paths outputs/eval/longmemeval_v3_10_cardce_guided_ctx50_top50summary/cardce_guided_topk_paths.json \
  --candidate-topn 120 \
  --topk 5 20 \
  --output-dir outputs/eval/longmemeval_v3_10_expand120_24_feature_cardce_selector_top50summary

echo "[v3.10] done"
