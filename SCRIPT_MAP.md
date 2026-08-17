# Script Map

The public repository keeps only the paper mainline and the scripts required
to inspect or regenerate selected artifacts. Earlier probes, parameter
searches, failed branches, and superseded implementations are intentionally
excluded from the public tree.

## Entry Points

| Script | Purpose |
| --- | --- |
| `run_locomo_v3_9_mainline.py` | Dispatches Relation Card construction, local expansion, selection, and LoCoMo QA. |
| `run_longmemeval_v3_9_mainline.py` | Prepares LongMemEval-S and builds its isolated base/semantic hierarchy. |
| `146_archive_hypathmem_longmemeval_final500.py` | Writes the final LongMemEval-S archive. |
| `154_archive_hypathmem_locomo_d0_time_v2_final.py` | Writes the final LoCoMo archive. |

## Data and Base Graph

| Script | Purpose |
| --- | --- |
| `00_prepare_locomo.py` | Converts LoCoMo to the internal schema. |
| `00_prepare_longmemeval.py` | Converts LongMemEval-S while preserving history isolation and provenance. |
| `01_extract_nodes.py` | Extracts typed memory nodes. |
| `02_build_relations.py` | Builds typed graph relations. |
| `54_build_semantic_hierarchy_v3.py` | Builds the semantic hierarchy used by the mainline. |

## HyperPath Retrieval and QA

| Script | Purpose |
| --- | --- |
| `94_build_v3_9_query_conditioned_relation_cards.py` | Produces query-conditioned Relation Cards. |
| `109_build_v3_9_card_guided_local_expansion.py` | Expands evidence around Card roles. |
| `30_run_loco_cv_selector.py` | Provides the grouped cross-validation utilities reused by the final selector. |
| `108_run_v3_9_24_feature_card_quota_cv.py` | Runs the route/relation-aware HyperPath selector. |
| `125_rerank_v3_9_candidates_with_qwen3_reranker.py` | Applies the configured Qwen3 reranker. |
| `06_run_qa_eval.py` | Generates and judges LoCoMo answers. |
| `121_run_longmemeval_qa_eval.py` | Generates and judges LongMemEval-S answers. |
| `11_eval_by_category.py` | Aggregates category-level metrics. |

## Evidence Reconstruction

| Script | Purpose |
| --- | --- |
| `128_audit_hypathmem_r_v0_1.py` | Audits provenance and support closure. |
| `129_build_hypathmem_r_v0_1_packs.py` | Builds structured and raw-grounded evidence packs. |
| `130_build_hypathmem_r_v0_1_node_cache.py` | Caches normalized evidence nodes. |
| `131_run_hypathmem_r_v0_1_paired_qa.py` | Runs paired reconstruction evaluation. |

## Temporal Grounding

| Script | Purpose |
| --- | --- |
| `134_build_hypathmem_temporal_oracle_v0_2.py` | Builds oracle operands for temporal error-bound diagnostics. |
| `135_compile_hypathmem_temporal_oracle_o1_o3.py` | Compiles oracle constraints and exposes the tested deterministic solver. |
| `136_run_hypathmem_temporal_oracle_qa.py` | Measures the temporal oracle upper bound. |
| `137_extract_hypathmem_temporal_qwen_joint_v0_2.py` | Extracts typed temporal operands and anchors. |
| `138_compile_hypathmem_temporal_q1_q4_v0_2.py` | Compiles executable temporal constraints. |
| `139_verify_hypathmem_temporal_q4_v0_2.py` | Applies fail-closed temporal verification. |
| `140_run_hypathmem_temporal_q1_q4_qa_v0_2.py` | Evaluates verified temporal answers. |
| `141_report_hypathmem_temporal_q4_metrics_v0_2.py` | Reports takeover, fix/break, coverage, and latency metrics. |
| `142_build_hypathmem_8_4time_v.py` | Builds the 8.4 temporal packet representation. |
| `143_run_hypathmem_8_4time_v_e1_e2.py` | Runs the selected E1/E2 diagnostics. |
| `144_run_hypathmem_8_5time_v_conversion.py` | Converts grounded eligible packets. |
| `145_eval_hypathmem_8_5time_v_candidates.py` | Evaluates safe LongMemEval-S takeovers. |
| `149_prepare_hypathmem_locomo_time_v8_5.py` | Prepares LoCoMo Cat2 temporal inputs. |
| `152_compile_hypathmem_locomo_time_v8_5_v2.py` | Compiles the selected LoCoMo temporal branch. |
| `153_eval_hypathmem_locomo_time_v8_5_v2.py` | Evaluates the selected LoCoMo temporal branch. |

All reusable graph, geometry, retrieval, reconstruction, and temporal logic is
implemented under `src/hytopomem/`. The scripts are orchestration and
experiment entry points rather than duplicate implementations.
