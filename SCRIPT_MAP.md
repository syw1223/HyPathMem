# Script Map

The numbered scripts are retained deliberately. They encode the base graph and
retrieval line as well as the controlled ablations used to choose the final
route. Do not delete a numbered runner solely because it is not selected for a
headline result.

## Selected Final Archive Writers

| Script | Purpose |
| --- | --- |
| `146_archive_hypathmem_longmemeval_final500.py` | Writes the frozen LongMemEval-S final500 archive. |
| `154_archive_hypathmem_locomo_d0_time_v2_final.py` | Writes the frozen LoCoMo final1540 archive. |

## LongMemEval-S Selected Temporal Line

| Script | Purpose |
| --- | --- |
| `142_build_hypathmem_8_4time_v.py` | Builds the fail-closed temporal packet/constraint representation. |
| `143_run_hypathmem_8_4time_v_e1_e2.py` | Runs the E1/E2 paired diagnostics. |
| `144_run_hypathmem_8_5time_v_conversion.py` | Converts eligible non-executable packets to grounded executable constraints. |
| `145_eval_hypathmem_8_5time_v_candidates.py` | Judges safe 8.5time_v candidate takeovers. |

## LoCoMo Selected/Control Lines

| Script | Purpose |
| --- | --- |
| `147_build_hypathmem_locomo_d2_packs.py` | Builds D2 structured-claim plus raw-quote packs. |
| `148_run_hypathmem_locomo_d2_qa.py` | Runs the D2 paired negative control. |
| `149_prepare_hypathmem_locomo_time_v8_5.py` | Prepares LoCoMo Cat2 temporal inputs. |
| `150_compile_hypathmem_locomo_time_v8_5.py` | Original temporal compiler retained as a development artifact. |
| `151_eval_hypathmem_locomo_time_v8_5.py` | Original temporal evaluator retained as a development artifact. |
| `152_compile_hypathmem_locomo_time_v8_5_v2.py` | Final independent LoCoMo temporal compiler. |
| `153_eval_hypathmem_locomo_time_v8_5_v2.py` | Final independent LoCoMo temporal evaluator. |

## Base Graph and Retrieval Lines

- `run_locomo_v3_9_mainline.py` dispatches the frozen LoCoMo graph/path/QA
  mainline. Its numbered dependencies are retained in `scripts/`.
- `run_longmemeval_v3_9_mainline.py` dispatches the LongMemEval-S base graph
  construction. Fine-grained path and answer-context stages live in the
  numbered runners around `111` through `125`.
- `129` through `141` are controlled D1/D2 and temporal diagnostic runners.
  Their outputs support the paper's ablation claims and remain intentionally.

## Deliberately Removed Only

The release excludes Python bytecode caches and pytest caches. The parent
working tree also removed one zero-byte console log. No source runner, JSON
result, process log with content, final graph, path artifact, or ablation
artifact was deleted during release preparation.
