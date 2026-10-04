# Frozen-candidate selector protocol audit

CPU only; no generation or judge. This is a post-hoc protocol correction, not an untouched test evaluation.

## locomo: completed

| Protocol | K | All questions | With gold | Hit (%) | Recall (%) | FullCover (%) |
|---|---:|---:|---:|---:|---:|---:|
| Original | 20 | 1540 | 1536 | 90.6494 | 84.1491 | 77.5974 |
| Same_pool_CE | 20 | 1540 | 1536 | 89.2857 | 83.0425 | 76.2987 |
| Corrected | 20 | 1540 | 1536 | 90.7792 | 84.4022 | 77.9221 |

LoCoMo uses all Cat1-4 questions; LongMemEval retrieval uses questions with gold evidence (all-question diagnostics also saved). LME @20 truncates the Top50 output order with the same quota=5.

## longmemeval: grouped_cv_blocked

| Protocol | K | All questions | With gold | Hit (%) | Recall (%) | FullCover (%) |
|---|---:|---:|---:|---:|---:|---:|
| Original | 20 | 500 | 479 | 95.8246 | 88.3264 | 81.2109 |
| Same_pool_CE | 20 | 500 | 479 | 95.6159 | 90.3967 | 84.7599 |
| Original | 50 | 500 | 479 | 97.0772 | 92.6862 | 87.2651 |
| Same_pool_CE | 50 | 500 | 479 | 97.2860 | 93.6013 | 88.7265 |

LoCoMo uses all Cat1-4 questions; LongMemEval retrieval uses questions with gold evidence (all-question diagnostics also saved). LME @20 truncates the Top50 output order with the same quota=5.
