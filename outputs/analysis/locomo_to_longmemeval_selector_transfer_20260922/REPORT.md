# LoCoMo-trained frozen selector on LongMemEval-S

500 questions retained. Retrieval metrics below use the subset with annotated evidence; all-question diagnostics are in JSON.
No generation/judge or GPU calls. Original is the historical in-benchmark CV comparison, not a clean transfer result.

| Protocol | K | Questions with gold | Hit (%) | Recall (%) | FullCover (%) |
|---|---:|---:|---:|---:|---:|
| Original_inbenchmark_CV | 20 | 479 | 95.8246 | 88.3264 | 81.2109 |
| Same_pool_CE | 20 | 479 | 95.6159 | 90.3967 | 84.7599 |
| LoCoMo_frozen_transfer | 20 | 479 | 95.1983 | 90.0000 | 83.7161 |
| Original_inbenchmark_CV | 50 | 479 | 97.0772 | 92.6862 | 87.2651 |
| Same_pool_CE | 50 | 479 | 97.2860 | 93.6013 | 88.7265 |
| LoCoMo_frozen_transfer | 50 | 479 | 97.2860 | 93.4969 | 88.3090 |

This post-hoc experiment removes LongMemEval labels from selector fitting, not prior benchmark-informed development.
