# LoCoMo Evidence Overlap Analysis

Frozen pre-Card route outputs; 1540 Cat1-4 questions in 10 conversations. No retraining, LLM calls, or GPU inference.

## Fact and Source-Turn Overlap

| Comparison | K | Fact Jaccard | Raw-turn Jaccard | Eu-only / union | Lor-only / union | Shared / union | Questions with additional Lorentz gold |
|---|---:|---:|---:|---:|---:|---:|---:|
| BU | 20 | 0.2137 | 0.2474 | 39.31% | 39.31% | 21.37% | 253/1536 |
| BU | 50 | 0.1268 | 0.1618 | 43.74% | 43.59% | 12.68% | 290/1536 |
| TD | 5 | 0.7038 | 0.7053 | 14.81% | 14.81% | 70.38% | 106/1536 |
| Geometry_union | 20 | 0.5339 | 0.5665 | 23.31% | 23.31% | 53.39% | 80/1536 |
| Geometry_union | 50 | 0.4536 | 0.4810 | 27.38% | 27.26% | 45.36% | 89/1536 |

Geometry_union compares BU-E + TD-E against BU-H + TD-H; each group uses stored CE scores before taking K. BU preserves route-local order. TD-H is limited to five saved candidates, so only TD@5 is valid. Geometry_union is a deployed-route diagnostic and inherits this asymmetric TD budget.

Jaccard = |E intersection H| / |E union H|. All percentages are macro averages over questions. The three union shares sum to 100%. Fact identity can distinguish paraphrases from the same source turn, so raw-turn Jaccard is reported separately. Questions without gold supporting turns remain in overlap analysis and are excluded only from gold-coverage denominators.

## Added Evidence and Fixed Budget

| Comparison | Eu Recall | Lor Recall | Joint CE Recall (K fixed) | Delta vs Eu (pp) | 95% cluster CI (pp) | Added recall, union up to 2K (pp) |
|---|---:|---:|---:|---:|---:|---:|
| BU@20 | 74.24 | 73.07 | 82.05 | +7.81 | [+5.97, +9.52] | +10.81 |
| BU@50 | 77.17 | 77.63 | 87.19 | +10.01 | [+8.23, +11.85] | +12.68 |
| TD@5 | 60.33 | 60.58 | 64.49 | +4.17 | [+3.09, +5.17] | +5.17 |
| Geometry_union@20 | 81.41 | 74.74 | 82.53 | +1.12 | [+0.06, +2.16] | +2.74 |
| Geometry_union@50 | 86.65 | 79.27 | 88.29 | +1.64 | [+0.83, +2.62] | +3.02 |

Confidence intervals use 2,000 paired conversation-cluster bootstrap resamples (seed 13, 10 conversation clusters). Expanded-union coverage has a larger budget and must not be presented as a same-budget accuracy gain. These are retrieval/provenance metrics, not QA accuracy.

## Lorentz-only Characterization

| Comparison | Subset | Fact-query occurrences | Query lexical coverage | Zero lexical overlap | Has Event/Topic ancestor | Shares ancestor with Eu | Novel raw support |
|---|---|---:|---:|---:|---:|---:|---:|
| BU@20 | all_lor_only | 20807 | 26.12% | 1.18% | 97.48% | 64.57% | 92.62% |
| BU@20 | gold_novel_lor_only | 372 | 31.16% | 0.00% | 98.12% | 70.70% | 100.00% |
| BU@50 | all_lor_only | 60441 | 23.67% | 1.02% | 97.36% | 72.81% | 92.26% |
| BU@50 | gold_novel_lor_only | 471 | 30.00% | 0.21% | 98.30% | 75.80% | 100.00% |
| TD@5 | all_lor_only | 1578 | 30.41% | 1.39% | 100.00% | 50.70% | 99.24% |
| TD@5 | gold_novel_lor_only | 111 | 48.97% | 0.00% | 100.00% | 50.45% | 100.00% |
| Geometry_union@20 | all_lor_only | 9947 | 25.38% | 1.91% | 94.72% | 79.54% | 91.44% |
| Geometry_union@20 | gold_novel_lor_only | 95 | 31.62% | 1.05% | 92.63% | 70.53% | 100.00% |
| Geometry_union@50 | all_lor_only | 30000 | 22.64% | 1.70% | 94.62% | 89.06% | 92.23% |
| Geometry_union@50 | gold_novel_lor_only | 110 | 30.28% | 0.00% | 92.73% | 87.27% | 100.00% |

All target evidence nodes are Facts. This analysis cannot show that Lorentz-only results are higher-level nodes. Ancestor measures refer to same-conversation IS_SPECIFIC_OF paths to Event/Topic nodes in the frozen graph. Lexical query coverage uses unique lowercase alphanumeric non-stopwords; it is not a semantic similarity measure. Shared ancestors show structural association, not causal attribution to hyperbolic geometry.

## Auditable Examples

Examples below are selected for low lexical overlap and additional gold-turn coverage, not for QA outcome.

### conv-30:q0010

Question: Which city have both Jean and John visited?

Lorentz-only Fact (rank 2): Jon recently took a short trip to Rome to clear his mind.

Additional gold turns: D15:1. Lexical query coverage: 0.0%.

Saved path labels: Jon's Recent Travels: Jon recently visited Paris and Rome, exploring these cities during his travels. -> Jon's Short Trip to Rome: Jon recently traveled to Rome for a short trip to clear his mind. -> Jon recently took a short trip to Rome to clear his mind.

### conv-44:q0027

Question: What is the biggest stressor in Andrew's life besides not being able to hike frequently?

Lorentz-only Fact (rank 7): Andrew enjoys hiking in the great outdoors.

Additional gold turns: D16:1. Lexical query coverage: 9.1%.

Saved path labels: Andrew enjoys hiking in the great outdoors.

### conv-50:q0059

Question: Which events in Dave's life inspired him to take up auto engineering?

Lorentz-only Fact (rank 19): Dave has fond memories of restoring an old car with his dad during one summer, finding it hard work but satisfying.

Additional gold turns: D12:4. Lexical query coverage: 10.0%.

Saved path labels: Dave has fond memories of restoring an old car with his dad during one summer, finding it hard work but satisfying.

### conv-41:q0108

Question: What new activity did Maria start recently, as mentioned on 3 June, 2023?

Lorentz-only Fact (rank 6): Maria started volunteering at a local dog shelter once a month, showing her compassion towards animals and her willingness to help.

Additional gold turns: D17:12. Lexical query coverage: 11.1%.

Saved path labels: Maria has been doing charity work recently, which she finds rewarding and fulfilling. -> Maria Reflects and Builds Community: Maria has been engaging in charity work, joined a church for community and faith, and is reflecting on life to find balance. -> Maria's Journey of Community and Growth: Maria is actively engaging in charity work, personal reflection, and community-building activities while finding balance and peace through meaningful experiences and relationships. -> Maria Volunteers at Dog Shelter: Maria began volunteering monthly at a local dog shelter, reflecting her compassion for animals and willingness to help. -> Maria started volunteering at a local dog shelter once a month, showing her compassion towards animals and her willingness to help.

### conv-41:q0115

Question: How does John describe the support he received during his journey to becoming assistant manager?

Lorentz-only Fact (rank 17): John shares about facing various challenges including tech issues, workplace hurdles, and self-doubt on his path to promotion.

Additional gold turns: D19:12. Lexical query coverage: 11.1%.

Saved path labels: John shares about facing various challenges including tech issues, workplace hurdles, and self-doubt on his path to promotion.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' /home/anonymous/miniconda3/envs/python311/bin/python scripts/analyze_locomo_geometry_overlap.py
```

Input hashes and exact paths are in manifest.json; per_question.jsonl records every set difference and novel gold turn. Existing method code/results are not modified.
