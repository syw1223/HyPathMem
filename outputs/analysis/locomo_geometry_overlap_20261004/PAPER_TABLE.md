# Minimal Evidence Overlap Experiment

LoCoMo Cat1-4, 1,540 questions. E = BU-E + TD-E; H = BU-H + TD-H. Both geometry unions are deduplicated and sorted by frozen CE relevance scores before selecting K.

| K | Fact Jaccard | Eu-only (%) | Lorentz-only (%) | Shared (%) | Raw-turn Jaccard | Extra-gold questions |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.5339 | 23.31 | 23.31 | 53.39 | 0.5665 | 80 |
| 50 | 0.4536 | 27.38 | 27.26 | 45.36 | 0.4810 | 89 |

Eu-only, Lorentz-only, and shared use the union as denominator and are macro-averaged across queries. Gold-coverage statistics use 1,536 questions with nonempty supporting-turn annotations; four questions remain in overlap analysis only. At K=50, seven H lists contain fewer than 50 Facts (mean H size 49.91).

## Evidence of Utility

At K=20, adding Lorentz-route candidates and reranking to the same 20-Fact budget raises source-turn Recall from 81.41% to 82.53% (+1.12 pp; 95% conversation-cluster bootstrap CI [+0.06, +2.16] pp). Lorentz contributes additional annotated turns for 80/1,536 questions. Among 95 Lorentz-only Fact-query occurrences adding novel gold support, 70.53% share an Event/Topic ancestor with Euclidean-selected Facts.

These results support complementary evidence coverage in the deployed routes, especially for multi-hop questions. They do not establish that Lorentz retrieval is stronger in isolation or that all novel candidates are useful. All retrieved targets are Facts, so describe them as hierarchy-associated Facts rather than higher-level evidence nodes. Weak lexical overlap is measured in REPORT.md; weak embedding-semantic similarity is not established by this experiment.

## Scope of the Comparison

The saved TD-H file has only five Facts per query. Therefore TD-E vs TD-H is additionally compared at K=5, and the combined-geometry result inherits the actual asymmetric TD budget. BU-E and BU-H differ in seed policy and hierarchy version as well as geometry. Use the title 'Evidence Complementarity of Euclidean and Lorentz Retrieval Routes'; do not present this as an isolated metric-only intervention.

## Suggested Paper Text

We analyze evidence overlap between the deployed Euclidean and Lorentz retrieval routes using frozen pre-composition candidates. Their mean Fact-level Jaccard overlap is 0.5339 at K=20 and 0.4536 at K=50. At K=20, Lorentz routes supply additional annotated supporting turns for 80 queries, and combining both route families with the existing relevance scorer at the same final budget improves retrieval recall by 1.12 percentage points. These results indicate complementary coverage rather than uniform superiority of either geometry. The comparison reflects the deployed route configurations, including the saved five-candidate TD-H output.
