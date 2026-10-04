# Random-Control Ancestor Overlap Analysis

The analysis reuses the preceding Geometry_union@20 result: 95 Lorentz-only Fact-query pairs supplying novel gold support, across 80 questions and 10 conversations. The observed Event/Topic overlap is 67/95 = 70.53%.

Each random draw samples the same number of Facts per query. Sampling is within the same conversation and excludes Euclidean Top20 Fact IDs. Facts are sampled without replacement within each draw. Actual Lorentz-only targets remain eligible, keeping the null population inclusive.

## Main Table

| Random control | Expected overlap | Lorentz-only gold overlap | Difference (pp) | 95% paired cluster CI (pp) | Conditional randomization p |
|---|---:|---:|---:|---:|---:|
| Random_same_conversation | 34.15% | 70.53% | +36.38 | [+23.26, +46.00] | 0.00010 |
| Random_hierarchy_attached | 47.01% | 70.53% | +23.52 | [+10.66, +32.81] | 0.00010 |
| Random_source_ancestor_degree_matched | 43.56% | 70.53% | +26.96 | [+15.24, +35.68] | 0.00010 |

The first null samples any Fact; the second samples only hierarchy-attached Facts; the third matches Fact source and the exact numbers of Event-type and Topic ancestors. The matched control preserves the hierarchy-connectivity profile of the 95 observed pairs.

## Ancestor-Level Breakdown

| Control | Ancestor level | Random expectation | Observed |
|---|---|---:|---:|
| Random_same_conversation | Event_or_Topic | 34.15% | 70.53% |
| Random_same_conversation | Event_only | 25.28% | 55.79% |
| Random_same_conversation | Topic_only | 34.15% | 70.53% |
| Random_hierarchy_attached | Event_or_Topic | 47.01% | 70.53% |
| Random_hierarchy_attached | Event_only | 34.81% | 55.79% |
| Random_hierarchy_attached | Topic_only | 47.01% | 70.53% |
| Random_source_ancestor_degree_matched | Event_or_Topic | 43.56% | 70.53% |
| Random_source_ancestor_degree_matched | Event_only | 31.24% | 55.79% |
| Random_source_ancestor_degree_matched | Topic_only | 43.56% | 70.53% |

## Interpretation

Use the matched control when assessing enrichment. A higher observed rate shows that added gold-supported Lorentz evidence is concentrated in branches related to Euclidean-selected evidence beyond generic hierarchy attachment/source/degree effects. It does not establish causality: the Lorentz subset is selected for gold support, whereas random controls are not matched on gold relevance or semantic relevance. A hierarchy-removal intervention would be needed to claim that hierarchy itself caused the gain.

Both Event and Topic measures are included because sharing a broad Topic is easier than sharing a finer Event. The graph represents some episode-like nodes with the EVENT type; Event_only follows this stored schema. This protocol matches the previous 70.53% measure exactly.

## Statistical Details

Uniform fixed-size sampling success counts are drawn from their exact hypergeometric distributions for 10,000 repetitions, seed 13. Results include analytical null expectations, Monte Carlo means, and null 95% intervals. A separate paired bootstrap resamples the 10 conversation clusters and preserves per-query sample counts. P-values are one-sided conditional descriptive tests; all three controls and level breakdowns are exploratory.

## Files

summary.json: full protocol, exact expected rates and confidence intervals. table.csv: table values. per_question.jsonl: query-level observed and expected counts. sampling_audit.json: every eligible Fact ID and explicit example draws. random_null_counts.npz: all simulated success counts. manifest.json: input and code hashes.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' /home/anonymous/miniconda3/envs/python311/bin/python scripts/analyze_locomo_ancestor_random_control.py
```
