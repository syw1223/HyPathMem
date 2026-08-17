# HyPathMem Result Analysis

> This document summarizes the **completed experimental results currently available** for HyPathMem.
> Unless otherwise stated, all QA results are reported as accuracy (%).
> Planned, incomplete, or provisional experiments are intentionally excluded.

## 1. Main Findings

Across both evaluation benchmarks and both generators, HyPathMem achieves the highest **overall** accuracy among the currently evaluated methods.

| Benchmark | Generator | HyPathMem | Best Baseline | Gain |
|---|---|---:|---:|---:|
| LoCoMo | GPT-4.1-mini | **91.62** | Mnemis 87.04 | **+4.58** |
| LoCoMo | Qwen3-30B | **87.84** | EverMemOS / Hindsight 83.90 | **+3.94** |
| LongMemEval-S | GPT-4.1-mini | **82.41** | EverMemOS 79.87 | **+2.54** |
| LongMemEval-S | Qwen3-30B | **79.69** | Mnemis 74.78 | **+4.91** |

The overall advantage is therefore not tied to a single answer generator. Replacing GPT-4.1-mini with Qwen3-30B lowers absolute accuracy, but HyPathMem remains the strongest overall method on both benchmarks.

---

## 2. LoCoMo Main Results

### 2.1 GPT-4.1-mini Generator

| Method | Overall | Cat1 | Cat2 | Cat3 | Cat4 |
|---|---:|---:|---:|---:|---:|
| FullText | 85.30 | 86.74 | 85.66 | 65.00 | 87.00 |
| NaiveRAG | 78.88 | 76.25 | 68.90 | 62.58 | 85.44 |
| BM25 Fact + CE | 77.99 | 64.18 | 79.75 | 61.46 | 83.83 |
| Graph Retrieval | 80.42 | 78.56 | 54.62 | 67.85 | 92.34 |
| LightMem | 72.14 | 72.05 | 54.56 | 56.74 | 80.64 |
| MECo | 78.40 | 67.67 | 79.44 | 58.33 | 83.12 |
| HyperMem | 84.22 | 82.98 | 77.57 | 61.46 | 89.77 |
| EverMemOS | 86.82 | 83.33 | 81.93 | 43.75 | **94.77** |
| Hindsight | 86.26 | 81.62 | 82.22 | 75.00 | 90.64 |
| Mnemis | 87.04 | 84.40 | 79.75 | 63.04 | 93.34 |
| **HyPathMem** | **91.62** | **90.07** | **90.34** | **77.08** | 94.29 |

HyPathMem improves overall accuracy by **4.58 percentage points** over the strongest baseline, Mnemis (87.04 → 91.62).

The category-level results show that the gain is broad rather than being produced by only one question type:

- **Cat1:** 90.07, **+3.33 pp** over the strongest baseline result (FullText, 86.74).
- **Cat2:** 90.34, **+4.68 pp** over FullText (85.66).
- **Cat3:** 77.08, **+2.08 pp** over Hindsight (75.00).
- **Cat4:** 94.29, slightly below EverMemOS (94.77) by **0.48 pp**.

Thus, the main LoCoMo improvement comes from stronger performance on Cat1–Cat3 while retaining near-best Cat4 performance.

### 2.2 Qwen3-30B Generator

| Method | Overall | Cat1 | Cat2 | Cat3 | Cat4 |
|---|---:|---:|---:|---:|---:|
| FullText | 81.30 | 74.47 | 75.08 | 66.67 | 87.63 |
| NaiveRAG | 75.78 | 72.70 | 65.42 | 59.38 | 82.64 |
| BM25 Fact + CE | 75.71 | 64.18 | 69.47 | 64.58 | 83.23 |
| Graph Retrieval | 76.62 | 75.18 | 49.84 | 64.58 | 88.70 |
| LightMem | 70.50 | 70.45 | 49.87 | 53.84 | 80.29 |
| MECo | 77.40 | 76.24 | 65.42 | 58.33 | 84.54 |
| HyperMem | 81.69 | 80.14 | 72.59 | 60.42 | 88.11 |
| EverMemOS | 83.90 | 76.95 | 77.88 | 52.08 | 92.15 |
| Hindsight | 83.90 | 78.37 | 81.00 | **71.88** | 88.23 |
| Mnemis | 83.64 | 82.51 | 73.78 | 60.23 | 90.45 |
| **HyPathMem** | **87.84** | **85.27** | **83.26** | 70.83 | **92.39** |

With Qwen3-30B as the generator, HyPathMem remains the best overall method at **87.84%**, exceeding the strongest baseline result (83.90%) by **3.94 pp**.

The category pattern is slightly different from GPT-4.1-mini:

- Cat1 improves by **2.76 pp** over Mnemis.
- Cat2 improves by **2.26 pp** over Hindsight.
- Cat4 is marginally best at 92.39.
- Cat3 is **1.05 pp below** Hindsight.

This suggests that the overall advantage persists across generators even though the strongest category-specific competitor changes.

---

## 3. LongMemEval-S Main Results

### 3.1 GPT-4.1-mini Generator

| Method | Overall | User | Assistant | Update | Temporal | Multi-session | Preference |
|---|---:|---:|---:|---:|---:|---:|---:|
| FullText | 61.06 | 90.20 | 95.75 | 75.49 | 45.56 | 41.78 | 45.00 |
| NaiveRAG | 74.05 | 92.54 | 95.27 | 79.89 | 55.36 | 70.45 | 75.67 |
| BM25 Fact + CE | 67.31 | 87.36 | 93.64 | 78.46 | 46.28 | 58.74 | 73.50 |
| Graph Retrieval | 62.56 | 95.26 | 67.58 | 70.25 | 37.68 | 60.44 | 76.67 |
| LightMem | 66.86 | 85.14 | 35.64 | 83.52 | 61.85 | 66.87 | 65.84 |
| MECo | 71.40 | **97.14** | 57.14 | 80.77 | 66.92 | 67.67 | 50.00 |
| Mnemis | 78.90 | 92.86 | **100.00** | 79.43 | 75.59 | 72.22 | 50.00 |
| HyperMem | 79.40 | 95.71 | 91.07 | 83.33 | 73.68 | 71.43 | 70.00 |
| EverMemOS | 79.87 | 95.71 | 82.14 | **88.46** | 66.17 | **76.44** | **92.22** |
| Hindsight | 69.20 | 90.00 | **100.00** | 78.21 | 54.89 | 54.14 | 70.00 |
| **HyPathMem** | **82.41** | 96.29 | **100.00** | 83.33 | **77.44** | 71.92 | 83.33 |

HyPathMem reaches **82.41% overall**, outperforming the strongest baseline, EverMemOS (79.87), by **2.54 pp**.

The category results reveal both strengths and remaining weaknesses:

- **Assistant:** 100.00%, tied for the best result.
- **Temporal:** 77.44%, **+1.85 pp** over Mnemis (75.59), the strongest baseline in this category.
- **User:** 96.29%, close to the best baseline result of 97.14.
- **Update:** 83.33%, below EverMemOS by 5.13 pp.
- **Multi-session:** 71.92%, below EverMemOS by 4.52 pp.
- **Preference:** 83.33%, below EverMemOS by 8.89 pp.

Therefore, the overall LongMemEval-S gain does not mean that HyPathMem dominates every memory ability. The clearest current advantages are Assistant and Temporal, while knowledge updates, multi-session aggregation, and preference questions remain important directions for improvement.

### 3.2 Qwen3-30B Generator

| Method | Overall | User | Assistant | Update | Temporal | Multi-session | Preference |
|---|---:|---:|---:|---:|---:|---:|---:|
| FullText | 57.00 | 87.14 | 89.29 | 71.79 | 42.11 | 37.59 | 40.00 |
| NaiveRAG | 72.10 | 90.86 | 93.33 | 77.77 | 54.14 | 67.67 | 73.33 |
| BM25 Fact + CE | 65.40 | 85.71 | 96.64 | 76.92 | 43.61 | 56.39 | 70.00 |
| Graph Retrieval | 60.40 | **95.71** | 64.29 | 67.95 | 35.34 | 57.14 | 76.67 |
| LightMem | 63.46 | 84.76 | 34.86 | 77.48 | 57.33 | 62.86 | 60.48 |
| MECo | 69.28 | 94.56 | 57.33 | 78.41 | 59.13 | 66.37 | 66.83 |
| Mnemis | 74.78 | 94.64 | 95.76 | 79.37 | 67.28 | 63.63 | 60.00 |
| HyperMem | 71.00 | **95.71** | 89.29 | 70.51 | 63.91 | 55.64 | 80.00 |
| EverMemOS | 74.20 | 88.57 | 83.93 | **85.90** | 65.41 | 60.90 | **90.00** |
| Hindsight | 66.69 | 87.68 | 95.45 | 75.44 | 52.89 | 52.58 | 64.98 |
| **HyPathMem** | **79.69** | **95.71** | **98.64** | 83.67 | **70.57** | **69.98** | 80.00 |

HyPathMem reaches **79.69% overall**, improving over the strongest baseline, Mnemis (74.78), by **4.91 pp**.

Compared with the GPT-4.1-mini setting, the Qwen3-30B results show a particularly strong relative advantage in:

- **Assistant:** 98.64%, **+2.00 pp** over the strongest baseline.
- **Temporal:** 70.57%, **+3.29 pp** over Mnemis.
- **Multi-session:** 69.98%, **+2.31 pp** over NaiveRAG.
- **User:** 95.71%, tied for the best result.

However, Update remains 2.23 pp below EverMemOS, and Preference remains 10.00 pp below EverMemOS.

---

## 4. Retrieval Quality on LoCoMo

| Method | Hit@5 | Recall@5 | FullCover@5 | Hit@20 | Recall@20 | FullCover@20 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 Fact | 51.36 | 46.24 | 42.53 | 66.43 | 59.55 | 54.09 |
| BM25 + CE | 67.92 | 60.78 | 55.19 | 74.22 | 67.19 | 60.97 |
| Graph Retrieval + CE | 76.30 | 68.35 | 62.08 | 85.26 | 78.16 | 71.56 |
| HyperMem | 79.81 | 72.30 | 66.04 | 88.64 | 82.98 | 76.95 |
| **HyPathMem** | **83.77** | **75.71** | **68.83** | **90.56** | **84.15** | **77.60** |

The retrieval results provide direct evidence that the downstream QA improvement is accompanied by stronger evidence retrieval.

Compared with HyperMem, HyPathMem improves:

- Hit@5 by **+3.96 pp**
- Recall@5 by **+3.41 pp**
- FullCover@5 by **+2.79 pp**
- Hit@20 by **+1.92 pp**
- Recall@20 by **+1.17 pp**
- FullCover@20 by **+0.65 pp**

Compared with Graph Retrieval + CE, the gains at Top-20 are larger: **+5.30 Hit@20**, **+5.99 Recall@20**, and **+6.04 FullCover@20**.

The improvement is therefore visible before answer generation and is not explained solely by a stronger generator.

---


## 5. Dual-Direction × Dual-Geometry Ablation on LoCoMo

| Retrieval Variant | Hit@20 | Recall@20 | FullCover@20 | QA Acc. |
|---|---:|---:|---:|---:|
| BU-E only | 81.64 | 74.24 | 67.45 | 83.68 |
| BU-H only | 79.82 | 73.07 | 66.15 | 84.20 |
| TD-E only | 76.43 | 68.95 | 61.65 | 82.45 |
| TD-H only | 67.58 | 60.58 | 54.69 | 78.96 |
| Euclidean only: BU-E + TD-E | 86.85 | 80.91 | 72.98 | 87.04 |
| Hyperbolic only: BU-H + TD-H | 81.58 | 74.74 | 67.71 | 84.67 |
| Bottom-up only: BU-E + BU-H | 87.04 | 80.55 | **73.76** | 85.83 |
| Top-down only: TD-E + TD-H | 80.73 | 73.50 | 66.47 | 83.66 |
| **Full: BU-E + BU-H + TD-E + TD-H** | **88.87** | **80.92** | 72.87 | **91.62** |

This ablation separates two design axes in HyPathMem:

- **Direction:** bottom-up (BU) versus top-down (TD);
- **Geometry:** Euclidean (E) versus hyperbolic (H).

The results do not support the claim that any single direction or geometry is uniformly superior. Instead, the strongest evidence is for **complementarity across routes**.

### Single-route behavior

Among the four individual routes, **BU-E** gives the strongest retrieval metrics:

- Hit@20 = 81.64
- Recall@20 = 74.24
- FullCover@20 = 67.45

However, **BU-H obtains slightly higher QA accuracy than BU-E** (84.20 vs 83.68) despite lower Hit, Recall, and FullCover. This is an important observation: aggregate retrieval coverage is not perfectly aligned with downstream answer quality. A route can retrieve fewer gold-support items overall while still surfacing evidence that is more useful for answering a subset of questions.

The top-down routes are weaker when used alone. TD-E reaches 82.45 QA accuracy, while TD-H reaches 78.96. In particular, TD-H is the weakest isolated route in both retrieval and QA. Therefore, the current results should **not** be interpreted as showing that hyperbolic top-down routing is independently stronger than Euclidean retrieval.

### Direction-level comparison

Combining the two bottom-up routes produces:

- Hit@20: 87.04
- Recall@20: 80.55
- FullCover@20: **73.76**
- QA: 85.83

This is substantially stronger in retrieval than the top-down-only combination:

- Hit@20: 80.73
- Recall@20: 73.50
- FullCover@20: 66.47
- QA: 83.66

Thus, **bottom-up retrieval provides the stronger standalone retrieval backbone**. It is particularly effective at retrieving and covering directly relevant evidence.

However, the full model improves QA from 85.83 to **91.62** over bottom-up only, a gain of **+5.79 pp**, even though FullCover@20 decreases slightly from 73.76 to 72.87. This indicates that the benefit of adding top-down routes is not simply "retrieving more gold evidence." Instead, top-down routes appear to provide **complementary structural signals or alternative access paths** that improve the usefulness of the final evidence set and/or its downstream selection.

### Geometry-level comparison

The Euclidean-only combination is clearly stronger than the hyperbolic-only combination when each geometry is used in isolation:

| Setting | Hit@20 | Recall@20 | FullCover@20 | QA |
|---|---:|---:|---:|---:|
| Euclidean only | 86.85 | 80.91 | 72.98 | 87.04 |
| Hyperbolic only | 81.58 | 74.74 | 67.71 | 84.67 |

Euclidean-only therefore exceeds Hyperbolic-only by:

- **+5.27 pp** Hit@20
- **+6.17 pp** Recall@20
- **+5.27 pp** FullCover@20
- **+2.37 pp** QA accuracy

This means that hyperbolic retrieval should not be framed as a replacement for Euclidean retrieval. The stronger interpretation is that hyperbolic routes contribute **non-redundant information** when combined with Euclidean routes.

This is most visible when comparing the full model with Euclidean-only:

- Hit@20: 86.85 → 88.87 (**+2.02 pp**)
- Recall@20: 80.91 → 80.92 (**+0.01 pp**)
- FullCover@20: 72.98 → 72.87 (**−0.11 pp**)
- QA: 87.04 → 91.62 (**+4.58 pp**)

The very large QA improvement occurs with almost no change in Recall@20 and a negligible decrease in FullCover@20. Therefore, the added hyperbolic routes are unlikely to help merely by increasing the quantity of retrieved gold evidence. A more plausible interpretation is that they improve the **composition, structural diversity, routing provenance, or ranking utility** of the candidate set in ways that are not captured by standard set-level retrieval metrics.

### Why the full model is better

The full four-route system achieves the best QA accuracy (**91.62%**) and the best Hit@20 (**88.87%**), but it does not achieve the best FullCover@20; bottom-up only is slightly higher (73.76 vs 72.87). This distinction is important.

The ablation suggests that HyPathMem benefits from **route complementarity rather than metric-wise dominance**:

1. **Bottom-up Euclidean retrieval** provides strong direct semantic access to relevant facts.
2. **Bottom-up hyperbolic retrieval** contributes a different evidence ordering: although its aggregate retrieval scores are lower, its QA accuracy is slightly higher than BU-E alone.
3. **Top-down routes** are weak in isolation but add useful hierarchical or structural access paths when combined with bottom-up retrieval.
4. **Hyperbolic routes** are weaker than Euclidean routes as standalone retrievers, but their addition to Euclidean routes yields a large downstream QA gain.
5. The final gain therefore cannot be explained by simply increasing Hit/Recall/FullCover. It is more consistent with **complementary evidence organization and route-aware selection**.

### Main takeaway

The key conclusion from this ablation is not that hyperbolic geometry outperforms Euclidean geometry, nor that top-down retrieval outperforms bottom-up retrieval. The evidence supports a more precise claim:

> **Euclidean and bottom-up routes form the strongest standalone retrieval backbone, while hyperbolic and top-down routes provide complementary structural signals that substantially improve final QA when jointly integrated.**

This interpretation is consistent with the full system achieving **+4.58 pp QA over Euclidean-only** and **+5.79 pp over bottom-up-only**, despite only small or even negative changes in some conventional retrieval metrics.

Accordingly, the role of the dual-geometry, dual-direction design should be described as **complementary multi-route evidence access**, rather than as four independently strong retrievers.

---

## 6. Component Ablation: Retrieval and Selection

| Variant | Hit@20 | Recall@20 | FullCover@20 | QA Acc. |
|---|---:|---:|---:|---:|
| CE Reranking | 88.87 | 80.92 | 72.87 | 87.94 |
| + Semantic LGBM | 89.03 | 81.79 | 73.50 | 88.61 |
| + Route Features | 89.68 | 82.44 | 75.15 | 90.22 |
| + Card Expansion | 90.42 | 83.59 | 76.24 | 90.87 |
| + Relation Selection | 90.56 | 84.15 | 77.60 | **91.62** |

Starting from CE reranking, the complete retrieval-and-selection pipeline increases:

- Hit@20 from 88.87 to 90.56 (**+1.69 pp**)
- Recall@20 from 80.92 to 84.15 (**+3.23 pp**)
- FullCover@20 from 72.87 to 77.60 (**+4.73 pp**)
- QA accuracy from 87.94 to 91.62 (**+3.68 pp**)

The stepwise pattern is informative:

1. **Semantic LGBM** gives a modest initial improvement (+0.67 pp QA).
2. **Route Features** produce the largest single QA gain in this chain (+1.61 pp), indicating that route-origin and route-related information provide useful signals beyond semantic reranking alone.
3. **Card Expansion** further improves evidence coverage and adds +0.65 pp QA.
4. **Relation Selection** produces the largest final FullCover improvement (+1.36 pp) and adds another +0.75 pp QA.

The larger cumulative improvement in FullCover@20 than in Hit@20 suggests that the later components are especially useful for retrieving a more complete evidence set rather than merely retrieving at least one relevant item.

---

## 7. Evidence Grounding and Reconstruction on LongMemEval-S

| Variant | Overall | User | Assistant | Update | Temporal | Multi-session | Preference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selected Facts | 76.59 | 93.58 | 73.21 | 80.05 | 72.18 | 68.42 | **90.00** |
| + RAW Support Closure | 78.96 | 94.84 | 90.50 | 82.65 | 70.24 | 70.56 | 83.33 |
| + Context & Provenance | 80.34 | 96.29 | **100.00** | 83.33 | 70.58 | 71.00 | 83.33 |
| + Temporal Grounding | **82.41** | **96.29** | **100.00** | **83.33** | **77.44** | **71.92** | 83.33 |

The evidence reconstruction pipeline raises overall accuracy from **76.59% to 82.41%**, a total improvement of **5.82 pp**.

### Raw Support Closure

Adding raw support increases overall accuracy by **2.37 pp**. The dominant effect is on Assistant questions:

- Assistant: 73.21 → 90.50 (**+17.29 pp**)
- Update: 80.05 → 82.65 (**+2.60 pp**)
- Multi-session: 68.42 → 70.56 (**+2.14 pp**)

This result shows that normalized selected facts alone can lose answer-critical surface information, especially for questions grounded in assistant utterances. Restoring raw supporting text substantially reduces this loss.

However, this stage also decreases Temporal by 1.94 pp and Preference by 6.67 pp. Raw grounding is therefore helpful but not uniformly beneficial across question types.

### Context and Provenance

Adding contextual and provenance information further improves overall accuracy from 78.96 to 80.34 (**+1.38 pp**).

The largest additional gain again appears in Assistant:

- Assistant: 90.50 → 100.00 (**+9.50 pp**)

User also improves by 1.45 pp. This supports retaining conversational context and provenance around evidence rather than passing isolated claims alone.

### Temporal Grounding

Temporal grounding raises overall accuracy from 80.34 to **82.41** (**+2.07 pp**).

Its effect is highly concentrated:

- Temporal: 70.58 → 77.44 (**+6.86 pp**)
- Multi-session: 71.00 → 71.92 (**+0.92 pp**)
- Other categories remain unchanged in this ablation.

This is the clearest evidence in the current experiments that explicit temporal grounding addresses a specific weakness that general raw-evidence reconstruction does not solve by itself.

### Overall Reconstruction Effect

Relative to Selected Facts, the complete reconstruction pipeline changes:

- Overall: **+5.82 pp**
- User: **+2.71 pp**
- Assistant: **+26.79 pp**
- Update: **+3.28 pp**
- Temporal: **+5.26 pp**
- Multi-session: **+3.50 pp**
- Preference: **−6.67 pp**

The major positive effect is therefore on Assistant and Temporal questions, while Preference remains a clear unresolved weakness.

---

## 8. Cross-Benchmark Interpretation

The two benchmarks expose different aspects of the method.

On **LoCoMo**, HyPathMem's strongest evidence is the combination of:

- highest overall QA accuracy under both generators;
- improved Hit, Recall, and FullCover retrieval metrics;
- stepwise improvements from route features, card expansion, and relation selection.

This pattern is consistent with the intended role of multi-route hierarchical retrieval: improving both access to relevant memory and completeness of multi-item evidence.

On **LongMemEval-S**, retrieval alone is not sufficient to explain the final result. The evidence reconstruction ablation shows an additional **+5.82 pp** overall improvement from Selected Facts to the complete grounded context. The largest effects occur on Assistant and Temporal questions, showing that preserving raw evidence and explicitly grounding temporal information are important after retrieval.

Taken together, the current results support a two-part interpretation:

1. **Retrieval quality matters:** HyPathMem improves evidence access and coverage on LoCoMo.
2. **Evidence usability matters:** raw support, context/provenance, and temporal grounding further improve the ability of the generator to use retrieved memory on LongMemEval-S.

---

## 9. Remaining Weaknesses

The current results also identify several limitations that should not be hidden by the overall score.

### Preference

Preference is the clearest LongMemEval-S weakness.

- GPT-4.1-mini: HyPathMem 83.33 vs EverMemOS 92.22
- Qwen3-30B: HyPathMem 80.00 vs EverMemOS 90.00

Moreover, the evidence reconstruction ablation decreases Preference from 90.00 to 83.33 after adding raw support, and later stages do not recover the loss. This indicates that simply adding more grounded evidence is insufficient for preference questions.

### Knowledge Update

HyPathMem is also below EverMemOS on Update under both generators:

- GPT-4.1-mini: 83.33 vs 88.46
- Qwen3-30B: 83.67 vs 85.90

This suggests that current retrieval and reconstruction do not yet fully resolve state changes or competing versions of the same information.

### Multi-session

The GPT-4.1-mini result remains below EverMemOS:

- HyPathMem: 71.92
- EverMemOS: 76.44

The Qwen3-30B setting improves to the best current result (69.98), but the absolute accuracy still leaves substantial room for improvement.

These categories should be treated as the main targets for further analysis rather than claiming uniform superiority across all memory abilities.

---

## 10. Summary

The completed experiments currently support four main conclusions:

1. **HyPathMem achieves the best overall accuracy on both LoCoMo and LongMemEval-S under both tested generators.**
2. **The LoCoMo gain is accompanied by measurable improvements in Hit, Recall, and FullCover, indicating that the benefit occurs at the retrieval stage rather than only during answer generation.**
3. **Route-aware selection, card expansion, and relation selection progressively improve evidence coverage and QA accuracy.**
4. **On LongMemEval-S, raw evidence reconstruction and temporal grounding provide substantial additional gains, especially for Assistant and Temporal questions.**

At the same time, Preference, Knowledge Update, and some Multi-session cases remain weaker than the strongest competing systems. These results define a clear boundary for the current method and motivate future improvements in state tracking, preference modeling, and cross-session evidence aggregation.
