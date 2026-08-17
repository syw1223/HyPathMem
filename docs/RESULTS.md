# Experimental Results

All values are percentages. The tables reproduce the consolidated paper
experiment sheet.

## LoCoMo Main Results

### GPT-4.1-mini Generator

| Method | Overall | Cat1 | Cat2 | Cat3 | Cat4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FullText | 85.30 | 86.74 | 85.66 | 65.00 | 87.00 |
| NaiveRAG | 78.88 | 76.25 | 68.90 | 62.58 | 85.44 |
| BM25 Fact + CE | 77.99 | 64.18 | 79.75 | 61.46 | 83.83 |
| Graph Retrieval | 80.42 | 78.56 | 54.62 | 67.85 | 92.34 |
| LightMem | 72.14 | 72.05 | 54.56 | 56.74 | 80.64 |
| MECo | 78.40 | 67.67 | 79.44 | 58.33 | 83.12 |
| HyperMem | 84.22 | 82.98 | 77.57 | 61.46 | 89.77 |
| EverMemOS | 86.82 | 83.33 | 81.93 | 43.75 | 94.77 |
| Hindsight | 86.26 | 81.62 | 82.22 | 75.00 | 90.64 |
| Mnemis | 87.04 | 84.40 | 79.75 | 63.04 | 93.34 |
| **HyPathMem** | **91.62** | **90.07** | **90.34** | **77.08** | **94.29** |

### Qwen3-30B Generator

| Method | Overall | Cat1 | Cat2 | Cat3 | Cat4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FullText | 81.30 | 74.47 | 75.08 | 66.67 | 87.63 |
| NaiveRAG | 75.78 | 72.70 | 65.42 | 59.38 | 82.64 |
| BM25 Fact + CE | 75.71 | 64.18 | 69.47 | 64.58 | 83.23 |
| Graph Retrieval | 76.62 | 75.18 | 49.84 | 64.58 | 88.70 |
| LightMem | 70.50 | 70.45 | 49.87 | 53.84 | 80.29 |
| MECo | 77.40 | 76.24 | 65.42 | 58.33 | 84.54 |
| HyperMem | 81.69 | 80.14 | 72.59 | 60.42 | 88.11 |
| EverMemOS | 83.90 | 76.95 | 77.88 | 52.08 | 92.15 |
| Hindsight | 83.90 | 78.37 | 81.00 | 71.88 | 88.23 |
| Mnemis | 83.64 | 82.51 | 73.78 | 60.23 | 90.45 |
| **HyPathMem** | **87.84** | **85.27** | **83.26** | **70.83** | **92.39** |

## LongMemEval-S Main Results

### GPT-4.1-mini Generator

| Method | Overall | User | Assistant | Update | Temporal | Multi-session | Preference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FullText | 61.06 | 90.20 | 95.75 | 75.49 | 45.56 | 41.78 | 45.00 |
| NaiveRAG | 74.05 | 92.54 | 95.27 | 79.89 | 55.36 | 70.45 | 75.67 |
| BM25 Fact + CE | 67.31 | 87.36 | 93.64 | 78.46 | 46.28 | 58.74 | 73.50 |
| Graph Retrieval | 62.56 | 95.26 | 67.58 | 70.25 | 37.68 | 60.44 | 76.67 |
| LightMem | 66.86 | 85.14 | 35.64 | 83.52 | 61.85 | 66.87 | 65.84 |
| MECo | 71.40 | 97.14 | 57.14 | 80.77 | 66.92 | 67.67 | 50.00 |
| Mnemis | 78.90 | 92.86 | 100.00 | 79.43 | 75.59 | 72.22 | 50.00 |
| HyperMem | 79.40 | 95.71 | 91.07 | 83.33 | 73.68 | 71.43 | 70.00 |
| EverMemOS | 79.87 | 95.71 | 82.14 | 88.46 | 66.17 | 76.44 | 92.22 |
| Hindsight | 69.20 | 90.00 | 100.00 | 78.21 | 54.89 | 54.14 | 70.00 |
| **HyPathMem** | **82.41** | **96.29** | **100.00** | **83.33** | **77.44** | **71.92** | **83.33** |

### Qwen3-30B Generator

| Method | Overall | User | Assistant | Update | Temporal | Multi-session | Preference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FullText | 57.00 | 87.14 | 89.29 | 71.79 | 42.11 | 37.59 | 40.00 |
| NaiveRAG | 72.10 | 90.86 | 93.33 | 77.77 | 54.14 | 67.67 | 73.33 |
| BM25 Fact + CE | 65.40 | 85.71 | 96.64 | 76.92 | 43.61 | 56.39 | 70.00 |
| Graph Retrieval | 60.40 | 95.71 | 64.29 | 67.95 | 35.34 | 57.14 | 76.67 |
| LightMem | 63.46 | 84.76 | 34.86 | 77.48 | 57.33 | 62.86 | 60.48 |
| MECo | 69.28 | 94.56 | 57.33 | 78.41 | 59.13 | 66.37 | 66.83 |
| Mnemis | 74.78 | 94.64 | 95.76 | 79.37 | 67.28 | 63.63 | 60.00 |
| HyperMem | 71.00 | 95.71 | 89.29 | 70.51 | 63.91 | 55.64 | 80.00 |
| EverMemOS | 74.20 | 88.57 | 83.93 | 85.90 | 65.41 | 60.90 | 90.00 |
| Hindsight | 66.69 | 87.68 | 95.45 | 75.44 | 52.89 | 52.58 | 64.98 |
| **HyPathMem** | **79.69** | **95.71** | **98.64** | **83.67** | **70.57** | **69.98** | **80.00** |

## LoCoMo Retrieval Quality

| Method | Hit@5 | Recall@5 | FullCover@5 | Hit@20 | Recall@20 | FullCover@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 Fact | 51.36 | 46.24 | 42.53 | 66.43 | 59.55 | 54.09 |
| BM25 + CE | 67.92 | 60.78 | 55.19 | 74.22 | 67.19 | 60.97 |
| Graph Retrieval + CE | 76.30 | 68.35 | 62.08 | 85.26 | 78.16 | 71.56 |
| HyperMem | 79.81 | 72.30 | 66.04 | 88.64 | 82.98 | 76.95 |
| **HyPathMem** | **83.77** | **75.71** | **68.83** | **90.65** | **84.15** | **77.60** |

## Dual-Direction and Dual-Geometry Ablation

| Retrieval variant | Hit@20 | Recall@20 | FullCover@20 | QA Acc. |
| --- | ---: | ---: | ---: | ---: |
| BU-E only | 81.64 | 74.24 | 67.45 | 83.68 |
| BU-H only | 79.82 | 73.07 | 66.15 | 84.20 |
| TD-E only | 76.43 | 68.95 | 61.65 | 82.45 |
| TD-H only | 67.58 | 60.58 | 54.69 | 78.96 |
| Euclidean only: BU-E + TD-E | 86.85 | 80.91 | 72.98 | 87.04 |
| Hyperbolic only: BU-H + TD-H | 81.58 | 74.74 | 67.71 | 84.67 |
| Bottom-up only: BU-E + BU-H | 87.04 | 80.55 | 73.76 | 85.83 |
| Top-down only: TD-E + TD-H | 80.73 | 73.50 | 66.47 | 83.66 |
| **Full: BU-E + BU-H + TD-E + TD-H** | **88.87** | **80.92** | **72.87** | **91.62** |

The four routes are complementary rather than independently dominant. The
full system benefits from bottom-up anchors, top-down hierarchy traversal, and
Euclidean/hyperbolic geometry.

## HyperPath Retrieval and Selection Ablation

| Variant | Hit@20 | Recall@20 | FullCover@20 | QA Acc. |
| --- | ---: | ---: | ---: | ---: |
| CE Reranking | 88.87 | 80.92 | 72.87 | 87.94 |
| + Semantic LGBM | 89.03 | 81.79 | 73.50 | 88.61 |
| + Route Features | 89.68 | 82.44 | 75.15 | 90.22 |
| + Card Expansion | 90.42 | 83.59 | 76.24 | 90.87 |
| **+ Relation Selection** | **90.56** | **84.15** | **77.60** | **91.62** |

The selector gains progressively as route provenance, agreement, Card-guided
expansion, and relation evidence are incorporated into final evidence
selection.

## Evidence Grounding and Reconstruction Ablation

| Method | Overall | User | Assistant | Update | Temporal | Multi-session | Preference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected Facts | 76.59 | 93.58 | 73.21 | 80.05 | 72.18 | 68.42 | 90.00 |
| + RAW Support Closure | 78.96 | 94.84 | 90.50 | 82.65 | 70.24 | 70.56 | 83.33 |
| + Context & Provenance | 80.34 | 96.29 | 100.00 | 83.33 | 70.58 | 71.00 | 83.33 |
| **+ Temporal Grounding** | **82.41** | **96.29** | **100.00** | **83.33** | **77.44** | **71.92** | **83.33** |

Raw support closure primarily improves evidence-faithful answer generation.
Context and provenance restoration further help assistant and update
questions, while explicit temporal grounding recovers the temporal category.
