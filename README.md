# HyPathMem

HyPathMem is a conversational memory framework that retrieves and compiles
query-conditioned evidence through hierarchical paths. It combines bottom-up
and top-down retrieval in Euclidean and hyperbolic spaces, preserves route
provenance, constructs query-conditioned Relation Cards, and restores raw
support before answer generation.

## Method at a Glance

```text
Dialogue history
  -> typed Topic / Episode / Event / Fact memory hierarchy
  -> BU-E, BU-H, TD-E, and TD-H candidate routes
  -> query-conditioned Relation Card and local expansion
  -> semantic + route + relation-aware HyperPath selection
  -> raw support closure and provenance restoration
  -> verified temporal grounding when applicable
  -> compact evidence context -> answer generator
```

The four retrieval routes provide complementary views:

- **BU-E:** semantic evidence discovery from fact-level cues.
- **BU-H:** bottom-up traversal informed by hierarchical geometry.
- **TD-E:** semantic coarse-to-fine search over the hierarchy.
- **TD-H:** Lorentz-space routing over parent-child and branch structure.

The final selector uses semantic relevance, route provenance and agreement,
and relation features. For LongMemEval-S, the evidence compiler additionally
restores exact raw support and uses a fail-closed temporal branch.

## Main Results

All values below are accuracy percentages from the consolidated paper tables.
The answer judge is GPT-4o-mini.

| Dataset | Generator | HyPathMem |
| --- | --- | ---: |
| LoCoMo (1,540 Cat1-4 questions) | GPT-4.1-mini | **91.62** |
| LoCoMo (1,540 Cat1-4 questions) | Qwen3-30B | **87.84** |
| LongMemEval-S (500 questions) | GPT-4.1-mini | **82.40** |
| LongMemEval-S (500 questions) | Qwen3-30B | **79.60** |

Detailed baseline, category, retrieval, and ablation results are in
[docs/RESULTS.md](docs/RESULTS.md). Interpretation of the main findings,
component gains, and remaining weaknesses is available in
[docs/RESULT_ANALYSIS.md](docs/RESULT_ANALYSIS.md). The exact experiment
protocol is in [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Repository Layout

```text
configs/         Selected experiment configurations
docs/            Method, experiment, and result documentation
results/         Paper tables and frozen per-question result artifacts
scripts/         Selected paper pipeline and artifact runners
src/hytopomem/   Core implementation
tests/           Unit and contract tests
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Model-backed runs additionally require PyTorch, a sentence-transformer or
Qwen3 embedding endpoint, and the configured reranker/LLM endpoints.

## Validate the Release

The included frozen artifacts can be checked without model calls:

```bash
python verify_frozen_results.py
pytest -q tests/test_reconstruction.py tests/test_retrieval.py \
  tests/test_temporal_time_v8_5.py tests/test_locomo_time_v8_5_v2.py
```

## Reproduction

1. Download LoCoMo and LongMemEval-S from their official sources.
2. Set the raw dataset paths in `configs/locomo_mvp.yaml` and
   `configs/longmemeval_s.yaml`.
3. Configure model endpoints through environment variables. API credentials
   must never be committed.
4. Follow [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) for the graph,
   retrieval, evidence compilation, temporal, generation, and judging stages.

The selected paper protocol uses Qwen3-30B for process-time LLM operations,
Qwen3-Embedding-0.6B for dense embeddings, and the recorded local reranking
configuration. Answer-generation tables report GPT-4.1-mini and Qwen3-30B
separately, with GPT-4o-mini as judge.

For LongMemEval-S, Top-K values refer to distinct pipeline stages: 150 initial
candidates are retained after multi-route retrieval, the selector exposes a
Top-50 pool to evidence reconstruction, and the QA compiler consumes up to 50
whole evidence/path units subject to its token/character budget. Top-20 is the
common retrieval-metric cutoff, not the final 8.5 reader truncation.

The installable distribution and Python import path remain `hytopomem` for
backward compatibility with the original research code. The paper method and
repository are named **HyPathMem**.

## Data and Checkpoints

Datasets and model checkpoints are not redistributed. Large graph/path
artifacts and their checksums are listed in [ARTIFACTS.md](ARTIFACTS.md).

## Citation

See [CITATION.cff](CITATION.cff). Citation metadata can be updated with the
final paper title, author list, venue, and DOI when available.

## License

The code is released under the MIT License. Dataset, model, and baseline
licenses remain with their respective owners.
