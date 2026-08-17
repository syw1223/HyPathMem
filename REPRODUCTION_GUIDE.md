# HyPathMem Method and Usage Guide

## Method

HyPathMem is a graph memory QA system that separates memory construction,
query-time path retrieval, evidence compilation, and answer-time reasoning.

### 1. Provenance-preserving memory graph

Each dialogue history is represented as a typed memory graph. The graph stores
semantic units and hierarchical/event structure while retaining the exact raw
dialogue quote and source metadata for each answer-bearing claim. LoCoMo uses
one graph per conversation. LongMemEval-S uses one isolated graph per unique
history, so no information crosses user histories.

### 2. Hierarchical hyperpath retrieval

At question time, HyPathMem retrieves anchor-to-evidence paths rather than
independent text chunks. The retrieval stack combines semantic matching with
typed graph expansion and hyperbolic/topological path features. The frozen
LoCoMo D0 system selects Top20 paths. The LongMemEval-S line retains a larger
Top50 candidate pool before answer-context compilation.

### 3. Evidence compilation

The compiler turns selected paths into evidence units and query requirements.
It keeps provenance, source speaker/session/time, structured claims, and exact
raw quotes. D2 is the raw-grounded context compiler: it gives the answer model
claims together with their supporting quotes rather than treating a path label
as evidence.

### 4. Conditional temporal branch

Temporal questions are not answered from the full D2 context directly. The
temporal sidecar builds a small typed packet with event operands, anchors,
message time, occurrence time/intervals, and temporal constraints. A solver
may take over only when every safety condition holds:

1. required operands are complete and grounded by exact quotes;
2. identities and anchors are trusted;
3. relative/absolute time normalization is deterministic and consistent;
4. the requested operation is supported;
5. reverse verification and the Q4 verifier accept the result.

Otherwise HyPathMem fails closed and reuses the frozen baseline answer. This
prevents a temporal branch from replacing an answer merely because a packet is
well formatted.

### 5. Dataset-specific selected routing

- **LongMemEval-S:** D2 is used for 367 non-temporal questions. Temporal
  questions use the frozen Q4 branch plus only five verified safe overrides
  from 8.4/8.5time_v; all unsafe cases fall back to D0.
- **LoCoMo:** selected reporting uses the original frozen D0 Top20 answer path.
  D2 was evaluated as a paired negative control and not selected. The
  `8.5time_locomo_v2` branch reused frozen D0 for unsafe cases and produced 36
  verified Cat2 takeovers with no paired break.

## Installation

Python 3.10+ is required. The lightweight package dependencies are declared
in `pyproject.toml`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[dev]'
```

For model-backed graph construction/retrieval, additionally install the local
PyTorch, embedding, and reranking runtime matching the serving stack. These
are intentionally not pinned into the lightweight package because hardware and
local serving endpoints differ between machines.

## Offline Validation

Run these before using any hosted model:

```bash
python3 verify_frozen_results.py
python3 -m pytest -q tests/test_reconstruction.py tests/test_retrieval.py \
  tests/test_temporal_time_v8_4.py tests/test_temporal_time_v8_5.py \
  tests/test_locomo_time_v8_5_v2.py
```

## Final-Artifact Regeneration

The final archive writers are deterministic with respect to their recorded
source artifacts. They intentionally refuse to run when an expected input is
missing.

```bash
python3 scripts/146_archive_hypathmem_longmemeval_final500.py
python3 scripts/154_archive_hypathmem_locomo_d0_time_v2_final.py
```

The source artifacts are already copied into each final archive. The original
working-tree paths referenced by the archivers are listed in the manifests.

## Development Runners

The recommended order when reproducing from raw data is:

1. construct a typed graph and candidate paths using the v3.9 mainline;
2. run answer generation and judging with the same model/protocol recorded in
   the relevant result metadata;
3. compile D2 packs only for the intended ablation or selected route;
4. run temporal 8.4/8.5 processing without modifying the semantic graph;
5. archive the final selected route and verify its manifest.

Useful entry points:

```bash
python3 scripts/run_locomo_v3_9_mainline.py --help
python3 scripts/run_longmemeval_v3_9_mainline.py --help
python3 scripts/142_build_hypathmem_8_4time_v.py --help
python3 scripts/144_run_hypathmem_8_5time_v_conversion.py --help
python3 scripts/129_build_hypathmem_r_v0_1_packs.py --help
python3 scripts/149_prepare_hypathmem_locomo_time_v8_5.py --help
python3 scripts/152_compile_hypathmem_locomo_time_v8_5_v2.py --help
```

Hosted QA/judge runners read their credentials from environment variables such
as `E_MEM_API_KEY`; never commit a key or endpoint credential into this
release.

## Reporting Guidance

Use the frozen `summary.json` files for headline numbers and the per-question
JSON files for category/type analysis. Keep the qualification that both chosen
systems are post-hoc development results. The final manifests identify source
hashes, and the LoCoMo manifest separately records the large external graph
and path artifacts.
