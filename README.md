# HyPathMem

Anonymous review snapshot for query-conditioned structured evidence retrieval
and composition in long-term conversational memory.

## Method

Offline construction retains source-turn provenance in a hierarchical memory
graph. Retrieval combines BU-E, BU-H, TD-E and TD-H, constructs query-conditioned
Relation Cards, expands local candidates, and applies a HyperPath-aware
selector. Evidence compilation preserves raw support and uses verified
temporal grounding when applicable.

LoCoMo QA uses Top20 evidence units. LongMemEval-S exposes a Top50 pool to the
evidence compiler; @20 is a separate retrieval reporting cutoff.

## Installation and Validation

```bash
python -m pip install -e '.[dev]'
python scripts/restore_release_artifacts.py
python verify_frozen_results.py
python -m pytest -q tests/test_paper_release.py
```

The import package remains `hytopomem` for compatibility. No hosted LLM is
required for artifact validation or the two structural analyses.

## Models

Process-time LLM: Qwen3-30B. Embedding: Qwen3-Embedding-0.6B.
Answer generator: GPT-4.1-mini or Qwen3-30B, reported separately.
Answer judge: GPT-4o-mini. Keys and serving URLs must be supplied locally.

```bash
python scripts/run_locomo_v3_9_mainline.py --stage qa-hybrid --dry-run
python scripts/run_locomo_v3_9_mainline.py --stage qa-hybrid \
  --generator-model qwen3-30b-a3b-instruct-2507 --judge-model gpt-4o-mini --dry-run
```

## Selector Protocol

The LoCoMo selector entrypoint filters Category 5 before constructing training
features. Ten conversation-disjoint folds train on nine conversations and
evaluate on the remaining conversation. Evidence annotations supply relevance
labels; reference answers and judge outputs are not selector training targets.

`scripts/191_run_selector_protocol_audit.py` contains the corrected protocol
audit. `scripts/192_run_locomo_to_longmemeval_selector_transfer.py` trains on
LoCoMo Cat1-4, freezes the model, and performs label-free LongMemEval inference
before attaching evaluation annotations. The saved model, training manifest,
and retrieval results are in `outputs/analysis/`. Historical in-benchmark
LongMemEval CV code is retained as a comparison, not a leakage-free transfer
protocol.

The protocol runners require the graph/candidate sources named in their
manifests. Not all original LongMemEval graph inputs are bundled; in particular,
the multi-gigabyte graph is not distributed in this review snapshot. Historical
QA archives are not automatically replaced by the corrected retrieval tests.

## Structural Analyses

```bash
python scripts/analyze_locomo_geometry_overlap.py
python scripts/analyze_locomo_ancestor_random_control.py
```

The frozen four-route pre-Card candidate pool, graph and Top20 path artifact
are included as hash-verified compressed parts. Analysis outputs include
per-query audit records, sampling controls, summary tables and manifests.

## Results and Provenance

`results/final/` preserves historical per-question predictions and judgments.
`outputs/analysis/` contains later selector protocol tests and structural
controls. Historical table summaries remain attached to their original
artifacts; this release does not rewrite judgments to match a draft.
Identity-bearing local paths have been normalized, and distributed manifests
hash sanitized bytes. `ANONYMIZATION_AUDIT.json` records this operation.

See `REPRODUCTION_GUIDE.md` for historical pipeline entrypoints and
`ARTIFACTS.md` for packaged and unbundled dependencies. Obtain datasets and
model checkpoints from their official sources under their own licenses.
