# Cleanup Record

Date: 2026-08-07

## Removed from the working tree

- `.pytest_cache/`
- `__pycache__/` directories under `src/hytopomem/`, `scripts/`, and `tests/`
- `outputs/reconstruction/hypathmem_locomo_8.5time_v2/cat2_q4.console.log`
  because it was zero bytes.

## Intentionally retained

- all `outputs/final/hypathmem_*` archives;
- all D1/D2, temporal, oracle, paired-control, and other ablation JSON
  artifacts under `outputs/qa/hypathmem_*` and
  `outputs/reconstruction/hypathmem_*`;
- non-empty process logs, because they retain cost, failure, and provenance
  details potentially needed during paper preparation;
- base graph/path files referenced by the final LoCoMo manifest.

This is a conservative cleanup. It removes only reproducibility-irrelevant
cache files and does not discard process evidence needed for later analyses.
