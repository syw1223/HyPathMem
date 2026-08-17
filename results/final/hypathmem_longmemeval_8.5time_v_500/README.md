# HyPathMem LongMemEval-S Final 500

This directory is the frozen HyPathMem LongMemEval-S result selected for
reporting.

## Result

```text
Non-temporal D2: 305/367
Temporal 8.5time_v: 103/133
Overall: 408/500 = 81.60%
Frozen D0: 385/500 = 77.00%
Delta: +23 correct, +4.60 percentage points
```

The answer model is GPT-4.1-mini and the judge model is GPT-4o-mini. The final
router uses D2 raw-grounded context for non-temporal questions, verified
temporal solver answers where every safety gate passes, and frozen D0 fallback
otherwise.

## Reporting status

This is a **post-hoc development result**, not an independent held-out result.
That qualification must remain attached to the 81.60% score.

## Contents

- `hypathmem_longmemeval_final500.json`: all 500 final predictions, judgments,
  routes, and provenance.
- `summary.json`: aggregate, type-level, and route-level metrics.
- `source_artifacts/`: frozen copies of the five answer/judge inputs and the
  8.5 conversion record used to construct the result.
- `method_snapshot/`: frozen config, implementation, runners, and method note.
- `MANIFEST.json`: original paths, file sizes, and SHA-256 hashes.

Regenerate and validate the archive with:

```bash
conda run -n python311 python \
  scripts/146_archive_hypathmem_longmemeval_final500.py
```
