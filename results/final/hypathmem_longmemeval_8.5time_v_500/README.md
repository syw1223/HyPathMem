# HyPathMem LongMemEval-S Final 500

This directory contains the frozen HyPathMem LongMemEval-S development
snapshot and records the final paper-table result.

## Result

```text
Final paper-table result: 412/500 = 82.40%
```

The bundled per-question `8.5time_v` development snapshot predates the final
paper-table consolidation and contains 408/500 correct judgments (81.60%). It
is retained unchanged for provenance; it must not be presented as the source
of the four additional final judgments.

The answer model is GPT-4.1-mini and the judge model is GPT-4o-mini. The final
router uses D2 raw-grounded context for non-temporal questions, verified
temporal solver answers where every safety gate passes, and frozen D0 fallback
otherwise.

## Reporting status

The bundled 408/500 snapshot is a **post-hoc development result**, not an
independent held-out result. That qualification remains attached to the
snapshot, while the paper reports the consolidated single-judge result of
412/500 (82.40%).

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
