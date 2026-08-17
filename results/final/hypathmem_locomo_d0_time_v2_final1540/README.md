# HyPathMem LoCoMo Final Result

Selected system: frozen D0 Top20 retrieval and GPT-4.1-mini answer generation, with fail-closed `8.5time_locomo_v2` replacement for 36 verified Cat2 answers.

- Accuracy: **1411/1540 = 91.6234%**
- Time V2: 36/36 takeover answers correct; fix=0, break=0.
- D2 was excluded: paired60 scored 49/60 versus frozen D0 51/60.
- Generation model: GPT-4.1-mini.
- Judge model: GPT-4o-mini, one judgment per answer.
- Retrieval: frozen HyPathMem Top20 paths over one graph per LoCoMo conversation.

## Reporting status

This is the final selected post-hoc development result. Time V2 was developed after diagnosing LoCoMo V1 failures. It is not a preregistered or untouched test-set result.

The graph and path files are referenced by path and SHA256 in `MANIFEST.json`; they are not duplicated in this archive. Temporal extraction token usage was not recorded by the source runner, so token accounting is explicitly marked incomplete.
