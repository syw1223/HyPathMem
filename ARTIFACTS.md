# Research Artifacts

The LoCoMo graph, frozen Top20 paths, four-route pre-Card union candidate pool
and large structural-analysis audit records are distributed as gzip-split
parts under `results/external/`. Every part is below 8 MB; the manifest records
part hashes, restored hashes and repository-relative destinations.

```bash
python scripts/restore_release_artifacts.py
```

Restoration verifies hashes before writing and refuses to overwrite a
different existing artifact. Restored files are generated outputs and need
not be committed again.

The frozen LoCoMo-trained transfer model and protocol metadata are included.
Full original graph/candidate dependencies for retraining or rerunning the
LongMemEval transfer test are not all bundled. Their relative paths and
checksums remain in protocol manifests; the multi-gigabyte LongMemEval graph
must be reconstructed using the dataset pipeline or supplied locally.

Original datasets and model weights are not redistributed. Historical QA
archives and newer protocol analyses remain separate; results must be traced
to actual per-question judgments and evaluation denominators.
