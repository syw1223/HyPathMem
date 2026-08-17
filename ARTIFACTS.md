# External Artifacts

The release includes all final predictions, source artifacts used for final
selection, method snapshots, and manifests. The following large frozen
artifacts remain in the parent working tree to avoid duplicating more than
140 MB inside the source package:

| Artifact | Parent-tree path | SHA-256 |
| --- | --- | --- |
| LoCoMo graph | `outputs/graphs/locomo_graph_v3_6b_qwen_all.json` | `5352a47ec98869bfbbdcb2ccfaa647937acb708f7bef5f9f7ac516ee211ed7a6` |
| LoCoMo Top20 paths | `outputs/paths/full_v3_9_card_guided_expand120_light_quota_top20.json` | `16c4df5832d0802ef6c7720231ecb7a589d0249fba58f97ca02c889d490064af` |

To make a fully self-contained artifact archive, add these two files under
`results/external/` before packaging, then recompute the release checksum.
The final LoCoMo archive records the same paths and hashes in its
`MANIFEST.json`.

Original datasets and model checkpoints are not redistributed. Obtain them
from their official sources and record the exact local snapshots used for any
new experiment.
