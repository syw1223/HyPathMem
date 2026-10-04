# HyPathMem 8.4time_v

`8.4time_v` is a non-destructive temporal branch over the frozen HyPathMem
semantic graph and frozen Top50 evidence packs. It does not retrain or modify
the Lorentz router.

## Pipeline

1. **H1 temporal sidecar**: query-local `EVENT`, `STATE`, and `ANCHOR` nodes,
   semantic time, state validity intervals, and temporal relations.
2. **H2 operand closure**: set-level required-role matching over frozen Top50.
3. **H3 relation expansion**: anchor and state-version closure with conflict
   checks.
4. **H4 supplementary retrieval interface**: an injectable retriever for
   roles absent from Top50. It is disabled in the frozen-Top50 build.
5. **H5 constraint compiler**: deterministic ordering, elapsed/recency, and
   conservative state-at-time solving, followed by the existing Q4 verifier.

Already executable source-Q4 candidates are protected. New candidates must
have complete operands, consistent temporal relations, a successful solver
result, and grounded restrictive clauses. Otherwise the branch falls back.

## Current LongMemEval Temporal133 Build

- Source operand full coverage: `89/133` (66.9%).
- `8.4time_v` Top50 role coverage: `105/133` (78.9%).
- Source Q4 consensus-executable: `68/133` (51.1%).
- `8.4time_v` safely executable: `68/133` (51.1%).

## Trial 3: conservative temporal normalization

The independent `trial3` artifact adds three fail-closed rules without reading
gold answers or mutating the frozen semantic graph:

- sentence-local explicit month/day normalization;
- same-day normalization for a grounded return from a visit/tour/trip;
- start/end operand expansion for a single-role duration question.

It also distinguishes calendar semantics: `days passed between` uses the date
difference, while `days spent` over a bounded trip counts covered calendar
days inclusively.

Results on the same frozen Temporal133 inputs:

- operand full coverage: `105/133` (78.9%), unchanged;
- safely executable: `70/133` (52.6%), up from `68/133`;
- retained prior safe candidates: `68/68`;
- newly safe candidates: `2`;
- newly compiled answers: MoMA-to-Met exhibit `7 days`; Yosemite trip `3 days`.

Artifact:

```text
outputs/reconstruction/hypathmem_8.4time_v/
  longmemeval_temporal133_h1_h5_trial3.json
```

## E1: Newly executable candidates

The two new candidates were evaluated with the frozen D0 answers as a paired
control. Qwen3-30B on GPU2 performed the Q4 verification, the deterministic
solver supplied the candidate answer, and GPT-4o-mini judged each answer once.

- verified takeover rate: `2/2` (100%);
- takeover precision: `2/2` (100%);
- solver executable rate: `2/2` (100%);
- operand full coverage: `2/2` (100%);
- anchor binding accepted by the verifier: `2/2` (100%);
- frozen D0 accuracy: `1/2` (50%);
- verified `8.4time_v` accuracy: `2/2` (100%);
- paired outcome: `1` fix, `0` breaks, `1` unchanged-correct;
- fallback: `0/2`.

The Qwen verifier consumed 2,471 tokens in 7.89 seconds total. The two answer
judgments consumed 756 tokens in 4.46 seconds total. This is a real paired gain
on the two newly executable questions, but the sample is too small to claim a
stable aggregate improvement.

Artifact:

```text
outputs/qa/hypathmem_8.4time_v/
  e1_new2_qwen_gpu2_verifier_solver_judge_gpt4omini.json
```

## E2: Non-executable packet-reader diagnosis

The 35 full-cover but non-executable questions were evaluated with identical
frozen D0 answers and a GPT-4.1-mini reader that received only the typed
temporal packet. GPT-4o-mini judged each packet-reader answer once.

- packet-reader accuracy: `15/35` (42.86%);
- frozen D0 accuracy: `22/35` (62.86%);
- paired delta: `-20.00` percentage points;
- outcomes: `4` fixes, `11` breaks, `11` unchanged-correct, and `9`
  unchanged-wrong;
- explicit reader refusals: `15/35` (42.86%).

Accuracy by compiler failure reason:

| Failure reason | N | Packet reader | Frozen D0 | Fix | Break | Refusal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| missing occurrence time | 15 | 46.67% | 66.67% | 1 | 4 | 5 |
| missing pairwise operands | 12 | 33.33% | 50.00% | 2 | 4 | 7 |
| temporal conflict | 5 | 60.00% | 80.00% | 1 | 2 | 2 |
| safety gate rejected | 1 | 0.00% | 100.00% | 0 | 1 | 0 |
| unresolved state target time | 1 | 0.00% | 0.00% | 0 | 0 | 1 |
| unsupported temporal operation | 1 | 100.00% | 100.00% | 0 | 0 | 0 |

The reader used 46,322 tokens and 94.82 seconds total; judging used 13,700
tokens and 97.01 seconds total. This controlled result rejects direct packet
takeover for non-executable questions. The packet remains useful as a
diagnostic and abstention signal, while answer generation should fall back to
D0 unless a verifier confirms an executable, fully grounded constraint.

Artifact:

```text
outputs/qa/hypathmem_8.4time_v/
  e2_nonexecutable35_packetreader_gpt41mini_judge_gpt4omini.json
```
- No source-Q4 safe candidate is lost.

The first build therefore validates the representation and increases role
coverage, but does not yet add a safe takeover. The remaining bottlenecks are
generic list-query roles, missing semantic occurrence times, and evidence that
lies outside Top50. H4 must be evaluated with an external candidate source
before claiming retrieval or accuracy gains.

## Reproduction

```bash
python scripts/142_build_hypathmem_8_4time_v.py \
  --compiled outputs/reconstruction/hypathmem_temporal_v0_2_qwen/paired20_q1_q4_compiled.json \
  --compiled outputs/reconstruction/hypathmem_temporal_v0_2_qwen/holdout113_q1_q4_compiled.json \
  --output outputs/reconstruction/hypathmem_8.4time_v/longmemeval_temporal133_h1_h5.json
```

The output records source hashes and explicitly declares that no benchmark
answer or gold evidence is consumed.
