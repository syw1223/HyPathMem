# HyPathMem 8.5time_v

> Reporting note: this document records the bundled 8.5 development snapshot
> (`408/500 = 81.60%`). The consolidated single-judge result used in the paper
> table is `412/500 = 82.40%`; its four additional correct judgments are not
> reconstructed from this older snapshot.

`8.5time_v` is an independent post-hoc development branch over the frozen
`8.4time_v` artifact. It converts a small subset of non-executable temporal
questions into executable constraints. It never sends a non-executable packet
to an answer reader and does not mutate the semantic graph, frozen Top50 packs,
or D0 answers.

## Decision rule

```text
complete and grounded constraint
  -> deterministic solver
  -> reverse verification
  -> Qwen Q4 verifier
  -> solver answer only if every check passes

otherwise -> frozen D0
```

## Failure funnel

The 35 structurally complete but non-executable `8.4time_v` questions separate
as follows:

| Failure type | N |
| --- | ---: |
| Missing occurred time with trusted identity and RAW | 12 |
| Operand missing or outside current Top50 binding | 12 |
| Event identity unreliable | 4 |
| Hypothesis conflict | 3 |
| Solver type unsupported | 3 |
| Safety gate rejected | 1 |

`Structural Operand Coverage` is 35/35. Only 2/35 already satisfy the stricter
`Validated Operand Coverage`; structural role filling must not be reported as
complete temporal evidence.

## Implemented conversion

- Qwen sees only the question, selected event identities, exact RAW quotes,
  message times, and allowed anchors; it never receives the gold answer.
- A time expression must occur verbatim in RAW.
- Message time is not treated as occurrence time without an explicit relative
  expression.
- Relative and absolute times are computed deterministically.
- `mid-February`, `last weekend`, `last month`, and `last year` remain bounded
  intervals rather than guessed timestamps.
- Ordering is executable only when all operand intervals are strictly disjoint.
- Symbolic offset propagation supports multi-edge chains and rejects cycles or
  inconsistent paths.
- Reverse verification recomputes ordering/arithmetic before Q4 verification.

## Conversion result

Twelve repairable questions were locally enriched with Qwen3-30B on GPU2:

- Qwen extraction calls: 12;
- extraction tokens: 17,452;
- extraction latency: 47.94 seconds total;
- validated new constraints: 3;
- new deterministic solver successes: 3;
- new Q4-verifier-safe candidates: 3/3;
- verifier tokens: 3,794;
- verifier latency: 9.18 seconds total.

GPT-4o-mini judged each new solver answer once:

- candidate accuracy: 3/3;
- frozen D0 on the same questions: 2/3;
- paired result: 1 fix, 0 breaks, 2 unchanged-correct;
- judge tokens: 1,083;
- judge latency: 10.04 seconds total.

The three converted questions concern vehicle ordering, book completion
ordering, and trip ordering. Their expressions are grounded as non-overlapping
calendar intervals rather than fabricated point dates.

## Aggregate post-hoc result

The prior Q4 branch scored 101/133. `8.4time_v` added one net fix, reaching
102/133. `8.5time_v` adds one further net fix without a break:

```text
Temporal133: 103/133 = 77.44%
D2 non-temporal + 8.5time_v: 408/500 = 81.60%
```

This must be reported as a **Temporal133 post-hoc development result**, not an
independent holdout result.

## Artifacts

```text
outputs/reconstruction/hypathmem_8.5time_v/
  non_executable35_failure_funnel_v2.json
  non_executable35_conversion_qwen_gpu2.json
  non_executable35_conversion_interval_v2_qwen_gpu2.json

outputs/qa/hypathmem_8.5time_v/
  new_verified3_solver_judge_gpt4omini.json
```

StateAt, H4 supplementary retrieval, and list/set solving remain inactive.
They require their own paired go/no-go experiments before entering routing.
