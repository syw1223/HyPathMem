# Experiment Protocol

## Benchmarks

### LoCoMo

- Evaluation set: 1,540 questions from Cat1-Cat4.
- Categories: single-hop, temporal, open-domain, and multi-hop.
- Memory isolation: one memory graph per conversation.
- Final HyPathMem evidence budget: Top20.

### LongMemEval-S

- Evaluation set: 500 questions across six official types.
- Types: single-session user, single-session assistant, knowledge update,
  temporal, multi-session, and preference.
- Memory isolation: one graph per unique haystack history.
- Retrieval budgets are stage-specific: **150 initial candidates -> Top-50
  selector/reconstruction pool -> token-budgeted QA context from up to 50
  evidence/path units**. The first value bounds the fused multi-route
  candidate set. Top-50 is frozen for evidence reconstruction and temporal
  grounding; the compiler drops low-ranked whole units only when its context
  budget is exceeded. Top-20 is reported as a common retrieval metric cutoff
  and must not be interpreted as the final 8.5 reader truncation.

## Unified Model Setting

| Role | Model |
| --- | --- |
| Process LLM for extraction, cards, and structured reasoning | Qwen3-30B-A3B-Instruct-2507 |
| Dense embedding | Qwen3-Embedding-0.6B |
| Answer generator, setting A | GPT-4.1-mini |
| Answer generator, setting B | Qwen3-30B-A3B-Instruct-2507 |
| Judge | GPT-4o-mini, one judgment per answer |

Reranking follows the recorded configuration for each experiment. Controlled
BM25/graph baselines use the same fixed candidate and final evidence budgets
reported in the paper tables.

## HyPathMem Pipeline

1. Convert each history into provenance-preserving semantic units.
2. Build a Topic/Episode/Event/Fact hierarchy and typed graph relations.
3. Retrieve four candidate routes: BU-E, BU-H, TD-E, and TD-H.
4. Construct a query-conditioned Relation Card.
5. Expand local evidence around Card roles and supporting facts.
6. Fuse semantic, route, agreement, and relation features in the HyperPath
   selector.
7. Restore exact raw support and speaker/session/time provenance.
8. Route temporal questions through typed temporal grounding and verification.
9. Generate an answer from the compiled evidence and judge it once.

## Retrieval Metrics

- **Hit@K:** at least one gold evidence item occurs in the first K items.
- **Recall@K:** fraction of all gold evidence items present in the first K.
- **FullCover@K:** all gold evidence items occur in the first K.

Metrics are macro-averaged over questions and reported as percentages.

## Baseline Boundaries

- **FullText:** raw history, no memory selection.
- **NaiveRAG:** dense retrieval over raw chunks/messages.
- **BM25 Fact + CE:** lexical fact seeds followed by a cross-encoder.
- **Graph Retrieval:** fixed event/topic expansion followed by reranking.
- Published memory baselines retain their own memory representation and
  retrieval method while using the unified answer/judge reporting protocol.

## Reproducibility Notes

- Keep dataset histories isolated.
- Freeze candidate pools before answer-generation comparisons.
- Use identical retrieved evidence across generator comparisons.
- Record build and online stages separately.
- Record model names, context limits, TopK values, and judge repetitions.
- Save per-question predictions and judgments before computing aggregates.
