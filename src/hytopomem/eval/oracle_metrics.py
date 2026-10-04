from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

from hytopomem.eval.retrieval_metrics import count_tokens, evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.schema import MemoryGraph


@dataclass(frozen=True)
class OracleEvalResult:
    question_id: str
    gold_evidence_ids: List[str]
    candidate_node_ids: List[str]
    matched_evidence_ids: List[str]
    hit: bool
    recall: float
    full_cover: bool
    tokens: int
    avg_candidates: int


def evidence_ids_for_nodes(graph: MemoryGraph, node_ids: Iterable[str]) -> Set[str]:
    evidence_ids: Set[str] = set()
    for node_id in node_ids:
        evidence_ids.update(evidence_ids_for_node(graph, node_id))
    return evidence_ids


def evaluate_candidate_pool(
    graph: MemoryGraph,
    *,
    question_id: str,
    gold_evidence: Sequence[str],
    candidate_node_ids: Sequence[str],
) -> OracleEvalResult:
    gold = {normalize_evidence_id(eid) for eid in gold_evidence}
    predicted = evidence_ids_for_nodes(graph, candidate_node_ids)
    matched = sorted(gold & predicted)
    recall = len(matched) / len(gold) if gold else 0.0
    return OracleEvalResult(
        question_id=question_id,
        gold_evidence_ids=sorted(gold),
        candidate_node_ids=list(candidate_node_ids),
        matched_evidence_ids=matched,
        hit=bool(matched),
        recall=recall,
        full_cover=bool(gold) and gold.issubset(predicted),
        tokens=count_tokens(graph, candidate_node_ids),
        avg_candidates=len(candidate_node_ids),
    )


def summarize_oracle(results: Sequence[OracleEvalResult]) -> dict:
    if not results:
        return {
            "num_questions": 0,
            "hit": 0.0,
            "recall": 0.0,
            "full_cover": 0.0,
            "avg_tokens": 0.0,
            "avg_candidates": 0.0,
        }
    n = len(results)
    return {
        "num_questions": n,
        "hit": sum(float(item.hit) for item in results) / n,
        "recall": sum(item.recall for item in results) / n,
        "full_cover": sum(float(item.full_cover) for item in results) / n,
        "avg_tokens": sum(item.tokens for item in results) / n,
        "avg_candidates": sum(item.avg_candidates for item in results) / n,
    }

