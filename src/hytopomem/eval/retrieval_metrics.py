from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from hytopomem.memory.schema import MemoryGraph, Node, NodeType


@dataclass(frozen=True)
class RetrievalEvalResult:
    question_id: str
    gold_evidence_ids: List[str]
    selected_path_node_ids: List[str]
    matched_evidence_ids: List[str]
    hit: bool
    recall: float
    full_cover: bool
    tokens: int
    path_len: float


def normalize_evidence_id(value: str) -> str:
    value = str(value).strip()
    if ":raw:" in value:
        return value.rsplit(":raw:", 1)[1]
    if ":fact:" in value:
        return value.rsplit(":fact:", 1)[1]
    return value


def conversation_id_from_question(question_id: str) -> str:
    return question_id.split(":q", 1)[0]


def gold_raw_node_id(question_id: str, evidence_id: str) -> str:
    return f"{conversation_id_from_question(question_id)}:raw:{normalize_evidence_id(evidence_id)}"


def build_gold_raw_map(items: Sequence[dict]) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    for item in items:
        question_id = item["question_id"]
        gold = [normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])]
        mapping[question_id] = {
            "gold_evidence_ids": gold,
            "gold_raw_node_ids": [gold_raw_node_id(question_id, eid) for eid in gold],
        }
    return mapping


def evidence_ids_for_node(graph: MemoryGraph, node_id: str) -> Set[str]:
    node = graph.nodes.get(node_id)
    if node is None:
        return set()
    evidence_ids: Set[str] = set()
    if node.type == NodeType.RAW:
        turn_id = node.metadata.get("turn_id") or node.node_id.rsplit(":raw:", 1)[-1]
        evidence_ids.add(normalize_evidence_id(str(turn_id)))
    elif node.type == NodeType.FACT:
        own_turn = node.metadata.get("turn_id")
        if own_turn:
            evidence_ids.add(normalize_evidence_id(str(own_turn)))
        for support_id in node.support_ids:
            evidence_ids.add(normalize_evidence_id(support_id))
    return evidence_ids


def selected_node_ids(paths: Sequence[dict], k: int) -> List[str]:
    node_ids: List[str] = []
    seen = set()
    for path in paths[:k]:
        for node_id in path.get("node_ids", []):
            if node_id not in seen:
                seen.add(node_id)
                node_ids.append(node_id)
    return node_ids


def count_tokens(graph: MemoryGraph, node_ids: Iterable[str]) -> int:
    total = 0
    for node_id in node_ids:
        node = graph.nodes.get(node_id)
        if node is not None:
            total += len(node.text.split())
    return total


def average_path_len(paths: Sequence[dict], k: int) -> float:
    selected = paths[:k]
    if not selected:
        return 0.0
    return sum(len(path.get("node_ids", [])) for path in selected) / len(selected)


def evaluate_item(graph: MemoryGraph, item: dict, k: int) -> RetrievalEvalResult:
    gold = {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}
    paths = item.get("paths", [])
    node_ids = selected_node_ids(paths, k)
    predicted: Set[str] = set()
    for node_id in node_ids:
        predicted.update(evidence_ids_for_node(graph, node_id))
    matched = sorted(gold & predicted)
    recall = len(matched) / len(gold) if gold else 0.0
    return RetrievalEvalResult(
        question_id=item["question_id"],
        gold_evidence_ids=sorted(gold),
        selected_path_node_ids=node_ids,
        matched_evidence_ids=matched,
        hit=bool(matched),
        recall=recall,
        full_cover=bool(gold) and gold.issubset(predicted),
        tokens=count_tokens(graph, node_ids),
        path_len=average_path_len(paths, k),
    )


def summarize(results: Sequence[RetrievalEvalResult]) -> dict:
    if not results:
        return {
            "num_questions": 0,
            "hit": 0.0,
            "recall": 0.0,
            "full_cover": 0.0,
            "avg_tokens": 0.0,
            "avg_path_len": 0.0,
        }
    n = len(results)
    return {
        "num_questions": n,
        "hit": sum(float(item.hit) for item in results) / n,
        "recall": sum(item.recall for item in results) / n,
        "full_cover": sum(float(item.full_cover) for item in results) / n,
        "avg_tokens": sum(item.tokens for item in results) / n,
        "avg_path_len": sum(item.path_len for item in results) / n,
    }

