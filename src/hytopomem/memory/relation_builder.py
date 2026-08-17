from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from hytopomem.memory.node_extractor import content_terms
from hytopomem.memory.schema import Edge, Node, NodeStatus, NodeType, RelationType


def jaccard_similarity(left: str, right: str) -> float:
    left_terms = set(content_terms(left))
    right_terms = set(content_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def jaccard_from_terms(left_terms: Set[str], right_terms: Set[str]) -> float:
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


@dataclass(frozen=True)
class RelationBuilderConfig:
    min_specific_score: float = 0.08
    min_support_score: float = 0.12
    max_anchor_links_per_fact: int = 3
    update_window: int = 20
    max_anchor_candidates: int = 40


class WeakRelationBuilder:
    """Build a typed graph with deterministic weak signals.

    This is the MVP relation induction layer. A later LLM/cross-encoder judge can
    replace `score_relation` while preserving the output schema.
    """

    def __init__(self, config: RelationBuilderConfig | None = None):
        self.config = config or RelationBuilderConfig()

    def build(self, nodes: Sequence[Node]) -> List[Edge]:
        raw_by_support = {node.node_id: node for node in nodes if node.type == NodeType.RAW}
        facts = [node for node in nodes if node.type == NodeType.FACT]
        anchors = [node for node in nodes if node.type == NodeType.ANCHOR]
        anchors_by_conv = self._bucket_by_conversation(anchors, marker=":anchor:")
        anchor_terms = {anchor.node_id: set(content_terms(anchor.text)) for anchor in anchors}
        fact_terms = {fact.node_id: set(content_terms(fact.text)) for fact in facts}
        anchor_indexes = {
            conv_id: self._build_anchor_index(bucket, anchor_terms)
            for conv_id, bucket in anchors_by_conv.items()
        }
        edges: List[Edge] = []

        for fact in facts:
            for raw_id in fact.support_ids:
                if raw_id in raw_by_support:
                    edges.append(
                        Edge(
                            src=fact.node_id,
                            dst=raw_id,
                            relation=RelationType.SUPPORTS,
                            confidence=0.98,
                        )
                    )

            conversation_id = fact.node_id.split(":fact:", 1)[0]
            candidate_anchors = anchors_by_conv.get(conversation_id, anchors)
            candidate_anchors = self._candidate_anchors(
                candidate_anchors,
                fact_terms[fact.node_id],
                anchor_indexes.get(conversation_id, {}),
            )
            ranked = sorted(
                (
                    (
                        anchor,
                        jaccard_from_terms(fact_terms[fact.node_id], anchor_terms[anchor.node_id]),
                    )
                    for anchor in candidate_anchors
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            for anchor, score in ranked[: self.config.max_anchor_links_per_fact]:
                if score >= self.config.min_specific_score:
                    edges.append(
                        Edge(
                            src=fact.node_id,
                            dst=anchor.node_id,
                            relation=RelationType.IS_SPECIFIC_OF,
                            confidence=min(0.95, 0.55 + score),
                            metadata={"similarity": score},
                        )
                    )
                elif score >= self.config.min_support_score:
                    edges.append(
                        Edge(
                            src=fact.node_id,
                            dst=anchor.node_id,
                            relation=RelationType.SUPPORTS,
                            confidence=min(0.9, 0.45 + score),
                            metadata={"similarity": score},
                        )
                    )

        edges.extend(self._build_update_and_conflict_edges(facts))
        return dedupe_edges(edges)

    def _build_update_and_conflict_edges(self, facts: Sequence[Node]) -> List[Edge]:
        edges: List[Edge] = []
        for bucket in self._bucket_by_conversation(facts, marker=":fact:").values():
            for idx, right in enumerate(bucket):
                start = max(0, idx - self.config.update_window)
                for left in bucket[start:idx]:
                    sim = jaccard_similarity(left.text, right.text)
                    if sim < 0.18:
                        continue
                    if right.status == NodeStatus.OUTDATED or self._looks_like_update(right.text):
                        edges.append(
                            Edge(
                                src=right.node_id,
                                dst=left.node_id,
                                relation=RelationType.UPDATES,
                                confidence=min(0.9, 0.5 + sim),
                                metadata={"similarity": sim},
                            )
                        )
                    elif self._looks_conflicting(left.text, right.text):
                        edges.append(
                            Edge(
                                src=right.node_id,
                                dst=left.node_id,
                                relation=RelationType.CONFLICTS_WITH,
                                confidence=min(0.85, 0.45 + sim),
                                metadata={"similarity": sim},
                            )
                        )
        return edges

    def _bucket_by_conversation(self, facts: Sequence[Node], marker: str) -> Dict[str, List[Node]]:
        buckets: Dict[str, List[Node]] = {}
        for fact in facts:
            conversation_id = fact.node_id.split(marker, 1)[0]
            buckets.setdefault(conversation_id, []).append(fact)
        return buckets

    def _build_anchor_index(
        self,
        anchors: Sequence[Node],
        anchor_terms: Dict[str, Set[str]],
    ) -> Dict[str, List[Node]]:
        index: Dict[str, List[Node]] = {}
        for anchor in anchors:
            for term in anchor_terms[anchor.node_id]:
                index.setdefault(term, []).append(anchor)
        return index

    def _candidate_anchors(
        self,
        fallback_anchors: Sequence[Node],
        terms: Set[str],
        anchor_index: Dict[str, List[Node]],
    ) -> List[Node]:
        counts: Dict[str, int] = {}
        by_id: Dict[str, Node] = {}
        for term in terms:
            for anchor in anchor_index.get(term, []):
                counts[anchor.node_id] = counts.get(anchor.node_id, 0) + 1
                by_id[anchor.node_id] = anchor
        if not counts:
            return list(fallback_anchors[: self.config.max_anchor_candidates])
        ranked_ids = sorted(counts, key=lambda node_id: counts[node_id], reverse=True)
        return [by_id[node_id] for node_id in ranked_ids[: self.config.max_anchor_candidates]]

    def _looks_like_update(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in ["changed", "updated", "now", "instead", "no longer"])

    def _looks_conflicting(self, left: str, right: str) -> bool:
        combined = f"{left.lower()} {right.lower()}"
        return any(token in combined for token in ["not ", "never", "contradict", "wrong"])


def dedupe_edges(edges: Iterable[Edge]) -> List[Edge]:
    seen = set()
    deduped: List[Edge] = []
    for edge in edges:
        key = (edge.src, edge.dst, edge.relation)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped
