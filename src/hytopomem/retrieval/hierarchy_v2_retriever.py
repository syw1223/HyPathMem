from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable

from hytopomem.memory.node_extractor import content_terms
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType
from hytopomem.retrieval.bm25_retriever import BM25Retriever
from hytopomem.retrieval.cross_encoder_reranker import RerankCandidate


@dataclass(frozen=True)
class HierarchyV2RetrieverConfig:
    seed_topn: int = 20
    preselect_topn: int = 100
    mode: str = "event_topic"
    same_event_limit: int = 30
    topic_event_limit: int = 8
    topic_fact_limit_per_event: int = 12
    max_topic_degree: int = 160
    min_bm25_norm: float = 0.0
    w_bm25: float = 1.0
    w_seed: float = 0.8
    w_event: float = 0.65
    w_topic: float = 0.35
    w_overlap: float = 0.45
    w_edge: float = 0.35
    w_degree: float = 0.25
    w_hop: float = 0.20


@dataclass
class HierarchyCandidate:
    node: Node
    score: float
    bm25_norm: float
    is_seed: bool
    source: str
    seed_node_id: str
    event_node_id: str
    topic_node_id: str
    hop: int
    edge_confidence: float
    event_degree: int
    topic_degree: int
    path_node_ids: list[str]


class HierarchyV2Retriever:
    def __init__(self, graph: MemoryGraph, config: HierarchyV2RetrieverConfig | None = None):
        self.graph = graph
        self.config = config or HierarchyV2RetrieverConfig()
        self.facts = list(graph.iter_nodes(NodeType.FACT))
        self.fact_retriever = BM25Retriever(self.facts)
        self._incoming: Dict[str, list[Edge]] = {}
        self._outgoing: Dict[str, list[Edge]] = {}
        for edge in graph.edges:
            self._incoming.setdefault(edge.dst, []).append(edge)
            self._outgoing.setdefault(edge.src, []).append(edge)
        self.fact_to_event = self._fact_to_event()
        self.event_to_topic = self._event_to_topic()
        self.event_to_facts = self._event_to_facts()
        self.topic_to_events = self._topic_to_events()

    def candidate_node_ids(self, query: str) -> list[str]:
        return [candidate.node.node_id for candidate in self._ranked_candidates(query)]

    def candidates(self, query: str) -> list[RerankCandidate]:
        return [
            RerankCandidate(
                node=candidate.node,
                base_score=candidate.score,
                path_node_ids=candidate.path_node_ids,
                metadata={
                    "candidate_source": candidate.source,
                    "is_seed": str(candidate.is_seed),
                    "seed_node_id": candidate.seed_node_id,
                    "event_node_id": candidate.event_node_id,
                    "topic_node_id": candidate.topic_node_id,
                    "bm25_norm": f"{candidate.bm25_norm:.6f}",
                    "hop": str(candidate.hop),
                    "edge_confidence": f"{candidate.edge_confidence:.6f}",
                    "event_degree": str(candidate.event_degree),
                    "topic_degree": str(candidate.topic_degree),
                    "hierarchy_v2_mode": self.config.mode,
                },
            )
            for candidate in self._ranked_candidates(query)
        ]

    def candidate_node_ids_from_seed_hits(
        self,
        query: str,
        seed_hits: list[tuple[Node, float]],
        bm25_scores: dict[str, float],
    ) -> list[str]:
        return [
            candidate.node.node_id
            for candidate in self._ranked_candidates_from_seed_hits(query, seed_hits, bm25_scores)
        ]

    def _ranked_candidates(self, query: str) -> list[HierarchyCandidate]:
        seed_hits = self.fact_retriever.search(query, top_k=self.config.seed_topn)
        bm25_scores = self._normalized_bm25_scores(query, top_k=max(self.config.preselect_topn * 4, 300))
        return self._ranked_candidates_from_seed_hits(query, seed_hits, bm25_scores)

    def _ranked_candidates_from_seed_hits(
        self,
        query: str,
        seed_hits: list[tuple[Node, float]],
        bm25_scores: dict[str, float],
    ) -> list[HierarchyCandidate]:
        query_terms = set(content_terms(query))
        candidates: dict[str, HierarchyCandidate] = {}
        for seed, _seed_score in seed_hits:
            event_id = self.fact_to_event.get(seed.node_id, "")
            topic_id = self.event_to_topic.get(event_id, "") if event_id else ""
            self._upsert(
                candidates,
                self._make_candidate(
                    node=seed,
                    query_terms=query_terms,
                    bm25_norm=bm25_scores.get(seed.node_id, 0.0),
                    is_seed=True,
                    source="seed",
                    seed_node_id=seed.node_id,
                    event_node_id=event_id,
                    topic_node_id=topic_id,
                    hop=0,
                    edge_confidence=1.0,
                    path_node_ids=[seed.node_id],
                ),
            )
            if event_id and self.config.mode in {"same_event", "event_topic"}:
                for fact_id in self.event_to_facts.get(event_id, [])[: self.config.same_event_limit]:
                    if fact_id == seed.node_id:
                        continue
                    fact = self.graph.nodes.get(fact_id)
                    if fact is not None and fact.type == NodeType.FACT:
                        self._upsert(
                            candidates,
                            self._make_candidate(
                                node=fact,
                                query_terms=query_terms,
                                bm25_norm=bm25_scores.get(fact.node_id, 0.0),
                                is_seed=False,
                                source="same_event",
                                seed_node_id=seed.node_id,
                                event_node_id=event_id,
                                topic_node_id=topic_id,
                                hop=2,
                                edge_confidence=self._fact_event_confidence(fact.node_id, event_id),
                                path_node_ids=[seed.node_id, event_id, fact.node_id],
                            ),
                        )
            if topic_id and self.config.mode in {"same_topic", "same_topic_only", "event_topic"}:
                topic_degree = self._topic_degree(topic_id)
                if topic_degree > self.config.max_topic_degree:
                    continue
                topic_events = self._rank_topic_events(topic_id, seed.node_id, query_terms)
                for event_id2 in topic_events[: self.config.topic_event_limit]:
                    if self.config.mode == "same_topic_only" and event_id2 == event_id:
                        continue
                    fact_ids = self.event_to_facts.get(event_id2, [])[: self.config.topic_fact_limit_per_event]
                    for fact_id in fact_ids:
                        if fact_id == seed.node_id:
                            continue
                        fact = self.graph.nodes.get(fact_id)
                        bm25_norm = bm25_scores.get(fact_id, 0.0)
                        if fact is None or fact.type != NodeType.FACT or bm25_norm < self.config.min_bm25_norm:
                            continue
                        source = "same_event" if event_id2 == self.fact_to_event.get(seed.node_id) else "same_topic"
                        self._upsert(
                            candidates,
                            self._make_candidate(
                                node=fact,
                                query_terms=query_terms,
                                bm25_norm=bm25_norm,
                                is_seed=False,
                                source=source,
                                seed_node_id=seed.node_id,
                                event_node_id=event_id2,
                                topic_node_id=topic_id,
                                hop=2 if source == "same_event" else 4,
                                edge_confidence=self._fact_event_confidence(fact_id, event_id2),
                                path_node_ids=[seed.node_id, self.fact_to_event.get(seed.node_id, ""), topic_id, event_id2, fact_id],
                            ),
                        )
        selected = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        return selected[: self.config.preselect_topn]

    def _make_candidate(
        self,
        *,
        node: Node,
        query_terms: set[str],
        bm25_norm: float,
        is_seed: bool,
        source: str,
        seed_node_id: str,
        event_node_id: str,
        topic_node_id: str,
        hop: int,
        edge_confidence: float,
        path_node_ids: list[str],
    ) -> HierarchyCandidate:
        c = self.config
        event_degree = len(self.event_to_facts.get(event_node_id, [])) if event_node_id else 0
        topic_degree = self._topic_degree(topic_node_id) if topic_node_id else 0
        overlap = overlap_score(query_terms, node.text)
        source_score = c.w_event if source == "same_event" else c.w_topic if source == "same_topic" else 0.0
        score = (
            c.w_bm25 * bm25_norm
            + c.w_seed * float(is_seed)
            + source_score
            + c.w_overlap * overlap
            + c.w_edge * edge_confidence
            - c.w_degree * math.log1p(max(event_degree, 1))
            - c.w_hop * hop
        )
        return HierarchyCandidate(
            node=node,
            score=score,
            bm25_norm=bm25_norm,
            is_seed=is_seed,
            source=source,
            seed_node_id=seed_node_id,
            event_node_id=event_node_id,
            topic_node_id=topic_node_id,
            hop=hop,
            edge_confidence=edge_confidence,
            event_degree=event_degree,
            topic_degree=topic_degree,
            path_node_ids=[node_id for node_id in path_node_ids if node_id],
        )

    def _normalized_bm25_scores(self, query: str, top_k: int) -> dict[str, float]:
        hits = self.fact_retriever.search(query, top_k=top_k)
        if not hits:
            return {}
        max_score = max(score for _node, score in hits) or 1.0
        return {node.node_id: score / max_score for node, score in hits}

    def _fact_to_event(self) -> dict[str, str]:
        mapping = {}
        for edge in self.graph.edges:
            if not is_hierarchy_edge(edge, "fact_event"):
                continue
            src = self.graph.nodes.get(edge.src)
            dst = self.graph.nodes.get(edge.dst)
            if src is not None and dst is not None and src.type == NodeType.FACT and dst.type == NodeType.EVENT:
                mapping[edge.src] = edge.dst
        return mapping

    def _event_to_topic(self) -> dict[str, str]:
        mapping = {}
        for edge in self.graph.edges:
            if not is_hierarchy_edge(edge, "event_topic"):
                continue
            src = self.graph.nodes.get(edge.src)
            dst = self.graph.nodes.get(edge.dst)
            if src is not None and dst is not None and src.type == NodeType.EVENT and dst.type == NodeType.TOPIC:
                mapping[edge.src] = edge.dst
        return mapping

    def _event_to_facts(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for fact_id, event_id in self.fact_to_event.items():
            mapping.setdefault(event_id, []).append(fact_id)
        return mapping

    def _topic_to_events(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for event_id, topic_id in self.event_to_topic.items():
            mapping.setdefault(topic_id, []).append(event_id)
        return mapping

    def _topic_degree(self, topic_id: str) -> int:
        total = 0
        for event_id in self.topic_to_events.get(topic_id, []):
            total += len(self.event_to_facts.get(event_id, []))
        return total

    def _fact_event_confidence(self, fact_id: str, event_id: str) -> float:
        for edge in self._outgoing.get(fact_id, []):
            if edge.dst == event_id and is_hierarchy_edge(edge, "fact_event"):
                return edge.confidence
        return 0.0

    def _rank_topic_events(self, topic_id: str, seed_id: str, query_terms: set[str]) -> list[str]:
        seed_event = self.fact_to_event.get(seed_id, "")
        events = self.topic_to_events.get(topic_id, [])
        return sorted(
            events,
            key=lambda event_id: (
                event_id == seed_event,
                overlap_score(query_terms, self.graph.nodes[event_id].text) if event_id in self.graph.nodes else 0.0,
                len(self.event_to_facts.get(event_id, [])),
            ),
            reverse=True,
        )

    def _upsert(self, candidates: dict[str, HierarchyCandidate], candidate: HierarchyCandidate) -> None:
        existing = candidates.get(candidate.node.node_id)
        if existing is None or candidate.score > existing.score:
            candidates[candidate.node.node_id] = candidate


def is_hierarchy_edge(edge: Edge, role: str) -> bool:
    return edge.relation == RelationType.IS_SPECIFIC_OF and edge.metadata.get("hierarchy_v2") == role


def overlap_score(query_terms: set[str], text: str) -> float:
    node_terms = set(content_terms(text))
    if not query_terms or not node_terms:
        return 0.0
    return min(1.0, len(query_terms & node_terms) / max(len(query_terms), 1))
