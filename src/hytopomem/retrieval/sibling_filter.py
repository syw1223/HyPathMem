from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from hytopomem.memory.node_extractor import content_terms
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeStatus, NodeType, RelationType
from hytopomem.retrieval.bm25_retriever import BM25Retriever
from hytopomem.retrieval.cross_encoder_reranker import RerankCandidate


_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_ENTITY_STOPWORDS = {
    "What",
    "When",
    "Where",
    "Which",
    "Would",
    "Could",
    "Should",
    "The",
    "How",
    "Why",
}


@dataclass(frozen=True)
class SiblingFilterConfig:
    seed_topn: int = 20
    max_anchor_degree: int = 50
    sibling_limit_per_anchor: int = 40
    preselect_topn: int = 100
    min_bm25_norm: float = 0.0
    require_entity_overlap: bool = False
    w_bm25: float = 1.0
    w_seed: float = 0.8
    w_entity: float = 0.6
    w_edge: float = 0.4
    w_time: float = 0.3
    w_degree: float = 0.4
    w_hop: float = 0.3


@dataclass
class FilteredSiblingCandidate:
    node: Node
    score: float
    bm25_score: float
    is_seed: bool
    entity_overlap: float
    edge_confidence: float
    anchor_degree: int
    hop: int
    seed_node_id: str
    anchor_node_id: str | None = None


class FilteredSiblingRetriever:
    def __init__(self, graph: MemoryGraph, config: SiblingFilterConfig | None = None):
        self.graph = graph
        self.config = config or SiblingFilterConfig()
        self.facts = list(graph.iter_nodes(NodeType.FACT))
        self.fact_retriever = BM25Retriever(self.facts)
        self._incoming: Dict[str, List[Edge]] = {}
        self._outgoing: Dict[str, List[Edge]] = {}
        for edge in graph.edges:
            self._incoming.setdefault(edge.dst, []).append(edge)
            self._outgoing.setdefault(edge.src, []).append(edge)

    def candidates(self, query: str) -> list[RerankCandidate]:
        seed_hits = self.fact_retriever.search(query, top_k=self.config.seed_topn)
        bm25_scores = self._normalized_bm25_scores(query, top_k=max(self.config.preselect_topn * 4, 300))
        query_entities = extract_entities(query)
        query_terms = set(content_terms(query))
        candidates: Dict[str, FilteredSiblingCandidate] = {}

        for seed, seed_score in seed_hits:
            self._upsert(
                candidates,
                self._score_candidate(
                    node=seed,
                    query_entities=query_entities,
                    query_terms=query_terms,
                    bm25_score=bm25_scores.get(seed.node_id, 0.0),
                    is_seed=True,
                    edge_confidence=1.0,
                    anchor_degree=0,
                    hop=0,
                    seed_node_id=seed.node_id,
                    anchor_node_id=None,
                ),
            )
            for anchor_edge in self._seed_anchor_edges(seed.node_id):
                anchor_degree = self._anchor_degree(anchor_edge.dst)
                if anchor_degree > self.config.max_anchor_degree:
                    continue
                sibling_edges = [
                    edge
                    for edge in self._incoming.get(anchor_edge.dst, [])
                    if edge.relation == RelationType.IS_SPECIFIC_OF and edge.src != seed.node_id
                ]
                sibling_edges = sorted(sibling_edges, key=lambda edge: edge.confidence, reverse=True)
                for sibling_edge in sibling_edges[: self.config.sibling_limit_per_anchor]:
                    sibling = self.graph.nodes.get(sibling_edge.src)
                    if sibling is None or sibling.type != NodeType.FACT:
                        continue
                    bm25_score = bm25_scores.get(sibling.node_id, 0.0)
                    if bm25_score < self.config.min_bm25_norm:
                        continue
                    candidate = self._score_candidate(
                        node=sibling,
                        query_entities=query_entities,
                        query_terms=query_terms,
                        bm25_score=bm25_score,
                        is_seed=False,
                        edge_confidence=min(anchor_edge.confidence, sibling_edge.confidence),
                        anchor_degree=anchor_degree,
                        hop=2,
                        seed_node_id=seed.node_id,
                        anchor_node_id=anchor_edge.dst,
                    )
                    if self.config.require_entity_overlap and candidate.entity_overlap <= 0:
                        continue
                    self._upsert(candidates, candidate)

        selected = sorted(candidates.values(), key=lambda item: item.score, reverse=True)[: self.config.preselect_topn]
        return [
            RerankCandidate(
                node=candidate.node,
                base_score=candidate.score,
                path_node_ids=[candidate.node.node_id],
                metadata={
                    "candidate_source": "filtered_sibling",
                    "is_seed": str(candidate.is_seed),
                    "seed_node_id": candidate.seed_node_id,
                    "anchor_node_id": candidate.anchor_node_id or "",
                    "bm25_norm": f"{candidate.bm25_score:.6f}",
                    "entity_overlap": f"{candidate.entity_overlap:.6f}",
                    "anchor_degree": str(candidate.anchor_degree),
                    "hop": str(candidate.hop),
                },
            )
            for candidate in selected
        ]

    def _normalized_bm25_scores(self, query: str, top_k: int) -> Dict[str, float]:
        hits = self.fact_retriever.search(query, top_k=top_k)
        if not hits:
            return {}
        max_score = max(score for _node, score in hits) or 1.0
        return {node.node_id: score / max_score for node, score in hits}

    def _seed_anchor_edges(self, seed_id: str) -> list[Edge]:
        return [
            edge
            for edge in self._outgoing.get(seed_id, [])
            if edge.relation == RelationType.IS_SPECIFIC_OF
            and self.graph.nodes.get(edge.dst) is not None
            and self.graph.nodes[edge.dst].type == NodeType.ANCHOR
        ]

    def _anchor_degree(self, anchor_id: str) -> int:
        return sum(
            1
            for edge in self._incoming.get(anchor_id, [])
            if edge.relation == RelationType.IS_SPECIFIC_OF
        )

    def _score_candidate(
        self,
        *,
        node: Node,
        query_entities: set[str],
        query_terms: set[str],
        bm25_score: float,
        is_seed: bool,
        edge_confidence: float,
        anchor_degree: int,
        hop: int,
        seed_node_id: str,
        anchor_node_id: str | None,
    ) -> FilteredSiblingCandidate:
        c = self.config
        entity_overlap = entity_overlap_score(query_entities, query_terms, node.text)
        temporal_score = 1.0 if node.status == NodeStatus.ACTIVE else 0.2
        score = (
            c.w_bm25 * bm25_score
            + c.w_seed * float(is_seed)
            + c.w_entity * entity_overlap
            + c.w_edge * edge_confidence
            + c.w_time * temporal_score
            - c.w_degree * math.log1p(anchor_degree)
            - c.w_hop * hop
        )
        return FilteredSiblingCandidate(
            node=node,
            score=score,
            bm25_score=bm25_score,
            is_seed=is_seed,
            entity_overlap=entity_overlap,
            edge_confidence=edge_confidence,
            anchor_degree=anchor_degree,
            hop=hop,
            seed_node_id=seed_node_id,
            anchor_node_id=anchor_node_id,
        )

    def _upsert(self, candidates: Dict[str, FilteredSiblingCandidate], candidate: FilteredSiblingCandidate) -> None:
        existing = candidates.get(candidate.node.node_id)
        if existing is None or candidate.score > existing.score:
            candidates[candidate.node.node_id] = candidate


def extract_entities(text: str) -> set[str]:
    return {token for token in _ENTITY_RE.findall(text) if token not in _ENTITY_STOPWORDS}


def entity_overlap_score(query_entities: set[str], query_terms: set[str], text: str) -> float:
    node_entities = extract_entities(text)
    if query_entities and node_entities:
        return len(query_entities & node_entities) / len(query_entities)
    node_terms = set(content_terms(text))
    if not query_terms or not node_terms:
        return 0.0
    return min(1.0, len(query_terms & node_terms) / max(len(query_terms), 1))

