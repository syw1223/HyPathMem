from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from hytopomem.memory.schema import Edge, EvidencePath, MemoryGraph, Node, NodeStatus, NodeType, RelationType
from hytopomem.retrieval.bm25_retriever import BM25Retriever
from hytopomem.retrieval.mmr_selector import select_mmr


EDGE_WEIGHTS = {
    RelationType.SUPPORTS: 1.0,
    RelationType.IS_SPECIFIC_OF: 0.65,
    RelationType.UPDATES: 0.85,
    RelationType.EXCEPTION_OF: 0.8,
    RelationType.CONFLICTS_WITH: -0.6,
}


@dataclass
class SeededCandidate:
    node_id: str
    node_type: NodeType
    text: str
    score: float = 0.0
    seed_score: float = 0.0
    bm25_score: float = 0.0
    edge_score: float = 0.0
    time_score: float = 0.0
    type_score: float = 0.0
    noise_score: float = 0.0
    path_node_ids: List[str] = field(default_factory=list)
    path_edge_ids: List[str] = field(default_factory=list)
    seed_node_id: Optional[str] = None
    metadata: Dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class SeededGraphConfig:
    seed_topn: int = 20
    expand_hops: int = 1
    final_topk: int = 5
    use_mmr: bool = True
    lambda_div: float = 0.3
    sibling_limit_per_anchor: int = 8
    support_raw: bool = True
    update_exception: bool = True
    conflict: bool = False
    sibling_fact: bool = True
    w_seed: float = 1.0
    w_bm25: float = 0.8
    w_edge: float = 0.4
    w_time: float = 0.3
    w_type: float = 0.2
    w_noise: float = 0.5


class BM25SeededGraphRetriever:
    """Use flat evidence retrieval as seeds, then use graph topology for expansion.

    The graph is an indexing/expansion layer here. Final returned nodes are FACT/RAW
    evidence nodes; anchors are kept in paths for explanation but are not selected as
    final evidence.
    """

    def __init__(self, graph: MemoryGraph, config: SeededGraphConfig | None = None):
        self.graph = graph
        self.config = config or SeededGraphConfig()
        self.facts = list(graph.iter_nodes(NodeType.FACT))
        self.raws = list(graph.iter_nodes(NodeType.RAW))
        self.evidence_nodes = self.facts + self.raws
        self.fact_retriever = BM25Retriever(self.facts)
        self.evidence_retriever = BM25Retriever(self.evidence_nodes)
        self._incoming: Dict[str, List[Edge]] = {}
        self._outgoing: Dict[str, List[Edge]] = {}
        for edge in graph.edges:
            self._incoming.setdefault(edge.dst, []).append(edge)
            self._outgoing.setdefault(edge.src, []).append(edge)

    def retrieve(self, query_id: str, query: str) -> List[EvidencePath]:
        seeds = self.fact_retriever.search(query, top_k=self.config.seed_topn)
        if not seeds:
            return []
        max_seed = max(score for _node, score in seeds) or 1.0
        bm25_scores = self._normalized_bm25_scores(query, top_k=max(self.config.seed_topn * 4, 100))
        candidates: Dict[str, SeededCandidate] = {}

        for seed, raw_seed_score in seeds:
            seed_score = raw_seed_score / max_seed
            self._upsert_candidate(
                candidates,
                node=seed,
                seed_score=seed_score,
                bm25_score=bm25_scores.get(seed.node_id, 0.0),
                edge_score=1.0,
                path_node_ids=[seed.node_id],
                path_edge_ids=[],
                seed_node_id=seed.node_id,
            )
            if self.config.expand_hops >= 1:
                self._expand_one_hop(candidates, seed, seed_score, bm25_scores)
            if self.config.expand_hops >= 2:
                self._expand_two_hop(candidates, seed, seed_score, bm25_scores)

        evidence_candidates = [
            candidate
            for candidate in candidates.values()
            if candidate.node_type in {NodeType.FACT, NodeType.RAW}
        ]
        for candidate in evidence_candidates:
            self._finalize_score(candidate)

        ranked = sorted(evidence_candidates, key=lambda item: item.score, reverse=True)
        if self.config.use_mmr:
            selected = select_mmr(
                ranked,
                score_fn=lambda item: item.score,
                text_fn=lambda item: item.text,
                id_fn=lambda item: item.node_id,
                k=self.config.final_topk,
                lambda_div=self.config.lambda_div,
            )
        else:
            selected = ranked[: self.config.final_topk]
        return [self._candidate_to_path(query_id, candidate) for candidate in selected]

    def _normalized_bm25_scores(self, query: str, top_k: int) -> Dict[str, float]:
        hits = self.evidence_retriever.search(query, top_k=top_k)
        if not hits:
            return {}
        max_score = max(score for _node, score in hits) or 1.0
        return {node.node_id: score / max_score for node, score in hits}

    def _expand_one_hop(
        self,
        candidates: Dict[str, SeededCandidate],
        seed: Node,
        seed_score: float,
        bm25_scores: Dict[str, float],
    ) -> None:
        for edge in self._outgoing.get(seed.node_id, []):
            if self.config.support_raw and edge.relation == RelationType.SUPPORTS:
                raw = self.graph.nodes.get(edge.dst)
                if raw is not None and raw.type == NodeType.RAW:
                    self._upsert_candidate(
                        candidates,
                        node=raw,
                        seed_score=seed_score * 0.95,
                        bm25_score=bm25_scores.get(raw.node_id, 0.0),
                        edge_score=self._edge_score([edge]),
                        path_node_ids=[seed.node_id, raw.node_id],
                        path_edge_ids=[edge.edge_id or ""],
                        seed_node_id=seed.node_id,
                    )
            elif (
                self.config.update_exception
                and edge.relation in {RelationType.UPDATES, RelationType.EXCEPTION_OF}
            ) or (self.config.conflict and edge.relation == RelationType.CONFLICTS_WITH):
                neighbor = self.graph.nodes.get(edge.dst)
                if neighbor is not None and neighbor.type == NodeType.FACT:
                    self._upsert_candidate(
                        candidates,
                        node=neighbor,
                        seed_score=seed_score * 0.75,
                        bm25_score=bm25_scores.get(neighbor.node_id, 0.0),
                        edge_score=self._edge_score([edge]),
                        path_node_ids=[seed.node_id, neighbor.node_id],
                        path_edge_ids=[edge.edge_id or ""],
                        seed_node_id=seed.node_id,
                    )
        for edge in self._incoming.get(seed.node_id, []):
            if (
                self.config.update_exception
                and edge.relation in {RelationType.UPDATES, RelationType.EXCEPTION_OF}
            ) or (self.config.conflict and edge.relation == RelationType.CONFLICTS_WITH):
                neighbor = self.graph.nodes.get(edge.src)
                if neighbor is not None and neighbor.type == NodeType.FACT:
                    self._upsert_candidate(
                        candidates,
                        node=neighbor,
                        seed_score=seed_score * 0.75,
                        bm25_score=bm25_scores.get(neighbor.node_id, 0.0),
                        edge_score=self._edge_score([edge]),
                        path_node_ids=[seed.node_id, neighbor.node_id],
                        path_edge_ids=[edge.edge_id or ""],
                        seed_node_id=seed.node_id,
                    )

    def _expand_two_hop(
        self,
        candidates: Dict[str, SeededCandidate],
        seed: Node,
        seed_score: float,
        bm25_scores: Dict[str, float],
    ) -> None:
        if not self.config.sibling_fact:
            return
        parent_edges = [
            edge
            for edge in self._outgoing.get(seed.node_id, [])
            if edge.relation == RelationType.IS_SPECIFIC_OF
        ]
        for parent_edge in parent_edges:
            anchor = self.graph.nodes.get(parent_edge.dst)
            if anchor is None or anchor.type != NodeType.ANCHOR:
                continue
            siblings = [
                edge
                for edge in self._incoming.get(anchor.node_id, [])
                if edge.relation == RelationType.IS_SPECIFIC_OF and edge.src != seed.node_id
            ]
            siblings = sorted(siblings, key=lambda edge: edge.confidence, reverse=True)
            for sibling_edge in siblings[: self.config.sibling_limit_per_anchor]:
                sibling = self.graph.nodes.get(sibling_edge.src)
                if sibling is None or sibling.type != NodeType.FACT:
                    continue
                self._upsert_candidate(
                    candidates,
                    node=sibling,
                    seed_score=seed_score * 0.45,
                    bm25_score=bm25_scores.get(sibling.node_id, 0.0),
                    edge_score=self._edge_score([parent_edge, sibling_edge]),
                    path_node_ids=[seed.node_id, anchor.node_id, sibling.node_id],
                    path_edge_ids=[parent_edge.edge_id or "", sibling_edge.edge_id or ""],
                    seed_node_id=seed.node_id,
                )

    def _upsert_candidate(
        self,
        candidates: Dict[str, SeededCandidate],
        *,
        node: Node,
        seed_score: float,
        bm25_score: float,
        edge_score: float,
        path_node_ids: List[str],
        path_edge_ids: List[str],
        seed_node_id: str,
    ) -> None:
        time_score = self._time_score(node)
        type_score = self._type_score(node)
        noise_score = self._noise_score(node, path_edge_ids)
        existing = candidates.get(node.node_id)
        if existing is None:
            candidates[node.node_id] = SeededCandidate(
                node_id=node.node_id,
                node_type=node.type,
                text=node.text,
                seed_score=seed_score,
                bm25_score=bm25_score,
                edge_score=edge_score,
                time_score=time_score,
                type_score=type_score,
                noise_score=noise_score,
                path_node_ids=path_node_ids,
                path_edge_ids=path_edge_ids,
                seed_node_id=seed_node_id,
            )
            return
        if seed_score + bm25_score + edge_score > existing.seed_score + existing.bm25_score + existing.edge_score:
            existing.seed_score = seed_score
            existing.bm25_score = bm25_score
            existing.edge_score = edge_score
            existing.time_score = time_score
            existing.type_score = type_score
            existing.noise_score = noise_score
            existing.path_node_ids = path_node_ids
            existing.path_edge_ids = path_edge_ids
            existing.seed_node_id = seed_node_id

    def _finalize_score(self, candidate: SeededCandidate) -> None:
        c = self.config
        candidate.score = (
            c.w_seed * candidate.seed_score
            + c.w_bm25 * candidate.bm25_score
            + c.w_edge * candidate.edge_score
            + c.w_time * candidate.time_score
            + c.w_type * candidate.type_score
            - c.w_noise * candidate.noise_score
        )

    def _edge_score(self, edges: Sequence[Edge]) -> float:
        if not edges:
            return 0.0
        scores = [EDGE_WEIGHTS.get(edge.relation, 0.0) * edge.confidence for edge in edges]
        return sum(scores) / len(scores)

    def _time_score(self, node: Node) -> float:
        if node.status == NodeStatus.ACTIVE:
            return 1.0
        if node.status == NodeStatus.EXCEPTION:
            return 0.75
        if node.status == NodeStatus.DISPUTED:
            return 0.35
        return 0.1

    def _type_score(self, node: Node) -> float:
        if node.type == NodeType.FACT:
            return 1.0
        if node.type == NodeType.RAW:
            return 0.85
        return 0.0

    def _noise_score(self, node: Node, edge_ids: Sequence[str]) -> float:
        score = 0.0
        if node.status in {NodeStatus.OUTDATED, NodeStatus.DISPUTED}:
            score += 0.5
        if any("CONFLICTS_WITH" in edge_id for edge_id in edge_ids):
            score += 0.7
        score += max(0, len(edge_ids) - 2) * 0.15
        return score

    def _candidate_to_path(self, query_id: str, candidate: SeededCandidate) -> EvidencePath:
        return EvidencePath(
            query_id=query_id,
            anchor_id=None,
            node_ids=candidate.path_node_ids,
            edge_ids=candidate.path_edge_ids,
            score=candidate.score,
            scores={
                "seed": candidate.seed_score,
                "bm25": candidate.bm25_score,
                "edge": candidate.edge_score,
                "time": candidate.time_score,
                "type": candidate.type_score,
                "noise": candidate.noise_score,
            },
            metadata={
                "retriever": "bm25_seeded_graph",
                "seed_node_id": candidate.seed_node_id or "",
                "evidence_node_id": candidate.node_id,
                "evidence_node_type": candidate.node_type.value,
            },
        )
