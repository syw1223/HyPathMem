from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from hytopomem.geometry.lorentz import LorentzManifold, cone_violation
from hytopomem.memory.node_extractor import content_terms
from hytopomem.memory.schema import Edge, EvidencePath, MemoryGraph, NodeType, RelationType
from hytopomem.models.hyperbolic_mapper import HyperbolicMapper
from hytopomem.models.path_scorer import PathScoreWeights


def lexical_overlap(query: str, texts: Iterable[str]) -> float:
    q_terms = set(content_terms(query))
    p_terms = set()
    for text in texts:
        p_terms.update(content_terms(text))
    if not q_terms or not p_terms:
        return 0.0
    return len(q_terms & p_terms) / len(q_terms)


@dataclass
class HeuristicPathSelector:
    graph: MemoryGraph
    mapper: HyperbolicMapper
    weights: PathScoreWeights = PathScoreWeights()

    def select(
        self,
        query_id: str,
        query: str,
        paths: Sequence[tuple[list[str], list[str]]],
        top_k: int = 5,
    ) -> List[EvidencePath]:
        if not paths:
            return []
        path_node_ids = sorted({node_id for node_ids, _edge_ids in paths for node_id in node_ids})
        path_nodes = [self.graph.nodes[node_id] for node_id in path_node_ids if node_id in self.graph.nodes]
        node_embeddings = self.mapper.encode_nodes(path_nodes)
        query_embedding = self.mapper.encode_query(query)
        scored = [
            self._score_path(query_id, query, query_embedding, node_embeddings, node_ids, edge_ids)
            for node_ids, edge_ids in paths
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _score_path(
        self,
        query_id: str,
        query: str,
        query_embedding: np.ndarray,
        node_embeddings: Dict[str, np.ndarray],
        node_ids: list[str],
        edge_ids: list[str],
    ) -> EvidencePath:
        nodes = [self.graph.nodes[node_id] for node_id in node_ids if node_id in self.graph.nodes]
        texts = [node.text for node in nodes]
        rel = lexical_overlap(query, texts)
        geo = self._geometry_score(query_embedding, node_embeddings, node_ids)
        temp = sum(1.0 for node in nodes if node.status == "active") / max(len(nodes), 1)
        support = sum(1.0 for node in nodes if node.type in {NodeType.FACT, NodeType.RAW}) / max(len(nodes), 1)
        conflict = self._relation_fraction(edge_ids, RelationType.CONFLICTS_WITH)
        redundancy = 1.0 - (len(set(texts)) / max(len(texts), 1))
        token_cost = sum(len(text.split()) for text in texts)
        length_cost = len(node_ids)
        score = (
            self.weights.relevance * rel
            + self.weights.geometry * geo
            + self.weights.temporal * temp
            + self.weights.support * support
            - self.weights.conflict * conflict
            - self.weights.redundancy * redundancy
            - self.weights.token_cost * token_cost
            - self.weights.length_cost * length_cost
        )
        anchor_id = node_ids[0] if node_ids else None
        return EvidencePath(
            query_id=query_id,
            anchor_id=anchor_id,
            node_ids=node_ids,
            edge_ids=edge_ids,
            score=score,
            scores={
                "relevance": rel,
                "geometry": geo,
                "temporal": temp,
                "support": support,
                "conflict": conflict,
                "redundancy": redundancy,
                "token_cost": float(token_cost),
                "length_cost": float(length_cost),
            },
        )

    def _geometry_score(
        self,
        query_embedding: np.ndarray,
        node_embeddings: Dict[str, np.ndarray],
        node_ids: list[str],
    ) -> float:
        manifold = LorentzManifold(curvature=self.mapper.curvature)
        if not node_ids:
            return 0.0
        distances = [
            float(manifold.distance(query_embedding, node_embeddings[node_id]))
            for node_id in node_ids
            if node_id in node_embeddings
        ]
        if not distances:
            return 0.0
        distance_score = 1.0 / (1.0 + sum(distances) / len(distances))
        violations = []
        for edge in self._edges_from_ids(node_ids):
            if edge.relation != RelationType.IS_SPECIFIC_OF:
                continue
            parent = self.graph.nodes[edge.dst]
            violations.append(
                cone_violation(
                    node_embeddings[edge.dst],
                    node_embeddings[edge.src],
                    parent.confidence,
                )
            )
        violation_score = 1.0 / (1.0 + sum(violations) / max(len(violations), 1))
        return 0.5 * distance_score + 0.5 * violation_score

    def _edges_from_ids(self, node_ids: list[str]) -> List[Edge]:
        node_set = set(node_ids)
        return [edge for edge in self.graph.edges if edge.src in node_set and edge.dst in node_set]

    def _relation_fraction(self, edge_ids: list[str], relation: RelationType) -> float:
        edge_id_set = set(edge_ids)
        if not edge_id_set:
            return 0.0
        matching = [
            edge
            for edge in self.graph.edges
            if (edge.edge_id in edge_id_set and edge.relation == relation)
        ]
        return len(matching) / len(edge_id_set)
