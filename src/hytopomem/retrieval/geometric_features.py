from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hytopomem.memory.schema import MemoryGraph, NodeType, RelationType
from hytopomem.models.hyperbolic_mapper import HyperbolicMapper
from hytopomem.models.text_encoder import HashTextEncoder


EUCLIDEAN_FEATURE_NAMES = [
    "euc_cos_query_candidate",
    "euc_cos_seed_candidate",
    "euc_cos_anchor_candidate",
    "euc_anchor_child_coherence",
]

HYPERBOLIC_FEATURE_NAMES = [
    "hyp_neg_dist_query_candidate",
    "hyp_neg_dist_seed_candidate",
    "hyp_neg_dist_anchor_candidate",
    "hyp_candidate_radius",
    "hyp_anchor_radius",
    "hyp_radius_gap_candidate_anchor",
    "hyp_radial_order_valid",
    "hyp_anchor_child_coherence",
]


@dataclass
class GeometryFeatureExtractor:
    graph: MemoryGraph
    dim: int = 32
    max_coherence_children: int = 50

    def __post_init__(self) -> None:
        self.euclidean_encoder = HashTextEncoder(dim=self.dim - 1)
        self.hyperbolic_mapper = HyperbolicMapper(dim=self.dim)
        self.manifold = self.hyperbolic_mapper.manifold
        self._node_euclidean = {
            node_id: self.euclidean_encoder.encode_one(node.text)
            for node_id, node in self.graph.nodes.items()
        }
        self._node_hyperbolic = {
            node_id: self.hyperbolic_mapper.encode_node(node)
            for node_id, node in self.graph.nodes.items()
        }
        self._anchor_children = self._build_anchor_children()
        self._euclidean_coherence = {
            anchor_id: self._mean_child_cosine(child_ids)
            for anchor_id, child_ids in self._anchor_children.items()
        }
        self._hyperbolic_coherence = {
            anchor_id: self._negative_mean_child_hyp_distance(child_ids)
            for anchor_id, child_ids in self._anchor_children.items()
        }

    def feature_names(self, kind: str) -> list[str]:
        if kind == "euclidean":
            return EUCLIDEAN_FEATURE_NAMES
        if kind == "hyperbolic":
            return HYPERBOLIC_FEATURE_NAMES
        if kind == "both":
            return EUCLIDEAN_FEATURE_NAMES + HYPERBOLIC_FEATURE_NAMES
        raise ValueError(f"unknown geometry feature kind: {kind}")

    def extract(self, item: dict, path: dict) -> dict[str, float]:
        metadata = path.get("metadata", {})
        candidate_id = evidence_node_id(path)
        seed_id = str(metadata.get("seed_node_id") or "")
        anchor_id = str(metadata.get("anchor_node_id") or "")
        query = item.get("question", "")

        query_euc = self.euclidean_encoder.encode_one(query)
        query_hyp = self.hyperbolic_mapper.encode_query(query)
        candidate_euc = self._node_euclidean.get(candidate_id)
        seed_euc = self._node_euclidean.get(seed_id)
        anchor_euc = self._node_euclidean.get(anchor_id)
        candidate_hyp = self._node_hyperbolic.get(candidate_id)
        seed_hyp = self._node_hyperbolic.get(seed_id)
        anchor_hyp = self._node_hyperbolic.get(anchor_id)

        candidate_radius = self._radius(candidate_hyp)
        anchor_radius = self._radius(anchor_hyp)
        radius_gap = candidate_radius - anchor_radius if anchor_hyp is not None and candidate_hyp is not None else 0.0

        return {
            "euc_cos_query_candidate": cosine(query_euc, candidate_euc),
            "euc_cos_seed_candidate": cosine(seed_euc, candidate_euc),
            "euc_cos_anchor_candidate": cosine(anchor_euc, candidate_euc),
            "euc_anchor_child_coherence": self._euclidean_coherence.get(anchor_id, 0.0),
            "hyp_neg_dist_query_candidate": -self._distance(query_hyp, candidate_hyp),
            "hyp_neg_dist_seed_candidate": -self._distance(seed_hyp, candidate_hyp),
            "hyp_neg_dist_anchor_candidate": -self._distance(anchor_hyp, candidate_hyp),
            "hyp_candidate_radius": candidate_radius,
            "hyp_anchor_radius": anchor_radius,
            "hyp_radius_gap_candidate_anchor": radius_gap,
            "hyp_radial_order_valid": float(radius_gap > 0.0),
            "hyp_anchor_child_coherence": self._hyperbolic_coherence.get(anchor_id, 0.0),
        }

    def _build_anchor_children(self) -> dict[str, list[str]]:
        children: dict[str, list[str]] = {}
        for edge in self.graph.edges:
            if edge.relation != RelationType.IS_SPECIFIC_OF:
                continue
            src = self.graph.nodes.get(edge.src)
            dst = self.graph.nodes.get(edge.dst)
            if src is None or dst is None or src.type != NodeType.FACT or dst.type != NodeType.ANCHOR:
                continue
            children.setdefault(edge.dst, []).append(edge.src)
        return children

    def _mean_child_cosine(self, child_ids: Sequence[str]) -> float:
        child_ids = list(child_ids)[: self.max_coherence_children]
        if len(child_ids) < 2:
            return 0.0
        values = [
            cosine(self._node_euclidean.get(left), self._node_euclidean.get(right))
            for left, right in itertools.combinations(child_ids, 2)
        ]
        return float(np.mean(values)) if values else 0.0

    def _negative_mean_child_hyp_distance(self, child_ids: Sequence[str]) -> float:
        child_ids = list(child_ids)[: self.max_coherence_children]
        if len(child_ids) < 2:
            return 0.0
        values = [
            self._distance(self._node_hyperbolic.get(left), self._node_hyperbolic.get(right))
            for left, right in itertools.combinations(child_ids, 2)
        ]
        return -float(np.mean(values)) if values else 0.0

    def _distance(self, left: np.ndarray | None, right: np.ndarray | None) -> float:
        if left is None or right is None:
            return 0.0
        return float(self.manifold.distance(left, right))

    def _radius(self, point: np.ndarray | None) -> float:
        if point is None:
            return 0.0
        return float(self.manifold.radius(point))


def cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""
