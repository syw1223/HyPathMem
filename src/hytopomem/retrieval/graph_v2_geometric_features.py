from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hytopomem.geometry.lorentz import LorentzManifold
from hytopomem.memory.schema import MemoryGraph, Node, NodeType, RelationType
from hytopomem.models.text_encoder import HashTextEncoder
from hytopomem.retrieval.geometric_features import evidence_node_id


GRAPH_V2_EUCLIDEAN_FEATURE_NAMES = [
    "g2_euc_cos_query_fact",
    "g2_euc_cos_query_event",
    "g2_euc_cos_query_topic",
    "g2_euc_cos_fact_event",
    "g2_euc_cos_event_topic",
    "g2_euc_event_child_coherence",
    "g2_euc_topic_event_coherence",
]

GRAPH_V2_HYPERBOLIC_FEATURE_NAMES = [
    "g2_hyp_neg_dist_query_fact",
    "g2_hyp_neg_dist_query_event",
    "g2_hyp_neg_dist_query_topic",
    "g2_hyp_neg_dist_fact_event",
    "g2_hyp_neg_dist_event_topic",
    "g2_hyp_fact_radius",
    "g2_hyp_event_radius",
    "g2_hyp_topic_radius",
    "g2_hyp_radius_gap_fact_event",
    "g2_hyp_radius_gap_event_topic",
    "g2_hyp_radial_order_fact_event",
    "g2_hyp_radial_order_event_topic",
    "g2_hyp_event_child_coherence",
    "g2_hyp_topic_event_coherence",
]


TYPE_RADIUS_SCALE = {
    NodeType.TOPIC: 0.36,
    NodeType.EVENT: 0.58,
    NodeType.FACT: 0.80,
    NodeType.RAW: 0.95,
    NodeType.ANCHOR: 0.42,
}


@dataclass
class GraphV2GeometryFeatureExtractor:
    graph: MemoryGraph
    dim: int = 32
    max_coherence_children: int = 50

    def __post_init__(self) -> None:
        self.encoder = HashTextEncoder(dim=self.dim - 1)
        self.manifold = LorentzManifold()
        self._node_euclidean = {
            node_id: self.encoder.encode_one(node.text)
            for node_id, node in self.graph.nodes.items()
        }
        self._node_hyperbolic = {
            node_id: self._encode_node(node)
            for node_id, node in self.graph.nodes.items()
        }
        self.fact_to_event, self.event_to_topic = self._build_hierarchy_maps()
        self.event_children = self._invert(self.fact_to_event)
        self.topic_events = self._invert(self.event_to_topic)
        self._event_euc_coherence = {
            event_id: self._mean_child_cosine(child_ids)
            for event_id, child_ids in self.event_children.items()
        }
        self._topic_euc_coherence = {
            topic_id: self._mean_child_cosine(event_ids)
            for topic_id, event_ids in self.topic_events.items()
        }
        self._event_hyp_coherence = {
            event_id: self._negative_mean_child_hyp_distance(child_ids)
            for event_id, child_ids in self.event_children.items()
        }
        self._topic_hyp_coherence = {
            topic_id: self._negative_mean_child_hyp_distance(event_ids)
            for topic_id, event_ids in self.topic_events.items()
        }

    def feature_names(self, kind: str = "both") -> list[str]:
        if kind == "euclidean":
            return GRAPH_V2_EUCLIDEAN_FEATURE_NAMES
        if kind == "hyperbolic":
            return GRAPH_V2_HYPERBOLIC_FEATURE_NAMES
        if kind == "both":
            return GRAPH_V2_EUCLIDEAN_FEATURE_NAMES + GRAPH_V2_HYPERBOLIC_FEATURE_NAMES
        raise ValueError(f"unknown graph v2 geometry kind: {kind}")

    def extract(self, item: dict, path: dict) -> dict[str, float]:
        metadata = path.get("metadata", {})
        fact_id = evidence_node_id(path)
        event_id = str(metadata.get("event_node_id") or self.fact_to_event.get(fact_id, ""))
        topic_id = str(metadata.get("topic_node_id") or self.event_to_topic.get(event_id, ""))
        query = item.get("question", "")

        query_euc = self.encoder.encode_one(query)
        query_hyp = self._encode_query(query)
        fact_euc = self._node_euclidean.get(fact_id)
        event_euc = self._node_euclidean.get(event_id)
        topic_euc = self._node_euclidean.get(topic_id)
        fact_hyp = self._node_hyperbolic.get(fact_id)
        event_hyp = self._node_hyperbolic.get(event_id)
        topic_hyp = self._node_hyperbolic.get(topic_id)

        fact_radius = self._radius(fact_hyp)
        event_radius = self._radius(event_hyp)
        topic_radius = self._radius(topic_hyp)
        gap_fact_event = fact_radius - event_radius if fact_hyp is not None and event_hyp is not None else 0.0
        gap_event_topic = event_radius - topic_radius if event_hyp is not None and topic_hyp is not None else 0.0

        return {
            "g2_euc_cos_query_fact": cosine(query_euc, fact_euc),
            "g2_euc_cos_query_event": cosine(query_euc, event_euc),
            "g2_euc_cos_query_topic": cosine(query_euc, topic_euc),
            "g2_euc_cos_fact_event": cosine(fact_euc, event_euc),
            "g2_euc_cos_event_topic": cosine(event_euc, topic_euc),
            "g2_euc_event_child_coherence": self._event_euc_coherence.get(event_id, 0.0),
            "g2_euc_topic_event_coherence": self._topic_euc_coherence.get(topic_id, 0.0),
            "g2_hyp_neg_dist_query_fact": -self._distance(query_hyp, fact_hyp),
            "g2_hyp_neg_dist_query_event": -self._distance(query_hyp, event_hyp),
            "g2_hyp_neg_dist_query_topic": -self._distance(query_hyp, topic_hyp),
            "g2_hyp_neg_dist_fact_event": -self._distance(fact_hyp, event_hyp),
            "g2_hyp_neg_dist_event_topic": -self._distance(event_hyp, topic_hyp),
            "g2_hyp_fact_radius": fact_radius,
            "g2_hyp_event_radius": event_radius,
            "g2_hyp_topic_radius": topic_radius,
            "g2_hyp_radius_gap_fact_event": gap_fact_event,
            "g2_hyp_radius_gap_event_topic": gap_event_topic,
            "g2_hyp_radial_order_fact_event": float(gap_fact_event > 0.0),
            "g2_hyp_radial_order_event_topic": float(gap_event_topic > 0.0),
            "g2_hyp_event_child_coherence": self._event_hyp_coherence.get(event_id, 0.0),
            "g2_hyp_topic_event_coherence": self._topic_hyp_coherence.get(topic_id, 0.0),
        }

    def _encode_node(self, node: Node) -> np.ndarray:
        tangent = self.encoder.encode_one(node.text)
        scale = TYPE_RADIUS_SCALE.get(node.type, 0.65)
        scale += 0.06 * max(0.0, min(1.0, node.confidence))
        return self.manifold.expmap0(tangent * scale)

    def _encode_query(self, query: str) -> np.ndarray:
        tangent = self.encoder.encode_one(query)
        return self.manifold.expmap0(tangent * 0.72)

    def _build_hierarchy_maps(self) -> tuple[dict[str, str], dict[str, str]]:
        fact_to_event: dict[str, str] = {}
        event_to_topic: dict[str, str] = {}
        for edge in self.graph.edges:
            if edge.relation != RelationType.IS_SPECIFIC_OF:
                continue
            src = self.graph.nodes.get(edge.src)
            dst = self.graph.nodes.get(edge.dst)
            if src is None or dst is None:
                continue
            role = edge.metadata.get("hierarchy_v2")
            if src.type == NodeType.FACT and dst.type == NodeType.EVENT and role == "fact_event":
                fact_to_event[edge.src] = edge.dst
            elif src.type == NodeType.EVENT and dst.type == NodeType.TOPIC and role == "event_topic":
                event_to_topic[edge.src] = edge.dst
        return fact_to_event, event_to_topic

    def _invert(self, mapping: dict[str, str]) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {}
        for child, parent in mapping.items():
            output.setdefault(parent, []).append(child)
        return output

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
