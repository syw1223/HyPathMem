from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from hytopomem.memory.schema import MemoryGraph, NodeType, RelationType
from hytopomem.models.graph_v2_hyperbolic import (
    GraphV2HyperbolicMapper,
    lorentz_distance,
    lorentz_radius,
)
from hytopomem.models.text_encoder import HashTextEncoder
from hytopomem.retrieval.geometric_features import evidence_node_id


GRAPH_V2_EUCLIDEAN_STRUCTURE_FEATURE_NAMES = [
    "g2s_euc_cos_query_fact",
    "g2s_euc_cos_query_event",
    "g2s_euc_cos_query_topic",
    "g2s_euc_cos_seed_fact",
    "g2s_euc_cos_fact_event",
    "g2s_euc_cos_event_topic",
    "g2s_euc_event_child_coherence",
    "g2s_euc_topic_event_coherence",
]

GRAPH_V2_HYPERBOLIC_STRUCTURE_FEATURE_NAMES = [
    "g2s_hyp_neg_dist_query_fact",
    "g2s_hyp_neg_dist_query_event",
    "g2s_hyp_neg_dist_query_topic",
    "g2s_hyp_neg_dist_seed_fact",
    "g2s_hyp_neg_dist_fact_event",
    "g2s_hyp_neg_dist_event_topic",
    "g2s_hyp_fact_radius",
    "g2s_hyp_event_radius",
    "g2s_hyp_topic_radius",
    "g2s_hyp_radius_gap_fact_event",
    "g2s_hyp_radius_gap_event_topic",
    "g2s_hyp_radial_order_fact_event",
    "g2s_hyp_radial_order_event_topic",
    "g2s_hyp_event_child_coherence",
    "g2s_hyp_topic_event_coherence",
]


@dataclass
class GraphV2StructureGeometryExtractor:
    graph: MemoryGraph
    checkpoint_path: str
    device: str = "cpu"
    max_coherence_children: int = 50

    def __post_init__(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.input_dim = int(checkpoint["input_dim"])
        self.tangent_dim = int(checkpoint["tangent_dim"])
        self.encoder = HashTextEncoder(dim=self.input_dim)
        self.torch_device = torch.device(
            self.device if torch.cuda.is_available() or not self.device.startswith("cuda") else "cpu"
        )
        self.model = GraphV2HyperbolicMapper(
            input_dim=self.input_dim,
            tangent_dim=self.tangent_dim,
            scale=float(checkpoint.get("scale", 0.9)),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.torch_device)
        self.model.eval()
        self._query_euclidean: dict[str, np.ndarray] = {}
        self._query_hyperbolic: dict[str, torch.Tensor] = {}
        self._node_euclidean = {
            node_id: self.encoder.encode_one(node.text)
            for node_id, node in self.graph.nodes.items()
        }
        with torch.no_grad():
            self._node_hyperbolic = {
                node_id: self._encode_text(node.text).cpu()
                for node_id, node in self.graph.nodes.items()
                if node.type in {NodeType.FACT, NodeType.EVENT, NodeType.TOPIC}
            }
        self.fact_to_event, self.event_to_topic = self._build_hierarchy_maps()
        self.event_children = invert(self.fact_to_event)
        self.topic_events = invert(self.event_to_topic)
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
            return GRAPH_V2_EUCLIDEAN_STRUCTURE_FEATURE_NAMES
        if kind == "hyperbolic":
            return GRAPH_V2_HYPERBOLIC_STRUCTURE_FEATURE_NAMES
        if kind == "both":
            return GRAPH_V2_EUCLIDEAN_STRUCTURE_FEATURE_NAMES + GRAPH_V2_HYPERBOLIC_STRUCTURE_FEATURE_NAMES
        raise ValueError(f"unknown graph v2 structure geometry kind: {kind}")

    def extract(self, item: dict, path: dict) -> dict[str, float]:
        metadata = path.get("metadata", {})
        fact_id = evidence_node_id(path)
        seed_id = str(metadata.get("seed_node_id") or "")
        event_id = str(metadata.get("event_node_id") or self.fact_to_event.get(fact_id, ""))
        topic_id = str(metadata.get("topic_node_id") or self.event_to_topic.get(event_id, ""))

        question = item.get("question", "")
        query_euc = self._query_euc(question)
        query_hyp = self._query_hyp(question)
        fact_euc = self._node_euclidean.get(fact_id)
        seed_euc = self._node_euclidean.get(seed_id)
        event_euc = self._node_euclidean.get(event_id)
        topic_euc = self._node_euclidean.get(topic_id)
        fact_hyp = self._node_hyperbolic.get(fact_id)
        seed_hyp = self._node_hyperbolic.get(seed_id)
        event_hyp = self._node_hyperbolic.get(event_id)
        topic_hyp = self._node_hyperbolic.get(topic_id)

        fact_radius = self._radius(fact_hyp)
        event_radius = self._radius(event_hyp)
        topic_radius = self._radius(topic_hyp)
        gap_fact_event = fact_radius - event_radius if fact_hyp is not None and event_hyp is not None else 0.0
        gap_event_topic = event_radius - topic_radius if event_hyp is not None and topic_hyp is not None else 0.0

        return {
            "g2s_euc_cos_query_fact": cosine(query_euc, fact_euc),
            "g2s_euc_cos_query_event": cosine(query_euc, event_euc),
            "g2s_euc_cos_query_topic": cosine(query_euc, topic_euc),
            "g2s_euc_cos_seed_fact": cosine(seed_euc, fact_euc),
            "g2s_euc_cos_fact_event": cosine(fact_euc, event_euc),
            "g2s_euc_cos_event_topic": cosine(event_euc, topic_euc),
            "g2s_euc_event_child_coherence": self._event_euc_coherence.get(event_id, 0.0),
            "g2s_euc_topic_event_coherence": self._topic_euc_coherence.get(topic_id, 0.0),
            "g2s_hyp_neg_dist_query_fact": -self._distance(query_hyp, fact_hyp),
            "g2s_hyp_neg_dist_query_event": -self._distance(query_hyp, event_hyp),
            "g2s_hyp_neg_dist_query_topic": -self._distance(query_hyp, topic_hyp),
            "g2s_hyp_neg_dist_seed_fact": -self._distance(seed_hyp, fact_hyp),
            "g2s_hyp_neg_dist_fact_event": -self._distance(fact_hyp, event_hyp),
            "g2s_hyp_neg_dist_event_topic": -self._distance(event_hyp, topic_hyp),
            "g2s_hyp_fact_radius": fact_radius,
            "g2s_hyp_event_radius": event_radius,
            "g2s_hyp_topic_radius": topic_radius,
            "g2s_hyp_radius_gap_fact_event": gap_fact_event,
            "g2s_hyp_radius_gap_event_topic": gap_event_topic,
            "g2s_hyp_radial_order_fact_event": float(gap_fact_event > 0.0),
            "g2s_hyp_radial_order_event_topic": float(gap_event_topic > 0.0),
            "g2s_hyp_event_child_coherence": self._event_hyp_coherence.get(event_id, 0.0),
            "g2s_hyp_topic_event_coherence": self._topic_hyp_coherence.get(topic_id, 0.0),
        }

    def _encode_text(self, text: str) -> torch.Tensor:
        vector = torch.tensor(self.encoder.encode_one(text), dtype=torch.float32, device=self.torch_device)
        with torch.no_grad():
            return self.model(vector[None, :]).squeeze(0)

    def _query_euc(self, text: str) -> np.ndarray:
        if text not in self._query_euclidean:
            self._query_euclidean[text] = self.encoder.encode_one(text)
        return self._query_euclidean[text]

    def _query_hyp(self, text: str) -> torch.Tensor:
        if text not in self._query_hyperbolic:
            self._query_hyperbolic[text] = self._encode_text(text).cpu()
        return self._query_hyperbolic[text]

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

    def _distance(self, left: torch.Tensor | None, right: torch.Tensor | None) -> float:
        if left is None or right is None:
            return 0.0
        return float(lorentz_distance(left, right).detach().cpu())

    def _radius(self, point: torch.Tensor | None) -> float:
        if point is None:
            return 0.0
        return float(lorentz_radius(point).detach().cpu())


def invert(mapping: dict[str, str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for child, parent in mapping.items():
        output.setdefault(parent, []).append(child)
    return output


def cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)
