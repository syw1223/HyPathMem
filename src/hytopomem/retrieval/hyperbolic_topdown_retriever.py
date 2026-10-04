from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from hytopomem.memory.schema import MemoryGraph, NodeType
from hytopomem.models.graph_v2_hyperbolic import GraphV2HyperbolicMapper, lorentz_distance
from hytopomem.retrieval.cross_encoder_reranker import RerankCandidate
from hytopomem.retrieval.topdown_semantic_retriever import (
    SentenceTransformerEncoder,
    TopDownSemanticConfig,
    bucket_ids_by_conversation,
    conversation_id_from_question,
    default_embedder,
    merge_route_metadata,
)


@dataclass(frozen=True)
class HyperbolicRouterCheckpoint:
    model: GraphV2HyperbolicMapper
    input_dim: int
    tangent_dim: int
    metadata: dict


class HyperbolicTopDownRetriever:
    def __init__(
        self,
        graph: MemoryGraph,
        *,
        encoder: SentenceTransformerEncoder,
        checkpoint_path: Path,
        config: TopDownSemanticConfig | None = None,
        embedding_cache_path: Path | None = None,
        device: str | None = None,
        batch_size: int = 1024,
    ):
        self.graph = graph
        self.encoder = encoder
        self.config = config or TopDownSemanticConfig()
        self.device = torch.device(device if device and torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        checkpoint = load_hyperbolic_router(checkpoint_path, self.device)
        self.model = checkpoint.model
        self.model.eval()
        self.checkpoint_metadata = checkpoint.metadata
        self.fact_to_event = self._fact_to_event()
        self.event_to_topic = self._event_to_topic()
        self.event_to_facts = self._event_to_facts()
        self.topic_to_events = self._topic_to_events()
        self.events = sorted(
            (
                node
                for node in graph.iter_nodes(NodeType.EVENT)
                if not (self.config.hierarchy_version == "v3_3" and node.metadata.get("hierarchy_v3_3") == "episode")
            ),
            key=lambda node: node.node_id,
        )
        self.topics = sorted(graph.iter_nodes(NodeType.TOPIC), key=lambda node: node.node_id)
        self.event_ids = [node.node_id for node in self.events]
        self.topic_ids = [node.node_id for node in self.topics]
        event_embeddings, topic_embeddings = self._load_or_encode_embeddings(embedding_cache_path)
        self.event_points = self._map_embeddings(event_embeddings)
        self.topic_points = self._map_embeddings(topic_embeddings)
        self.events_by_conversation = bucket_ids_by_conversation(self.event_ids)
        self.topics_by_conversation = bucket_ids_by_conversation(self.topic_ids)
        self.event_index = {node_id: idx for idx, node_id in enumerate(self.event_ids)}
        self.topic_index = {node_id: idx for idx, node_id in enumerate(self.topic_ids)}

    def candidates(self, question_id: str, query: str) -> list[RerankCandidate]:
        query_vector = self.encoder.encode([query])[0]
        return self.candidates_from_vector(question_id, query_vector)

    def candidates_from_vector(self, question_id: str, query_vector: np.ndarray) -> list[RerankCandidate]:
        query_point = self._map_embeddings(np.asarray([query_vector], dtype=np.float32))[0]
        candidates: dict[str, RerankCandidate] = {}
        if self.config.mode in {"event", "both"}:
            for rank, (event_id, score) in enumerate(self._top_events(question_id, query_point), start=1):
                self._add_event_facts(
                    candidates,
                    event_id=event_id,
                    query_score=score,
                    route_source="hyp_event",
                    route_rank=rank,
                    path_prefix=[event_id],
                )
        if self.config.mode in {"topic", "both"}:
            for topic_rank, (topic_id, topic_score) in enumerate(self._top_topics(question_id, query_point), start=1):
                events = self._rank_topic_events(topic_id, query_point)
                for event_rank, (event_id, event_score) in enumerate(events[: self.config.events_per_topic], start=1):
                    combined = 0.55 * topic_score + 0.45 * event_score
                    self._add_event_facts(
                        candidates,
                        event_id=event_id,
                        query_score=combined,
                        route_source="hyp_topic",
                        route_rank=topic_rank,
                        path_prefix=[topic_id, event_id],
                        event_rank=event_rank,
                    )
        ranked = sorted(candidates.values(), key=lambda item: item.base_score, reverse=True)
        return ranked[: self.config.max_candidates]

    def _top_events(self, question_id: str, query_point: np.ndarray) -> list[tuple[str, float]]:
        event_ids = self._candidate_event_ids(question_id)
        return top_by_lorentz_score(event_ids, self.event_index, self.event_points, query_point, self.config.event_topk)

    def _top_topics(self, question_id: str, query_point: np.ndarray) -> list[tuple[str, float]]:
        topic_ids = self._candidate_topic_ids(question_id)
        return top_by_lorentz_score(topic_ids, self.topic_index, self.topic_points, query_point, self.config.topic_topk)

    def _rank_topic_events(self, topic_id: str, query_point: np.ndarray) -> list[tuple[str, float]]:
        event_ids = self.topic_to_events.get(topic_id, [])
        return top_by_lorentz_score(event_ids, self.event_index, self.event_points, query_point, len(event_ids))

    def _add_event_facts(
        self,
        candidates: dict[str, RerankCandidate],
        *,
        event_id: str,
        query_score: float,
        route_source: str,
        route_rank: int,
        path_prefix: list[str],
        event_rank: int = 0,
    ) -> None:
        topic_id = self.event_to_topic.get(event_id, "")
        fact_ids = self.event_to_facts.get(event_id, [])[: self.config.facts_per_event]
        for fact_offset, fact_id in enumerate(fact_ids):
            node = self.graph.nodes.get(fact_id)
            if node is None or node.type != NodeType.FACT:
                continue
            score = float(query_score) - 0.002 * fact_offset
            metadata = {
                "candidate_source": route_source,
                "route_source": route_source,
                "hyperbolic_score": f"{float(query_score):.6f}",
                "semantic_score": f"{float(query_score):.6f}",
                "route_rank": str(route_rank),
                "event_rank": str(event_rank),
                "fact_offset": str(fact_offset),
                "event_node_id": event_id,
                "topic_node_id": topic_id,
                "retriever": "hyperbolic_topdown",
            }
            if route_source == "hyp_event":
                metadata["hyp_event_score"] = f"{float(query_score):.6f}"
                metadata["hyp_event_rank"] = str(route_rank)
            elif route_source == "hyp_topic":
                metadata["hyp_topic_score"] = f"{float(query_score):.6f}"
                metadata["hyp_topic_rank"] = str(route_rank)
                metadata["hyp_topic_event_rank"] = str(event_rank)
            path_node_ids = [*path_prefix, fact_id]
            previous = candidates.get(fact_id)
            if previous is not None:
                merged_metadata = merge_route_metadata(previous.metadata or {}, metadata)
                if score <= previous.base_score:
                    previous.metadata = merged_metadata
                    continue
                metadata = merged_metadata
            candidates[fact_id] = RerankCandidate(
                node=node,
                base_score=score,
                path_node_ids=path_node_ids,
                metadata=metadata,
            )

    def _candidate_event_ids(self, question_id: str) -> list[str]:
        if not self.config.restrict_conversation:
            return self.event_ids
        conv_id = conversation_id_from_question(question_id)
        return self.events_by_conversation.get(conv_id, [])

    def _candidate_topic_ids(self, question_id: str) -> list[str]:
        if not self.config.restrict_conversation:
            return self.topic_ids
        conv_id = conversation_id_from_question(question_id)
        return self.topics_by_conversation.get(conv_id, [])

    def _load_or_encode_embeddings(self, cache_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
        if cache_path is not None and cache_path.exists():
            payload = np.load(cache_path, allow_pickle=False)
            event_ids = [str(item) for item in payload["event_ids"]]
            topic_ids = [str(item) for item in payload["topic_ids"]]
            cached_model = str(payload["model"][0]) if "model" in payload.files else ""
            if event_ids == self.event_ids and topic_ids == self.topic_ids and cached_model == self.encoder.model_name_or_path:
                return np.asarray(payload["event_embeddings"], dtype=np.float32), np.asarray(
                    payload["topic_embeddings"],
                    dtype=np.float32,
                )
        event_embeddings = self.encoder.encode([node.text for node in self.events])
        topic_embeddings = self.encoder.encode([node.text for node in self.topics])
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                event_ids=np.asarray(self.event_ids),
                topic_ids=np.asarray(self.topic_ids),
                event_embeddings=event_embeddings,
                topic_embeddings=topic_embeddings,
                model=np.asarray([self.encoder.model_name_or_path]),
            )
        return event_embeddings, topic_embeddings

    def _map_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(embeddings), self.batch_size):
                batch = torch.tensor(embeddings[start : start + self.batch_size], dtype=torch.float32, device=self.device)
                outputs.append(self.model(batch).detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32) if outputs else np.empty((0, 0), dtype=np.float32)

    def _fact_to_event(self) -> dict[str, str]:
        mapping = {}
        hierarchy_key = f"hierarchy_{self.config.hierarchy_version}"
        for edge in self.graph.edges:
            if edge.metadata.get(hierarchy_key) != "fact_event":
                continue
            src = self.graph.nodes.get(edge.src)
            dst = self.graph.nodes.get(edge.dst)
            if src is not None and dst is not None and src.type == NodeType.FACT and dst.type == NodeType.EVENT:
                mapping[edge.src] = edge.dst
        return mapping

    def _event_to_topic(self) -> dict[str, str]:
        mapping = {}
        hierarchy_key = f"hierarchy_{self.config.hierarchy_version}"
        if self.config.hierarchy_version == "v3_3":
            event_to_episode: dict[str, str] = {}
            episode_to_topic: dict[str, str] = {}
            for edge in self.graph.edges:
                role = edge.metadata.get(hierarchy_key)
                src = self.graph.nodes.get(edge.src)
                dst = self.graph.nodes.get(edge.dst)
                if src is None or dst is None:
                    continue
                if role == "event_episode" and src.type == NodeType.EVENT and dst.type == NodeType.EVENT:
                    event_to_episode[edge.src] = edge.dst
                elif role == "episode_topic" and src.type == NodeType.EVENT and dst.type == NodeType.TOPIC:
                    episode_to_topic[edge.src] = edge.dst
            return {
                event_id: episode_to_topic[episode_id]
                for event_id, episode_id in event_to_episode.items()
                if episode_id in episode_to_topic
            }
        for edge in self.graph.edges:
            if edge.metadata.get(hierarchy_key) != "event_topic":
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


def load_hyperbolic_router(path: Path, device: torch.device) -> HyperbolicRouterCheckpoint:
    checkpoint = torch.load(path, map_location="cpu")
    input_dim = int(checkpoint["input_dim"])
    tangent_dim = int(checkpoint["tangent_dim"])
    model = GraphV2HyperbolicMapper(
        input_dim=input_dim,
        tangent_dim=tangent_dim,
        scale=float(checkpoint.get("scale", 0.9)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return HyperbolicRouterCheckpoint(
        model=model,
        input_dim=input_dim,
        tangent_dim=tangent_dim,
        metadata=dict(checkpoint.get("metadata", {})),
    )


def top_by_lorentz_score(
    node_ids: list[str],
    index: dict[str, int],
    points: np.ndarray,
    query_point: np.ndarray,
    top_k: int,
) -> list[tuple[str, float]]:
    if not node_ids or top_k <= 0:
        return []
    valid_node_ids = [node_id for node_id in node_ids if node_id in index]
    indices = np.asarray([index[node_id] for node_id in valid_node_ids], dtype=np.int64)
    if indices.size == 0:
        return []
    selected = points[indices]
    scores = -lorentz_distance_numpy(selected, query_point)
    limit = min(top_k, len(scores))
    if limit >= len(scores):
        local_order = np.argsort(-scores)
    else:
        local_order = np.argpartition(-scores, limit - 1)[:limit]
        local_order = local_order[np.argsort(-scores[local_order])]
    return [(valid_node_ids[int(pos)], float(scores[pos])) for pos in local_order]


def lorentz_distance_numpy(points: np.ndarray, query_point: np.ndarray) -> np.ndarray:
    query = np.asarray(query_point, dtype=np.float32)
    prod = points[:, 0] * query[0] - points[:, 1:] @ query[1:]
    prod = np.clip(prod, 1.0 + 1e-7, None)
    return np.arccosh(prod)


__all__ = [
    "HyperbolicTopDownRetriever",
    "HyperbolicRouterCheckpoint",
    "load_hyperbolic_router",
    "top_by_lorentz_score",
    "lorentz_distance_numpy",
    "default_embedder",
]
