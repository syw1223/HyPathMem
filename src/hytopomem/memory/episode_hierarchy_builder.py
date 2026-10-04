from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from hytopomem.memory.hierarchy_builder import cosine, jaccard
from hytopomem.memory.node_extractor import content_terms
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType


class TextEncoder(Protocol):
    model_name_or_path: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class EpisodeHierarchyConfig:
    episode_similarity_threshold: float = 0.50
    episode_max_events: int = 6
    min_episode_events: int = 2
    max_entities: int = 8
    max_keywords: int = 10


@dataclass(frozen=True)
class EventUnit:
    node: Node
    vector: np.ndarray
    terms: set[str]
    entities: set[str]
    sessions: set[str]
    fact_ids: tuple[str, ...]
    order: str


@dataclass
class EpisodeCluster:
    members: list[EventUnit]
    centroid: np.ndarray
    terms: set[str] = field(default_factory=set)
    entities: set[str] = field(default_factory=set)
    sessions: set[str] = field(default_factory=set)

    @classmethod
    def from_event(cls, event: EventUnit) -> "EpisodeCluster":
        return cls(
            members=[event],
            centroid=event.vector.copy(),
            terms=set(event.terms),
            entities=set(event.entities),
            sessions=set(event.sessions),
        )

    def add(self, event: EventUnit) -> None:
        count = len(self.members)
        self.centroid = normalize_vector((self.centroid * count + event.vector) / (count + 1))
        self.members.append(event)
        self.terms.update(event.terms)
        self.entities.update(event.entities)
        self.sessions.update(event.sessions)


class EpisodeHierarchyBuilder:
    """Insert an Episode/Subtopic layer between Topic and Event.

    This is intentionally a narrow V3.3-A transformation. It takes an existing
    semantic V3 graph and adds a new hierarchy key:

        FACT --fact_event--> EVENT --event_episode--> EPISODE --episode_topic--> TOPIC

    Episode nodes reuse NodeType.EVENT to avoid a schema migration. They are
    identifiable by source="semantic_episode_cluster_v1" and
    metadata["hierarchy_v3_3"] == "episode".
    """

    def __init__(self, encoder: TextEncoder, config: EpisodeHierarchyConfig | None = None):
        self.encoder = encoder
        self.config = config or EpisodeHierarchyConfig()

    def build(self, graph: MemoryGraph, graph_id: str | None = None) -> MemoryGraph:
        output = strip_existing_v3_3(graph)
        output.graph_id = graph_id or f"{graph.graph_id}_episode_v3_3"

        event_to_topic, topic_to_events = self._event_topic_maps(output)
        event_to_facts = self._event_fact_map(output)
        event_ids = sorted(event_to_topic)
        vectors = self.encoder.encode([output.nodes[event_id].text for event_id in event_ids]) if event_ids else np.empty((0, 0), dtype=np.float32)
        vector_by_event = {
            event_id: normalize_vector(vector)
            for event_id, vector in zip(event_ids, vectors)
        }

        # Mark canonical fact->event edges as part of the V3.3 hierarchy rather
        # than duplicating identical edge IDs.
        hierarchy_edge_count = 0
        for edge in output.edges:
            if edge.metadata.get("hierarchy_v3") == "fact_event":
                edge.metadata["hierarchy_v3_3"] = "fact_event"
                edge.metadata["builder_v3_3"] = "episode_semantic_v1"
                hierarchy_edge_count += 1

        episode_count = 0
        event_episode_edges = 0
        episode_topic_edges = 0
        episode_sizes: list[int] = []
        for topic_id in sorted(topic_to_events):
            topic_node = output.nodes[topic_id]
            conversation_id = str(topic_node.metadata.get("conversation_id") or topic_id.split(":", 1)[0])
            units = [
                self._event_unit(output.nodes[event_id], vector_by_event[event_id], event_to_facts.get(event_id, []))
                for event_id in topic_to_events[topic_id]
                if event_id in vector_by_event
            ]
            for episode_index, cluster in enumerate(self._episode_clusters(units), start=1):
                episode_count += 1
                episode_id = f"{conversation_id}:episodev3_3:{episode_count:04d}"
                episode_node = self._episode_node(episode_id, conversation_id, topic_id, cluster)
                output.add_node(episode_node)
                episode_sizes.append(len(cluster))
                output.add_edge(
                    Edge(
                        src=episode_id,
                        dst=topic_id,
                        relation=RelationType.IS_SPECIFIC_OF,
                        confidence=episode_node.confidence,
                        metadata={
                            "hierarchy_v3_3": "episode_topic",
                            "builder": "episode_semantic_v1",
                        },
                    )
                )
                hierarchy_edge_count += 1
                episode_topic_edges += 1
                for event in cluster:
                    output.add_edge(
                        Edge(
                            src=event.node.node_id,
                            dst=episode_id,
                            relation=RelationType.IS_SPECIFIC_OF,
                            confidence=episode_node.confidence,
                            metadata={
                                "hierarchy_v3_3": "event_episode",
                                "builder": "episode_semantic_v1",
                                "topic_id": topic_id,
                            },
                        )
                    )
                    output.nodes[event.node.node_id].metadata["episode_id"] = episode_id
                    hierarchy_edge_count += 1
                    event_episode_edges += 1

        metadata = dict(output.metadata)
        metadata["hierarchy_v3_3"] = {
            "builder": "episode_semantic_v1",
            "encoder": self.encoder.model_name_or_path,
            "config": self.config.__dict__,
            "source_graph_id": graph.graph_id,
            "episode_nodes": episode_count,
            "event_episode_edges": event_episode_edges,
            "episode_topic_edges": episode_topic_edges,
            "hierarchy_edges": hierarchy_edge_count,
            "mean_events_per_episode": float(np.mean(episode_sizes)) if episode_sizes else 0.0,
            "singleton_episode_ratio": sum(size == 1 for size in episode_sizes) / len(episode_sizes) if episode_sizes else 0.0,
            "episode_size_max": max(episode_sizes, default=0),
        }
        output.metadata = metadata
        return output

    def _event_topic_maps(self, graph: MemoryGraph) -> tuple[dict[str, str], dict[str, list[str]]]:
        event_to_topic: dict[str, str] = {}
        topic_to_events: dict[str, list[str]] = {}
        for edge in graph.edges:
            if edge.metadata.get("hierarchy_v3") != "event_topic":
                continue
            src = graph.nodes.get(edge.src)
            dst = graph.nodes.get(edge.dst)
            if src is None or dst is None or src.type != NodeType.EVENT or dst.type != NodeType.TOPIC:
                continue
            if src.metadata.get("hierarchy_v3_3") == "episode":
                continue
            event_to_topic[edge.src] = edge.dst
            topic_to_events.setdefault(edge.dst, []).append(edge.src)
        return event_to_topic, topic_to_events

    def _event_fact_map(self, graph: MemoryGraph) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for edge in graph.edges:
            if edge.metadata.get("hierarchy_v3") != "fact_event":
                continue
            src = graph.nodes.get(edge.src)
            dst = graph.nodes.get(edge.dst)
            if src is not None and dst is not None and src.type == NodeType.FACT and dst.type == NodeType.EVENT:
                mapping.setdefault(edge.dst, []).append(edge.src)
        return mapping

    def _event_unit(self, node: Node, vector: np.ndarray, fact_ids: Sequence[str]) -> EventUnit:
        metadata = node.metadata
        return EventUnit(
            node=node,
            vector=normalize_vector(vector),
            terms=set(content_terms(node.text)) | set(str(item).lower() for item in metadata.get("keywords", [])),
            entities=set(str(item) for item in metadata.get("entities", [])),
            sessions=set(str(item) for item in metadata.get("session_ids", [])),
            fact_ids=tuple(fact_ids),
            order=str(metadata.get("session_ids", [""])[0] if metadata.get("session_ids") else node.node_id),
        )

    def _episode_clusters(self, events: Sequence[EventUnit]) -> list[list[EventUnit]]:
        clusters: list[EpisodeCluster] = []
        for event in sorted(events, key=lambda item: item.order):
            best_index = -1
            best_score = -1.0
            for index, cluster in enumerate(clusters):
                if len(cluster.members) >= self.config.episode_max_events:
                    continue
                score = self._event_episode_similarity(event, cluster)
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index >= 0 and best_score >= self.config.episode_similarity_threshold:
                clusters[best_index].add(event)
            else:
                clusters.append(EpisodeCluster.from_event(event))
        return [sorted(cluster.members, key=lambda item: item.order) for cluster in clusters]

    def _event_episode_similarity(self, event: EventUnit, cluster: EpisodeCluster) -> float:
        return (
            0.70 * cosine(event.vector, cluster.centroid)
            + 0.15 * jaccard(event.entities, cluster.entities)
            + 0.10 * jaccard(event.terms, cluster.terms)
            + 0.05 * jaccard(event.sessions, cluster.sessions)
        )

    def _episode_node(
        self,
        node_id: str,
        conversation_id: str,
        topic_id: str,
        events: Sequence[EventUnit],
    ) -> Node:
        vectors = [event.vector for event in events]
        centroid = mean_vector(vectors)
        representatives = sorted(events, key=lambda event: cosine(event.vector, centroid), reverse=True)[:2]
        entities = ranked_values(events, "entities", self.config.max_entities)
        keywords = ranked_values(events, "terms", self.config.max_keywords)
        coherence = cluster_coherence(vectors)
        text_parts = [event.node.text for event in representatives]
        label = "; ".join(text_parts)
        if len(label) > 360:
            label = label[:357].rstrip() + "..."
        return Node(
            node_id=node_id,
            type=NodeType.EVENT,
            text=f"Episode: {label or ', '.join(keywords) or 'memory episode'}",
            source="semantic_episode_cluster_v1",
            confidence=min(0.96, 0.58 + 0.36 * coherence),
            support_ids=[event.node.node_id for event in events],
            metadata={
                "conversation_id": conversation_id,
                "topic_id": topic_id,
                "event_ids": [event.node.node_id for event in events],
                "fact_ids": [fact_id for event in events for fact_id in event.fact_ids],
                "session_ids": sorted({session for event in events for session in event.sessions}),
                "entities": entities,
                "keywords": keywords,
                "coherence": coherence,
                "hierarchy_v3_3": "episode",
                "label_type": "template",
            },
        )


def strip_existing_v3_3(graph: MemoryGraph) -> MemoryGraph:
    output = graph.model_copy(deep=True)
    generated_ids = {
        node_id
        for node_id, node in output.nodes.items()
        if node.metadata.get("hierarchy_v3_3") == "episode" or node.source == "semantic_episode_cluster_v1"
    }
    output.nodes = {node_id: node for node_id, node in output.nodes.items() if node_id not in generated_ids}
    cleaned_edges = []
    for edge in output.edges:
        if edge.src in generated_ids or edge.dst in generated_ids:
            continue
        if "hierarchy_v3_3" in edge.metadata:
            edge.metadata = dict(edge.metadata)
            edge.metadata.pop("hierarchy_v3_3", None)
            edge.metadata.pop("builder_v3_3", None)
        cleaned_edges.append(edge)
    output.edges = cleaned_edges
    metadata = dict(output.metadata)
    metadata.pop("hierarchy_v3_3", None)
    output.metadata = metadata
    return output


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.asarray(vector, dtype=np.float32)
    return np.asarray(vector / norm, dtype=np.float32)


def mean_vector(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.empty((0,), dtype=np.float32)
    return normalize_vector(np.mean(np.asarray(vectors, dtype=np.float32), axis=0))


def cluster_coherence(vectors: Sequence[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.50
    centroid = mean_vector(vectors)
    values = [cosine(vector, centroid) for vector in vectors]
    return max(0.0, min(1.0, float(np.mean(values))))


def ranked_values(events: Sequence[EventUnit], attr: str, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for event in events:
        for value in getattr(event, attr):
            if value:
                counts[str(value)] = counts.get(str(value), 0) + 1
    return [
        item
        for item, _count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    ]
