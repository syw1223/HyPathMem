from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from hytopomem.memory.node_extractor import content_terms
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType
from hytopomem.models.text_encoder import HashTextEncoder


_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


@dataclass(frozen=True)
class HierarchyBuilderConfig:
    embedding_dim: int = 128
    topic_boundary_similarity: float = 0.10
    event_boundary_similarity: float = 0.18
    min_topic_facts: int = 2
    max_topic_facts: int = 36
    min_event_facts: int = 1
    max_event_facts: int = 8
    max_keywords: int = 8


@dataclass(frozen=True)
class FactView:
    node: Node
    vector: np.ndarray
    terms: set[str]
    entities: set[str]
    session: str
    order: tuple


class HierarchicalGraphBuilder:
    """Add a lightweight Topic -> Event -> Fact hierarchy to an existing graph.

    This v1 intentionally avoids LLM calls. It builds topic segments from
    sequential lexical/hashed-embedding coherence, then splits each topic into
    smaller event blocks. The output is meant for oracle/retrieval diagnosis
    before replacing rule summaries with learned or LLM summaries.
    """

    def __init__(self, config: HierarchyBuilderConfig | None = None):
        self.config = config or HierarchyBuilderConfig()
        self.encoder = HashTextEncoder(dim=self.config.embedding_dim)

    def build(self, graph: MemoryGraph, graph_id: str | None = None) -> MemoryGraph:
        output = graph.model_copy(deep=True)
        output.graph_id = graph_id or f"{graph.graph_id}_hierarchy_v2"
        metadata = dict(output.metadata)
        metadata["hierarchy_v2"] = {
            "builder": "rule_topic_event_fact_v1",
            "config": self.config.__dict__,
            "source_graph_id": graph.graph_id,
        }
        output.metadata = metadata

        fact_buckets = self._facts_by_conversation(output)
        topic_count = 0
        event_count = 0
        edge_count = 0
        for conversation_id, facts in fact_buckets.items():
            views = [self._fact_view(fact) for fact in sorted(facts, key=self._node_order)]
            topic_segments = self._topic_segments(views)
            event_index = 0
            for topic_index, segment in enumerate(topic_segments, start=1):
                topic_count += 1
                topic_id = f"{conversation_id}:topicv2:{topic_index:04d}"
                topic_node = self._topic_node(topic_id, conversation_id, segment)
                output.add_node(topic_node)
                for event_segment in self._event_segments(segment):
                    event_index += 1
                    event_count += 1
                    event_id = f"{conversation_id}:eventv2:{event_index:04d}"
                    event_node = self._event_node(event_id, conversation_id, topic_id, event_segment)
                    output.add_node(event_node)
                    output.add_edge(
                        Edge(
                            src=event_id,
                            dst=topic_id,
                            relation=RelationType.IS_SPECIFIC_OF,
                            confidence=self._edge_confidence(event_segment),
                            metadata={
                                "hierarchy_v2": "event_topic",
                                "builder": "rule_topic_event_fact_v1",
                            },
                        )
                    )
                    edge_count += 1
                    for fact_view in event_segment:
                        output.add_edge(
                            Edge(
                                src=fact_view.node.node_id,
                                dst=event_id,
                                relation=RelationType.IS_SPECIFIC_OF,
                                confidence=self._edge_confidence(event_segment),
                                metadata={
                                    "hierarchy_v2": "fact_event",
                                    "builder": "rule_topic_event_fact_v1",
                                    "topic_id": topic_id,
                                },
                            )
                        )
                        edge_count += 1
        output.metadata["hierarchy_v2"]["topic_nodes"] = topic_count
        output.metadata["hierarchy_v2"]["event_nodes"] = event_count
        output.metadata["hierarchy_v2"]["hierarchy_edges"] = edge_count
        return output

    def _facts_by_conversation(self, graph: MemoryGraph) -> dict[str, list[Node]]:
        buckets: dict[str, list[Node]] = {}
        for node in graph.iter_nodes(NodeType.FACT):
            conversation_id = node.node_id.split(":fact:", 1)[0]
            buckets.setdefault(conversation_id, []).append(node)
        return buckets

    def _fact_view(self, node: Node) -> FactView:
        text = node.text
        return FactView(
            node=node,
            vector=self.encoder.encode_one(text),
            terms=set(content_terms(text)),
            entities=extract_entities(text),
            session=str(node.metadata.get("session") or session_from_turn(node.metadata.get("turn_id"))),
            order=self._node_order(node),
        )

    def _node_order(self, node: Node) -> tuple:
        turn_id = str(node.metadata.get("turn_id") or "")
        if turn_id:
            return parse_turn_order(turn_id)
        session = str(node.metadata.get("session") or "")
        suffix = node.node_id.rsplit(":", 1)[-1]
        try:
            number = int(suffix)
        except ValueError:
            number = 999999
        return (session_order(session), 1, number, node.node_id)

    def _topic_segments(self, facts: Sequence[FactView]) -> list[list[FactView]]:
        segments: list[list[FactView]] = []
        current: list[FactView] = []
        for fact in facts:
            if not current:
                current = [fact]
                continue
            previous = current[-1]
            should_break = False
            if fact.session != previous.session:
                should_break = True
            elif len(current) >= self.config.max_topic_facts:
                should_break = True
            elif (
                len(current) >= self.config.min_topic_facts
                and cosine(previous.vector, fact.vector) < self.config.topic_boundary_similarity
                and jaccard(previous.terms, fact.terms) < 0.08
            ):
                should_break = True
            if should_break:
                segments.append(current)
                current = [fact]
            else:
                current.append(fact)
        if current:
            segments.append(current)
        return merge_tiny_segments(segments, self.config.min_topic_facts, self.config.max_topic_facts)

    def _event_segments(self, topic_facts: Sequence[FactView]) -> list[list[FactView]]:
        segments: list[list[FactView]] = []
        current: list[FactView] = []
        for fact in topic_facts:
            if not current:
                current = [fact]
                continue
            previous = current[-1]
            overlap = max(jaccard(previous.terms, fact.terms), jaccard(previous.entities, fact.entities))
            should_break = False
            if len(current) >= self.config.max_event_facts:
                should_break = True
            elif (
                len(current) >= self.config.min_event_facts
                and cosine(previous.vector, fact.vector) < self.config.event_boundary_similarity
                and overlap < 0.12
            ):
                should_break = True
            if should_break:
                segments.append(current)
                current = [fact]
            else:
                current.append(fact)
        if current:
            segments.append(current)
        return segments

    def _topic_node(self, node_id: str, conversation_id: str, facts: Sequence[FactView]) -> Node:
        keywords = top_keywords(facts, self.config.max_keywords)
        coherence = segment_coherence([fact.vector for fact in facts])
        return Node(
            node_id=node_id,
            type=NodeType.TOPIC,
            text=f"Topic: {', '.join(keywords) if keywords else 'general memory'}",
            source="hierarchy_v2_topic",
            confidence=min(0.92, 0.55 + 0.35 * coherence),
            support_ids=[fact.node.node_id for fact in facts],
            metadata={
                "conversation_id": conversation_id,
                "fact_ids": [fact.node.node_id for fact in facts],
                "turn_ids": [str(fact.node.metadata.get("turn_id") or "") for fact in facts],
                "session_ids": sorted({fact.session for fact in facts if fact.session}),
                "keywords": keywords,
                "coherence": coherence,
                "hierarchy_v2": "topic",
            },
        )

    def _event_node(self, node_id: str, conversation_id: str, topic_id: str, facts: Sequence[FactView]) -> Node:
        keywords = top_keywords(facts, self.config.max_keywords)
        entities = sorted(set().union(*(fact.entities for fact in facts)))
        coherence = segment_coherence([fact.vector for fact in facts])
        label_terms = entities[:3] + [keyword for keyword in keywords if keyword not in {item.lower() for item in entities}]
        return Node(
            node_id=node_id,
            type=NodeType.EVENT,
            text=f"Event: {', '.join(label_terms[: self.config.max_keywords]) if label_terms else 'memory episode'}",
            source="hierarchy_v2_event",
            confidence=min(0.94, 0.58 + 0.34 * coherence),
            support_ids=[fact.node.node_id for fact in facts],
            metadata={
                "conversation_id": conversation_id,
                "topic_id": topic_id,
                "fact_ids": [fact.node.node_id for fact in facts],
                "turn_ids": [str(fact.node.metadata.get("turn_id") or "") for fact in facts],
                "session_ids": sorted({fact.session for fact in facts if fact.session}),
                "entities": entities[:12],
                "keywords": keywords,
                "coherence": coherence,
                "hierarchy_v2": "event",
            },
        )

    def _edge_confidence(self, facts: Sequence[FactView]) -> float:
        return min(0.96, 0.62 + 0.32 * segment_coherence([fact.vector for fact in facts]))


def parse_turn_order(turn_id: str) -> tuple:
    match = re.match(r"([A-Za-z]+)(\d+):(\d+)", turn_id)
    if match:
        prefix, session_number, turn_number = match.groups()
        return (f"{prefix}{int(session_number):04d}", 0, int(turn_number), turn_id)
    return (turn_id, 0, 0, turn_id)


def session_order(session: str) -> str:
    match = re.match(r"([A-Za-z]+)(\d+)", session)
    if match:
        prefix, number = match.groups()
        return f"{prefix}{int(number):04d}"
    return session or "zzzz"


def session_from_turn(turn_id: object) -> str:
    if not turn_id:
        return ""
    return str(turn_id).split(":", 1)[0]


def extract_entities(text: str) -> set[str]:
    return set(_ENTITY_RE.findall(text))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def segment_coherence(vectors: Sequence[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.5
    values = [cosine(vectors[index - 1], vectors[index]) for index in range(1, len(vectors))]
    if not values:
        return 0.5
    return max(0.0, min(1.0, float(np.mean(values))))


def merge_tiny_segments(
    segments: list[list[FactView]],
    min_size: int,
    max_size: int,
) -> list[list[FactView]]:
    merged: list[list[FactView]] = []
    for segment in segments:
        if merged and len(segment) < min_size and len(merged[-1]) + len(segment) <= max_size:
            merged[-1].extend(segment)
        else:
            merged.append(list(segment))
    return merged


def top_keywords(facts: Sequence[FactView], limit: int) -> list[str]:
    counts: dict[str, float] = {}
    for fact in facts:
        for term in fact.terms:
            counts[term] = counts.get(term, 0.0) + 1.0
        for entity in fact.entities:
            key = entity.lower()
            counts[key] = counts.get(key, 0.0) + 1.5
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _count in ranked[:limit]]
