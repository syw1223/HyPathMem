from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Iterable, Sequence

import numpy as np

from hytopomem.memory.hierarchy_builder import cosine, extract_entities, jaccard
from hytopomem.memory.multiview_graph_builder import canonical_session_id, conversation_id
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType
from hytopomem.models.text_encoder import HashTextEncoder
from hytopomem.retrieval.bm25_retriever import tokenize


ENTITY_VIEW_VERSION = "v3_5_entity_state_v1"

_GENERIC_ENTITY_STOPWORDS = {
    "A",
    "An",
    "And",
    "Anything",
    "Are",
    "As",
    "At",
    "But",
    "Can",
    "Did",
    "Do",
    "Does",
    "For",
    "Good",
    "Great",
    "Have",
    "Hey",
    "Hi",
    "How",
    "I",
    "If",
    "In",
    "It",
    "Its",
    "Me",
    "My",
    "No",
    "Oh",
    "Ok",
    "Okay",
    "So",
    "That",
    "The",
    "Then",
    "There",
    "They",
    "This",
    "To",
    "Was",
    "We",
    "Well",
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "Wow",
    "Yes",
    "You",
}

_TERM_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "any",
    "are",
    "because",
    "been",
    "but",
    "can",
    "did",
    "for",
    "from",
    "get",
    "got",
    "had",
    "has",
    "have",
    "her",
    "him",
    "his",
    "how",
    "into",
    "just",
    "like",
    "not",
    "now",
    "out",
    "said",
    "she",
    "that",
    "the",
    "their",
    "them",
    "then",
    "they",
    "this",
    "was",
    "were",
    "what",
    "when",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class EntityViewConfig:
    min_state_facts: int = 2
    max_state_facts: int = 20
    cluster_threshold: float = 0.50
    max_entities_per_fact: int = 4
    max_state_keywords: int = 8
    graph_id_suffix: str = "entity_view_v3_5"


@dataclass
class FactUnit:
    node: Node
    entity: str
    normalized_entity: str
    terms: set[str]
    vector: np.ndarray
    event_id: str = ""
    episode_id: str = ""
    session_id: str = ""


@dataclass
class EntityStateCluster:
    entity: str
    normalized_entity: str
    facts: list[FactUnit] = field(default_factory=list)
    terms: set[str] = field(default_factory=set)
    event_ids: set[str] = field(default_factory=set)
    episode_ids: set[str] = field(default_factory=set)
    session_ids: set[str] = field(default_factory=set)

    @classmethod
    def from_fact(cls, fact: FactUnit) -> "EntityStateCluster":
        cluster = cls(entity=fact.entity, normalized_entity=fact.normalized_entity)
        cluster.add(fact)
        return cluster

    def add(self, fact: FactUnit) -> None:
        self.facts.append(fact)
        self.terms.update(fact.terms)
        if fact.event_id:
            self.event_ids.add(fact.event_id)
        if fact.episode_id:
            self.episode_ids.add(fact.episode_id)
        if fact.session_id:
            self.session_ids.add(fact.session_id)


class EntityViewBuilder:
    """Add an entity-state hierarchy view.

    The graph stores hierarchy edges in the same direction as the existing
    semantic hierarchy: specific node -> abstract node.
    """

    def __init__(self, encoder: HashTextEncoder | None = None, config: EntityViewConfig | None = None):
        self.encoder = encoder or HashTextEncoder(dim=256)
        self.config = config or EntityViewConfig()

    def build(self, graph: MemoryGraph, *, graph_id: str | None = None) -> tuple[MemoryGraph, dict]:
        output = strip_existing_v3_5_entity(graph)
        output.graph_id = graph_id or f"{graph.graph_id}_{self.config.graph_id_suffix}"

        event_index = SemanticEventIndex.from_graph(output)
        fact_units = self._fact_units(output, event_index)
        stats = add_entity_view(output, fact_units, self.config)
        stats["source_graph_id"] = graph.graph_id
        stats["version"] = ENTITY_VIEW_VERSION
        stats["diagnostics"] = entity_view_diagnostics(output)

        metadata = dict(output.metadata)
        metadata["hierarchy_v3_5"] = stats
        output.metadata = metadata
        return output, stats

    def _fact_units(self, graph: MemoryGraph, event_index: "SemanticEventIndex") -> list[FactUnit]:
        semantic_fact_ids = set(event_index.fact_to_event)
        units: list[FactUnit] = []
        vectors = self.encoder.encode([graph.nodes[fact_id].text for fact_id in sorted(semantic_fact_ids)])
        vector_by_fact = dict(zip(sorted(semantic_fact_ids), vectors))
        for fact_id in sorted(semantic_fact_ids):
            node = graph.nodes[fact_id]
            entities = ranked_entities_for_fact(node, graph, event_index)
            if not entities:
                continue
            terms = content_terms(node.text)
            event_id = event_index.fact_to_event.get(fact_id, "")
            episode_id = event_index.event_to_episode.get(event_id, "")
            session_id = first_session_id(node)
            for entity in entities[: self.config.max_entities_per_fact]:
                normalized = normalize_entity(entity)
                if not normalized:
                    continue
                entity_terms = set(tokenize(entity))
                filtered_terms = {term for term in terms if term not in entity_terms}
                units.append(
                    FactUnit(
                        node=node,
                        entity=entity,
                        normalized_entity=normalized,
                        terms=filtered_terms,
                        vector=vector_by_fact[fact_id],
                        event_id=event_id,
                        episode_id=episode_id,
                        session_id=session_id,
                    )
                )
        return units


@dataclass(frozen=True)
class SemanticEventIndex:
    fact_to_event: dict[str, str]
    event_to_episode: dict[str, str]
    event_entities: dict[str, list[str]]
    episode_entities: dict[str, list[str]]

    @classmethod
    def from_graph(cls, graph: MemoryGraph) -> "SemanticEventIndex":
        fact_to_event: dict[str, str] = {}
        event_to_episode: dict[str, str] = {}
        event_entities: dict[str, list[str]] = {}
        episode_entities: dict[str, list[str]] = {}
        for node in graph.iter_nodes(NodeType.EVENT):
            entities = [str(item) for item in node.metadata.get("entities", [])]
            if node.metadata.get("hierarchy_v3_3") == "episode":
                episode_entities[node.node_id] = entities
            elif node.metadata.get("hierarchy_v3") == "event":
                event_entities[node.node_id] = entities
        for edge in graph.edges:
            src = graph.nodes.get(edge.src)
            dst = graph.nodes.get(edge.dst)
            if src is None or dst is None:
                continue
            role = edge.metadata.get("hierarchy_v3_3") or edge.metadata.get("hierarchy_v3")
            if src.type == NodeType.FACT and dst.type == NodeType.EVENT and role == "fact_event":
                fact_to_event[src.node_id] = dst.node_id
            elif (
                src.type == NodeType.EVENT
                and dst.type == NodeType.EVENT
                and edge.metadata.get("hierarchy_v3_3") == "event_episode"
            ):
                event_to_episode[src.node_id] = dst.node_id
        return cls(fact_to_event, event_to_episode, event_entities, episode_entities)


def add_entity_view(graph: MemoryGraph, fact_units: Sequence[FactUnit], config: EntityViewConfig) -> dict:
    by_conversation_entity: dict[tuple[str, str], list[FactUnit]] = defaultdict(list)
    entity_display: dict[tuple[str, str], str] = {}
    for unit in fact_units:
        key = (conversation_id(unit.node.node_id), unit.normalized_entity)
        by_conversation_entity[key].append(unit)
        entity_display.setdefault(key, unit.entity)

    existing_edges = {(edge.src, edge.dst, edge.metadata.get("hierarchy_v3_5_entity")) for edge in graph.edges}
    entity_nodes_added = 0
    state_nodes_added = 0
    fact_state_edges_added = 0
    state_entity_edges_added = 0
    direct_fact_entity_edges_added = 0
    direct_fact_units = 0
    clustered_fact_units = 0
    high_freq_entities = 0
    split_high_freq_entities = 0

    for (conv_id, normalized), units in sorted(by_conversation_entity.items()):
        entity = entity_display[(conv_id, normalized)]
        entity_id = entity_node_id(conv_id, normalized)
        if entity_id not in graph.nodes:
            graph.add_node(entity_node(conv_id, entity, normalized, units))
            entity_nodes_added += 1

        clusters, direct_units = cluster_entity_facts(units, config)
        if len(units) > config.max_state_facts:
            high_freq_entities += 1
            if len(clusters) > 1:
                split_high_freq_entities += 1

        for unit in direct_units:
            edge_key = (unit.node.node_id, entity_id, "fact_entity_direct")
            if edge_key in existing_edges:
                continue
            graph.add_edge(
                Edge(
                    src=unit.node.node_id,
                    dst=entity_id,
                    relation=RelationType.IS_SPECIFIC_OF,
                    confidence=0.72,
                    metadata={
                        "hierarchy_v3_5": "fact_entity_direct",
                        "hierarchy_v3_5_entity": "fact_entity_direct",
                        "view": "entity",
                        "builder": ENTITY_VIEW_VERSION,
                        "entity": entity,
                        "normalized_entity": normalized,
                    },
                )
            )
            existing_edges.add(edge_key)
            direct_fact_entity_edges_added += 1
            direct_fact_units += 1

        for state_index, cluster in enumerate(clusters, start=1):
            state_id = entity_state_node_id(conv_id, normalized, state_index)
            state = entity_state_node(state_id, conv_id, entity, cluster, config)
            if state_id not in graph.nodes:
                graph.add_node(state)
                state_nodes_added += 1
            edge_key = (state_id, entity_id, "entity_state")
            if edge_key not in existing_edges:
                graph.add_edge(
                    Edge(
                        src=state_id,
                        dst=entity_id,
                        relation=RelationType.IS_SPECIFIC_OF,
                        confidence=state.confidence,
                        metadata={
                            "hierarchy_v3_5": "entity_state",
                            "hierarchy_v3_5_entity": "entity_state",
                            "view": "entity",
                            "builder": ENTITY_VIEW_VERSION,
                            "entity": entity,
                            "normalized_entity": normalized,
                        },
                    )
                )
                existing_edges.add(edge_key)
                state_entity_edges_added += 1
            for unit in cluster.facts:
                edge_key = (unit.node.node_id, state_id, "fact_entity_state")
                if edge_key in existing_edges:
                    continue
                graph.add_edge(
                    Edge(
                        src=unit.node.node_id,
                        dst=state_id,
                        relation=RelationType.IS_SPECIFIC_OF,
                        confidence=state.confidence,
                        metadata={
                            "hierarchy_v3_5": "fact_entity_state",
                            "hierarchy_v3_5_entity": "fact_entity_state",
                            "view": "entity",
                            "builder": ENTITY_VIEW_VERSION,
                            "entity": entity,
                            "normalized_entity": normalized,
                        },
                    )
                )
                existing_edges.add(edge_key)
                fact_state_edges_added += 1
                clustered_fact_units += 1

    return {
        "input_fact_entity_units": len(fact_units),
        "entity_nodes_added": entity_nodes_added,
        "entity_state_nodes_added": state_nodes_added,
        "fact_state_edges_added": fact_state_edges_added,
        "state_entity_edges_added": state_entity_edges_added,
        "direct_fact_entity_edges_added": direct_fact_entity_edges_added,
        "clustered_fact_units": clustered_fact_units,
        "direct_fact_units": direct_fact_units,
        "high_freq_entities": high_freq_entities,
        "split_high_freq_entities": split_high_freq_entities,
        "hub_entity_split_ratio": split_high_freq_entities / high_freq_entities if high_freq_entities else 1.0,
    }


def cluster_entity_facts(
    units: Sequence[FactUnit],
    config: EntityViewConfig,
) -> tuple[list[EntityStateCluster], list[FactUnit]]:
    ordered = sorted(units, key=fact_sort_key)
    if len(ordered) < config.min_state_facts:
        return [], list(ordered)
    clusters: list[EntityStateCluster] = []
    direct: list[FactUnit] = []
    for unit in ordered:
        best_index = -1
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            if len(cluster.facts) >= config.max_state_facts:
                continue
            score = entity_state_similarity(unit, cluster)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index >= 0 and best_score >= config.cluster_threshold:
            clusters[best_index].add(unit)
        else:
            clusters.append(EntityStateCluster.from_fact(unit))

    kept: list[EntityStateCluster] = []
    for cluster in clusters:
        if len(cluster.facts) >= config.min_state_facts:
            kept.append(cluster)
        else:
            direct.extend(cluster.facts)
    return kept, direct


def entity_state_similarity(unit: FactUnit, cluster: EntityStateCluster) -> float:
    vectors = [fact.vector for fact in cluster.facts]
    centroid = np.mean(vectors, axis=0) if vectors else unit.vector
    same_episode = 1.0 if unit.episode_id and unit.episode_id in cluster.episode_ids else 0.0
    same_event = 1.0 if unit.event_id and unit.event_id in cluster.event_ids else 0.0
    same_session = 1.0 if unit.session_id and unit.session_id in cluster.session_ids else 0.0
    return (
        0.35 * max(0.0, cosine(unit.vector, centroid))
        + 0.25 * jaccard(unit.terms, cluster.terms)
        + 0.18 * same_episode
        + 0.12 * same_event
        + 0.10 * same_session
    )


def ranked_entities_for_fact(node: Node, graph: MemoryGraph, index: SemanticEventIndex) -> list[str]:
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for entity in extract_entities(node.text):
        if entity in _GENERIC_ENTITY_STOPWORDS:
            continue
        normalized = normalize_entity(entity)
        if not normalized:
            continue
        counts[normalized] += 2.0
        display.setdefault(normalized, entity)
    speaker = speaker_for_fact(node, graph)
    if speaker:
        normalized = normalize_entity(speaker)
        counts[normalized] += 3.0
        display.setdefault(normalized, speaker)
    event_id = index.fact_to_event.get(node.node_id, "")
    for entity in index.event_entities.get(event_id, []):
        normalized = normalize_entity(entity)
        if normalized:
            counts[normalized] += 1.5
            display.setdefault(normalized, entity)
    episode_id = index.event_to_episode.get(event_id, "")
    for entity in index.episode_entities.get(episode_id, []):
        normalized = normalize_entity(entity)
        if normalized:
            counts[normalized] += 0.75
            display.setdefault(normalized, entity)
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    return [display[key] for key in ranked]


def speaker_for_fact(node: Node, graph: MemoryGraph) -> str:
    raw_ids = list(node.metadata.get("support_raw_ids") or node.support_ids)
    for raw_id in raw_ids:
        raw = graph.nodes.get(str(raw_id))
        if raw is not None and raw.metadata.get("speaker"):
            return str(raw.metadata["speaker"])
    match = re.match(r"([A-Z][A-Za-z]{2,}) said:", node.text)
    return match.group(1) if match else ""


def first_session_id(node: Node) -> str:
    for turn_id in node.metadata.get("support_turn_ids", []):
        session_id = canonical_session_id(turn_id)
        if session_id:
            return session_id
    for raw_id in node.metadata.get("support_raw_ids", []):
        session_id = canonical_session_id(raw_id)
        if session_id:
            return session_id
    return ""


def entity_node(conv_id: str, entity: str, normalized: str, units: Sequence[FactUnit]) -> Node:
    support_ids = sorted({unit.node.node_id for unit in units})
    sessions = sorted({unit.session_id for unit in units if unit.session_id})
    return Node(
        node_id=entity_node_id(conv_id, normalized),
        type=NodeType.ENTITY,
        text=f"Entity: {entity} in {conv_id}.",
        source="entity_view_v3_5",
        confidence=0.94,
        support_ids=support_ids,
        metadata={
            "conversation_id": conv_id,
            "entity": entity,
            "normalized_entity": normalized,
            "fact_ids": support_ids,
            "session_ids": sessions,
            "hierarchy_v3_5": "entity",
            "view": "entity",
            "builder": ENTITY_VIEW_VERSION,
        },
    )


def entity_state_node(
    state_id: str,
    conv_id: str,
    entity: str,
    cluster: EntityStateCluster,
    config: EntityViewConfig,
) -> Node:
    keywords = ranked_terms(cluster.facts, config.max_state_keywords)
    fact_ids = [fact.node.node_id for fact in cluster.facts]
    coherence = cluster_coherence(cluster)
    label = ", ".join(keywords[:4]) or "state"
    return Node(
        node_id=state_id,
        type=NodeType.ENTITY_STATE,
        text=f"EntityState: {entity} - {label}.",
        source="entity_state_view_v3_5",
        confidence=max(0.55, min(0.98, 0.55 + 0.40 * coherence)),
        support_ids=fact_ids,
        metadata={
            "conversation_id": conv_id,
            "entity": entity,
            "normalized_entity": cluster.normalized_entity,
            "aspect_keywords": keywords,
            "fact_ids": fact_ids,
            "event_ids": sorted(cluster.event_ids),
            "episode_ids": sorted(cluster.episode_ids),
            "session_ids": sorted(cluster.session_ids),
            "num_facts": len(cluster.facts),
            "coherence": coherence,
            "hierarchy_v3_5": "entity_state",
            "view": "entity",
            "builder": ENTITY_VIEW_VERSION,
        },
    )


def entity_view_diagnostics(graph: MemoryGraph) -> dict:
    entity_nodes = list(graph.iter_nodes(NodeType.ENTITY))
    state_nodes = list(graph.iter_nodes(NodeType.ENTITY_STATE))
    state_sizes = [int(node.metadata.get("num_facts", len(node.support_ids))) for node in state_nodes]
    coherences = [float(node.metadata.get("coherence", 0.0)) for node in state_nodes]
    semantic_fact_ids = semantic_fact_ids_from_graph(graph)
    fact_entity_edges = [
        edge
        for edge in graph.edges
        if edge.metadata.get("hierarchy_v3_5_entity") in {"fact_entity_state", "fact_entity_direct"}
    ]
    facts_with_entity_path = {edge.src for edge in fact_entity_edges}
    entity_state_edges = [
        edge for edge in graph.edges if edge.metadata.get("hierarchy_v3_5_entity") == "entity_state"
    ]
    states_by_entity: dict[str, int] = defaultdict(int)
    for edge in entity_state_edges:
        states_by_entity[edge.dst] += 1
    high_freq_entities = [
        node
        for node in entity_nodes
        if len(node.metadata.get("fact_ids", node.support_ids)) > 20
    ]
    split_high_freq = [node for node in high_freq_entities if states_by_entity.get(node.node_id, 0) > 1]
    return {
        "entity_count": len(entity_nodes),
        "entity_state_count": len(state_nodes),
        "semantic_fact_count": len(semantic_fact_ids),
        "fact_with_entity_path_count": len(facts_with_entity_path & semantic_fact_ids),
        "fact_with_entity_path_ratio": len(facts_with_entity_path & semantic_fact_ids) / len(semantic_fact_ids)
        if semantic_fact_ids
        else 0.0,
        "mean_facts_per_entity_state": safe_mean(state_sizes),
        "median_facts_per_entity_state": safe_median(state_sizes),
        "max_facts_per_entity_state": max(state_sizes, default=0),
        "singleton_state_ratio": ratio_equal(state_sizes, 1),
        "entity_state_coherence_mean": safe_mean(coherences),
        "entity_state_coherence_median": safe_median(coherences),
        "entity_state_coherence_min": min(coherences, default=0.0),
        "entity_state_coherence_max": max(coherences, default=0.0),
        "direct_fact_entity_edge_count": sum(
            1 for edge in fact_entity_edges if edge.metadata.get("hierarchy_v3_5_entity") == "fact_entity_direct"
        ),
        "fact_entity_state_edge_count": sum(
            1 for edge in fact_entity_edges if edge.metadata.get("hierarchy_v3_5_entity") == "fact_entity_state"
        ),
        "high_freq_entity_count": len(high_freq_entities),
        "split_high_freq_entity_count": len(split_high_freq),
        "hub_entity_split_ratio": len(split_high_freq) / len(high_freq_entities) if high_freq_entities else 1.0,
        "largest_entity_fact_counts": largest_entity_counts(entity_nodes, 10),
        "largest_entity_state_sizes": sorted(state_sizes, reverse=True)[:10],
    }


def strip_existing_v3_5_entity(graph: MemoryGraph) -> MemoryGraph:
    output = graph.model_copy(deep=True)
    generated_node_ids = {
        node_id
        for node_id, node in output.nodes.items()
        if node.metadata.get("hierarchy_v3_5") in {"entity", "entity_state"}
        or node.source in {"entity_view_v3_5", "entity_state_view_v3_5"}
    }
    for node_id in generated_node_ids:
        output.nodes.pop(node_id, None)
    output.edges = [
        edge
        for edge in output.edges
        if edge.src not in generated_node_ids
        and edge.dst not in generated_node_ids
        and not edge.metadata.get("hierarchy_v3_5_entity")
    ]
    metadata = dict(output.metadata)
    metadata.pop("hierarchy_v3_5", None)
    output.metadata = metadata
    return output


def semantic_fact_ids_from_graph(graph: MemoryGraph) -> set[str]:
    output: set[str] = set()
    for edge in graph.edges:
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        role = edge.metadata.get("hierarchy_v3_3") or edge.metadata.get("hierarchy_v3")
        if src is not None and dst is not None and src.type == NodeType.FACT and dst.type == NodeType.EVENT and role == "fact_event":
            output.add(src.node_id)
    return output


def content_terms(text: str) -> set[str]:
    return {
        term
        for term in tokenize(text)
        if len(term) >= 3 and term not in _TERM_STOPWORDS and not term.isdigit()
    }


def ranked_terms(facts: Sequence[FactUnit], limit: int) -> list[str]:
    counts: Counter[str] = Counter()
    for fact in facts:
        counts.update(fact.terms)
    return [term for term, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def cluster_coherence(cluster: EntityStateCluster) -> float:
    if len(cluster.facts) < 2:
        return 0.5
    values: list[float] = []
    for index, left in enumerate(cluster.facts):
        for right in cluster.facts[index + 1 :]:
            values.append(max(0.0, cosine(left.vector, right.vector)))
    return max(0.0, min(1.0, float(mean(values)))) if values else 0.5


def normalize_entity(entity: str) -> str:
    value = " ".join(str(entity).strip().split())
    if not value or value in _GENERIC_ENTITY_STOPWORDS:
        return ""
    tokens = [token.lower() for token in tokenize(value)]
    tokens = [token for token in tokens if token and token not in _TERM_STOPWORDS]
    if not tokens:
        return ""
    return "_".join(tokens)


def entity_node_id(conv_id: str, normalized: str) -> str:
    return f"{conv_id}:entityv3_5:{slug(normalized)}"


def entity_state_node_id(conv_id: str, normalized: str, index: int) -> str:
    return f"{conv_id}:entitystatev3_5:{slug(normalized)}:{index:04d}"


def slug(value: str) -> str:
    clean = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return clean or "entity"


def fact_sort_key(unit: FactUnit) -> tuple[str, str, str, str]:
    return (unit.episode_id or "zzzz", unit.session_id or "zzzz", unit.event_id or "zzzz", unit.node.node_id)


def largest_entity_counts(nodes: Sequence[Node], limit: int) -> list[dict]:
    rows = []
    for node in nodes:
        rows.append(
            {
                "entity": node.metadata.get("entity", node.text),
                "node_id": node.node_id,
                "fact_count": len(node.metadata.get("fact_ids", node.support_ids)),
            }
        )
    return sorted(rows, key=lambda item: (-int(item["fact_count"]), str(item["entity"])))[:limit]


def ratio_equal(values: Sequence[int | float], target: int | float) -> float:
    return sum(1 for value in values if value == target) / len(values) if values else 0.0


def safe_mean(values: Sequence[int | float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_median(values: Sequence[int | float]) -> float:
    return float(median(values)) if values else 0.0
