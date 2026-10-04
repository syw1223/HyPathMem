from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from statistics import mean, median
from typing import Iterable, Sequence

import numpy as np

from hytopomem.memory.entity_view_builder import ENTITY_VIEW_VERSION
from hytopomem.memory.hierarchy_builder import cosine
from hytopomem.memory.multiview_graph_builder import conversation_id
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType
from hytopomem.models.text_encoder import HashTextEncoder


EVIDENCE_PACK_VERSION = "v3_5_evidence_pack_v1"


@dataclass(frozen=True)
class EvidencePackConfig:
    min_bridge_facts: int = 2
    max_pack_facts: int = 24
    graph_id_suffix: str = "evidence_pack_v3_5"


@dataclass(frozen=True)
class PackSpec:
    pack_id: str
    pack_type: str
    text: str
    fact_ids: tuple[str, ...]
    raw_ids: tuple[str, ...]
    member_ids: tuple[str, ...]
    metadata: dict


class EvidencePackBuilder:
    def __init__(self, encoder: HashTextEncoder | None = None, config: EvidencePackConfig | None = None):
        self.encoder = encoder or HashTextEncoder(dim=256)
        self.config = config or EvidencePackConfig()

    def build(self, graph: MemoryGraph, *, graph_id: str | None = None) -> tuple[MemoryGraph, dict]:
        output = strip_existing_v3_5_packs(graph)
        output.graph_id = graph_id or f"{graph.graph_id}_{self.config.graph_id_suffix}"

        index = PackIndex.from_graph(output)
        specs = []
        specs.extend(episode_pack_specs(output, index, self.config))
        specs.extend(entity_state_pack_specs(output, index, self.config))
        specs.extend(bridge_pack_specs(output, index, self.config))
        stats = add_pack_nodes(output, specs, self.encoder, index)
        stats["source_graph_id"] = graph.graph_id
        stats["version"] = EVIDENCE_PACK_VERSION
        stats["diagnostics"] = evidence_pack_diagnostics(output)

        metadata = dict(output.metadata)
        metadata["evidence_packs_v3_5"] = stats
        output.metadata = metadata
        return output, stats


@dataclass(frozen=True)
class PackIndex:
    fact_to_event: dict[str, str]
    fact_event_confidence: dict[tuple[str, str], float]
    event_to_facts: dict[str, list[str]]
    event_to_episode: dict[str, str]
    event_episode_confidence: dict[tuple[str, str], float]
    episode_to_events: dict[str, list[str]]
    entity_state_to_facts: dict[str, list[str]]
    fact_entity_state_confidence: dict[tuple[str, str], float]
    entity_state_to_entity: dict[str, str]
    fact_raw_ids: dict[str, list[str]]

    @classmethod
    def from_graph(cls, graph: MemoryGraph) -> "PackIndex":
        fact_to_event: dict[str, str] = {}
        fact_event_confidence: dict[tuple[str, str], float] = {}
        event_to_facts: dict[str, list[str]] = defaultdict(list)
        event_to_episode: dict[str, str] = {}
        event_episode_confidence: dict[tuple[str, str], float] = {}
        episode_to_events: dict[str, list[str]] = defaultdict(list)
        entity_state_to_facts: dict[str, list[str]] = defaultdict(list)
        fact_entity_state_confidence: dict[tuple[str, str], float] = {}
        entity_state_to_entity: dict[str, str] = {}
        for edge in graph.edges:
            src = graph.nodes.get(edge.src)
            dst = graph.nodes.get(edge.dst)
            if src is None or dst is None:
                continue
            role = edge.metadata.get("hierarchy_v3_3") or edge.metadata.get("hierarchy_v3")
            if src.type == NodeType.FACT and dst.type == NodeType.EVENT and role == "fact_event":
                fact_to_event[src.node_id] = dst.node_id
                fact_event_confidence[(src.node_id, dst.node_id)] = float(edge.confidence)
                event_to_facts[dst.node_id].append(src.node_id)
            elif (
                src.type == NodeType.EVENT
                and dst.type == NodeType.EVENT
                and edge.metadata.get("hierarchy_v3_3") == "event_episode"
            ):
                event_to_episode[src.node_id] = dst.node_id
                event_episode_confidence[(src.node_id, dst.node_id)] = float(edge.confidence)
                episode_to_events[dst.node_id].append(src.node_id)
            elif (
                src.type == NodeType.FACT
                and dst.type == NodeType.ENTITY_STATE
                and edge.metadata.get("hierarchy_v3_5_entity") == "fact_entity_state"
            ):
                entity_state_to_facts[dst.node_id].append(src.node_id)
                fact_entity_state_confidence[(src.node_id, dst.node_id)] = float(edge.confidence)
            elif (
                src.type == NodeType.ENTITY_STATE
                and dst.type == NodeType.ENTITY
                and edge.metadata.get("hierarchy_v3_5_entity") == "entity_state"
            ):
                entity_state_to_entity[src.node_id] = dst.node_id
        fact_raw_ids = {
            node.node_id: list(node.metadata.get("support_raw_ids") or node.support_ids)
            for node in graph.iter_nodes(NodeType.FACT)
        }
        return cls(
            fact_to_event=dict(fact_to_event),
            fact_event_confidence=dict(fact_event_confidence),
            event_to_facts={key: sorted(set(values)) for key, values in event_to_facts.items()},
            event_to_episode=dict(event_to_episode),
            event_episode_confidence=dict(event_episode_confidence),
            episode_to_events={key: sorted(set(values)) for key, values in episode_to_events.items()},
            entity_state_to_facts={key: sorted(set(values)) for key, values in entity_state_to_facts.items()},
            fact_entity_state_confidence=dict(fact_entity_state_confidence),
            entity_state_to_entity=dict(entity_state_to_entity),
            fact_raw_ids=fact_raw_ids,
        )


def episode_pack_specs(graph: MemoryGraph, index: PackIndex, config: EvidencePackConfig) -> list[PackSpec]:
    specs: list[PackSpec] = []
    for episode_id, event_ids in sorted(index.episode_to_events.items()):
        episode = graph.nodes.get(episode_id)
        if episode is None:
            continue
        fact_ids = dedupe(
            fact_id
            for event_id in event_ids
            for fact_id in index.event_to_facts.get(event_id, [])
        )[: config.max_pack_facts]
        if not fact_ids:
            continue
        raw_ids = raw_ids_for_facts(fact_ids, index)
        specs.append(
            PackSpec(
                pack_id=f"{conversation_id(episode_id)}:packv3_5:episode:{episode_id.rsplit(':', 1)[-1]}",
                pack_type="episode",
                text=f"EpisodePack: {episode.text}",
                fact_ids=tuple(fact_ids),
                raw_ids=tuple(raw_ids),
                member_ids=tuple([episode_id, *event_ids, *fact_ids]),
                metadata={
                    "episode_id": episode_id,
                    "event_ids": event_ids,
                    "entities": list(episode.metadata.get("entities", [])),
                    "keywords": list(episode.metadata.get("keywords", [])),
                },
            )
        )
    return specs


def entity_state_pack_specs(graph: MemoryGraph, index: PackIndex, config: EvidencePackConfig) -> list[PackSpec]:
    specs: list[PackSpec] = []
    for state_id, fact_ids_all in sorted(index.entity_state_to_facts.items()):
        state = graph.nodes.get(state_id)
        if state is None:
            continue
        fact_ids = list(fact_ids_all)[: config.max_pack_facts]
        event_ids = sorted({index.fact_to_event.get(fact_id, "") for fact_id in fact_ids if index.fact_to_event.get(fact_id)})
        entity_id = index.entity_state_to_entity.get(state_id, "")
        raw_ids = raw_ids_for_facts(fact_ids, index)
        specs.append(
            PackSpec(
                pack_id=state_id.replace(":entitystatev3_5:", ":packv3_5:entitystate:"),
                pack_type="entity_state",
                text=f"EntityStatePack: {state.text}",
                fact_ids=tuple(fact_ids),
                raw_ids=tuple(raw_ids),
                member_ids=tuple([item for item in [entity_id, state_id, *event_ids, *fact_ids] if item]),
                metadata={
                    "entity_id": entity_id,
                    "entity_state_id": state_id,
                    "event_ids": event_ids,
                    "entity": state.metadata.get("entity"),
                    "aspect_keywords": list(state.metadata.get("aspect_keywords", [])),
                },
            )
        )
    return specs


def bridge_pack_specs(graph: MemoryGraph, index: PackIndex, config: EvidencePackConfig) -> list[PackSpec]:
    specs: list[PackSpec] = []
    for state_id, state_fact_ids in sorted(index.entity_state_to_facts.items()):
        facts_by_episode: dict[str, list[str]] = defaultdict(list)
        for fact_id in state_fact_ids:
            event_id = index.fact_to_event.get(fact_id, "")
            episode_id = index.event_to_episode.get(event_id, "")
            if episode_id:
                facts_by_episode[episode_id].append(fact_id)
        for episode_id, fact_ids_all in sorted(facts_by_episode.items()):
            fact_ids = sorted(set(fact_ids_all))
            if len(fact_ids) < config.min_bridge_facts:
                continue
            fact_ids = fact_ids[: config.max_pack_facts]
            state = graph.nodes.get(state_id)
            episode = graph.nodes.get(episode_id)
            if state is None or episode is None:
                continue
            raw_ids = raw_ids_for_facts(fact_ids, index)
            event_ids = sorted({index.fact_to_event.get(fact_id, "") for fact_id in fact_ids if index.fact_to_event.get(fact_id)})
            suffix = f"{state_id.rsplit(':', 2)[-2]}_{state_id.rsplit(':', 1)[-1]}_{episode_id.rsplit(':', 1)[-1]}"
            specs.append(
                PackSpec(
                    pack_id=f"{conversation_id(state_id)}:packv3_5:bridge:{suffix}",
                    pack_type="bridge_entity_episode",
                    text=f"BridgePack: {state.text} within {episode.text}",
                    fact_ids=tuple(fact_ids),
                    raw_ids=tuple(raw_ids),
                    member_ids=tuple([state_id, episode_id, *event_ids, *fact_ids]),
                    metadata={
                        "entity_state_id": state_id,
                        "episode_id": episode_id,
                        "event_ids": event_ids,
                        "entity": state.metadata.get("entity"),
                        "aspect_keywords": list(state.metadata.get("aspect_keywords", [])),
                        "episode_keywords": list(episode.metadata.get("keywords", [])),
                    },
                )
            )
    return specs


def add_pack_nodes(graph: MemoryGraph, specs: Sequence[PackSpec], encoder: HashTextEncoder, index: PackIndex) -> dict:
    existing_edges = {(edge.src, edge.dst, edge.metadata.get("hierarchy_v3_5_pack")) for edge in graph.edges}
    added_nodes = 0
    added_edges = 0
    fact_texts = {node.node_id: node.text for node in graph.iter_nodes(NodeType.FACT)}
    vectors_by_fact = encode_fact_vectors(fact_texts, encoder)
    for spec in specs:
        coherence = pack_coherence(spec.fact_ids, vectors_by_fact)
        centrality = fact_pack_centrality(spec.fact_ids, vectors_by_fact)
        if spec.pack_id not in graph.nodes:
            graph.add_node(pack_node(spec, coherence))
            added_nodes += 1
        for member_id in spec.member_ids:
            if member_id not in graph.nodes:
                continue
            member_weight = membership_weight(graph, spec, member_id, coherence, centrality, index)
            edge_key = (member_id, spec.pack_id, "pack_member")
            if edge_key in existing_edges:
                continue
            graph.add_edge(
                Edge(
                    src=member_id,
                    dst=spec.pack_id,
                    relation=RelationType.MEMBER_OF,
                    confidence=member_weight,
                    metadata={
                        "hierarchy_v3_5": "pack_member",
                        "hierarchy_v3_5_pack": "pack_member",
                        "view": "hyperedge_pack",
                        "builder": EVIDENCE_PACK_VERSION,
                        "pack_type": spec.pack_type,
                        "membership_weight": member_weight,
                        "membership_weight_source": "structure_embedding_v1",
                    },
                )
            )
            existing_edges.add(edge_key)
            added_edges += 1
    return {
        "pack_nodes_added": added_nodes,
        "pack_member_edges_added": added_edges,
        "pack_specs": len(specs),
        "pack_type_counts": pack_type_counts(specs),
    }


def pack_node(spec: PackSpec, coherence: float) -> Node:
    metadata = dict(spec.metadata)
    metadata.update(
        {
            "pack_type": spec.pack_type,
            "fact_ids": list(spec.fact_ids),
            "raw_ids": list(spec.raw_ids),
            "member_ids": list(spec.member_ids),
            "num_facts": len(spec.fact_ids),
            "num_raw": len(spec.raw_ids),
            "coherence": coherence,
            "hierarchy_v3_5": "evidence_pack",
            "view": "hyperedge_pack",
            "builder": EVIDENCE_PACK_VERSION,
        }
    )
    return Node(
        node_id=spec.pack_id,
        type=NodeType.EVIDENCE_PACK,
        text=spec.text,
        source="evidence_pack_v3_5",
        confidence=max(0.55, min(0.98, 0.55 + 0.40 * coherence)),
        support_ids=list(spec.fact_ids),
        metadata=metadata,
    )


def evidence_pack_diagnostics(graph: MemoryGraph) -> dict:
    packs = list(graph.iter_nodes(NodeType.EVIDENCE_PACK))
    sizes = [int(pack.metadata.get("num_facts", len(pack.support_ids))) for pack in packs]
    coherences = [float(pack.metadata.get("coherence", 0.0)) for pack in packs]
    member_weights = [
        float(edge.metadata.get("membership_weight", edge.confidence))
        for edge in graph.edges
        if edge.relation == RelationType.MEMBER_OF and edge.metadata.get("hierarchy_v3_5_pack") == "pack_member"
    ]
    by_type: dict[str, list[Node]] = defaultdict(list)
    for pack in packs:
        by_type[str(pack.metadata.get("pack_type", "unknown"))].append(pack)
    return {
        "pack_count": len(packs),
        "pack_type_counts": {key: len(value) for key, value in sorted(by_type.items())},
        "mean_pack_size": safe_mean(sizes),
        "median_pack_size": safe_median(sizes),
        "pack_size_p90": percentile(sizes, 0.90),
        "max_pack_size": max(sizes, default=0),
        "pack_coherence_mean": safe_mean(coherences),
        "pack_coherence_median": safe_median(coherences),
        "pack_coherence_min": min(coherences, default=0.0),
        "pack_coherence_max": max(coherences, default=0.0),
        "membership_weight_mean": safe_mean(member_weights),
        "membership_weight_median": safe_median(member_weights),
        "membership_weight_min": min(member_weights, default=0.0),
        "membership_weight_max": max(member_weights, default=0.0),
        "largest_pack_sizes": sorted(sizes, reverse=True)[:10],
        "pack_type_size_summary": {
            key: {
                "count": len(nodes),
                "mean_size": safe_mean([int(node.metadata.get("num_facts", 0)) for node in nodes]),
                "max_size": max([int(node.metadata.get("num_facts", 0)) for node in nodes], default=0),
                "mean_coherence": safe_mean([float(node.metadata.get("coherence", 0.0)) for node in nodes]),
            }
            for key, nodes in sorted(by_type.items())
        },
    }


def strip_existing_v3_5_packs(graph: MemoryGraph) -> MemoryGraph:
    output = graph.model_copy(deep=True)
    generated_node_ids = {
        node_id
        for node_id, node in output.nodes.items()
        if node.metadata.get("hierarchy_v3_5") == "evidence_pack"
        or node.source == "evidence_pack_v3_5"
    }
    for node_id in generated_node_ids:
        output.nodes.pop(node_id, None)
    output.edges = [
        edge
        for edge in output.edges
        if edge.src not in generated_node_ids
        and edge.dst not in generated_node_ids
        and not edge.metadata.get("hierarchy_v3_5_pack")
    ]
    metadata = dict(output.metadata)
    metadata.pop("evidence_packs_v3_5", None)
    output.metadata = metadata
    return output


def raw_ids_for_facts(fact_ids: Sequence[str], index: PackIndex) -> list[str]:
    return dedupe(raw_id for fact_id in fact_ids for raw_id in index.fact_raw_ids.get(fact_id, []))


def encode_fact_vectors(texts: dict[str, str], encoder: HashTextEncoder) -> dict[str, np.ndarray]:
    ids = sorted(texts)
    if not ids:
        return {}
    matrix = encoder.encode([texts[fact_id] for fact_id in ids])
    return dict(zip(ids, matrix))


def pack_coherence(fact_ids: Sequence[str], vectors_by_fact: dict[str, np.ndarray]) -> float:
    vectors = [vectors_by_fact[fact_id] for fact_id in fact_ids if fact_id in vectors_by_fact]
    if len(vectors) < 2:
        return 0.5
    values = []
    for left, right in combinations(vectors, 2):
        values.append(max(0.0, cosine(left, right)))
    return max(0.0, min(1.0, float(mean(values)))) if values else 0.5


def fact_pack_centrality(fact_ids: Sequence[str], vectors_by_fact: dict[str, np.ndarray]) -> dict[str, float]:
    ids = [fact_id for fact_id in fact_ids if fact_id in vectors_by_fact]
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: 1.0}
    matrix = np.vstack([vectors_by_fact[fact_id] for fact_id in ids])
    centroid = matrix.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    raw_scores = np.maximum(0.0, matrix @ centroid)
    lo = float(raw_scores.min())
    hi = float(raw_scores.max())
    if hi <= lo:
        return {fact_id: 0.5 for fact_id in ids}
    return {fact_id: float((score - lo) / (hi - lo)) for fact_id, score in zip(ids, raw_scores)}


def membership_weight(
    graph: MemoryGraph,
    spec: PackSpec,
    member_id: str,
    coherence: float,
    centrality: dict[str, float],
    index: PackIndex,
) -> float:
    node = graph.nodes.get(member_id)
    if node is None:
        return 0.0
    if node.type != NodeType.FACT:
        return clamp01(0.70 + 0.20 * coherence)
    fact_confidence = float(node.confidence)
    event_id = index.fact_to_event.get(member_id, "")
    episode_id = str(spec.metadata.get("episode_id", ""))
    state_id = str(spec.metadata.get("entity_state_id", ""))
    fact_event_conf = index.fact_event_confidence.get((member_id, event_id), 0.5)
    event_episode_conf = index.event_episode_confidence.get((event_id, episode_id), 0.5) if episode_id else 0.5
    fact_state_conf = index.fact_entity_state_confidence.get((member_id, state_id), 0.5) if state_id else 0.5
    hierarchy_conf = {
        "episode": 0.55 * fact_event_conf + 0.45 * event_episode_conf,
        "entity_state": 0.60 * fact_state_conf + 0.40 * fact_event_conf,
        "bridge_entity_episode": 0.40 * fact_state_conf + 0.30 * fact_event_conf + 0.30 * event_episode_conf,
    }.get(spec.pack_type, fact_event_conf)
    weight = (
        0.30 * hierarchy_conf
        + 0.25 * fact_confidence
        + 0.25 * coherence
        + 0.20 * centrality.get(member_id, 0.5)
    )
    return clamp01(0.05 + 0.95 * weight)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def pack_type_counts(specs: Sequence[PackSpec]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for spec in specs:
        counts[spec.pack_type] += 1
    return dict(sorted(counts.items()))


def dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def percentile(values: Sequence[int | float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def safe_mean(values: Sequence[int | float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_median(values: Sequence[int | float]) -> float:
    return float(median(values)) if values else 0.0
