from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable, Sequence

from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType


TEMPORAL_VIEW_VERSION = "v3_4_temporal_session_v1"
_SESSION_RE = re.compile(r"\b(D\d+)\b")


@dataclass(frozen=True)
class MultiViewGraphConfig:
    add_raw_provenance: bool = True
    add_temporal_view: bool = True
    graph_id_suffix: str = "multiview_v3_4"


class MultiViewGraphBuilder:
    """Build V3.4 multi-view graph additions without changing the semantic hierarchy."""

    def __init__(self, config: MultiViewGraphConfig | None = None):
        self.config = config or MultiViewGraphConfig()

    def build(self, graph: MemoryGraph, *, graph_id: str | None = None) -> tuple[MemoryGraph, dict]:
        output = strip_existing_v3_4(graph)
        output.graph_id = graph_id or f"{graph.graph_id}_{self.config.graph_id_suffix}"

        raw_index = RawIndex.from_graph(output)
        stats: dict = {
            "source_graph_id": graph.graph_id,
            "version": TEMPORAL_VIEW_VERSION,
        }
        if self.config.add_raw_provenance:
            stats["raw_provenance"] = ensure_fact_raw_provenance(output, raw_index)
        if self.config.add_temporal_view:
            stats["temporal_view"] = add_temporal_view(output, raw_index)
        stats["diagnostics"] = graph_v3_4_diagnostics(output)

        metadata = dict(output.metadata)
        metadata["hierarchy_v3_4"] = stats
        output.metadata = metadata
        return output, stats


@dataclass
class RawIndex:
    raw_by_id: dict[str, Node]
    raw_by_turn: dict[tuple[str, str], str]
    raw_by_session: dict[tuple[str, str], list[str]]

    @classmethod
    def from_graph(cls, graph: MemoryGraph) -> "RawIndex":
        raw_by_id: dict[str, Node] = {}
        raw_by_turn: dict[tuple[str, str], str] = {}
        raw_by_session: dict[tuple[str, str], list[str]] = defaultdict(list)
        for node in graph.iter_nodes(NodeType.RAW):
            raw_by_id[node.node_id] = node
            conv_id = conversation_id(node.node_id)
            turn_id = str(node.metadata.get("turn_id") or node.node_id.split(":raw:", 1)[-1])
            raw_by_turn[(conv_id, turn_id)] = node.node_id
            session_id = canonical_session_id(turn_id)
            if session_id:
                raw_by_session[(conv_id, session_id)].append(node.node_id)
        return cls(
            raw_by_id=raw_by_id,
            raw_by_turn=dict(raw_by_turn),
            raw_by_session=dict(raw_by_session),
        )


def ensure_fact_raw_provenance(graph: MemoryGraph, raw_index: RawIndex) -> dict:
    existing_support_edges = {
        (edge.src, edge.dst)
        for edge in graph.edges
        if edge.relation == RelationType.SUPPORTS
    }
    fixed_facts = 0
    missing_raw_facts = 0
    added_edges = 0
    unresolved: set[str] = set()
    for node in graph.iter_nodes(NodeType.FACT):
        raw_ids, unresolved_values = support_raw_ids_for_node(node, graph, raw_index)
        unresolved.update(unresolved_values)
        if not raw_ids:
            missing_raw_facts += 1
            continue
        old_support = list(node.support_ids)
        old_metadata = dict(node.metadata)
        metadata = dict(node.metadata)
        metadata["support_raw_ids"] = raw_ids
        metadata["support_texts"] = [graph.nodes[raw_id].text for raw_id in raw_ids if raw_id in graph.nodes]
        metadata["support_timestamps"] = [
            str(graph.nodes[raw_id].time) for raw_id in raw_ids if raw_id in graph.nodes and graph.nodes[raw_id].time
        ]
        metadata["support_turn_ids"] = [
            str(graph.nodes[raw_id].metadata.get("turn_id") or raw_id.split(":raw:", 1)[-1])
            for raw_id in raw_ids
            if raw_id in graph.nodes
        ]
        node.support_ids = raw_ids
        node.metadata = metadata
        if node.time is None and metadata["support_timestamps"]:
            node.time = metadata["support_timestamps"][0]
        if node.support_ids != old_support or node.metadata != old_metadata:
            fixed_facts += 1
        for raw_id in raw_ids:
            if (node.node_id, raw_id) in existing_support_edges or raw_id not in graph.nodes:
                continue
            graph.edges.append(
                Edge(
                    src=node.node_id,
                    dst=raw_id,
                    relation=RelationType.SUPPORTS,
                    confidence=0.98,
                    metadata={
                        "hierarchy_v3_4": "fact_raw",
                        "view": "provenance",
                        "builder": TEMPORAL_VIEW_VERSION,
                    },
                )
            )
            existing_support_edges.add((node.node_id, raw_id))
            added_edges += 1
    total_facts = sum(1 for _ in graph.iter_nodes(NodeType.FACT))
    return {
        "total_facts": total_facts,
        "fixed_facts": fixed_facts,
        "missing_raw_facts": missing_raw_facts,
        "fact_with_raw_ratio": (total_facts - missing_raw_facts) / total_facts if total_facts else 0.0,
        "added_support_edges": added_edges,
        "unresolved_support_ids": len(unresolved),
        "unresolved_examples": sorted(unresolved)[:20],
    }


def add_temporal_view(graph: MemoryGraph, raw_index: RawIndex) -> dict:
    events = semantic_event_nodes(graph)
    event_sessions = {event.node_id: event_session_ids(event, graph, raw_index) for event in events}
    conversations = sorted({conversation_id(node.node_id) for node in graph.nodes.values()})
    sessions_by_conversation: dict[str, set[str]] = defaultdict(set)
    for conv_id, session_id in raw_index.raw_by_session:
        sessions_by_conversation[conv_id].add(session_id)
    for event in events:
        conv_id = conversation_id(event.node_id)
        sessions_by_conversation[conv_id].update(event_sessions[event.node_id])

    existing_edges = {
        (edge.src, edge.dst, edge.metadata.get("hierarchy_v3_4_temporal"))
        for edge in graph.edges
    }
    added_conversation_nodes = 0
    added_session_nodes = 0
    added_session_conversation_edges = 0
    added_event_session_edges = 0

    for conv_id in conversations:
        conv_node_id = conversation_node_id(conv_id)
        if conv_node_id not in graph.nodes:
            graph.add_node(conversation_node(conv_id, sorted(sessions_by_conversation.get(conv_id, []), key=session_sort_key)))
            added_conversation_nodes += 1
        for session_id in sorted(sessions_by_conversation.get(conv_id, []), key=session_sort_key):
            session_node_id = temporal_session_node_id(conv_id, session_id)
            raw_ids = raw_index.raw_by_session.get((conv_id, session_id), [])
            if session_node_id not in graph.nodes:
                graph.add_node(session_node(conv_id, session_id, raw_ids, graph))
                added_session_nodes += 1
            edge_key = (session_node_id, conv_node_id, "session_conversation")
            if edge_key not in existing_edges:
                graph.add_edge(
                    Edge(
                        src=session_node_id,
                        dst=conv_node_id,
                        relation=RelationType.IS_SPECIFIC_OF,
                        confidence=0.99,
                        metadata={
                            "hierarchy_v3_4": "session_conversation",
                            "hierarchy_v3_4_temporal": "session_conversation",
                            "view": "temporal",
                            "builder": TEMPORAL_VIEW_VERSION,
                        },
                    )
                )
                existing_edges.add(edge_key)
                added_session_conversation_edges += 1

    events_without_session = 0
    event_session_counts: list[int] = []
    for event in events:
        conv_id = conversation_id(event.node_id)
        session_ids = sorted(event_sessions[event.node_id], key=session_sort_key)
        if not session_ids:
            events_without_session += 1
            continue
        event_session_counts.append(len(session_ids))
        for session_id in session_ids:
            session_node_id = temporal_session_node_id(conv_id, session_id)
            if session_node_id not in graph.nodes:
                continue
            edge_key = (event.node_id, session_node_id, "event_session")
            if edge_key in existing_edges:
                continue
            graph.add_edge(
                Edge(
                    src=event.node_id,
                    dst=session_node_id,
                    relation=RelationType.IS_SPECIFIC_OF,
                    confidence=min(0.98, max(0.55, event.confidence)),
                    metadata={
                        "hierarchy_v3_4": "event_session",
                        "hierarchy_v3_4_temporal": "event_session",
                        "view": "temporal",
                        "builder": TEMPORAL_VIEW_VERSION,
                    },
                )
            )
            existing_edges.add(edge_key)
            added_event_session_edges += 1

    session_event_counts = temporal_session_event_counts(graph)
    return {
        "semantic_events": len(events),
        "conversation_nodes": added_conversation_nodes,
        "session_nodes": added_session_nodes,
        "session_conversation_edges": added_session_conversation_edges,
        "event_session_edges": added_event_session_edges,
        "events_without_session": events_without_session,
        "event_with_session_ratio": (len(events) - events_without_session) / len(events) if events else 0.0,
        "mean_sessions_per_event": safe_mean(event_session_counts),
        "mean_events_per_session": safe_mean(list(session_event_counts.values())),
        "median_events_per_session": safe_median(list(session_event_counts.values())),
        "max_events_per_session": max(session_event_counts.values(), default=0),
        "singleton_session_ratio": ratio_equal(list(session_event_counts.values()), 1),
    }


def graph_v3_4_diagnostics(graph: MemoryGraph) -> dict:
    node_counts: dict[str, int] = defaultdict(int)
    edge_counts: dict[str, int] = defaultdict(int)
    temporal_edge_counts: dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        node_counts[str(node.type)] += 1
    for edge in graph.edges:
        edge_counts[str(edge.relation)] += 1
        role = edge.metadata.get("hierarchy_v3_4_temporal")
        if role:
            temporal_edge_counts[str(role)] += 1
    facts = list(graph.iter_nodes(NodeType.FACT))
    facts_with_raw = sum(bool(node.metadata.get("support_raw_ids") or node.support_ids) for node in facts)
    semantic_events = semantic_event_nodes(graph)
    event_session_edges = temporal_event_session_edges(graph)
    event_with_temporal = {edge.src for edge in event_session_edges}
    return {
        "node_counts": dict(sorted(node_counts.items())),
        "edge_counts": dict(sorted(edge_counts.items())),
        "temporal_edge_counts": dict(sorted(temporal_edge_counts.items())),
        "fact_with_raw_ratio": facts_with_raw / len(facts) if facts else 0.0,
        "semantic_event_with_temporal_ratio": len(event_with_temporal) / len(semantic_events) if semantic_events else 0.0,
    }


def strip_existing_v3_4(graph: MemoryGraph) -> MemoryGraph:
    output = graph.model_copy(deep=True)
    generated_node_ids = {
        node_id
        for node_id, node in output.nodes.items()
        if node.metadata.get("hierarchy_v3_4") in {"conversation", "session"}
        or node.source in {"temporal_conversation_view_v1", "temporal_session_view_v1"}
    }
    for node_id in generated_node_ids:
        output.nodes.pop(node_id, None)
    output.edges = [
        edge
        for edge in output.edges
        if edge.src not in generated_node_ids
        and edge.dst not in generated_node_ids
        and not edge.metadata.get("hierarchy_v3_4_temporal")
    ]
    metadata = dict(output.metadata)
    metadata.pop("hierarchy_v3_4", None)
    output.metadata = metadata
    return output


def semantic_event_nodes(graph: MemoryGraph) -> list[Node]:
    return [
        node
        for node in graph.iter_nodes(NodeType.EVENT)
        if node.metadata.get("hierarchy_v3") == "event"
        and node.metadata.get("hierarchy_v3_3") != "episode"
    ]


def event_session_ids(event: Node, graph: MemoryGraph, raw_index: RawIndex) -> set[str]:
    session_ids: set[str] = set()
    raw_ids = list(event.metadata.get("support_raw_ids") or [])
    for raw_id in raw_ids:
        raw = graph.nodes.get(str(raw_id))
        if raw is None:
            continue
        turn_id = str(raw.metadata.get("turn_id") or raw.node_id.split(":raw:", 1)[-1])
        session_id = canonical_session_id(turn_id)
        if session_id:
            session_ids.add(session_id)
    if session_ids:
        return session_ids
    for value in event.metadata.get("session_ids", []):
        session_id = canonical_session_id(str(value))
        if session_id:
            session_ids.add(session_id)
    return session_ids


def support_raw_ids_for_node(node: Node, graph: MemoryGraph, raw_index: RawIndex) -> tuple[list[str], set[str]]:
    values = list(node.metadata.get("support_raw_ids") or node.support_ids)
    conv_id = conversation_id(node.node_id)
    raw_ids: list[str] = []
    unresolved: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if item in graph.nodes and graph.nodes[item].type == NodeType.RAW:
            raw_ids.append(item)
            continue
        turn_id = item.split(":raw:", 1)[-1] if ":raw:" in item else item
        raw_id = raw_index.raw_by_turn.get((conv_id, turn_id))
        if raw_id:
            raw_ids.append(raw_id)
        else:
            unresolved.add(item)
    return dedupe(raw_ids), unresolved


def conversation_node(conv_id: str, session_ids: Sequence[str]) -> Node:
    return Node(
        node_id=conversation_node_id(conv_id),
        type=NodeType.CONVERSATION,
        text=f"Conversation {conv_id} containing {len(session_ids)} sessions.",
        source="temporal_conversation_view_v1",
        confidence=1.0,
        support_ids=[temporal_session_node_id(conv_id, session_id) for session_id in session_ids],
        metadata={
            "conversation_id": conv_id,
            "session_ids": list(session_ids),
            "hierarchy_v3_4": "conversation",
            "view": "temporal",
            "builder": TEMPORAL_VIEW_VERSION,
        },
    )


def session_node(conv_id: str, session_id: str, raw_ids: Sequence[str], graph: MemoryGraph) -> Node:
    timestamps = [str(graph.nodes[raw_id].time) for raw_id in raw_ids if raw_id in graph.nodes and graph.nodes[raw_id].time]
    speakers = sorted(
        {
            str(graph.nodes[raw_id].metadata.get("speaker"))
            for raw_id in raw_ids
            if raw_id in graph.nodes and graph.nodes[raw_id].metadata.get("speaker")
        }
    )
    time_hint = f" at {timestamps[0]}" if timestamps else ""
    speaker_hint = f" involving {', '.join(speakers[:4])}" if speakers else ""
    return Node(
        node_id=temporal_session_node_id(conv_id, session_id),
        type=NodeType.SESSION,
        text=f"Session {session_id} in {conv_id}{time_hint}{speaker_hint}.",
        time=timestamps[0] if timestamps else None,
        source="temporal_session_view_v1",
        confidence=0.99,
        support_ids=list(raw_ids),
        metadata={
            "conversation_id": conv_id,
            "session_id": session_id,
            "support_raw_ids": list(raw_ids),
            "support_timestamps": timestamps,
            "speakers": speakers,
            "turn_count": len(raw_ids),
            "hierarchy_v3_4": "session",
            "view": "temporal",
            "builder": TEMPORAL_VIEW_VERSION,
        },
    )


def temporal_session_event_counts(graph: MemoryGraph) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for edge in temporal_event_session_edges(graph):
        counts[edge.dst] += 1
    return dict(counts)


def temporal_event_session_edges(graph: MemoryGraph) -> list[Edge]:
    return [
        edge
        for edge in graph.edges
        if edge.metadata.get("hierarchy_v3_4_temporal") == "event_session"
    ]


def canonical_session_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if ":raw:" in text:
        text = text.split(":raw:", 1)[-1]
    if ":" in text:
        text = text.split(":", 1)[0]
    match = _SESSION_RE.search(text)
    if match:
        return match.group(1)
    lower = text.lower()
    match = re.search(r"session[_ -]*(\d+)", lower)
    if match:
        return f"D{int(match.group(1))}"
    return text


def conversation_node_id(conv_id: str) -> str:
    return f"{conv_id}:conversationv3_4"


def temporal_session_node_id(conv_id: str, session_id: str) -> str:
    return f"{conv_id}:sessionv3_4:{session_id}"


def conversation_id(node_id: str) -> str:
    for marker in [":fact:", ":raw:", ":event", ":topic", ":episode", ":session", ":conversation"]:
        if marker in node_id:
            return node_id.split(marker, 1)[0]
    return node_id.split(":", 1)[0]


def session_sort_key(session_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", session_id)
    return (int(match.group(1)) if match else 999999, session_id)


def dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def safe_mean(values: Sequence[int | float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_median(values: Sequence[int | float]) -> float:
    return float(median(values)) if values else 0.0


def ratio_equal(values: Sequence[int | float], target: int | float) -> float:
    return sum(value == target for value in values) / len(values) if values else 0.0
