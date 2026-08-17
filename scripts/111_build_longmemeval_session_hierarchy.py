from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean

from common import load_config, resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/longmemeval_s.yaml")
    parser.add_argument("--graph", default=None)
    parser.add_argument("--output", default="outputs/longmemeval_s/graph_session_hierarchy_v1.json")
    parser.add_argument("--diagnostics", default="outputs/eval/longmemeval_session_hierarchy_v1_diagnostics.json")
    parser.add_argument("--graph-id", default="longmemeval_s_session_hierarchy_v1")
    parser.add_argument("--limit-conversations", type=int, default=0)
    parser.add_argument("--max-preview-facts", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    graph_path = resolve_path(args.graph or config["graph"]["graph_path"])
    graph = JsonGraphStore().load(graph_path)
    if args.limit_conversations:
        graph = limit_graph_conversations(graph, args.limit_conversations)

    output = build_session_hierarchy(graph, graph_id=args.graph_id, max_preview_facts=args.max_preview_facts)
    diagnostics = diagnose(output)

    output_path = resolve_path(args.output)
    diagnostics_path = resolve_path(args.diagnostics)
    JsonGraphStore().save(output, output_path)
    write_json(diagnostics, diagnostics_path)
    print(f"wrote {output_path}")
    print(f"wrote {diagnostics_path}")
    print(f"nodes={len(output.nodes)} edges={len(output.edges)}")
    print(f"structure={diagnostics['structure']}")


def build_session_hierarchy(graph: MemoryGraph, *, graph_id: str, max_preview_facts: int) -> MemoryGraph:
    output = graph.model_copy(deep=True)
    output.graph_id = graph_id

    facts_by_conv_session: dict[tuple[str, str], list[Node]] = defaultdict(list)
    facts_by_conv: dict[str, list[Node]] = defaultdict(list)
    for fact in output.iter_nodes(NodeType.FACT):
        conv_id = conversation_id(fact.node_id)
        session_key = session_key_from_fact(fact)
        facts_by_conv_session[(conv_id, session_key)].append(fact)
        facts_by_conv[conv_id].append(fact)

    event_ids_by_conv: dict[str, list[str]] = defaultdict(list)
    fact_event_edges = 0
    event_topic_edges = 0
    event_count = 0
    topic_count = 0

    for (conv_id, session_key), facts in sorted(facts_by_conv_session.items()):
        facts = sorted(facts, key=lambda node: str(node.metadata.get("turn_id") or node.node_id))
        event_count += 1
        event_id = f"{conv_id}:event_session:{safe_local_id(session_key)}"
        if event_id not in output.nodes:
            output.add_node(session_event_node(event_id, conv_id, session_key, facts, max_preview_facts))
        event_ids_by_conv[conv_id].append(event_id)
        confidence = session_confidence(facts)
        for fact in facts:
            fact.metadata["event_id"] = event_id
            output.add_edge(
                Edge(
                    src=fact.node_id,
                    dst=event_id,
                    relation=RelationType.IS_SPECIFIC_OF,
                    confidence=confidence,
                    metadata={
                        "hierarchy_v3": "fact_event",
                        "builder": "longmemeval_session_hierarchy_v1",
                        "session_key": session_key,
                    },
                )
            )
            fact_event_edges += 1

    for conv_id, facts in sorted(facts_by_conv.items()):
        topic_count += 1
        topic_id = f"{conv_id}:topic_instance:0001"
        if topic_id not in output.nodes:
            output.add_node(instance_topic_node(topic_id, conv_id, facts, event_ids_by_conv[conv_id], max_preview_facts))
        for event_id in sorted(set(event_ids_by_conv[conv_id])):
            output.nodes[event_id].metadata["topic_id"] = topic_id
            output.add_edge(
                Edge(
                    src=event_id,
                    dst=topic_id,
                    relation=RelationType.IS_SPECIFIC_OF,
                    confidence=0.95,
                    metadata={
                        "hierarchy_v3": "event_topic",
                        "builder": "longmemeval_session_hierarchy_v1",
                    },
                )
            )
            event_topic_edges += 1

    metadata = dict(output.metadata)
    metadata["hierarchy_v3"] = {
        "builder": "longmemeval_session_hierarchy_v1",
        "source_graph_id": graph.graph_id,
        "design": "FACT -> SESSION_EVENT -> INSTANCE_TOPIC",
        "event_nodes": event_count,
        "topic_nodes": topic_count,
        "fact_event_edges": fact_event_edges,
        "event_topic_edges": event_topic_edges,
        "cross_instance_edges": count_cross_instance_hierarchy_edges(output),
    }
    output.metadata = metadata
    return output


def session_event_node(event_id: str, conv_id: str, session_key: str, facts: list[Node], max_preview_facts: int) -> Node:
    timestamp = first_time(facts)
    preview = "; ".join(fact.text for fact in facts[:max_preview_facts])
    return Node(
        node_id=event_id,
        type=NodeType.EVENT,
        text=f"Session event {session_key}: {preview}",
        time=timestamp,
        source="longmemeval_session_boundary",
        confidence=session_confidence(facts),
        support_ids=[fact.node_id for fact in facts],
        metadata={
            "conversation_id": conv_id,
            "session_key": session_key,
            "num_facts": len(facts),
            "hierarchy_v3": "session_event",
            "coherence": 1.0,
        },
    )


def instance_topic_node(
    topic_id: str,
    conv_id: str,
    facts: list[Node],
    event_ids: list[str],
    max_preview_facts: int,
) -> Node:
    preview = "; ".join(fact.text for fact in sorted(facts, key=lambda node: node.node_id)[:max_preview_facts])
    return Node(
        node_id=topic_id,
        type=NodeType.TOPIC,
        text=f"LongMemEval instance topic for {conv_id}: {preview}",
        time=first_time(facts),
        source="longmemeval_instance_boundary",
        confidence=0.95,
        support_ids=sorted(set(event_ids)),
        metadata={
            "conversation_id": conv_id,
            "num_facts": len(facts),
            "num_events": len(set(event_ids)),
            "hierarchy_v3": "instance_topic",
            "coherence": 1.0,
        },
    )


def diagnose(graph: MemoryGraph) -> dict:
    event_to_facts: dict[str, list[str]] = defaultdict(list)
    topic_to_events: dict[str, list[str]] = defaultdict(list)
    cross_instance_edges = 0
    for edge in graph.edges:
        role = edge.metadata.get("hierarchy_v3")
        if role == "fact_event":
            event_to_facts[edge.dst].append(edge.src)
        elif role == "event_topic":
            topic_to_events[edge.dst].append(edge.src)
        if role in {"fact_event", "event_topic"} and conversation_id(edge.src) != conversation_id(edge.dst):
            cross_instance_edges += 1

    event_sizes = [len(values) for values in event_to_facts.values()]
    topic_sizes = [len(values) for values in topic_to_events.values()]
    node_counts = Counter(node.type.value for node in graph.nodes.values())
    edge_counts = Counter(edge.relation.value for edge in graph.edges)
    return {
        "graph_id": graph.graph_id,
        "node_counts": dict(node_counts),
        "edge_counts": dict(edge_counts),
        "structure": {
            "events": len(event_to_facts),
            "topics": len(topic_to_events),
            "fact_event_edges": sum(event_sizes),
            "event_topic_edges": sum(topic_sizes),
            "cross_instance_hierarchy_edges": cross_instance_edges,
            "avg_facts_per_event": float(mean(event_sizes)) if event_sizes else 0.0,
            "max_facts_per_event": max(event_sizes, default=0),
            "avg_events_per_topic": float(mean(topic_sizes)) if topic_sizes else 0.0,
            "max_events_per_topic": max(topic_sizes, default=0),
        },
        "metadata": graph.metadata.get("hierarchy_v3", {}),
    }


def limit_graph_conversations(graph: MemoryGraph, limit: int) -> MemoryGraph:
    conv_ids = sorted({conversation_id(node_id) for node_id in graph.nodes})[:limit]
    allowed = set(conv_ids)
    output = graph.model_copy(deep=True)
    output.nodes = {node_id: node for node_id, node in output.nodes.items() if conversation_id(node_id) in allowed}
    output.edges = [edge for edge in output.edges if edge.src in output.nodes and edge.dst in output.nodes]
    output.graph_id = f"{graph.graph_id}_first_{limit}"
    return output


def conversation_id(node_id: str) -> str:
    for marker in (":raw:", ":fact:", ":anchor:", ":event_", ":topic_"):
        if marker in node_id:
            return node_id.split(marker, 1)[0]
    return node_id.split(":", 1)[0]


def session_key_from_fact(fact: Node) -> str:
    turn_id = str(fact.metadata.get("turn_id") or "")
    if ":t" in turn_id:
        return turn_id.rsplit(":t", 1)[0]
    if fact.support_ids:
        raw_tail = fact.support_ids[0].split(":raw:", 1)[-1]
        if ":t" in raw_tail:
            return raw_tail.rsplit(":t", 1)[0]
    return "unknown_session"


def safe_local_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)[:120]


def first_time(nodes: list[Node]) -> str | None:
    times = [node.time for node in nodes if node.time]
    return min(times) if times else None


def session_confidence(facts: list[Node]) -> float:
    if not facts:
        return 0.0
    return min(0.98, 0.85 + min(len(facts), 20) / 200.0)


def count_cross_instance_hierarchy_edges(graph: MemoryGraph) -> int:
    count = 0
    for edge in graph.edges:
        if edge.metadata.get("hierarchy_v3") in {"fact_event", "event_topic"}:
            count += int(conversation_id(edge.src) != conversation_id(edge.dst))
    return count


if __name__ == "__main__":
    main()
