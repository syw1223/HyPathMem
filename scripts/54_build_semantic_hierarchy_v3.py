from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean, median

from common import load_config, resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, NodeType
from hytopomem.memory.semantic_hierarchy_builder import (
    SemanticHierarchyBuilder,
    SemanticHierarchyConfig,
)
from hytopomem.retrieval.topdown_semantic_retriever import (
    SentenceTransformerEncoder,
    default_embedder,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--output", default="outputs/graphs/locomo_graph_semantic_hierarchy_v3.json")
    parser.add_argument("--graph-id", default="locomo_semantic_hierarchy_v3")
    parser.add_argument("--diagnostics", default="outputs/eval/graph_v3_structure_diagnostics.json")
    parser.add_argument("--embedder", default=default_embedder())
    parser.add_argument("--device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--event-similarity-threshold", type=float, default=0.52)
    parser.add_argument("--topic-similarity-threshold", type=float, default=0.52)
    parser.add_argument("--event-max-facts", type=int, default=6)
    parser.add_argument("--topic-max-events", type=int, default=8)
    parser.add_argument("--event-max-session-gap", type=int, default=1)
    parser.add_argument("--include-uncovered-rule-facts", action="store_true")
    parser.add_argument("--rule-fact-policy", choices=["none", "filtered", "random", "all"], default="none")
    parser.add_argument("--rule-quality-threshold", type=float, default=0.60)
    parser.add_argument("--random-rule-count", type=int, default=944)
    parser.add_argument("--random-rule-seed", type=int, default=13)
    parser.add_argument("--limit-conversations", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    graph_path = resolve_path(args.graph or config["graph"]["graph_path"])
    graph = JsonGraphStore().load(graph_path)
    if args.limit_conversations > 0:
        graph = limit_graph_conversations(graph, args.limit_conversations)

    encoder = SentenceTransformerEncoder(
        args.embedder,
        device=args.device,
        batch_size=args.embedding_batch_size,
    )
    builder = SemanticHierarchyBuilder(
        encoder,
        SemanticHierarchyConfig(
            event_similarity_threshold=args.event_similarity_threshold,
            topic_similarity_threshold=args.topic_similarity_threshold,
            event_max_facts=args.event_max_facts,
            topic_max_events=args.topic_max_events,
            event_max_session_gap=args.event_max_session_gap,
            include_uncovered_rule_facts=args.include_uncovered_rule_facts,
            rule_fact_policy=args.rule_fact_policy,
            rule_quality_threshold=args.rule_quality_threshold,
            random_rule_count=args.random_rule_count,
            random_rule_seed=args.random_rule_seed,
        ),
    )
    output = builder.build(graph, graph_id=args.graph_id)
    output_path = resolve_path(args.output)
    JsonGraphStore().save(output, output_path)

    diagnostics = graph_v3_diagnostics(output)
    diagnostics_path = resolve_path(args.diagnostics)
    write_json(diagnostics, diagnostics_path)
    print(f"wrote {output_path}")
    print(f"wrote {diagnostics_path}")
    print(f"hierarchy_v3={output.metadata.get('hierarchy_v3', {})}")
    print(f"structure={diagnostics['structure']}")


def limit_graph_conversations(graph: MemoryGraph, limit: int) -> MemoryGraph:
    conversation_ids = sorted(
        {
            node_id.split(":", 1)[0]
            for node_id, node in graph.nodes.items()
            if node.type in {NodeType.RAW, NodeType.FACT}
        }
    )[:limit]
    allowed = set(conversation_ids)
    output = graph.model_copy(deep=True)
    output.nodes = {
        node_id: node
        for node_id, node in output.nodes.items()
        if node_id.split(":", 1)[0] in allowed
    }
    output.edges = [
        edge
        for edge in output.edges
        if edge.src in output.nodes and edge.dst in output.nodes
    ]
    output.graph_id = f"{graph.graph_id}_first_{limit}"
    return output


def graph_v3_diagnostics(graph: MemoryGraph) -> dict:
    event_to_facts: dict[str, list[str]] = defaultdict(list)
    topic_to_events: dict[str, list[str]] = defaultdict(list)
    alias_edges = 0
    fact_event_sources: set[str] = set()
    event_topic_sources: set[str] = set()
    cross_instance_fact_event_edges = 0
    cross_instance_event_topic_edges = 0
    for edge in graph.edges:
        role = edge.metadata.get("hierarchy_v3")
        if role == "fact_event":
            event_to_facts[edge.dst].append(edge.src)
            fact_event_sources.add(edge.src)
            if graph_conversation_id(edge.src) != graph_conversation_id(edge.dst):
                cross_instance_fact_event_edges += 1
        elif role == "event_topic":
            topic_to_events[edge.dst].append(edge.src)
            event_topic_sources.add(edge.src)
            if graph_conversation_id(edge.src) != graph_conversation_id(edge.dst):
                cross_instance_event_topic_edges += 1
        elif role == "lexical_alias_event":
            alias_edges += 1

    event_sizes = [len(values) for values in event_to_facts.values()]
    topic_sizes = [len(values) for values in topic_to_events.values()]
    event_coherence = [
        float(graph.nodes[event_id].metadata.get("coherence", 0.0))
        for event_id in event_to_facts
    ]
    topic_coherence = [
        float(graph.nodes[topic_id].metadata.get("coherence", 0.0))
        for topic_id in topic_to_events
    ]
    node_counts = Counter(node.type.value for node in graph.nodes.values())
    metadata = graph.metadata.get("hierarchy_v3", {})
    fact_ids = {node.node_id for node in graph.iter_nodes(NodeType.FACT)}
    event_ids = {
        node.node_id
        for node in graph.iter_nodes(NodeType.EVENT)
        if str(node.metadata.get("hierarchy_v3") or "") in {"event", "session_event"} or ":event" in node.node_id
    }
    return {
        "metadata": metadata,
        "node_counts": dict(node_counts),
        "structure": {
            "canonical_facts": int(metadata.get("canonical_fact_nodes", 0)),
            "canonical_source_counts": metadata.get("canonical_source_counts", {}),
            "lexical_alias_edges": alias_edges,
            "rule_filter": metadata.get("rule_filter", {}),
            "events": len(event_sizes),
            "topics": len(topic_sizes),
            "mean_facts_per_event": safe_mean(event_sizes),
            "median_facts_per_event": safe_median(event_sizes),
            "singleton_event_ratio": ratio_equal(event_sizes, 1),
            "event_size_p95": percentile(event_sizes, 0.95),
            "event_size_max": max(event_sizes, default=0),
            "mean_events_per_topic": safe_mean(topic_sizes),
            "median_events_per_topic": safe_median(topic_sizes),
            "singleton_topic_ratio": ratio_equal(topic_sizes, 1),
            "topic_size_p95": percentile(topic_sizes, 0.95),
            "topic_size_max": max(topic_sizes, default=0),
            "fact_without_event": len(fact_ids - fact_event_sources),
            "event_without_topic": len(event_ids - event_topic_sources),
            "cross_instance_fact_event_edges": cross_instance_fact_event_edges,
            "cross_instance_event_topic_edges": cross_instance_event_topic_edges,
        },
        "coherence": {
            "event_mean": safe_mean(event_coherence),
            "event_median": safe_median(event_coherence),
            "topic_mean": safe_mean(topic_coherence),
            "topic_median": safe_median(topic_coherence),
        },
    }


def safe_mean(values: list[float] | list[int]) -> float:
    return float(mean(values)) if values else 0.0


def safe_median(values: list[float] | list[int]) -> float:
    return float(median(values)) if values else 0.0


def ratio_equal(values: list[int], target: int) -> float:
    return sum(value == target for value in values) / len(values) if values else 0.0


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(quantile * (len(ordered) - 1))))
    return int(ordered[index])


def graph_conversation_id(node_id: str) -> str:
    for marker in (":raw:", ":fact_sent:", ":fact:", ":anchor:", ":event", ":topic", ":episode"):
        if marker in node_id:
            return node_id.split(marker, 1)[0]
    return node_id.split(":", 1)[0]


if __name__ == "__main__":
    main()
