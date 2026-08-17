from __future__ import annotations

import argparse
import json
from statistics import mean, median

from common import resolve_path
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import NodeType, RelationType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_hierarchy_v2.json")
    parser.add_argument("--output-json", default="outputs/eval/graph_v2_quality_diagnostics.json")
    parser.add_argument("--output-md", default="outputs/eval/graph_v2_quality_diagnostics.md")
    parser.add_argument("--large-topic-threshold", type=int, default=24)
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    event_to_facts: dict[str, list[str]] = {}
    topic_to_events: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.relation != RelationType.IS_SPECIFIC_OF:
            continue
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None:
            continue
        if src.type == NodeType.FACT and dst.type == NodeType.EVENT and edge.metadata.get("hierarchy_v2") == "fact_event":
            event_to_facts.setdefault(edge.dst, []).append(edge.src)
        elif src.type == NodeType.EVENT and dst.type == NodeType.TOPIC and edge.metadata.get("hierarchy_v2") == "event_topic":
            topic_to_events.setdefault(edge.dst, []).append(edge.src)

    event_sizes = [len(facts) for facts in event_to_facts.values()]
    topic_event_sizes = [len(events) for events in topic_to_events.values()]
    topic_fact_sizes = [
        sum(len(event_to_facts.get(event_id, [])) for event_id in events)
        for events in topic_to_events.values()
    ]
    event_coherence = [
        float(node.metadata.get("coherence", 0.0))
        for node in graph.iter_nodes(NodeType.EVENT)
        if node.metadata.get("hierarchy_v2") == "event"
    ]
    topic_coherence = [
        float(node.metadata.get("coherence", 0.0))
        for node in graph.iter_nodes(NodeType.TOPIC)
        if node.metadata.get("hierarchy_v2") == "topic"
    ]
    payload = {
        "graph": str(resolve_path(args.graph)),
        "num_topics": len(topic_to_events),
        "num_events": len(event_to_facts),
        "event_size": describe(event_sizes),
        "topic_event_size": describe(topic_event_sizes),
        "topic_fact_size": describe(topic_fact_sizes),
        "event_coherence": describe(event_coherence),
        "topic_coherence": describe(topic_coherence),
        "singleton_event_ratio": ratio(lambda value: value == 1, event_sizes),
        "large_topic_ratio": ratio(lambda value: value >= args.large_topic_threshold, topic_fact_sizes),
        "large_topic_threshold": args.large_topic_threshold,
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, resolve_path(args.output_md))
    print(resolve_path(args.output_md))
    print(json.dumps(payload, indent=2))


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": ordered[0],
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def ratio(predicate, values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if predicate(value)) / len(values)


def write_markdown(payload: dict, path) -> None:
    lines = [
        "# Graph v2 Quality Diagnostics",
        "",
        f"Graph: `{payload['graph']}`",
        f"Topics: {payload['num_topics']}",
        f"Events: {payload['num_events']}",
        f"Singleton event ratio: {payload['singleton_event_ratio']:.4f}",
        f"Large topic ratio >= {payload['large_topic_threshold']} facts: {payload['large_topic_ratio']:.4f}",
        "",
        "| Metric | Count | Mean | Median | Min | P90 | P95 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["event_size", "topic_event_size", "topic_fact_size", "event_coherence", "topic_coherence"]:
        item = payload[name]
        lines.append(
            f"| {name} | {item['count']} | {item['mean']:.4f} | {item['median']:.4f} | "
            f"{item['min']:.4f} | {item['p90']:.4f} | {item['p95']:.4f} | {item['max']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
