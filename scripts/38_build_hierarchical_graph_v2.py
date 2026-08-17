from __future__ import annotations

import argparse
from collections import Counter

from common import load_config, resolve_path
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.hierarchy_builder import HierarchicalGraphBuilder, HierarchyBuilderConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--output", default="outputs/graphs/locomo_graph_hierarchy_v2.json")
    parser.add_argument("--graph-id", default="locomo_hierarchy_v2")
    parser.add_argument("--topic-boundary-similarity", type=float, default=0.10)
    parser.add_argument("--event-boundary-similarity", type=float, default=0.18)
    parser.add_argument("--min-topic-facts", type=int, default=2)
    parser.add_argument("--max-topic-facts", type=int, default=36)
    parser.add_argument("--max-event-facts", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=128)
    args = parser.parse_args()

    config = load_config(args.config)
    graph_path = resolve_path(args.graph or config["graph"]["graph_path"])
    output_path = resolve_path(args.output)
    graph = JsonGraphStore().load(graph_path)
    builder = HierarchicalGraphBuilder(
        HierarchyBuilderConfig(
            embedding_dim=args.embedding_dim,
            topic_boundary_similarity=args.topic_boundary_similarity,
            event_boundary_similarity=args.event_boundary_similarity,
            min_topic_facts=args.min_topic_facts,
            max_topic_facts=args.max_topic_facts,
            max_event_facts=args.max_event_facts,
        )
    )
    output = builder.build(graph, graph_id=args.graph_id)
    JsonGraphStore().save(output, output_path)
    node_counts = Counter(node.type.value for node in output.nodes.values())
    edge_counts = Counter(edge.relation.value for edge in output.edges)
    print(f"wrote {output_path}")
    print(f"nodes={len(output.nodes)} edges={len(output.edges)}")
    print(f"node_counts={dict(node_counts)}")
    print(f"edge_counts={dict(edge_counts)}")
    print(f"hierarchy_v2={output.metadata.get('hierarchy_v2', {})}")


if __name__ == "__main__":
    main()
