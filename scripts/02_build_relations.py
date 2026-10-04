from __future__ import annotations

import argparse
from collections import Counter

from common import load_config, read_json, resolve_path
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.relation_builder import WeakRelationBuilder
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--nodes", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--graph-id", default=None)
    parser.add_argument(
        "--light-support-only",
        action="store_true",
        help="Only add FACT -> RAW SUPPORTS edges. Useful for large datasets such as LongMemEval-S.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = config.get("dataset", {}).get("name", "memory")
    nodes_path = resolve_path(args.nodes or config["graph"]["nodes_path"])
    output_path = resolve_path(args.output or config["graph"]["graph_path"])
    graph_id = args.graph_id or f"{dataset_name}_base_graph"

    nodes = [Node.model_validate(item) for item in read_json(nodes_path)]
    graph = MemoryGraph(graph_id=graph_id)
    for node in nodes:
        graph.add_node(node)
    edge_iterable = build_support_edges(nodes) if args.light_support_only else WeakRelationBuilder().build(nodes)
    for edge in edge_iterable:
        graph.add_edge(edge)

    JsonGraphStore().save(graph, output_path)
    node_counts = Counter(node.type.value for node in graph.nodes.values())
    edge_counts = Counter(edge.relation.value for edge in graph.edges)
    print(f"wrote {output_path}")
    print(f"nodes={len(graph.nodes)} edges={len(graph.edges)}")
    print(f"node_counts={dict(node_counts)}")
    print(f"edge_counts={dict(edge_counts)}")


def build_support_edges(nodes: list[Node]) -> list[Edge]:
    raw_ids = {node.node_id for node in nodes if node.type == NodeType.RAW}
    edges: list[Edge] = []
    for node in nodes:
        if node.type != NodeType.FACT:
            continue
        for raw_id in node.support_ids:
            if raw_id in raw_ids:
                edges.append(
                    Edge(
                        src=node.node_id,
                        dst=raw_id,
                        relation=RelationType.SUPPORTS,
                        confidence=0.98,
                        metadata={"builder": "light_support_only"},
                    )
                )
    return edges


if __name__ == "__main__":
    main()
