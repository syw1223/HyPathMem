from __future__ import annotations

import argparse
import ast
from pathlib import Path

from common import resolve_path
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backup", default=None)
    args = parser.parse_args()

    graph_path = resolve_path(args.graph)
    output_path = resolve_path(args.output)
    graph = JsonGraphStore().load(graph_path)
    if args.backup:
        JsonGraphStore().save(graph, resolve_path(args.backup))

    stats = repair_graph_provenance(graph)
    metadata = dict(graph.metadata)
    metadata["provenance_fix"] = {
        "version": "raw_support_ids_v1",
        "source_graph": str(graph_path),
        **stats,
    }
    graph.metadata = metadata
    JsonGraphStore().save(graph, output_path)
    print(stats)
    print(f"wrote {output_path}")


def repair_graph_provenance(graph: MemoryGraph) -> dict:
    raw_by_turn = raw_nodes_by_turn_id(graph)
    existing_support_edges = {
        (edge.src, edge.dst)
        for edge in graph.edges
        if edge.relation == RelationType.SUPPORTS
    }
    fixed_fact_nodes = 0
    fixed_hierarchy_nodes = 0
    added_support_edges = 0
    unresolved_support_ids: set[str] = set()

    for node in graph.nodes.values():
        if node.type == NodeType.FACT:
            fixed, unresolved = repair_fact_node(node, graph, raw_by_turn)
            if fixed:
                fixed_fact_nodes += 1
            unresolved_support_ids.update(unresolved)
            for raw_id in node.metadata.get("support_raw_ids", []):
                key = (node.node_id, raw_id)
                if key in existing_support_edges or raw_id not in graph.nodes:
                    continue
                graph.edges.append(
                    Edge(
                        src=node.node_id,
                        dst=raw_id,
                        relation=RelationType.SUPPORTS,
                        confidence=0.98,
                        metadata={"builder": "provenance_fix_raw_support_ids_v1"},
                    )
                )
                existing_support_edges.add(key)
                added_support_edges += 1
        elif node.type in {NodeType.EVENT, NodeType.TOPIC}:
            if repair_hierarchy_node_support_raw_ids(node, graph, raw_by_turn):
                fixed_hierarchy_nodes += 1

    return {
        "fixed_fact_nodes": fixed_fact_nodes,
        "fixed_hierarchy_nodes": fixed_hierarchy_nodes,
        "added_support_edges": added_support_edges,
        "unresolved_support_ids": len(unresolved_support_ids),
        "unresolved_examples": sorted(unresolved_support_ids)[:20],
    }


def repair_fact_node(
    node: Node,
    graph: MemoryGraph,
    raw_by_turn: dict[tuple[str, str], str],
) -> tuple[bool, set[str]]:
    original_support_ids = list(node.support_ids)
    original_metadata = dict(node.metadata)
    raw_ids, unresolved = normalize_support_ids(node.node_id, original_support_ids, graph, raw_by_turn)
    if not raw_ids:
        return False, unresolved
    timestamps: list[str] = []
    support_texts: list[str] = []
    turn_ids: list[str] = []
    for raw_id in raw_ids:
        raw = graph.nodes.get(raw_id)
        if raw is None:
            continue
        if raw.time:
            timestamps.append(str(raw.time))
        turn_id = str(raw.metadata.get("turn_id") or raw_id.split(":raw:", 1)[-1])
        if turn_id:
            turn_ids.append(turn_id)
        support_texts.append(raw.text)
    metadata = dict(node.metadata)
    metadata["support_raw_ids"] = raw_ids
    if timestamps:
        metadata["support_timestamps"] = timestamps
    if support_texts:
        metadata["support_texts"] = support_texts
    if turn_ids:
        metadata["support_turn_ids"] = turn_ids
    node.support_ids = raw_ids
    node.metadata = metadata
    if node.time is None and timestamps:
        node.time = timestamps[0]
    fixed = (
        node.support_ids != original_support_ids
        or node.metadata != original_metadata
    )
    return fixed, unresolved


def repair_hierarchy_node_support_raw_ids(
    node: Node,
    graph: MemoryGraph,
    raw_by_turn: dict[tuple[str, str], str],
) -> bool:
    metadata = dict(node.metadata)
    raw_ids: list[str] = []
    unresolved: set[str] = set()
    if metadata.get("support_raw_ids"):
        raw_ids, unresolved = normalize_support_ids(node.node_id, metadata["support_raw_ids"], graph, raw_by_turn)
    elif metadata.get("fact_ids"):
        for fact_id in metadata.get("fact_ids", []):
            fact = graph.nodes.get(str(fact_id))
            if fact is None:
                continue
            fact_raw_ids, fact_unresolved = normalize_support_ids(fact.node_id, fact.support_ids, graph, raw_by_turn)
            raw_ids.extend(fact_raw_ids)
            unresolved.update(fact_unresolved)
    deduped = dedupe(raw_ids)
    if not deduped:
        return False
    old = list(metadata.get("support_raw_ids", []))
    metadata["support_raw_ids"] = deduped
    node.metadata = metadata
    return old != deduped


def normalize_support_ids(
    node_id: str,
    values: list[str],
    graph: MemoryGraph,
    raw_by_turn: dict[tuple[str, str], str],
) -> tuple[list[str], set[str]]:
    conv_id = conversation_id(node_id)
    raw_ids: list[str] = []
    unresolved: set[str] = set()
    for item in expand_support_values(values):
        value = str(item).strip()
        if not value:
            continue
        if value in graph.nodes and graph.nodes[value].type == NodeType.RAW:
            raw_ids.append(value)
            continue
        turn_id = value.split(":raw:", 1)[-1] if ":raw:" in value else value
        candidate = raw_by_turn.get((conv_id, turn_id))
        if candidate:
            raw_ids.append(candidate)
        else:
            unresolved.add(value)
    return dedupe(raw_ids), unresolved


def expand_support_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for item in values:
        if isinstance(item, list):
            output.extend(str(value).strip() for value in item if str(value).strip())
            continue
        text = str(item).strip()
        if not text:
            continue
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                output.extend(str(value).strip() for value in parsed if str(value).strip())
                continue
        if "," in text and ":raw:" not in text:
            output.extend(part.strip().strip("'\"") for part in text.split(",") if part.strip().strip("'\""))
            continue
        output.append(text)
    return output


def raw_nodes_by_turn_id(graph: MemoryGraph) -> dict[tuple[str, str], str]:
    output: dict[tuple[str, str], str] = {}
    for node in graph.iter_nodes(NodeType.RAW):
        conv_id = conversation_id(node.node_id)
        turn_id = str(node.metadata.get("turn_id") or node.node_id.split(":raw:", 1)[-1])
        output[(conv_id, turn_id)] = node.node_id
    return output


def conversation_id(node_id: str) -> str:
    if ":fact:" in node_id:
        return node_id.split(":fact:", 1)[0]
    if ":raw:" in node_id:
        return node_id.split(":raw:", 1)[0]
    if ":event" in node_id:
        return node_id.split(":event", 1)[0]
    if ":topic" in node_id:
        return node_id.split(":topic", 1)[0]
    return node_id.split(":", 1)[0]


def dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


if __name__ == "__main__":
    main()
