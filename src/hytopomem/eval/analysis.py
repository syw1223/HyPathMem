from __future__ import annotations

from hytopomem.memory.schema import EvidencePath, MemoryGraph


def render_path(graph: MemoryGraph, path: EvidencePath) -> str:
    lines = []
    for node_id in path.node_ids:
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        lines.append(f"[{node.type}] {node.node_id}: {node.text}")
    return "\n".join(lines)

