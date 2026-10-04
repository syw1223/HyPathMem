from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List

from hytopomem.memory.schema import Edge, MemoryGraph, RelationType


FOLLOW_RELATIONS = {
    RelationType.IS_SPECIFIC_OF,
    RelationType.UPDATES,
    RelationType.EXCEPTION_OF,
    RelationType.CONFLICTS_WITH,
    RelationType.SUPPORTS,
}


@dataclass
class GraphExpander:
    graph: MemoryGraph
    max_path_len: int = 4
    max_paths: int = 20
    max_branch: int = 8

    def __post_init__(self) -> None:
        self._incoming: Dict[str, List] = {}
        self._outgoing: Dict[str, List] = {}
        for edge in self.graph.edges:
            self._incoming.setdefault(edge.dst, []).append(edge)
            self._outgoing.setdefault(edge.src, []).append(edge)

    def expand_from_anchor(self, anchor_id: str) -> List[tuple[list[str], list[str]]]:
        paths: List[tuple[list[str], list[str]]] = []
        queue = deque([(anchor_id, [anchor_id], [])])
        while queue and len(paths) < self.max_paths:
            current, node_path, edge_path = queue.popleft()
            incoming = self._incoming.get(current, [])
            outgoing = self._outgoing.get(current, [])
            next_edges = [edge for edge in incoming + outgoing if edge.relation in FOLLOW_RELATIONS]
            if not next_edges or len(node_path) >= self.max_path_len:
                paths.append((node_path, edge_path))
                continue
            expanded = False
            ranked_edges = sorted(next_edges, key=lambda item: item.confidence, reverse=True)
            for edge in ranked_edges[: self.max_branch]:
                nxt = edge.src if edge.dst == current else edge.dst
                if nxt in node_path:
                    continue
                expanded = True
                queue.append((nxt, node_path + [nxt], edge_path + [edge.edge_id or ""]))
            if not expanded:
                paths.append((node_path, edge_path))
        return paths
