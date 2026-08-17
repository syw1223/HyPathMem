from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType
from hytopomem.retrieval.bm25_retriever import BM25Retriever


@dataclass(frozen=True)
class ExpansionOptions:
    support_raw: bool = True
    parent_anchor: bool = True
    sibling_fact: bool = True
    update_exception: bool = True
    conflict: bool = False
    hops: int = 2
    sibling_limit_per_anchor: int = 20


@dataclass(frozen=True)
class CandidatePool:
    seed_node_ids: List[str]
    candidate_node_ids: List[str]
    expanded_node_ids: List[str]
    edge_type_counts: Dict[str, int]

    @property
    def expanded_only_node_ids(self) -> List[str]:
        seeds = set(self.seed_node_ids)
        return [node_id for node_id in self.candidate_node_ids if node_id not in seeds]


class GraphCandidateOracle:
    """Build seed and graph-expanded candidate pools for diagnostic upper bounds."""

    def __init__(self, graph: MemoryGraph):
        self.graph = graph
        self.facts = list(graph.iter_nodes(NodeType.FACT))
        self.fact_retriever = BM25Retriever(self.facts)
        self._seed_cache: Dict[tuple[str, int], List[str]] = {}
        self._incoming: Dict[str, List[Edge]] = {}
        self._outgoing: Dict[str, List[Edge]] = {}
        for edge in graph.edges:
            self._incoming.setdefault(edge.dst, []).append(edge)
            self._outgoing.setdefault(edge.src, []).append(edge)

    def bm25_fact_seed_ids(self, query: str, top_n: int) -> List[str]:
        key = (query, top_n)
        if key not in self._seed_cache:
            self._seed_cache[key] = [node.node_id for node, _score in self.fact_retriever.search(query, top_k=top_n)]
        return self._seed_cache[key]

    def build_pool(
        self,
        query: str,
        *,
        seed_topn: int,
        options: ExpansionOptions,
    ) -> CandidatePool:
        seed_ids = self.bm25_fact_seed_ids(query, seed_topn)
        candidate_ids: List[str] = []
        expanded_ids: List[str] = []
        seen: Set[str] = set()
        edge_type_counts: Dict[str, int] = {}

        def add_node(node_id: str, expanded: bool) -> None:
            if node_id not in self.graph.nodes or node_id in seen:
                return
            seen.add(node_id)
            candidate_ids.append(node_id)
            if expanded:
                expanded_ids.append(node_id)

        for seed_id in seed_ids:
            add_node(seed_id, expanded=False)
            if options.hops >= 1:
                for node_id, edge in self._one_hop(seed_id, options):
                    add_node(node_id, expanded=True)
                    edge_type_counts[edge.relation.value] = edge_type_counts.get(edge.relation.value, 0) + 1
            if options.hops >= 2:
                for node_id, edges in self._two_hop(seed_id, options):
                    add_node(node_id, expanded=True)
                    for edge in edges:
                        edge_type_counts[edge.relation.value] = edge_type_counts.get(edge.relation.value, 0) + 1

        return CandidatePool(
            seed_node_ids=seed_ids,
            candidate_node_ids=candidate_ids,
            expanded_node_ids=expanded_ids,
            edge_type_counts=edge_type_counts,
        )

    def _one_hop(self, seed_id: str, options: ExpansionOptions) -> Iterable[tuple[str, Edge]]:
        for edge in self._outgoing.get(seed_id, []):
            dst = self.graph.nodes.get(edge.dst)
            if dst is None:
                continue
            if options.support_raw and edge.relation == RelationType.SUPPORTS and dst.type == NodeType.RAW:
                yield dst.node_id, edge
            if options.parent_anchor and edge.relation == RelationType.IS_SPECIFIC_OF and dst.type == NodeType.ANCHOR:
                yield dst.node_id, edge
            if (
                options.update_exception
                and edge.relation in {RelationType.UPDATES, RelationType.EXCEPTION_OF}
                and dst.type == NodeType.FACT
            ):
                yield dst.node_id, edge
            if options.conflict and edge.relation == RelationType.CONFLICTS_WITH and dst.type == NodeType.FACT:
                yield dst.node_id, edge

        for edge in self._incoming.get(seed_id, []):
            src = self.graph.nodes.get(edge.src)
            if src is None:
                continue
            if (
                options.update_exception
                and edge.relation in {RelationType.UPDATES, RelationType.EXCEPTION_OF}
                and src.type == NodeType.FACT
            ):
                yield src.node_id, edge
            if options.conflict and edge.relation == RelationType.CONFLICTS_WITH and src.type == NodeType.FACT:
                yield src.node_id, edge

    def _two_hop(self, seed_id: str, options: ExpansionOptions) -> Iterable[tuple[str, List[Edge]]]:
        if not options.sibling_fact:
            return
        parent_edges = [
            edge
            for edge in self._outgoing.get(seed_id, [])
            if edge.relation == RelationType.IS_SPECIFIC_OF
            and self.graph.nodes.get(edge.dst) is not None
            and self.graph.nodes[edge.dst].type == NodeType.ANCHOR
        ]
        for parent_edge in parent_edges:
            siblings = [
                edge
                for edge in self._incoming.get(parent_edge.dst, [])
                if edge.relation == RelationType.IS_SPECIFIC_OF and edge.src != seed_id
            ]
            siblings = sorted(siblings, key=lambda edge: edge.confidence, reverse=True)
            for sibling_edge in siblings[: options.sibling_limit_per_anchor]:
                sibling = self.graph.nodes.get(sibling_edge.src)
                if sibling is not None and sibling.type == NodeType.FACT:
                    yield sibling.node_id, [parent_edge, sibling_edge]
