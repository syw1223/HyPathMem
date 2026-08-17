from __future__ import annotations

from dataclasses import dataclass
from typing import List

from hytopomem.memory.schema import EvidencePath, MemoryGraph, NodeType
from hytopomem.models.hyperbolic_mapper import HyperbolicMapper
from hytopomem.models.text_encoder import HashTextEncoder
from hytopomem.retrieval.bm25_retriever import BM25Retriever
from hytopomem.retrieval.dense_retriever import DenseRetriever
from hytopomem.retrieval.graph_expander import GraphExpander
from hytopomem.retrieval.path_selector import HeuristicPathSelector


@dataclass
class PathRetrievalPipeline:
    graph: MemoryGraph
    mapper: HyperbolicMapper
    top_anchor_k: int = 5
    max_path_len: int = 4
    max_paths: int = 20

    def __post_init__(self) -> None:
        self.anchors = list(self.graph.iter_nodes(NodeType.ANCHOR))
        encoder = self.mapper.text_encoder or HashTextEncoder(dim=self.mapper.dim - 1)
        self.dense_retriever = DenseRetriever(self.anchors, encoder)
        self.bm25_retriever = BM25Retriever(self.anchors)
        self.expander = GraphExpander(self.graph, max_path_len=self.max_path_len, max_paths=self.max_paths)
        self.selector = HeuristicPathSelector(self.graph, self.mapper)

    def retrieve(self, query_id: str, query: str, top_k: int = 5) -> List[EvidencePath]:
        if not self.anchors:
            return []
        dense = self.dense_retriever.search(query, top_k=self.top_anchor_k)
        bm25 = self.bm25_retriever.search(query, top_k=self.top_anchor_k)
        anchor_ids = []
        for node, _score in dense + bm25:
            if node.node_id not in anchor_ids:
                anchor_ids.append(node.node_id)
        raw_paths = []
        for anchor_id in anchor_ids[: self.top_anchor_k]:
            raw_paths.extend(self.expander.expand_from_anchor(anchor_id))
        return self.selector.select(query_id, query, raw_paths, top_k=top_k)
