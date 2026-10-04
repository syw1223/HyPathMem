from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType
from hytopomem.models.hyperbolic_mapper import HyperbolicMapper
from hytopomem.retrieval.path_builder import PathRetrievalPipeline


def test_path_retrieval_returns_anchor_fact_path():
    graph = MemoryGraph(graph_id="g")
    fact = Node(node_id="fact_1", type=NodeType.FACT, text="User wants hyperbolic memory for AAAI.")
    anchor = Node(node_id="anchor_1", type=NodeType.ANCHOR, text="Topic: hyperbolic memory research")
    graph.add_node(fact)
    graph.add_node(anchor)
    graph.add_edge(Edge(src=fact.node_id, dst=anchor.node_id, relation=RelationType.IS_SPECIFIC_OF))
    pipeline = PathRetrievalPipeline(graph=graph, mapper=HyperbolicMapper(dim=16))
    paths = pipeline.retrieve("q1", "What memory idea is the user pursuing?", top_k=1)
    assert paths
    assert "fact_1" in paths[0].node_ids

