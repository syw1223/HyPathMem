from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType


def test_memory_graph_adds_partial_order_edge():
    graph = MemoryGraph(graph_id="g")
    fact = Node(node_id="fact_1", type=NodeType.FACT, text="User studies memory graphs.")
    anchor = Node(node_id="anchor_1", type=NodeType.ANCHOR, text="Topic: memory research")
    graph.add_node(fact)
    graph.add_node(anchor)
    graph.add_edge(Edge(src=fact.node_id, dst=anchor.node_id, relation=RelationType.IS_SPECIFIC_OF))
    assert len(graph.partial_order_edges()) == 1

