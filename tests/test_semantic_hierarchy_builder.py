import numpy as np

from hytopomem.memory.schema import MemoryGraph, Node, NodeType
from hytopomem.memory.semantic_hierarchy_builder import (
    SemanticHierarchyBuilder,
    SemanticHierarchyConfig,
    extract_rule_statement,
)


class FakeEncoder:
    model_name_or_path = "fake-semantic-encoder"

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "support group" in lowered or "accepted" in lowered:
                vector = [1.0, 0.0, 0.0]
            elif "painting" in lowered:
                vector = [0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0]
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def test_event_first_builder_uses_observations_and_rule_fallbacks() -> None:
    graph = MemoryGraph(graph_id="tiny")
    graph.add_node(raw("conv-1:raw:D1:1", "Caroline joined a support group."))
    graph.add_node(raw("conv-1:raw:D1:2", "The group made Caroline feel accepted."))
    graph.add_node(raw("conv-1:raw:D2:1", "Melanie painted with her children."))
    graph.add_node(rule("conv-1:fact:D1:1", "conv-1:raw:D1:1", "Caroline said: support group"))
    graph.add_node(rule("conv-1:fact:D1:2", "conv-1:raw:D1:2", "Caroline said: accepted"))
    graph.add_node(rule("conv-1:fact:D2:1", "conv-1:raw:D2:1", "Melanie said: painting"))
    graph.add_node(
        observation(
            "conv-1:fact:obs:0001",
            ["conv-1:raw:D1:1", "conv-1:raw:D1:2"],
            "Caroline attended a support group and felt accepted.",
        )
    )

    output = SemanticHierarchyBuilder(
        FakeEncoder(),
        SemanticHierarchyConfig(
            event_similarity_threshold=0.40,
            topic_similarity_threshold=0.40,
            include_uncovered_rule_facts=True,
        ),
    ).build(graph)

    metadata = output.metadata["hierarchy_v3"]
    assert metadata["canonical_fact_nodes"] == 2
    assert metadata["canonical_source_counts"] == {
        "locomo_observation": 1,
        "rule_extracted": 1,
    }
    fact_event_edges = [
        edge
        for edge in output.edges
        if edge.metadata.get("hierarchy_v3") == "fact_event"
    ]
    alias_edges = [
        edge
        for edge in output.edges
        if edge.metadata.get("hierarchy_v3") == "lexical_alias_event"
    ]
    assert {edge.src for edge in fact_event_edges} == {
        "conv-1:fact:obs:0001",
        "conv-1:fact:D2:1",
    }
    assert {edge.src for edge in alias_edges} == {
        "conv-1:fact:D1:1",
        "conv-1:fact:D1:2",
    }
    assert len(list(output.iter_nodes(NodeType.EVENT))) == 2
    assert all(node.text.startswith("Event") for node in output.iter_nodes(NodeType.EVENT))
    assert all(node.text.startswith("Topic") for node in output.iter_nodes(NodeType.TOPIC))


def test_default_builder_keeps_uncovered_rules_out_of_semantic_clusters() -> None:
    graph = MemoryGraph(graph_id="tiny-default")
    graph.add_node(raw("conv-1:raw:D1:1", "Caroline joined a support group."))
    graph.add_node(raw("conv-1:raw:D2:1", "Melanie made a painting."))
    graph.add_node(rule("conv-1:fact:D1:1", "conv-1:raw:D1:1", "Caroline said: support group"))
    graph.add_node(rule("conv-1:fact:D2:1", "conv-1:raw:D2:1", "Melanie said: painting"))
    graph.add_node(
        observation(
            "conv-1:fact:obs:0001",
            ["conv-1:raw:D1:1"],
            "Caroline attended a support group.",
        )
    )

    output = SemanticHierarchyBuilder(FakeEncoder()).build(graph)

    metadata = output.metadata["hierarchy_v3"]
    assert metadata["canonical_fact_nodes"] == 1
    assert metadata["lexical_alias_edges"] == 1
    assert metadata["unclustered_rule_facts"] == 1
    semantic_fact_ids = {
        edge.src
        for edge in output.edges
        if edge.metadata.get("hierarchy_v3") == "fact_event"
    }
    assert semantic_fact_ids == {"conv-1:fact:obs:0001"}


def test_filtered_rule_policy_selects_informative_statement_and_keeps_alias() -> None:
    graph = MemoryGraph(graph_id="tiny-filtered")
    graph.add_node(raw("conv-1:raw:D1:1", "I went bowling yesterday and got 2 strikes."))
    graph.add_node(raw("conv-1:raw:D1:2", "Wow! What happened next?"))
    graph.add_node(rule("conv-1:fact:D1:1", "conv-1:raw:D1:1", "James said: I went bowling yesterday and got 2 strikes."))
    graph.add_node(rule("conv-1:fact:D1:2", "conv-1:raw:D1:2", "James said: Wow! What happened next?"))

    output = SemanticHierarchyBuilder(
        FakeEncoder(),
        SemanticHierarchyConfig(rule_fact_policy="filtered"),
    ).build(graph)

    derived = output.nodes["conv-1:fact:rule_sem:D1:1"]
    assert derived.source == "filtered_rule_statement"
    assert derived.text == "James went bowling yesterday and got 2 strikes."
    assert "conv-1:fact:rule_sem:D1:2" not in output.nodes
    assert output.metadata["hierarchy_v3"]["rule_filter"]["selected_uncovered_rules"] == 1
    assert output.metadata["hierarchy_v3"]["rule_filter"]["rejected_uncovered_rules"] == 1
    assert any(
        edge.src == derived.node_id
        and edge.dst == "conv-1:raw:D1:1"
        and edge.relation.value == "SUPPORTS"
        for edge in output.edges
    )
    assert any(
        edge.src == "conv-1:fact:D1:1"
        and edge.metadata.get("hierarchy_v3") == "lexical_alias_event"
        for edge in output.edges
    )


def test_rule_statement_extraction_keeps_fact_before_question() -> None:
    node = rule(
        "conv-1:fact:D1:3",
        "conv-1:raw:D1:3",
        "Evan said: We're having a family get-together tonight and enjoying homemade lasagna. What's on your menu tonight?",
    )

    statement = extract_rule_statement(node)

    assert statement is not None
    assert "family get-together tonight" in statement.text
    assert "menu" not in statement.text
    assert "time" in statement.signals


def test_random_rule_policy_selects_exact_reproducible_count() -> None:
    graph = MemoryGraph(graph_id="tiny-random")
    for index in range(5):
        raw_id = f"conv-1:raw:D1:{index + 1}"
        graph.add_node(raw(raw_id, f"Statement {index}"))
        graph.add_node(rule(f"conv-1:fact:D1:{index + 1}", raw_id, f"James said: Statement {index}"))

    config = SemanticHierarchyConfig(
        rule_fact_policy="random",
        random_rule_count=2,
        random_rule_seed=7,
    )
    first = SemanticHierarchyBuilder(FakeEncoder(), config).build(graph.model_copy(deep=True))
    second = SemanticHierarchyBuilder(FakeEncoder(), config).build(graph.model_copy(deep=True))

    first_ids = {
        edge.src
        for edge in first.edges
        if edge.metadata.get("hierarchy_v3") == "fact_event"
    }
    second_ids = {
        edge.src
        for edge in second.edges
        if edge.metadata.get("hierarchy_v3") == "fact_event"
    }
    assert len(first_ids) == 2
    assert first_ids == second_ids
    stats = first.metadata["hierarchy_v3"]["rule_filter"]
    assert stats["selected_uncovered_rules"] == 2
    assert stats["rejected_uncovered_rules"] == 3


def raw(node_id: str, text: str) -> Node:
    return Node(node_id=node_id, type=NodeType.RAW, text=text, source="raw_dialogue")


def rule(node_id: str, raw_id: str, text: str) -> Node:
    turn_id = raw_id.rsplit(":raw:", 1)[-1]
    return Node(
        node_id=node_id,
        type=NodeType.FACT,
        text=text,
        source="rule_extracted",
        support_ids=[raw_id],
        metadata={
            "turn_id": turn_id,
            "speaker": text.split(" ", 1)[0],
            "support_raw_ids": [raw_id],
        },
    )


def observation(node_id: str, raw_ids: list[str], text: str) -> Node:
    return Node(
        node_id=node_id,
        type=NodeType.FACT,
        text=text,
        source="locomo_observation",
        support_ids=raw_ids,
        metadata={
            "session": "session_1_observation",
            "speaker": "Caroline",
            "support_raw_ids": raw_ids,
            "support_turn_ids": [raw_id.rsplit(":raw:", 1)[-1] for raw_id in raw_ids],
        },
    )
