from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

from hytopomem.eval.retrieval_metrics import (
    conversation_id_from_question,
    evidence_ids_for_node,
    normalize_evidence_id,
)
from hytopomem.memory.node_extractor import content_terms
from hytopomem.memory.schema import MemoryGraph, NodeStatus, NodeType, RelationType


_TRUE_VALUES = {"1", "true", "yes", "y"}
_DATE_RE = re.compile(r"\bD(\d+):")


FEATURE_NAMES = [
    "ce_score",
    "base_score",
    "bm25_norm",
    "ce_rank",
    "ce_reciprocal_rank",
    "is_seed",
    "hop",
    "is_hop0",
    "is_hop2",
    "entity_overlap",
    "anchor_degree",
    "log_anchor_degree",
    "has_anchor",
    "anchor_confidence",
    "anchor_child_count",
    "edge_confidence_to_anchor",
    "same_session_as_seed",
    "same_speaker_as_seed",
    "same_source_as_seed",
    "support_overlap_with_seed",
    "same_conversation",
    "node_confidence",
    "node_is_active",
    "node_is_outdated",
    "node_is_exception",
    "support_count",
    "text_token_count",
    "query_token_count",
    "query_term_overlap",
    "question_category",
    "candidate_source_filtered_sibling",
    "node_source_observation",
    "node_source_summary",
    "support_day_gap_from_question",
    "candidate_source_seed",
    "candidate_source_same_event",
    "candidate_source_same_topic",
    "has_event",
    "event_degree",
    "log_event_degree",
    "event_confidence",
    "event_coherence",
    "edge_confidence_to_event",
    "same_event_as_seed",
    "has_topic",
    "topic_degree",
    "log_topic_degree",
    "topic_confidence",
    "topic_coherence",
    "event_topic_confidence",
    "same_topic_as_seed",
    "has_topdown_route",
    "has_bottom_up_route",
    "route_from_both",
    "route_source_eu_event",
    "route_source_eu_topic",
    "semantic_score",
    "eu_event_score",
    "eu_topic_score",
    "route_rank",
    "route_reciprocal_rank",
    "eu_event_rank",
    "eu_event_reciprocal_rank",
    "eu_topic_rank",
    "eu_topic_reciprocal_rank",
    "eu_topic_event_rank",
    "eu_topic_event_reciprocal_rank",
    "route_source_hyp_event",
    "route_source_hyp_topic",
    "hyperbolic_score",
    "hyp_event_score",
    "hyp_topic_score",
    "hyp_event_rank",
    "hyp_event_reciprocal_rank",
    "hyp_topic_rank",
    "hyp_topic_reciprocal_rank",
    "hyp_topic_event_rank",
    "hyp_topic_event_reciprocal_rank",
    "fact_offset",
    "bottom_up_rank",
    "bottom_up_reciprocal_rank",
    "is_eu_route",
    "is_hyp_route",
    "route_source_count",
    "route_overlap_score",
    "eu_hyp_agreement",
    "bottom_up_eu_agreement",
    "bottom_up_hyp_agreement",
    "route_consistency_entropy",
    "route_min_rank",
    "route_best_reciprocal_rank",
    "is_bottom_up_only",
    "is_eu_only",
    "is_hyp_only",
    "is_eu_hyp_only",
    "is_all_route_hit",
    "eu_best_rank",
    "hyp_best_rank",
    "topdown_best_rank",
    "eu_best_reciprocal_rank",
    "hyp_best_reciprocal_rank",
    "topdown_best_reciprocal_rank",
    "eu_best_score",
    "hyp_best_score",
    "topdown_best_score",
    "eu_hyp_score_gap",
    "eu_hyp_rank_gap",
    "bottom_up_eu_rank_gap",
    "bottom_up_hyp_rank_gap",
    "eu_rrf_score",
    "hyp_rrf_score",
    "bottom_up_rrf_score",
    "topdown_rrf_score",
    "route_rrf_score",
    "eu_path_complete",
    "hyp_path_complete",
    "topdown_path_complete",
    "best_event_rank",
    "best_topic_rank",
    "event_topic_rank_gap",
    "fact_offset_reciprocal",
    "has_episode",
    "episode_size_events",
    "log_episode_size_events",
    "episode_num_facts",
    "log_episode_num_facts",
    "episode_coherence",
    "is_singleton_episode",
    "event_episode_confidence",
    "episode_candidate_count_in_pool",
    "episode_best_ce_score",
    "candidate_ce_minus_episode_best",
    "episode_seen_by_bottom_up",
    "episode_seen_by_top_down",
    "episode_seen_by_eu",
    "episode_seen_by_hyp",
    "episode_bottom_top_agreement",
    "episode_eu_hyp_agreement",
    "has_temporal_route",
    "route_source_temporal_session",
    "temporal_rank",
    "temporal_reciprocal_rank",
    "temporal_session_degree",
    "log_temporal_session_degree",
    "temporal_event_session_count",
    "same_temporal_session_as_seed",
    "temporal_bottom_up_agreement",
    "temporal_topdown_agreement",
    "is_nary_completion",
    "nary_type_change",
    "nary_type_preference",
    "nary_type_state",
    "nary_type_plan_constraint",
    "nary_role_old_state",
    "nary_role_new_state",
    "nary_role_preference_value",
    "nary_role_polarity",
    "nary_role_state_value",
    "nary_role_plan_goal",
    "nary_role_constraint",
    "nary_role_temporal_scope",
    "nary_role_reason_or_trigger",
    "nary_role_exception",
    "nary_role_context",
    "nary_seed_fact_rank",
    "nary_seed_fact_score",
    "nary_seed_is_bottom_up",
    "nary_seed_is_topdown",
    "nary_seed_is_eu",
    "nary_seed_is_hyp",
    "nary_hyperedge_size",
    "nary_hyperedge_confidence",
    "nary_role_confidence",
    "nary_extractor_qwen",
    "nary_extractor_gpt4o",
    "nary_same_hyperedge_count_in_candidate_pool",
    "nary_role_coverage_potential",
    "nary_completion_rank",
    "nary_completion_reciprocal_rank",
    "nary_pool_covered_roles_count",
    "nary_pool_required_roles_covered",
    "nary_pool_has_preference_and_constraint",
    "nary_pool_has_old_and_new_state",
    "nary_pool_has_reason",
    "nary_pool_has_time_scope",
    "v39_card_ce_score",
    "v39_card_same_event_ratio",
    "v39_card_same_episode_ratio",
    "v39_card_same_topic_ratio",
    "v39_card_branch_entropy",
    "v39_card_bu_td_agreement",
    "v39_card_hyp_route_share",
    "v39_card_avg_hyp_distance",
    "v39_card_max_hyp_distance",
    "v39_fact_to_card_anchor_distance",
]


@dataclass(frozen=True)
class TopologyExample:
    question_id: str
    label: int
    features: list[float]


@dataclass(frozen=True)
class TopologyFeatureIndex:
    anchor_child_counts: dict[str, int]
    edge_confidence_to_anchor: dict[tuple[str, str], float]
    event_child_counts: dict[str, int]
    topic_child_counts: dict[str, int]
    topic_fact_counts: dict[str, int]
    edge_confidence_to_event: dict[tuple[str, str], float]
    edge_confidence_event_topic: dict[tuple[str, str], float]
    fact_event: dict[str, str]
    event_topic: dict[str, str]
    event_episode: dict[str, str]
    episode_topic: dict[str, str]
    episode_event_counts: dict[str, int]
    episode_fact_counts: dict[str, int]
    edge_confidence_event_episode: dict[tuple[str, str], float]
    event_sessions: dict[str, list[str]]
    session_event_counts: dict[str, int]
    node_terms: dict[str, set[str]]
    node_token_counts: dict[str, int]

    @classmethod
    def from_graph(cls, graph: MemoryGraph) -> "TopologyFeatureIndex":
        anchor_child_counts: dict[str, int] = {}
        edge_confidence_to_anchor: dict[tuple[str, str], float] = {}
        event_child_counts: dict[str, int] = {}
        topic_child_counts: dict[str, int] = {}
        topic_fact_counts: dict[str, int] = {}
        edge_confidence_to_event: dict[tuple[str, str], float] = {}
        edge_confidence_event_topic: dict[tuple[str, str], float] = {}
        fact_event: dict[str, str] = {}
        event_topic: dict[str, str] = {}
        event_episode: dict[str, str] = {}
        episode_topic: dict[str, str] = {}
        episode_event_counts: dict[str, int] = {}
        episode_fact_counts: dict[str, int] = {}
        edge_confidence_event_episode: dict[tuple[str, str], float] = {}
        event_sessions: dict[str, list[str]] = {}
        session_event_counts: dict[str, int] = {}
        for edge in graph.edges:
            if edge.relation != RelationType.IS_SPECIFIC_OF:
                continue
            src = graph.nodes.get(edge.src)
            dst = graph.nodes.get(edge.dst)
            if src is None or dst is None or src.type != NodeType.FACT or dst.type != NodeType.ANCHOR:
                if src is not None and dst is not None and src.type == NodeType.FACT and dst.type == NodeType.EVENT:
                    hierarchy_v3_role = edge.metadata.get("hierarchy_v3")
                    if hierarchy_v3_role != "lexical_alias_event":
                        event_child_counts[edge.dst] = event_child_counts.get(edge.dst, 0) + 1
                    fact_event[edge.src] = edge.dst
                    key = (edge.src, edge.dst)
                    edge_confidence_to_event[key] = max(edge_confidence_to_event.get(key, 0.0), edge.confidence)
                elif src is not None and dst is not None and src.type == NodeType.EVENT and dst.type == NodeType.TOPIC:
                    topic_child_counts[edge.dst] = topic_child_counts.get(edge.dst, 0) + 1
                    if edge.metadata.get("hierarchy_v3_3") == "episode_topic":
                        episode_topic[edge.src] = edge.dst
                    else:
                        event_topic[edge.src] = edge.dst
                    key = (edge.src, edge.dst)
                    edge_confidence_event_topic[key] = max(edge_confidence_event_topic.get(key, 0.0), edge.confidence)
                elif src is not None and dst is not None and src.type == NodeType.EVENT and dst.type == NodeType.EVENT:
                    if edge.metadata.get("hierarchy_v3_3") == "event_episode":
                        event_episode[edge.src] = edge.dst
                        episode_event_counts[edge.dst] = episode_event_counts.get(edge.dst, 0) + 1
                        key = (edge.src, edge.dst)
                        edge_confidence_event_episode[key] = max(
                            edge_confidence_event_episode.get(key, 0.0), edge.confidence
                        )
                elif src is not None and dst is not None and src.type == NodeType.EVENT and dst.type == NodeType.SESSION:
                    if edge.metadata.get("hierarchy_v3_4_temporal") == "event_session":
                        event_sessions.setdefault(edge.src, []).append(edge.dst)
                        session_event_counts[edge.dst] = session_event_counts.get(edge.dst, 0) + 1
                continue
            anchor_child_counts[edge.dst] = anchor_child_counts.get(edge.dst, 0) + 1
            key = (edge.src, edge.dst)
            edge_confidence_to_anchor[key] = max(edge_confidence_to_anchor.get(key, 0.0), edge.confidence)
        for fact_id, event_id in fact_event.items():
            episode_id = event_episode.get(event_id)
            if episode_id:
                episode_fact_counts[episode_id] = episode_fact_counts.get(episode_id, 0) + 1
            topic_id = event_topic.get(event_id) or episode_topic.get(episode_id or "")
            if topic_id:
                topic_fact_counts[topic_id] = topic_fact_counts.get(topic_id, 0) + 1
        return cls(
            anchor_child_counts=anchor_child_counts,
            edge_confidence_to_anchor=edge_confidence_to_anchor,
            event_child_counts=event_child_counts,
            topic_child_counts=topic_child_counts,
            topic_fact_counts=topic_fact_counts,
            edge_confidence_to_event=edge_confidence_to_event,
            edge_confidence_event_topic=edge_confidence_event_topic,
            fact_event=fact_event,
            event_topic=event_topic,
            event_episode=event_episode,
            episode_topic=episode_topic,
            episode_event_counts=episode_event_counts,
            episode_fact_counts=episode_fact_counts,
            edge_confidence_event_episode=edge_confidence_event_episode,
            event_sessions=event_sessions,
            session_event_counts=session_event_counts,
            node_terms={node_id: set(content_terms(node.text)) for node_id, node in graph.nodes.items()},
            node_token_counts={node_id: len(node.text.split()) for node_id, node in graph.nodes.items()},
        )


def build_feature_matrix(graph: MemoryGraph, items: Sequence[dict]) -> tuple[list[list[float]], list[int], list[int], list[str]]:
    index = TopologyFeatureIndex.from_graph(graph)
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    question_ids: list[str] = []
    for item in items:
        item = with_cached_query_terms(item)
        item_rows = []
        item_labels = []
        for rank, path in enumerate(item.get("paths", []), start=1):
            example = build_example(graph, item, path, rank, index)
            item_rows.append(example.features)
            item_labels.append(example.label)
        if item_rows:
            rows.extend(item_rows)
            labels.extend(item_labels)
            groups.append(len(item_rows))
            question_ids.append(item["question_id"])
    return rows, labels, groups, question_ids


def select_feature_names(exclude: Sequence[str] | None = None, include: Sequence[str] | None = None) -> list[str]:
    excluded = set(exclude or [])
    if include is not None:
        unknown = set(include) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown feature names: {sorted(unknown)}")
        return [name for name in FEATURE_NAMES if name in set(include) and name not in excluded]
    unknown = excluded - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature names: {sorted(unknown)}")
    return [name for name in FEATURE_NAMES if name not in excluded]


def feature_indices(feature_names: Sequence[str]) -> list[int]:
    name_to_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    unknown = set(feature_names) - set(name_to_index)
    if unknown:
        raise ValueError(f"unknown feature names: {sorted(unknown)}")
    return [name_to_index[name] for name in feature_names]


def project_feature_rows(rows: Sequence[Sequence[float]], feature_names: Sequence[str]) -> list[list[float]]:
    indices = feature_indices(feature_names)
    return [[float(row[index]) for index in indices] for row in rows]


def build_example(
    graph: MemoryGraph,
    item: dict,
    path: dict,
    ce_rank: int,
    index: TopologyFeatureIndex | None = None,
) -> TopologyExample:
    gold = {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}
    evidence_node_id = _evidence_node_id(path)
    matched = evidence_ids_for_node(graph, evidence_node_id) & gold
    features = extract_topology_features(graph, item, path, ce_rank, index)
    return TopologyExample(
        question_id=item["question_id"],
        label=int(bool(matched)),
        features=features,
    )


def extract_topology_features(
    graph: MemoryGraph,
    item: dict,
    path: dict,
    ce_rank: int,
    index: TopologyFeatureIndex | None = None,
) -> list[float]:
    if index is None:
        index = TopologyFeatureIndex.from_graph(graph)
    metadata = path.get("metadata", {})
    scores = path.get("scores", {})
    node_id = _evidence_node_id(path)
    node = graph.nodes.get(node_id)
    seed = graph.nodes.get(str(metadata.get("seed_node_id") or ""))
    anchor = graph.nodes.get(str(metadata.get("anchor_node_id") or ""))
    metadata_event_id = str(metadata.get("event_node_id") or "")
    event = graph.nodes.get(metadata_event_id)
    if event is None:
        event = graph.nodes.get(index.fact_event.get(node_id, ""))
    metadata_topic_id = str(metadata.get("topic_node_id") or "")
    topic = graph.nodes.get(metadata_topic_id)
    if topic is None:
        event_id_for_topic = event.node_id if event else ""
        episode_id_for_topic = index.event_episode.get(event_id_for_topic, "")
        topic = graph.nodes.get(index.event_topic.get(event_id_for_topic, "") or index.episode_topic.get(episode_id_for_topic, ""))
    seed_event_id = index.fact_event.get(seed.node_id if seed else "", "")
    seed_episode_id = index.event_episode.get(seed_event_id, "")
    seed_topic_id = index.event_topic.get(seed_event_id, "") or index.episode_topic.get(seed_episode_id, "")
    episode_event_id = _episode_event_id_for_path(path, node_id, index)
    episode_id = index.event_episode.get(episode_event_id, "")
    episode = graph.nodes.get(episode_id)

    ce_score = float(scores.get("cross_encoder", path.get("score", 0.0)))
    base_score = float(scores.get("base", 0.0))
    bm25_norm = _float(metadata.get("bm25_norm"))
    is_seed = float(_bool(metadata.get("is_seed")))
    hop = _float(metadata.get("hop"))
    anchor_degree = _float(metadata.get("anchor_degree"))
    entity_overlap = _float(metadata.get("entity_overlap"))
    anchor_confidence = float(anchor.confidence) if anchor is not None else 0.0
    anchor_child_count = float(index.anchor_child_counts.get(anchor.node_id, 0)) if anchor is not None else 0.0
    edge_confidence = (
        index.edge_confidence_to_anchor.get((node_id, anchor.node_id), 0.0) if anchor is not None else 0.0
    )
    event_degree = float(index.event_child_counts.get(event.node_id, 0)) if event is not None else 0.0
    topic_degree = float(index.topic_fact_counts.get(topic.node_id, 0)) if topic is not None else 0.0
    edge_confidence_to_event = (
        index.edge_confidence_to_event.get((node_id, event.node_id), 0.0) if event is not None else 0.0
    )
    event_topic_confidence = (
        index.edge_confidence_event_topic.get((event.node_id, topic.node_id), 0.0)
        if event is not None and topic is not None
        else 0.0
    )
    event_episode_confidence = (
        index.edge_confidence_event_episode.get((episode_event_id, episode_id), 0.0)
        if episode_event_id and episode_id
        else 0.0
    )

    query_terms = cached_query_terms(item)
    node_terms = index.node_terms.get(node_id, set())
    query_term_overlap = len(query_terms & node_terms) / max(len(query_terms), 1) if query_terms else 0.0

    same_session = _same_metadata(node, seed, "session")
    same_speaker = _same_metadata(node, seed, "speaker")
    same_source = float(node is not None and seed is not None and node.source == seed.source)
    support_overlap = _support_overlap(node.support_ids if node else [], seed.support_ids if seed else [])

    node_source = node.source if node is not None else ""
    support_gap = _support_day_gap_from_question(item["question_id"], node.support_ids if node else [])
    route_source = str(metadata.get("route_source") or metadata.get("candidate_source") or "")
    has_eu_event = "eu_event" in route_source
    has_eu_topic = "eu_topic" in route_source
    has_hyp_event = "hyp_event" in route_source
    has_hyp_topic = "hyp_topic" in route_source
    has_hyp_bottom = "hyp_bottom" in route_source
    has_temporal_route = "temporal_session" in route_source
    has_eu = has_eu_event or has_eu_topic
    has_hyp = has_hyp_event or has_hyp_topic or has_hyp_bottom
    has_topdown = has_eu_event or has_eu_topic or has_hyp_event or has_hyp_topic
    has_bottom_up = "bottom_up" in route_source
    route_rank = _float(metadata.get("route_rank"))
    eu_event_rank = _float(metadata.get("eu_event_rank"))
    eu_topic_rank = _float(metadata.get("eu_topic_rank"))
    eu_topic_event_rank = _float(metadata.get("eu_topic_event_rank"))
    hyp_event_rank = _float(metadata.get("hyp_event_rank"))
    hyp_topic_rank = _float(metadata.get("hyp_topic_rank"))
    hyp_topic_event_rank = _float(metadata.get("hyp_topic_event_rank"))
    hyp_bottom_rank = _float(metadata.get("hyp_bottom_seed_rank"))
    bottom_up_rank = _float(metadata.get("bottom_up_rank"))
    hyp_event_slot = has_hyp_event or has_hyp_bottom
    route_source_count = float(sum([has_bottom_up, has_eu_event, has_eu_topic, hyp_event_slot, has_hyp_topic]))
    route_overlap_score = route_source_count / 5.0
    route_entropy = _route_entropy([has_bottom_up, has_eu_event, has_eu_topic, hyp_event_slot, has_hyp_topic])
    positive_ranks = [
        rank
        for rank in [
            route_rank,
            eu_event_rank,
            eu_topic_rank,
            eu_topic_event_rank,
            hyp_event_rank,
            hyp_topic_rank,
            hyp_topic_event_rank,
            hyp_bottom_rank,
            bottom_up_rank,
        ]
        if rank > 0.0
    ]
    route_min_rank = min(positive_ranks) if positive_ranks else 0.0
    eu_ranks = [rank for rank in [eu_event_rank, eu_topic_rank, eu_topic_event_rank] if rank > 0.0]
    hyp_ranks = [rank for rank in [hyp_event_rank, hyp_topic_rank, hyp_topic_event_rank, hyp_bottom_rank] if rank > 0.0]
    topdown_ranks = [rank for rank in [*eu_ranks, *hyp_ranks, route_rank] if rank > 0.0]
    eu_best_rank = min(eu_ranks) if eu_ranks else 0.0
    hyp_best_rank = min(hyp_ranks) if hyp_ranks else 0.0
    topdown_best_rank = min(topdown_ranks) if topdown_ranks else 0.0
    bottom_up_only = has_bottom_up and not has_eu and not has_hyp
    eu_only = has_eu and not has_bottom_up and not has_hyp
    hyp_only = has_hyp and not has_bottom_up and not has_eu
    eu_hyp_only = has_eu and has_hyp and not has_bottom_up
    all_route_hit = has_bottom_up and has_eu and has_hyp
    eu_best_score = _max_positive_or_any(
        [
            _float(metadata.get("semantic_score")),
            _float(metadata.get("eu_event_score")),
            _float(metadata.get("eu_topic_score")),
        ]
    )
    hyp_best_score = _max_positive_or_any(
        [
            _float(metadata.get("hyperbolic_score")),
            _float(metadata.get("hyp_event_score")),
            _float(metadata.get("hyp_topic_score")),
            _float(metadata.get("hyp_bottom_score")),
        ]
    )
    topdown_best_score = _max_positive_or_any([eu_best_score, hyp_best_score])
    eu_path_complete = has_eu_event and has_eu_topic
    hyp_path_complete = has_hyp_event and has_hyp_topic
    topdown_path_complete = eu_path_complete or hyp_path_complete
    best_event_rank = _min_positive([eu_event_rank, hyp_event_rank])
    best_topic_rank = _min_positive([eu_topic_rank, hyp_topic_rank])
    fact_offset = _float(metadata.get("fact_offset"))
    episode_size_events = float(index.episode_event_counts.get(episode_id, 0)) if episode is not None else 0.0
    episode_num_facts = float(index.episode_fact_counts.get(episode_id, 0)) if episode is not None else 0.0
    episode_pool = _episode_pool_stats(graph, item, index).get(episode_id, {}) if episode_id else {}
    episode_candidate_count = _float(episode_pool.get("candidate_count"))
    episode_best_ce_score = _float(episode_pool.get("best_ce_score"))
    episode_route_flags = episode_pool.get("route_flags") if isinstance(episode_pool, dict) else None
    if not isinstance(episode_route_flags, dict):
        episode_route_flags = {}
    episode_seen_by_bottom_up = _bool(episode_route_flags.get("bottom_up"))
    episode_seen_by_top_down = _bool(episode_route_flags.get("top_down"))
    episode_seen_by_eu = _bool(episode_route_flags.get("eu"))
    episode_seen_by_hyp = _bool(episode_route_flags.get("hyp"))
    temporal_rank = _float(metadata.get("temporal_rank"))
    temporal_session_id = str(metadata.get("temporal_session_id") or "")
    temporal_session_degree = float(index.session_event_counts.get(temporal_session_id, 0)) if temporal_session_id else 0.0
    candidate_event_id = event.node_id if event is not None else index.fact_event.get(node_id, "")
    candidate_sessions = set(index.event_sessions.get(candidate_event_id, []))
    seed_sessions = set(index.event_sessions.get(seed_event_id, []))
    nary_type = _normalize_token(metadata.get("nary_hyperedge_type"))
    nary_role = _normalize_token(metadata.get("nary_role"))
    nary_seed_route = str(metadata.get("nary_seed_route_origin") or "")
    nary_extractor = str(metadata.get("nary_extractor_type") or "").lower()
    nary_completion_rank = _float(metadata.get("nary_completion_rank"))

    return [
        ce_score,
        base_score,
        bm25_norm,
        float(ce_rank),
        1.0 / max(float(ce_rank), 1.0),
        is_seed,
        hop,
        float(hop == 0.0),
        float(hop == 2.0),
        entity_overlap,
        anchor_degree,
        math.log1p(anchor_degree),
        float(anchor is not None),
        anchor_confidence,
        anchor_child_count,
        edge_confidence,
        same_session,
        same_speaker,
        same_source,
        support_overlap,
        float(node is not None and conversation_id_from_question(item["question_id"]) == conversation_id_from_question(node_id)),
        float(node.confidence) if node is not None else 0.0,
        float(node is not None and node.status == NodeStatus.ACTIVE),
        float(node is not None and node.status == NodeStatus.OUTDATED),
        float(node is not None and node.status == NodeStatus.EXCEPTION),
        float(len(node.support_ids)) if node is not None else 0.0,
        float(index.node_token_counts.get(node_id, 0)),
        float(len(item.get("question", "").split())),
        query_term_overlap,
        _float(item.get("category")),
        float(metadata.get("candidate_source") == "filtered_sibling"),
        float("observation" in node_source),
        float("summary" in node_source),
        support_gap,
        float(metadata.get("candidate_source") == "seed"),
        float(metadata.get("candidate_source") == "same_event"),
        float(metadata.get("candidate_source") == "same_topic"),
        float(event is not None),
        event_degree,
        math.log1p(event_degree),
        float(event.confidence) if event is not None else 0.0,
        _float(event.metadata.get("coherence") if event is not None else None),
        edge_confidence_to_event,
        float(event is not None and event.node_id == seed_event_id),
        float(topic is not None),
        topic_degree,
        math.log1p(topic_degree),
        float(topic.confidence) if topic is not None else 0.0,
        _float(topic.metadata.get("coherence") if topic is not None else None),
        event_topic_confidence,
        float(topic is not None and topic.node_id == seed_topic_id),
        float(has_topdown),
        float(has_bottom_up),
        float(_bool(metadata.get("from_both")) or (has_topdown and has_bottom_up)),
        float(has_eu_event),
        float(has_eu_topic),
        _float(metadata.get("semantic_score")),
        _float(metadata.get("eu_event_score")),
        _float(metadata.get("eu_topic_score")),
        route_rank,
        1.0 / route_rank if route_rank > 0.0 else 0.0,
        eu_event_rank,
        1.0 / eu_event_rank if eu_event_rank > 0.0 else 0.0,
        eu_topic_rank,
        1.0 / eu_topic_rank if eu_topic_rank > 0.0 else 0.0,
        eu_topic_event_rank,
        1.0 / eu_topic_event_rank if eu_topic_event_rank > 0.0 else 0.0,
        float(has_hyp_event),
        float(has_hyp_topic),
        _float(metadata.get("hyperbolic_score")),
        _float(metadata.get("hyp_event_score")),
        _float(metadata.get("hyp_topic_score")),
        hyp_event_rank,
        1.0 / hyp_event_rank if hyp_event_rank > 0.0 else 0.0,
        hyp_topic_rank,
        1.0 / hyp_topic_rank if hyp_topic_rank > 0.0 else 0.0,
        hyp_topic_event_rank,
        1.0 / hyp_topic_event_rank if hyp_topic_event_rank > 0.0 else 0.0,
        _float(metadata.get("fact_offset")),
        bottom_up_rank,
        1.0 / bottom_up_rank if bottom_up_rank > 0.0 else 0.0,
        float(has_eu),
        float(has_hyp),
        route_source_count,
        route_overlap_score,
        float(has_eu and has_hyp),
        float(has_bottom_up and has_eu),
        float(has_bottom_up and has_hyp),
        route_entropy,
        route_min_rank,
        1.0 / route_min_rank if route_min_rank > 0.0 else 0.0,
        float(bottom_up_only),
        float(eu_only),
        float(hyp_only),
        float(eu_hyp_only),
        float(all_route_hit),
        eu_best_rank,
        hyp_best_rank,
        topdown_best_rank,
        1.0 / eu_best_rank if eu_best_rank > 0.0 else 0.0,
        1.0 / hyp_best_rank if hyp_best_rank > 0.0 else 0.0,
        1.0 / topdown_best_rank if topdown_best_rank > 0.0 else 0.0,
        eu_best_score,
        hyp_best_score,
        topdown_best_score,
        eu_best_score - hyp_best_score,
        _rank_gap(eu_best_rank, hyp_best_rank),
        _rank_gap(bottom_up_rank, eu_best_rank),
        _rank_gap(bottom_up_rank, hyp_best_rank),
        _rrf(eu_best_rank),
        _rrf(hyp_best_rank),
        _rrf(bottom_up_rank),
        _rrf(topdown_best_rank),
        _rrf(eu_best_rank) + _rrf(hyp_best_rank) + _rrf(bottom_up_rank),
        float(eu_path_complete),
        float(hyp_path_complete),
        float(topdown_path_complete),
        best_event_rank,
        best_topic_rank,
        _rank_gap(best_event_rank, best_topic_rank),
        1.0 / (fact_offset + 1.0) if fact_offset >= 0.0 else 0.0,
        float(episode is not None),
        episode_size_events,
        math.log1p(episode_size_events),
        episode_num_facts,
        math.log1p(episode_num_facts),
        _float(episode.metadata.get("coherence") if episode is not None else None),
        float(episode_size_events == 1.0),
        event_episode_confidence,
        episode_candidate_count,
        episode_best_ce_score,
        ce_score - episode_best_ce_score if episode_best_ce_score else 0.0,
        float(episode_seen_by_bottom_up),
        float(episode_seen_by_top_down),
        float(episode_seen_by_eu),
        float(episode_seen_by_hyp),
        float(episode_seen_by_bottom_up and episode_seen_by_top_down),
        float(episode_seen_by_eu and episode_seen_by_hyp),
        float(has_temporal_route),
        float(has_temporal_route),
        temporal_rank,
        1.0 / temporal_rank if temporal_rank > 0.0 else 0.0,
        temporal_session_degree,
        math.log1p(temporal_session_degree),
        float(len(candidate_sessions)),
        float(bool(candidate_sessions and seed_sessions and candidate_sessions & seed_sessions)),
        float(has_temporal_route and has_bottom_up),
        float(has_temporal_route and has_topdown),
        float(_bool(metadata.get("is_nary_completion"))),
        float(nary_type == "change"),
        float(nary_type == "preference"),
        float(nary_type == "state"),
        float(nary_type == "plan_constraint"),
        float(nary_role == "old_state"),
        float(nary_role == "new_state"),
        float(nary_role == "preference_value"),
        float(nary_role == "polarity"),
        float(nary_role == "state_value"),
        float(nary_role == "plan_goal"),
        float(nary_role == "constraint"),
        float(nary_role == "temporal_scope"),
        float(nary_role == "reason_or_trigger"),
        float(nary_role == "exception"),
        float(nary_role == "context"),
        _float(metadata.get("nary_seed_fact_rank")),
        _float(metadata.get("nary_seed_fact_score")),
        float("bottom_up" in nary_seed_route),
        float(
            "eu_event" in nary_seed_route
            or "eu_topic" in nary_seed_route
            or "hyp_event" in nary_seed_route
            or "hyp_topic" in nary_seed_route
        ),
        float("eu_event" in nary_seed_route or "eu_topic" in nary_seed_route),
        float("hyp_event" in nary_seed_route or "hyp_topic" in nary_seed_route or "hyp_bottom" in nary_seed_route),
        _float(metadata.get("nary_hyperedge_size")),
        _float(metadata.get("nary_hyperedge_confidence")),
        _float(metadata.get("nary_role_confidence")),
        float("qwen" in nary_extractor),
        float("gpt" in nary_extractor or "4o" in nary_extractor),
        _float(metadata.get("nary_same_hyperedge_count_in_candidate_pool")),
        _float(metadata.get("nary_role_coverage_potential")),
        nary_completion_rank,
        1.0 / nary_completion_rank if nary_completion_rank > 0.0 else 0.0,
        _float(metadata.get("nary_pool_covered_roles_count")),
        _float(metadata.get("nary_pool_required_roles_covered")),
        _float(metadata.get("nary_pool_has_preference_and_constraint")),
        _float(metadata.get("nary_pool_has_old_and_new_state")),
        _float(metadata.get("nary_pool_has_reason")),
        _float(metadata.get("nary_pool_has_time_scope")),
        _float(metadata.get("v39_card_ce_score")),
        _float(metadata.get("v39_card_same_event_ratio")),
        _float(metadata.get("v39_card_same_episode_ratio")),
        _float(metadata.get("v39_card_same_topic_ratio")),
        _float(metadata.get("v39_card_branch_entropy")),
        _float(metadata.get("v39_card_bu_td_agreement")),
        _float(metadata.get("v39_card_hyp_route_share")),
        _float(metadata.get("v39_card_avg_hyp_distance")),
        _float(metadata.get("v39_card_max_hyp_distance")),
        _float(metadata.get("v39_fact_to_card_anchor_distance")),
    ]


def rerank_items_with_scores(items: Sequence[dict], scores_by_question: dict[str, list[float]], top_k: int) -> list[dict]:
    outputs = []
    for item in items:
        question_id = item["question_id"]
        scores = scores_by_question.get(question_id, [])
        paths = []
        for path, score in zip(item.get("paths", []), scores):
            new_path = dict(path)
            new_scores = dict(new_path.get("scores", {}))
            new_scores["topology_selector"] = float(score)
            new_path["scores"] = new_scores
            new_path["score"] = float(score)
            metadata = dict(new_path.get("metadata", {}))
            metadata["retriever"] = "topology_selector"
            new_path["metadata"] = metadata
            paths.append(new_path)
        paths.sort(key=lambda path: path.get("score", 0.0), reverse=True)
        new_item = dict(item)
        new_item["paths"] = paths[:top_k]
        new_metadata = dict(new_item.get("metadata", {}))
        new_metadata["method"] = "topology_selector"
        new_metadata["selector_input_size"] = len(item.get("paths", []))
        new_metadata["final_topk"] = top_k
        new_item["metadata"] = new_metadata
        outputs.append(new_item)
    return outputs


def split_items_by_conversation(items: Sequence[dict], holdout_conversations: int) -> tuple[list[dict], list[dict], list[str]]:
    conversation_order = []
    seen = set()
    for item in items:
        conversation_id = conversation_id_from_question(item["question_id"])
        if conversation_id not in seen:
            seen.add(conversation_id)
            conversation_order.append(conversation_id)
    holdout = set(conversation_order[-holdout_conversations:]) if holdout_conversations else set()
    train = [item for item in items if conversation_id_from_question(item["question_id"]) not in holdout]
    valid = [item for item in items if conversation_id_from_question(item["question_id"]) in holdout]
    return train, valid, conversation_order[-holdout_conversations:]


def group_scores_by_question(
    graph: MemoryGraph,
    items: Sequence[dict],
    model,
    feature_names: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    index = TopologyFeatureIndex.from_graph(graph)
    selected_features = list(feature_names or FEATURE_NAMES)
    selected_indices = feature_indices(selected_features)
    scores_by_question: dict[str, list[float]] = {}
    for item in items:
        item = with_cached_query_terms(item)
        rows = [
            extract_topology_features(graph, item, path, rank, index)
            for rank, path in enumerate(item.get("paths", []), start=1)
        ]
        rows = [[float(row[index]) for index in selected_indices] for row in rows]
        if rows:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                scores_by_question[item["question_id"]] = [float(score) for score in model.predict(rows)]
        else:
            scores_by_question[item["question_id"]] = []
    return scores_by_question


def with_cached_query_terms(item: dict) -> dict:
    if "_topology_query_terms" in item:
        return item
    copied = dict(item)
    copied["_topology_query_terms"] = list(content_terms(copied.get("question", "")))
    return copied


def cached_query_terms(item: dict) -> set[str]:
    cached = item.get("_topology_query_terms")
    if cached is not None:
        return set(cached)
    return set(content_terms(item.get("question", "")))


def _evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def _episode_event_id_for_path(path: dict, node_id: str, index: TopologyFeatureIndex) -> str:
    metadata = path.get("metadata", {})
    metadata_event_id = str(metadata.get("event_node_id") or "")
    if metadata_event_id in index.event_episode:
        return metadata_event_id
    graph_event_id = index.fact_event.get(node_id, "")
    if graph_event_id in index.event_episode:
        return graph_event_id
    return metadata_event_id or graph_event_id


def _episode_pool_stats(graph: MemoryGraph, item: dict, index: TopologyFeatureIndex) -> dict[str, dict]:
    cache_key = "_topology_episode_pool_stats"
    cached = item.get(cache_key)
    if isinstance(cached, dict):
        return cached
    stats: dict[str, dict] = {}
    for path in item.get("paths", []):
        node_id = _evidence_node_id(path)
        event_id = _episode_event_id_for_path(path, node_id, index)
        episode_id = index.event_episode.get(event_id, "")
        if not episode_id or graph.nodes.get(episode_id) is None:
            continue
        metadata = path.get("metadata", {})
        route_source = str(metadata.get("route_source") or metadata.get("candidate_source") or "")
        has_eu = "eu_event" in route_source or "eu_topic" in route_source
        has_hyp = "hyp_event" in route_source or "hyp_topic" in route_source or "hyp_bottom" in route_source
        has_topdown = "eu_event" in route_source or "eu_topic" in route_source or "hyp_event" in route_source or "hyp_topic" in route_source
        has_bottom_up = "bottom_up" in route_source
        scores = path.get("scores", {})
        ce_score = float(scores.get("cross_encoder", path.get("score", 0.0)))
        row = stats.setdefault(
            episode_id,
            {
                "candidate_count": 0.0,
                "best_ce_score": 0.0,
                "route_flags": {"bottom_up": False, "top_down": False, "eu": False, "hyp": False},
            },
        )
        row["candidate_count"] = float(row["candidate_count"]) + 1.0
        row["best_ce_score"] = max(float(row["best_ce_score"]), ce_score)
        flags = row["route_flags"]
        flags["bottom_up"] = bool(flags["bottom_up"] or has_bottom_up)
        flags["top_down"] = bool(flags["top_down"] or has_topdown)
        flags["eu"] = bool(flags["eu"] or has_eu)
        flags["hyp"] = bool(flags["hyp"] or has_hyp)
    item[cache_key] = stats
    return stats


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    return str(value).strip().lower() in _TRUE_VALUES


def _normalize_token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _anchor_child_count(graph: MemoryGraph, anchor_id: str) -> int:
    return sum(
        1
        for edge in graph.incoming(anchor_id, RelationType.IS_SPECIFIC_OF)
        if graph.nodes.get(edge.src) is not None and graph.nodes[edge.src].type == NodeType.FACT
    )


def _edge_confidence_to_anchor(graph: MemoryGraph, node_id: str, anchor_id: str) -> float:
    confidences = [
        edge.confidence
        for edge in graph.outgoing(node_id, RelationType.IS_SPECIFIC_OF)
        if edge.dst == anchor_id
    ]
    return max(confidences) if confidences else 0.0


def _same_metadata(node, seed, key: str) -> float:
    if node is None or seed is None:
        return 0.0
    left = node.metadata.get(key)
    right = seed.metadata.get(key)
    return float(bool(left) and left == right)


def _support_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {normalize_evidence_id(item) for item in left}
    right_set = {normalize_evidence_id(item) for item in right}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _route_entropy(flags: Sequence[bool]) -> float:
    active = sum(1 for flag in flags if flag)
    if active <= 1:
        return 0.0
    probability = 1.0 / float(active)
    return -float(active) * probability * math.log(probability)


def _min_positive(values: Sequence[float]) -> float:
    positive = [value for value in values if value > 0.0]
    return min(positive) if positive else 0.0


def _max_positive_or_any(values: Sequence[float]) -> float:
    positive = [value for value in values if value > 0.0]
    if positive:
        return max(positive)
    return max(values) if values else 0.0


def _rank_gap(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return left - right


def _rrf(rank: float, constant: float = 60.0) -> float:
    if rank <= 0.0:
        return 0.0
    return 1.0 / (constant + rank)


def _support_day_gap_from_question(question_id: str, support_ids: Sequence[str]) -> float:
    q_match = _DATE_RE.search(question_id)
    if not q_match:
        return 0.0
    q_day = int(q_match.group(1))
    days = []
    for support_id in support_ids:
        match = _DATE_RE.search(str(support_id))
        if match:
            days.append(abs(q_day - int(match.group(1))))
    return float(min(days)) if days else 0.0
