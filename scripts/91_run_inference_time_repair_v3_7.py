from __future__ import annotations

import argparse
import importlib.util
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import conversation_id_from_question, evaluate_item, summarize
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.node_extractor import content_terms
from hytopomem.memory.schema import MemoryGraph
from hytopomem.retrieval.topology_features import TopologyFeatureIndex, project_feature_rows
from hytopomem.retrieval.topology_selector import train_lightgbm_ranker


VARIANTS = [
    "base_selector",
    "hyp_path_rerank",
    "nary_verified_add",
    "hyp_nary_verified_add",
    "evidence_set_beam",
]

REQUIRED_ROLES = {
    "change": {"old_state", "new_state"},
    "preference": {"preference_value"},
    "state": {"state_value"},
    "plan_constraint": {"plan_goal", "constraint"},
}


@dataclass
class CandidateView:
    path: dict
    fact_id: str
    selector_score: float
    selector_norm: float = 0.0
    ce_norm: float = 0.0
    hyp_raw: float = 0.0
    hyp_norm: float = 0.0
    nary_score: float = 0.0
    verified_completion: bool = False
    relation_type: str = ""
    role: str = ""
    hyperedge_id: str = ""
    event_id: str = ""
    episode_id: str = ""
    topic_id: str = ""
    terms: set[str] = field(default_factory=set)


@dataclass
class QueryProfile:
    type_weights: dict[str, float]
    role_weights: dict[str, float]
    raw_terms: set[str]


@dataclass
class BeamState:
    selected: tuple[int, ...]
    score: float
    roles_by_hyperedge: dict[str, set[str]] = field(default_factory=dict)
    terms: set[str] = field(default_factory=set)
    topic_counts: Counter = field(default_factory=Counter)
    episode_counts: Counter = field(default_factory=Counter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--candidates", default="outputs/nary_v3_6c_selector/qwen_all_base100_completion50_paths.json")
    parser.add_argument("--output-dir", default="outputs/eval/v3_7_inference_time_repair_qwen_all")
    parser.add_argument("--final-topks", default="5,20")
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--base-pool-topn", type=int, default=100)
    parser.add_argument("--completion-pool-topn", type=int, default=50)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--alpha-hyp", type=float, default=0.18)
    parser.add_argument("--beta-nary", type=float, default=0.28)
    parser.add_argument("--role-complete-bonus", type=float, default=0.18)
    parser.add_argument("--branch-bonus", type=float, default=0.04)
    parser.add_argument("--redundancy-penalty", type=float, default=0.08)
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()

    cv = load_cv_helpers()
    graph_v2 = load_graph_v2_selector_module()
    feature_names = selector_feature_names(graph_v2)
    started = time.perf_counter()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph or config["graph"]["graph_path"]))
    index = TopologyFeatureIndex.from_graph(graph)
    candidates = read_json(resolve_path(args.candidates))
    output_dir = resolve_path(args.output_dir)
    final_topks = [int(value) for value in args.final_topks.split(",") if value.strip()]

    conversation_ids = cv.ordered_conversations(candidates)
    if args.max_conversations:
        conversation_ids = conversation_ids[: args.max_conversations]
        keep = set(conversation_ids)
        candidates = [item for item in candidates if conversation_id_from_question(item["question_id"]) in keep]
    if args.max_folds:
        conversation_ids = conversation_ids[: args.max_folds]
    print(
        f"loaded V3.7 repair: conversations={len(cv.ordered_conversations(candidates))} "
        f"folds={len(conversation_ids)} questions={len(candidates)} features={len(feature_names)}",
        flush=True,
    )
    feature_cache = cv.build_feature_cache(graph, candidates)
    print(f"built feature cache elapsed={time.perf_counter() - started:.1f}s", flush=True)

    aggregate_results: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    aggregate_completion: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    folds = []
    paths_by_variant_k: dict[tuple[str, int], list[dict]] = defaultdict(list)

    for fold_index, test_conversation in enumerate(conversation_ids):
        fold_started = time.perf_counter()
        fold_dir = output_dir / f"fold_{fold_index:02d}_{test_conversation}"
        train_items = [
            item for item in candidates if conversation_id_from_question(item["question_id"]) != test_conversation
        ]
        test_items = [
            item for item in candidates if conversation_id_from_question(item["question_id"]) == test_conversation
        ]
        train_rows_all, train_labels, train_groups, train_question_ids = cv.flatten_feature_cache(
            feature_cache,
            [item["question_id"] for item in train_items],
        )
        print(
            f"training fold={fold_index} train_questions={len(train_question_ids)} "
            f"examples={len(train_rows_all)} positives={sum(train_labels)}",
            flush=True,
        )
        model = train_lightgbm_ranker(
            project_feature_rows(train_rows_all, feature_names),
            train_labels,
            train_groups,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            n_jobs=args.n_jobs,
            random_state=args.random_state + fold_index,
        )
        scores = cv.score_items_with_cache(test_items, model, feature_cache, feature_names)
        fold_payload = {
            "fold": fold_index,
            "test_conversation": test_conversation,
            "num_test_questions": len(test_items),
            "methods": {},
        }
        for k in final_topks:
            variant_items: dict[str, list[dict]] = {variant: [] for variant in VARIANTS}
            completion_stats: dict[str, list[dict]] = {variant: [] for variant in VARIANTS}
            for item in test_items:
                views = build_candidate_views(graph, index, item, scores.get(item["question_id"], []))
                repaired = run_variants(item, views, k, args)
                for variant, paths in repaired.items():
                    out_item = dict(item)
                    out_item["paths"] = paths
                    metadata = dict(out_item.get("metadata", {}))
                    metadata.update(
                        {
                            "method": variant,
                            "v3_7": "inference_time_repair",
                            "final_topk": k,
                            "test_conversation": test_conversation,
                        }
                    )
                    out_item["metadata"] = metadata
                    variant_items[variant].append(out_item)
                    completion_stats[variant].append(completion_summary_for_item(paths, k))
            for variant, items in variant_items.items():
                payload = evaluate_items(graph, items, k, variant)
                fold_payload["methods"][f"{variant}@{k}"] = payload["summary"]
                aggregate_results[variant][k].extend(payload["per_question"])
                aggregate_completion[variant][k].extend(completion_stats[variant])
                write_json(items, fold_dir / f"{variant}_top{k}_paths.json")
                write_json(payload, fold_dir / f"{variant}_top{k}_eval.json")
                paths_by_variant_k[(variant, k)].extend(items)
        write_json(fold_payload, fold_dir / "fold_summary.json")
        folds.append(fold_payload)
        print_fold(fold_payload)
        print(f"finished fold={fold_index} elapsed={time.perf_counter() - fold_started:.1f}s", flush=True)

    summary = build_summary(
        aggregate_results=aggregate_results,
        aggregate_completion=aggregate_completion,
        folds=folds,
        output_dir=output_dir,
        feature_names=feature_names,
        args=args,
        final_topks=final_topks,
    )
    write_json(summary, output_dir / "v3_7_repair_summary.json")
    write_markdown_summary(summary, output_dir / "v3_7_repair_summary.md")
    for (variant, k), items in paths_by_variant_k.items():
        write_json(items, output_dir / f"{variant}_top{k}_paths.json")
    print(render_markdown_summary(summary))
    print(f"wrote {output_dir / 'v3_7_repair_summary.json'}")
    print(f"wrote {output_dir / 'v3_7_repair_summary.md'}")


def selector_feature_names(graph_v2) -> list[str]:
    retrieval = graph_v2.dedupe(graph_v2.RETRIEVAL_FEATURES)
    graph_features = graph_v2.dedupe(retrieval + graph_v2.GRAPH_V2_FEATURES)
    return graph_v2.dedupe(
        graph_features + graph_v2.ROUTE_ORIGIN_V1_FEATURES + graph_v2.ROUTE_AGREEMENT_V1_FEATURES
    )


def build_candidate_views(
    graph: MemoryGraph,
    index: TopologyFeatureIndex,
    item: dict,
    selector_scores: list[float],
) -> list[CandidateView]:
    query_terms = set(content_terms(item.get("question", "")))
    profile = query_profile(item.get("question", ""))
    views = []
    for path, selector_score in zip(item.get("paths", []), selector_scores):
        fact_id = evidence_node_id(path)
        if not fact_id or graph.nodes.get(fact_id) is None:
            continue
        event_id = event_id_for_path(path, fact_id, index)
        episode_id = index.event_episode.get(event_id, "")
        topic_id = index.event_topic.get(event_id, "") or index.episode_topic.get(episode_id, "")
        relation_type = str(path.get("metadata", {}).get("nary_hyperedge_type") or "")
        role = str(path.get("metadata", {}).get("nary_role") or "")
        view = CandidateView(
            path=path,
            fact_id=fact_id,
            selector_score=float(selector_score),
            relation_type=relation_type,
            role=role,
            hyperedge_id=str(path.get("metadata", {}).get("nary_hyperedge_id") or ""),
            event_id=event_id,
            episode_id=episode_id,
            topic_id=topic_id,
            terms=set(content_terms(graph.nodes[fact_id].text)),
        )
        view.hyp_raw = hyp_path_score(graph, index, query_terms, view)
        view.nary_score = nary_verification_score(path, profile)
        view.verified_completion = is_completion(path) and view.nary_score >= 0.36
        views.append(view)
    selector_values = [view.selector_score for view in views]
    ce_values = [path_ce_score(view.path) for view in views]
    hyp_values = [view.hyp_raw for view in views]
    for view in views:
        view.selector_norm = minmax_value(view.selector_score, selector_values)
        view.ce_norm = minmax_value(path_ce_score(view.path), ce_values)
        view.hyp_norm = minmax_value(view.hyp_raw, hyp_values)
    return views


def run_variants(item: dict, views: list[CandidateView], k: int, args) -> dict[str, list[dict]]:
    base_views = [view for view in views if not is_completion(view.path)]
    completion_views = [view for view in views if view.verified_completion]
    base_ranked = sorted(base_views, key=lambda view: view.selector_score, reverse=True)
    hyp_ranked = sorted(
        base_views,
        key=lambda view: view.selector_norm + args.alpha_hyp * view.hyp_norm,
        reverse=True,
    )
    nary_pool = prune_pool(base_ranked, completion_views, args)
    nary_ranked = sorted(
        nary_pool,
        key=lambda view: view.selector_norm + args.beta_nary * view.nary_score,
        reverse=True,
    )
    hyp_nary_ranked = sorted(
        nary_pool,
        key=lambda view: view.selector_norm + args.alpha_hyp * view.hyp_norm + args.beta_nary * view.nary_score,
        reverse=True,
    )
    beam_ranked = beam_select(nary_pool, k, args)
    return {
        "base_selector": paths_from_views(base_ranked[:k], "base_selector"),
        "hyp_path_rerank": paths_from_views(hyp_ranked[:k], "hyp_path_rerank"),
        "nary_verified_add": paths_from_views(nary_ranked[:k], "nary_verified_add"),
        "hyp_nary_verified_add": paths_from_views(hyp_nary_ranked[:k], "hyp_nary_verified_add"),
        "evidence_set_beam": paths_from_views(beam_ranked[:k], "evidence_set_beam"),
    }


def prune_pool(base_ranked: list[CandidateView], completion_views: list[CandidateView], args) -> list[CandidateView]:
    output = []
    seen = set()
    for view in base_ranked[: args.base_pool_topn]:
        if view.fact_id not in seen:
            seen.add(view.fact_id)
            output.append(view)
    completion_ranked = sorted(
        completion_views,
        key=lambda view: (
            view.nary_score,
            view.selector_norm,
            _float(view.path.get("metadata", {}).get("nary_seed_fact_score")),
        ),
        reverse=True,
    )
    for view in completion_ranked[: args.completion_pool_topn]:
        if view.fact_id not in seen:
            seen.add(view.fact_id)
            output.append(view)
    return output


def beam_select(pool: list[CandidateView], k: int, args) -> list[CandidateView]:
    pool = sorted(
        pool,
        key=lambda view: view.selector_norm + args.alpha_hyp * view.hyp_norm + args.beta_nary * view.nary_score,
        reverse=True,
    )[: max(args.base_pool_topn + args.completion_pool_topn, k)]
    beams = [BeamState(selected=tuple(), score=0.0)]
    for _ in range(k):
        candidates = []
        for state in beams:
            selected_set = set(state.selected)
            for idx, view in enumerate(pool):
                if idx in selected_set:
                    continue
                gain = candidate_gain(view, state, args)
                candidates.append(advance_state(state, idx, view, gain))
        candidates.sort(key=lambda state: state.score, reverse=True)
        beams = dedupe_states(candidates)[: args.beam_width]
        if not beams:
            break
    if not beams:
        return []
    best = beams[0]
    return [pool[idx] for idx in best.selected]


def candidate_gain(view: CandidateView, state: BeamState, args) -> float:
    base = view.selector_norm + args.alpha_hyp * view.hyp_norm
    if is_completion(view.path):
        base += args.beta_nary * view.nary_score
    if view.hyperedge_id and view.role:
        existing_roles = state.roles_by_hyperedge.get(view.hyperedge_id, set())
        if view.role not in existing_roles:
            base += args.role_complete_bonus * role_completion_gain(view, existing_roles)
    if view.topic_id and state.topic_counts.get(view.topic_id, 0):
        base += args.branch_bonus
    if view.episode_id and state.episode_counts.get(view.episode_id, 0):
        base += args.branch_bonus
    if state.terms and view.terms:
        redundancy = len(state.terms & view.terms) / max(len(view.terms), 1)
        base -= args.redundancy_penalty * redundancy
    return base


def advance_state(state: BeamState, idx: int, view: CandidateView, gain: float) -> BeamState:
    roles_by_hyperedge = {key: set(value) for key, value in state.roles_by_hyperedge.items()}
    if view.hyperedge_id and view.role:
        roles_by_hyperedge.setdefault(view.hyperedge_id, set()).add(view.role)
    topic_counts = Counter(state.topic_counts)
    episode_counts = Counter(state.episode_counts)
    if view.topic_id:
        topic_counts[view.topic_id] += 1
    if view.episode_id:
        episode_counts[view.episode_id] += 1
    return BeamState(
        selected=state.selected + (idx,),
        score=state.score + gain,
        roles_by_hyperedge=roles_by_hyperedge,
        terms=set(state.terms) | set(view.terms),
        topic_counts=topic_counts,
        episode_counts=episode_counts,
    )


def dedupe_states(states: list[BeamState]) -> list[BeamState]:
    output = []
    seen = set()
    for state in states:
        key = tuple(sorted(state.selected))
        if key in seen:
            continue
        seen.add(key)
        output.append(state)
    return output


def role_completion_gain(view: CandidateView, existing_roles: set[str]) -> float:
    required = REQUIRED_ROLES.get(view.relation_type, set())
    if view.role in required:
        before = required.issubset(existing_roles)
        after = required.issubset(existing_roles | {view.role})
        return 1.5 if after and not before else 1.0
    return 0.5


def hyp_path_score(graph: MemoryGraph, index: TopologyFeatureIndex, query_terms: set[str], view: CandidateView) -> float:
    if not query_terms:
        return 0.0
    event = graph.nodes.get(view.event_id)
    episode = graph.nodes.get(view.episode_id)
    topic = graph.nodes.get(view.topic_id)
    path_overlap = (
        0.45 * term_overlap(query_terms, topic.text if topic else "")
        + 0.75 * term_overlap(query_terms, episode.text if episode else "")
        + 0.60 * term_overlap(query_terms, event.text if event else "")
    )
    metadata = view.path.get("metadata", {})
    route_source = str(metadata.get("route_source") or metadata.get("candidate_source") or "")
    route_bonus = 0.0
    route_bonus += 0.10 if "bottom_up" in route_source else 0.0
    route_bonus += 0.10 if "eu_event" in route_source or "eu_topic" in route_source else 0.0
    route_bonus += 0.10 if "hyp_event" in route_source or "hyp_topic" in route_source or "hyp_bottom" in route_source else 0.0
    coherence = 0.0
    if event is not None:
        coherence += 0.05 * _float(event.metadata.get("coherence"))
    if episode is not None:
        coherence += 0.05 * _float(episode.metadata.get("coherence"))
    if topic is not None:
        coherence += 0.05 * _float(topic.metadata.get("coherence"))
    return path_overlap + route_bonus + coherence


def query_profile(query: str) -> QueryProfile:
    lowered = query.lower()
    terms = set(content_terms(query))
    type_weights = defaultdict(float)
    role_weights = defaultdict(float)

    if any(word in lowered for word in ["prefer", "preference", "favorite", "favourite", "like", "likes", "enjoy", "enjoys", "dislike"]):
        type_weights["preference"] = 1.0
        role_weights["preference_value"] = 1.0
        role_weights["polarity"] = 0.7
    if any(word in lowered for word in ["constraint", "limit", "restriction", "cannot", "can't", "avoid", "allergy", "requirement"]):
        type_weights["preference"] = max(type_weights["preference"], 0.8)
        type_weights["plan_constraint"] = max(type_weights["plan_constraint"], 0.8)
        role_weights["constraint"] = 1.0
        role_weights["exception"] = 0.8
    if any(word in lowered for word in ["change", "changed", "switch", "switched", "instead", "no longer", "used to", "previously", "before", "after"]):
        type_weights["change"] = 1.0
        role_weights["old_state"] = 0.8
        role_weights["new_state"] = 1.0
    if any(word in lowered for word in ["plan", "plans", "planned", "schedule", "trip", "travel", "meeting", "deadline", "appointment", "task"]):
        type_weights["plan_constraint"] = max(type_weights["plan_constraint"], 1.0)
        role_weights["plan_goal"] = 1.0
        role_weights["constraint"] = max(role_weights["constraint"], 0.7)
    if any(word in lowered for word in ["status", "state", "current", "currently", "where", "what is", "what was", "how is", "how was"]):
        type_weights["state"] = max(type_weights["state"], 0.8)
        role_weights["state_value"] = 1.0
        role_weights["context"] = 0.5
    if lowered.startswith("why") or " why " in lowered or "because" in lowered or "reason" in lowered:
        type_weights["change"] = max(type_weights["change"], 0.8)
        type_weights["plan_constraint"] = max(type_weights["plan_constraint"], 0.8)
        role_weights["reason_or_trigger"] = 1.0
    if lowered.startswith("when") or "date" in lowered or "time" in lowered:
        role_weights["temporal_scope"] = 1.0
    if not type_weights:
        type_weights["state"] = 0.35
        type_weights["preference"] = 0.25
        type_weights["plan_constraint"] = 0.25
        type_weights["change"] = 0.15
    return QueryProfile(dict(type_weights), dict(role_weights), terms)


def nary_verification_score(path: dict, profile: QueryProfile) -> float:
    if not is_completion(path):
        return 0.0
    metadata = path.get("metadata", {})
    relation_type = str(metadata.get("nary_hyperedge_type") or "")
    role = str(metadata.get("nary_role") or "")
    type_weight = profile.type_weights.get(relation_type, 0.0)
    role_weight = profile.role_weights.get(role, 0.0)
    if role_weight == 0.0 and role in REQUIRED_ROLES.get(relation_type, set()):
        role_weight = 0.25
    seed_rank = _float(metadata.get("nary_seed_fact_rank"))
    seed_rr = 1.0 / seed_rank if seed_rank > 0.0 else 0.0
    conf = max(_float(metadata.get("nary_hyperedge_confidence")), _float(metadata.get("nary_role_confidence")))
    pool_required = _float(metadata.get("nary_pool_required_roles_covered"))
    pool_roles = min(_float(metadata.get("nary_pool_covered_roles_count")) / 4.0, 1.0)
    return (
        0.38 * type_weight
        + 0.34 * role_weight
        + 0.10 * seed_rr
        + 0.08 * conf
        + 0.06 * pool_required
        + 0.04 * pool_roles
    )


def paths_from_views(views: list[CandidateView], method: str) -> list[dict]:
    output = []
    for rank, view in enumerate(views, start=1):
        path = dict(view.path)
        scores = dict(path.get("scores", {}))
        scores.update(
            {
                "v3_7_selector": view.selector_score,
                "v3_7_selector_norm": view.selector_norm,
                "v3_7_hyp_path": view.hyp_norm,
                "v3_7_nary_verification": view.nary_score,
            }
        )
        path["scores"] = scores
        path["score"] = float(view.selector_score)
        metadata = dict(path.get("metadata", {}))
        metadata.update(
            {
                "retriever": method,
                "v3_7_rank": rank,
                "v3_7_verified_completion": str(bool(view.verified_completion)).lower(),
            }
        )
        path["metadata"] = metadata
        output.append(path)
    return output


def completion_summary_for_item(paths: list[dict], k: int) -> dict:
    selected = paths[:k]
    completion_count = sum(1 for path in selected if is_completion(path))
    verified_count = sum(
        1 for path in selected if str(path.get("metadata", {}).get("v3_7_verified_completion", "")).lower() == "true"
    )
    return {
        "completion_count": completion_count,
        "verified_completion_count": verified_count,
    }


def evaluate_items(graph, items: list[dict], k: int, method: str) -> dict:
    results = [evaluate_item(graph, item, k) for item in items]
    return {
        "method": method,
        "k": k,
        "summary": summarize(results),
        "per_question": [result.__dict__ for result in results],
    }


def build_summary(
    *,
    aggregate_results: dict[str, dict[int, list]],
    aggregate_completion: dict[str, dict[int, list[dict]]],
    folds: list[dict],
    output_dir: Path,
    feature_names: list[str],
    args,
    final_topks: list[int],
) -> dict:
    aggregate = {}
    fold_stats = {}
    completion = {}
    for variant in VARIANTS:
        aggregate[variant] = {}
        fold_stats[variant] = {}
        completion[variant] = {}
        for k in final_topks:
            rows = aggregate_results[variant][k]
            aggregate[variant][f"top{k}"] = summarize_rows(rows)
            per_fold = [fold["methods"][f"{variant}@{k}"] for fold in folds]
            fold_stats[variant][f"top{k}"] = {
                "mean_hit": mean(metric["hit"] for metric in per_fold),
                "std_hit": pstdev(metric["hit"] for metric in per_fold),
                "mean_recall": mean(metric["recall"] for metric in per_fold),
                "std_recall": pstdev(metric["recall"] for metric in per_fold),
                "mean_full_cover": mean(metric["full_cover"] for metric in per_fold),
                "std_full_cover": pstdev(metric["full_cover"] for metric in per_fold),
            }
            comp_rows = aggregate_completion[variant][k]
            completion[variant][f"top{k}"] = {
                "avg_completion_selected": mean(row["completion_count"] for row in comp_rows) if comp_rows else 0.0,
                "avg_verified_completion_selected": mean(row["verified_completion_count"] for row in comp_rows)
                if comp_rows
                else 0.0,
                "questions_with_completion": sum(row["completion_count"] > 0 for row in comp_rows),
            }
    return {
        "method": "V3.7 inference-time geometric-relational evidence repair",
        "candidate_file": str(resolve_path(args.candidates)),
        "graph": str(resolve_path(args.graph)),
        "output_dir": str(output_dir),
        "folds": len(folds),
        "feature_names": feature_names,
        "params": {
            "base_pool_topn": args.base_pool_topn,
            "completion_pool_topn": args.completion_pool_topn,
            "beam_width": args.beam_width,
            "alpha_hyp": args.alpha_hyp,
            "beta_nary": args.beta_nary,
            "role_complete_bonus": args.role_complete_bonus,
            "branch_bonus": args.branch_bonus,
            "redundancy_penalty": args.redundancy_penalty,
        },
        "aggregate": aggregate,
        "fold_mean_std": fold_stats,
        "selected_completion": completion,
        "folds_detail": folds,
    }


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"num_questions": 0, "hit": 0.0, "recall": 0.0, "full_cover": 0.0}
    n = len(rows)
    return {
        "num_questions": n,
        "hit": sum(float(row["hit"]) for row in rows) / n,
        "recall": sum(float(row["recall"]) for row in rows) / n,
        "full_cover": sum(float(row["full_cover"]) for row in rows) / n,
        "avg_tokens": sum(float(row["tokens"]) for row in rows) / n,
        "avg_path_len": sum(float(row["path_len"]) for row in rows) / n,
    }


def print_fold(fold: dict) -> None:
    print(f"fold={fold['fold']} test={fold['test_conversation']} questions={fold['num_test_questions']}")
    for name, metrics in fold["methods"].items():
        print(f"  {name}: hit={metrics['hit']:.4f} recall={metrics['recall']:.4f} full_cover={metrics['full_cover']:.4f}")


def write_markdown_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_summary(summary), encoding="utf-8")


def render_markdown_summary(summary: dict) -> str:
    lines = [
        "# V3.7 Inference-time Geometric-Relational Evidence Repair",
        "",
        f"Candidates: `{summary['candidate_file']}`",
        f"Graph: `{summary['graph']}`",
        "",
        "## Aggregate",
        "",
        "| Variant | K | Hit | Recall | FullCover | Avg Completion | Avg Verified Completion | Q w/ Completion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, by_k in summary["aggregate"].items():
        for k_name, metrics in by_k.items():
            comp = summary["selected_completion"][variant][k_name]
            k = k_name.replace("top", "")
            lines.append(
                f"| {variant} | {k} | {metrics['hit']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['full_cover']:.4f} | {comp['avg_completion_selected']:.3f} | "
                f"{comp['avg_verified_completion_selected']:.3f} | {comp['questions_with_completion']} |"
            )
    lines.extend(["", "## Parameters", "", "```json", json_dumps(summary["params"]), "```", ""])
    return "\n".join(lines)


def event_id_for_path(path: dict, fact_id: str, index: TopologyFeatureIndex) -> str:
    metadata_event = str(path.get("metadata", {}).get("event_node_id") or "")
    return metadata_event or index.fact_event.get(fact_id, "")


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def is_completion(path: dict) -> bool:
    metadata = path.get("metadata", {})
    return str(metadata.get("is_nary_completion", "")).lower() == "true" or metadata.get("candidate_source") == "nary_completion"


def path_ce_score(path: dict) -> float:
    scores = path.get("scores", {})
    return _float(scores.get("cross_encoder", path.get("score", 0.0)))


def term_overlap(query_terms: set[str], text: str) -> float:
    terms = set(content_terms(text))
    if not query_terms or not terms:
        return 0.0
    return len(query_terms & terms) / max(len(query_terms), 1)


def minmax_value(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    low = min(values)
    high = max(values)
    if math.isclose(high, low):
        return 0.0
    return (value - low) / (high - low)


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


def load_cv_helpers():
    path = Path(__file__).resolve().parent / "30_run_loco_cv_selector.py"
    spec = importlib.util.spec_from_file_location("loco_cv_selector_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_graph_v2_selector_module():
    path = Path(__file__).resolve().parent / "41_run_graph_v2_selector_cv.py"
    spec = importlib.util.spec_from_file_location("graph_v2_selector_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
