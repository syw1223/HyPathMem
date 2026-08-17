from __future__ import annotations

import argparse
import importlib.util
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import conversation_id_from_question
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topology_features import project_feature_rows, select_feature_names
from hytopomem.retrieval.topology_selector import train_lightgbm_ranker


RETRIEVAL_FEATURES = [
    "ce_score",
    "base_score",
    "bm25_norm",
    "ce_rank",
    "ce_reciprocal_rank",
    "query_term_overlap",
    "text_token_count",
    "query_token_count",
]

GRAPH_V2_FEATURES = [
    "is_seed",
    "hop",
    "is_hop0",
    "is_hop2",
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
]

ENTITY_SESSION_FEATURES = [
    "same_session_as_seed",
    "same_speaker_as_seed",
    "same_source_as_seed",
    "support_overlap_with_seed",
    "same_conversation",
    "support_day_gap_from_question",
]

TOPDOWN_ROUTE_FEATURES = [
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
]

ROUTE_ORIGIN_V1_FEATURES = [
    "has_topdown_route",
    "has_bottom_up_route",
    "route_from_both",
    "route_source_eu_event",
    "route_source_eu_topic",
    "route_source_hyp_event",
    "route_source_hyp_topic",
    "is_eu_route",
    "is_hyp_route",
]

ROUTE_QUALITY_V1_FEATURES = [
    "semantic_score",
    "eu_event_score",
    "eu_topic_score",
    "hyperbolic_score",
    "hyp_event_score",
    "hyp_topic_score",
    "route_rank",
    "route_reciprocal_rank",
    "eu_event_rank",
    "eu_event_reciprocal_rank",
    "eu_topic_rank",
    "eu_topic_reciprocal_rank",
    "hyp_event_rank",
    "hyp_event_reciprocal_rank",
    "hyp_topic_rank",
    "hyp_topic_reciprocal_rank",
    "bottom_up_rank",
    "bottom_up_reciprocal_rank",
    "route_min_rank",
    "route_best_reciprocal_rank",
    "fact_offset",
]

ROUTE_AGREEMENT_V1_FEATURES = [
    "route_source_count",
    "route_overlap_score",
    "eu_hyp_agreement",
    "bottom_up_eu_agreement",
    "bottom_up_hyp_agreement",
    "route_consistency_entropy",
]

ROUTE_AGREEMENT_V2_FEATURES = [
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
]

EPISODE_STRUCT_FEATURES = [
    "has_episode",
    "episode_size_events",
    "log_episode_size_events",
    "episode_num_facts",
    "log_episode_num_facts",
    "episode_coherence",
    "is_singleton_episode",
    "event_episode_confidence",
]

EPISODE_ROUTE_FEATURES = [
    "episode_seen_by_bottom_up",
    "episode_seen_by_top_down",
    "episode_seen_by_eu",
    "episode_seen_by_hyp",
    "episode_bottom_top_agreement",
    "episode_eu_hyp_agreement",
]

EPISODE_POOL_FEATURES = [
    "episode_candidate_count_in_pool",
    "episode_best_ce_score",
    "candidate_ce_minus_episode_best",
]

EPISODE_LITE_V1_FEATURES = [
    "has_episode",
    "log_episode_size_events",
    "log_episode_num_facts",
    "episode_coherence",
    "is_singleton_episode",
    "event_episode_confidence",
    "episode_seen_by_bottom_up",
    "episode_seen_by_top_down",
    "episode_seen_by_eu",
    "episode_seen_by_hyp",
    "episode_bottom_top_agreement",
    "episode_eu_hyp_agreement",
    "episode_candidate_count_in_pool",
    "episode_best_ce_score",
    "candidate_ce_minus_episode_best",
]

TEMPORAL_ROUTE_V1_FEATURES = [
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
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_hierarchy_v2.json")
    parser.add_argument("--candidates", default="outputs/paths/full_graph_v2_event_topic_ce_top100.json")
    parser.add_argument("--bm25-paths", default="outputs/paths/full_bm25_fact50_ce_top5.json")
    parser.add_argument("--graph-v2-ce-paths", default="outputs/paths/full_graph_v2_event_topic_ce_top5.json")
    parser.add_argument("--output-dir", default="outputs/eval/cv/graph_v2_selector_lgbm_n80_top100")
    parser.add_argument("--candidate-topn", type=int, default=100)
    parser.add_argument("--final-topk", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-conversations", type=int, default=0, help="Debug only: keep first N conversations.")
    parser.add_argument("--max-folds", type=int, default=0, help="Debug only: run first N folds.")
    args = parser.parse_args()

    cv = load_cv_helpers()
    started = time.perf_counter()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph or config["graph"]["graph_path"]))
    candidates = read_json(resolve_path(args.candidates))
    if args.candidate_topn:
        candidates = cv.truncate_candidates(candidates, args.candidate_topn)
    bm25_items = {item["question_id"]: item for item in read_json(resolve_path(args.bm25_paths))}
    graph_v2_ce_items = {item["question_id"]: item for item in read_json(resolve_path(args.graph_v2_ce_paths))}
    output_dir = resolve_path(args.output_dir)

    conversation_ids = cv.ordered_conversations(candidates)
    if args.max_conversations:
        conversation_ids = conversation_ids[: args.max_conversations]
        keep = set(conversation_ids)
        candidates = [item for item in candidates if conversation_id_from_question(item["question_id"]) in keep]
    if args.max_folds:
        conversation_ids = conversation_ids[: args.max_folds]
    variants = selector_variants()
    print(
        f"loaded graph v2 selector CV: conversations={len(cv.ordered_conversations(candidates))} "
        f"folds={len(conversation_ids)} questions={len(candidates)} "
        f"candidate_topn={args.candidate_topn or 'all'} variants={len(variants)} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    feature_cache = cv.build_feature_cache(graph, candidates)
    print(f"built feature cache: elapsed={time.perf_counter() - started:.1f}s", flush=True)

    fold_payloads = []
    aggregate_results: dict[str, list] = defaultdict(list)
    aggregate_by_question: dict[str, dict[str, bool]] = defaultdict(dict)
    importances: dict[str, list[dict]] = defaultdict(list)

    for fold_index, test_conversation in enumerate(conversation_ids):
        fold_started = time.perf_counter()
        fold_dir = output_dir / f"fold_{fold_index:02d}_{test_conversation}"
        train_items = [
            item for item in candidates if conversation_id_from_question(item["question_id"]) != test_conversation
        ]
        test_items = [
            item for item in candidates if conversation_id_from_question(item["question_id"]) == test_conversation
        ]
        test_qids = {item["question_id"] for item in test_items}
        train_rows_all, train_labels, train_groups, train_question_ids = cv.flatten_feature_cache(
            feature_cache,
            [item["question_id"] for item in train_items],
        )
        fold = {
            "fold": fold_index,
            "test_conversation": test_conversation,
            "num_train_questions": len(train_question_ids),
            "num_test_questions": len(test_items),
            "num_train_examples": len(train_rows_all),
            "num_train_positive": int(sum(train_labels)),
            "methods": {},
        }

        for method_name, item_map in [
            ("bm25_fact50_ce", bm25_items),
            ("graph_v2_event_topic_ce", graph_v2_ce_items),
        ]:
            items = [item_map[qid] for qid in sorted(test_qids) if qid in item_map]
            payload = cv.evaluate_items(graph, items, args.final_topk, method_name)
            write_json(payload, fold_dir / f"{method_name}_eval.json")
            fold["methods"][method_name] = payload["summary"]
            aggregate_results[method_name].extend(payload["per_question"])
            for result in payload["per_question"]:
                aggregate_by_question[result["question_id"]][method_name] = bool(result["hit"])

        for variant_name, feature_names in variants.items():
            variant_started = time.perf_counter()
            print(
                f"training fold={fold_index} variant={variant_name} "
                f"features={len(feature_names)} train_examples={len(train_rows_all)}",
                flush=True,
            )
            train_rows = project_feature_rows(train_rows_all, feature_names)
            model = train_lightgbm_ranker(
                train_rows,
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
            reranked = cv.rerank_items_with_scores(test_items, scores, args.final_topk)
            for item in reranked:
                metadata = dict(item.get("metadata", {}))
                metadata["method"] = variant_name
                metadata["feature_count"] = len(feature_names)
                metadata["test_conversation"] = test_conversation
                item["metadata"] = metadata
            paths_output = fold_dir / f"{variant_name}_paths.json"
            eval_output = fold_dir / f"{variant_name}_eval.json"
            write_json(reranked, paths_output)
            payload = cv.evaluate_items(graph, reranked, args.final_topk, variant_name)
            payload["metadata"] = {
                "feature_names": feature_names,
                "feature_count": len(feature_names),
                "paths_output": str(paths_output),
            }
            write_json(payload, eval_output)
            fold["methods"][variant_name] = payload["summary"]
            aggregate_results[variant_name].extend(payload["per_question"])
            for result in payload["per_question"]:
                aggregate_by_question[result["question_id"]][variant_name] = bool(result["hit"])
            importances[variant_name].append(
                {
                    "fold": fold_index,
                    "test_conversation": test_conversation,
                    "feature_importance": cv.feature_importance_for_names(model, feature_names),
                }
            )
            print(
                f"finished fold={fold_index} variant={variant_name} "
                f"elapsed={time.perf_counter() - variant_started:.1f}s",
                flush=True,
            )

        write_json(fold, fold_dir / "fold_summary.json")
        fold_payloads.append(fold)
        cv.print_fold(fold)
        print(f"finished fold={fold_index} elapsed={time.perf_counter() - fold_started:.1f}s", flush=True)

    summary = build_summary(
        cv=cv,
        folds=fold_payloads,
        aggregate_results=aggregate_results,
        aggregate_by_question=aggregate_by_question,
        importances=importances,
        variants=variants,
        output_dir=output_dir,
        args=args,
    )
    write_json(summary, output_dir / "graph_v2_selector_summary.json")
    write_json(importances, output_dir / "feature_importance_by_fold.json")
    write_markdown_summary(summary, output_dir / "graph_v2_selector_summary.md")
    print("\nGraph v2 selector aggregate:")
    for method_name, metrics in summary["aggregate"].items():
        print(
            f"{method_name}: questions={metrics['num_questions']} "
            f"hit={metrics['hit']:.4f} recall={metrics['recall']:.4f} "
            f"full_cover={metrics['full_cover']:.4f}"
        )
    print(f"wrote {output_dir / 'graph_v2_selector_summary.json'}")
    print(f"wrote {output_dir / 'graph_v2_selector_summary.md'}")


def selector_variants() -> dict[str, list[str]]:
    retrieval = dedupe(RETRIEVAL_FEATURES)
    graph_v2 = dedupe(retrieval + GRAPH_V2_FEATURES)
    graph_v2_entity = dedupe(graph_v2 + ENTITY_SESSION_FEATURES)
    graph_v2_topdown = dedupe(graph_v2 + TOPDOWN_ROUTE_FEATURES)
    graph_v2_entity_topdown = dedupe(graph_v2_entity + TOPDOWN_ROUTE_FEATURES)
    route_origin = dedupe(graph_v2 + ROUTE_ORIGIN_V1_FEATURES)
    route_quality = dedupe(graph_v2 + ROUTE_ORIGIN_V1_FEATURES + ROUTE_QUALITY_V1_FEATURES)
    route_agreement = dedupe(graph_v2 + ROUTE_ORIGIN_V1_FEATURES + ROUTE_AGREEMENT_V1_FEATURES)
    route_agreement_v2 = dedupe(
        graph_v2
        + ROUTE_ORIGIN_V1_FEATURES
        + ROUTE_AGREEMENT_V1_FEATURES
        + ROUTE_AGREEMENT_V2_FEATURES
    )
    route_aware = dedupe(
        graph_v2
        + ROUTE_ORIGIN_V1_FEATURES
        + ROUTE_QUALITY_V1_FEATURES
        + ROUTE_AGREEMENT_V1_FEATURES
    )
    route_aware_v2 = dedupe(route_aware + ROUTE_AGREEMENT_V2_FEATURES)
    route_aware_entity = dedupe(route_aware + ENTITY_SESSION_FEATURES)
    route_aware_entity_v2 = dedupe(route_aware_v2 + ENTITY_SESSION_FEATURES)
    episode_base = route_agreement
    episode_struct = dedupe(episode_base + EPISODE_STRUCT_FEATURES)
    episode_route = dedupe(episode_base + EPISODE_ROUTE_FEATURES)
    episode_pool = dedupe(episode_base + EPISODE_POOL_FEATURES)
    episode_lite = dedupe(episode_base + EPISODE_LITE_V1_FEATURES)
    temporal_route = dedupe(route_agreement_v2 + TEMPORAL_ROUTE_V1_FEATURES)
    route_aware_entity_temporal = dedupe(route_aware_entity_v2 + TEMPORAL_ROUTE_V1_FEATURES)
    return {
        "ce_score_only": ["ce_score"],
        "retrieval_only": retrieval,
        "retrieval_plus_graph_v2": graph_v2,
        "retrieval_graph_v2_entity_session": graph_v2_entity,
        "retrieval_graph_v2_topdown_route": graph_v2_topdown,
        "retrieval_graph_v2_entity_session_topdown_route": graph_v2_entity_topdown,
        "route_origin_v1": route_origin,
        "route_quality_v1": route_quality,
        "route_agreement_v1": route_agreement,
        "route_agreement_v2": route_agreement_v2,
        "route_aware_v1": route_aware,
        "route_aware_v2": route_aware_v2,
        "route_aware_entity_v1": route_aware_entity,
        "route_aware_entity_v2": route_aware_entity_v2,
        "episode_struct_v1": episode_struct,
        "episode_route_v1": episode_route,
        "episode_pool_v1": episode_pool,
        "episode_lite_v1": episode_lite,
        "temporal_route_v1": temporal_route,
        "route_aware_entity_temporal_v1": route_aware_entity_temporal,
        "all_without_category": select_feature_names(exclude=["question_category"]),
    }


def build_summary(
    *,
    cv,
    folds: list[dict],
    aggregate_results: dict[str, list],
    aggregate_by_question: dict[str, dict[str, bool]],
    importances: dict[str, list[dict]],
    variants: dict[str, list[str]],
    output_dir: Path,
    args,
) -> dict:
    aggregate = {}
    fold_stats = {}
    for method_name, rows in aggregate_results.items():
        aggregate[method_name] = cv.summarize_rows(rows)
        per_fold = [fold["methods"][method_name] for fold in folds]
        fold_stats[method_name] = {
            "mean_hit": mean(metric["hit"] for metric in per_fold),
            "std_hit": pstdev(metric["hit"] for metric in per_fold),
            "mean_recall": mean(metric["recall"] for metric in per_fold),
            "std_recall": pstdev(metric["recall"] for metric in per_fold),
            "mean_full_cover": mean(metric["full_cover"] for metric in per_fold),
            "std_full_cover": pstdev(metric["full_cover"] for metric in per_fold),
        }
    paired = {}
    for variant_name in variants:
        paired[f"{variant_name}_vs_bm25_fact50_ce"] = cv.paired_compare(
            aggregate_by_question, variant_name, "bm25_fact50_ce"
        )
        paired[f"{variant_name}_vs_graph_v2_ce"] = cv.paired_compare(
            aggregate_by_question, variant_name, "graph_v2_event_topic_ce"
        )
    paired["graph_v2_features_vs_retrieval_only"] = cv.paired_compare(
        aggregate_by_question, "retrieval_plus_graph_v2", "retrieval_only"
    )
    paired["all_without_category_vs_graph_v2_features"] = cv.paired_compare(
        aggregate_by_question, "all_without_category", "retrieval_plus_graph_v2"
    )
    paired["topdown_route_vs_graph_v2_features"] = cv.paired_compare(
        aggregate_by_question, "retrieval_graph_v2_topdown_route", "retrieval_plus_graph_v2"
    )
    paired["entity_session_topdown_route_vs_entity_session"] = cv.paired_compare(
        aggregate_by_question,
        "retrieval_graph_v2_entity_session_topdown_route",
        "retrieval_graph_v2_entity_session",
    )
    paired["route_origin_v1_vs_graph_v2_features"] = cv.paired_compare(
        aggregate_by_question, "route_origin_v1", "retrieval_plus_graph_v2"
    )
    paired["route_quality_v1_vs_route_origin_v1"] = cv.paired_compare(
        aggregate_by_question, "route_quality_v1", "route_origin_v1"
    )
    paired["route_agreement_v1_vs_route_origin_v1"] = cv.paired_compare(
        aggregate_by_question, "route_agreement_v1", "route_origin_v1"
    )
    paired["route_agreement_v2_vs_route_agreement_v1"] = cv.paired_compare(
        aggregate_by_question, "route_agreement_v2", "route_agreement_v1"
    )
    paired["route_aware_v1_vs_graph_v2_features"] = cv.paired_compare(
        aggregate_by_question, "route_aware_v1", "retrieval_plus_graph_v2"
    )
    paired["route_aware_v2_vs_route_aware_v1"] = cv.paired_compare(
        aggregate_by_question, "route_aware_v2", "route_aware_v1"
    )
    paired["route_aware_entity_v1_vs_route_aware_v1"] = cv.paired_compare(
        aggregate_by_question, "route_aware_entity_v1", "route_aware_v1"
    )
    paired["route_aware_entity_v2_vs_route_aware_entity_v1"] = cv.paired_compare(
        aggregate_by_question, "route_aware_entity_v2", "route_aware_entity_v1"
    )
    return {
        "method": "LOCO-CV Graph v2 Selector LightGBM",
        "folds": len(folds),
        "k": args.final_topk,
        "candidate_file": str(resolve_path(args.candidates)),
        "candidate_topn": args.candidate_topn or None,
        "output_dir": str(output_dir),
        "lgbm": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_child_samples": args.min_child_samples,
            "n_jobs": args.n_jobs,
            "random_state": args.random_state,
        },
        "feature_sets": variants,
        "aggregate": aggregate,
        "fold_mean_std": fold_stats,
        "paired_compare": paired,
        "folds_detail": folds,
        "importance_top10_mean": {
            name: cv.average_importance_topk(items, top_k=10) for name, items in importances.items()
        },
    }


def write_markdown_summary(summary: dict, path: Path) -> None:
    lines = [
        "# Graph v2 Selector CV",
        "",
        f"Folds: {summary['folds']}",
        f"K: {summary['k']}",
        f"Candidate TopN: {summary['candidate_topn']}",
        "",
        "## Aggregate",
        "",
        "| Method | Features | Questions | Hit@5 | Recall@5 | FullCover@5 | AvgTokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method_name, metrics in summary["aggregate"].items():
        feature_count = len(summary["feature_sets"].get(method_name, []))
        lines.append(
            f"| {method_name} | {feature_count} | {metrics['num_questions']} | "
            f"{metrics['hit']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['full_cover']:.4f} | {metrics['avg_tokens']:.1f} |"
        )
    lines.extend(["", "## Paired Compare", ""])
    lines.append("| Compare | Compared | LeftOnly | RightOnly | Net |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, metrics in summary["paired_compare"].items():
        lines.append(
            f"| {name} | {metrics['compared']} | {metrics['left_only']} | "
            f"{metrics['right_only']} | {metrics['net_left_minus_right']} |"
        )
    lines.extend(["", "## Top Importance", ""])
    for method_name, items in summary["importance_top10_mean"].items():
        lines.append(f"### {method_name}")
        for item in items:
            lines.append(f"- {item['feature']}: {item['mean_importance']:.1f}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cv_helpers():
    path = Path(__file__).with_name("30_run_loco_cv_selector.py")
    spec = importlib.util.spec_from_file_location("loco_cv_selector_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load CV helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dedupe(features: list[str]) -> list[str]:
    selected = []
    for feature in features:
        if feature not in selected:
            selected.append(feature)
    return selected


if __name__ == "__main__":
    main()
