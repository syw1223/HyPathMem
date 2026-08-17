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
from hytopomem.retrieval.topology_features import project_feature_rows
from hytopomem.retrieval.topology_selector import train_lightgbm_ranker


NARY_POINT_FEATURES = [
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
]

NARY_SET_FEATURES = [
    "nary_pool_covered_roles_count",
    "nary_pool_required_roles_covered",
    "nary_pool_has_preference_and_constraint",
    "nary_pool_has_old_and_new_state",
    "nary_pool_has_reason",
    "nary_pool_has_time_scope",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-topn", type=int, default=0)
    parser.add_argument("--final-topk", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument(
        "--variant-profile",
        choices=["standard", "card_ablation", "card_additive"],
        default="standard",
    )
    args = parser.parse_args()

    cv = load_cv_helpers()
    graph_v2 = load_graph_v2_selector_module()
    variants = selector_variants(graph_v2, args.variant_profile)
    started = time.perf_counter()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph or config["graph"]["graph_path"]))
    candidates = read_json(resolve_path(args.candidates))
    if args.candidate_topn:
        candidates = cv.truncate_candidates(candidates, args.candidate_topn)
    output_dir = resolve_path(args.output_dir)

    conversation_ids = cv.ordered_conversations(candidates)
    if args.max_conversations:
        conversation_ids = conversation_ids[: args.max_conversations]
        keep = set(conversation_ids)
        candidates = [item for item in candidates if conversation_id_from_question(item["question_id"]) in keep]
    if args.max_folds:
        conversation_ids = conversation_ids[: args.max_folds]
    print(
        f"loaded nary selector CV: conversations={len(cv.ordered_conversations(candidates))} "
        f"folds={len(conversation_ids)} questions={len(candidates)} variants={len(variants)} "
        f"final_topk={args.final_topk} candidate_topn={args.candidate_topn or 'all'}",
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

        ce_payload = cv.evaluate_items(graph, test_items, args.final_topk, "candidate_ce_input_order")
        write_json(ce_payload, fold_dir / "candidate_ce_input_order_eval.json")
        fold["methods"]["candidate_ce_input_order"] = ce_payload["summary"]
        aggregate_results["candidate_ce_input_order"].extend(ce_payload["per_question"])
        for result in ce_payload["per_question"]:
            aggregate_by_question[result["question_id"]]["candidate_ce_input_order"] = bool(result["hit"])

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
    write_json(summary, output_dir / "nary_completion_selector_summary.json")
    write_json(importances, output_dir / "feature_importance_by_fold.json")
    write_markdown_summary(summary, output_dir / "nary_completion_selector_summary.md")
    print("\nN-ary completion selector aggregate:")
    for method_name, metrics in summary["aggregate"].items():
        print(
            f"{method_name}: questions={metrics['num_questions']} "
            f"hit={metrics['hit']:.4f} recall={metrics['recall']:.4f} "
            f"full_cover={metrics['full_cover']:.4f}"
        )
    print(f"wrote {output_dir / 'nary_completion_selector_summary.json'}")
    print(f"wrote {output_dir / 'nary_completion_selector_summary.md'}")


def selector_variants(graph_v2, profile: str = "standard") -> dict[str, list[str]]:
    retrieval = graph_v2.dedupe(graph_v2.RETRIEVAL_FEATURES)
    graph_features = graph_v2.dedupe(retrieval + graph_v2.GRAPH_V2_FEATURES)
    route_agreement_v1 = graph_v2.dedupe(
        graph_features + graph_v2.ROUTE_ORIGIN_V1_FEATURES + graph_v2.ROUTE_AGREEMENT_V1_FEATURES
    )
    nary_point = graph_v2.dedupe(route_agreement_v1 + NARY_POINT_FEATURES)
    nary_point_set = graph_v2.dedupe(route_agreement_v1 + NARY_POINT_FEATURES + NARY_SET_FEATURES)
    standard = {
        "base_completion_no_nary_features": route_agreement_v1,
        "base_completion_nary_point_features": nary_point,
        "base_completion_nary_point_set_features": nary_point_set,
    }
    if profile == "standard":
        return standard

    type_features = [
        "is_nary_completion",
        "nary_type_change",
        "nary_type_preference",
        "nary_type_state",
        "nary_type_plan_constraint",
    ]
    role_features = [name for name in NARY_POINT_FEATURES if name.startswith("nary_role_")]
    quality_features = [
        "nary_hyperedge_size",
        "nary_hyperedge_confidence",
        "nary_same_hyperedge_count_in_candidate_pool",
        "nary_role_coverage_potential",
    ]
    rank_features = ["nary_completion_rank", "nary_completion_reciprocal_rank"]
    extractor_features = ["nary_extractor_qwen", "nary_extractor_gpt4o"]
    full = graph_v2.dedupe(route_agreement_v1 + NARY_POINT_FEATURES + NARY_SET_FEATURES)

    def without(names: list[str]) -> list[str]:
        excluded = set(names)
        return [name for name in full if name not in excluded]

    if profile == "card_additive":
        membership = ["is_nary_completion"]
        return {
            "card_additive_base": route_agreement_v1,
            "card_additive_membership": graph_v2.dedupe(route_agreement_v1 + membership),
            "card_additive_type": graph_v2.dedupe(route_agreement_v1 + type_features),
            "card_additive_role": graph_v2.dedupe(route_agreement_v1 + membership + role_features),
            "card_additive_quality": graph_v2.dedupe(route_agreement_v1 + membership + quality_features),
            "card_additive_rank": graph_v2.dedupe(route_agreement_v1 + membership + rank_features),
            "card_additive_set": graph_v2.dedupe(route_agreement_v1 + membership + NARY_SET_FEATURES),
        }

    return {
        "card_ablation_base": route_agreement_v1,
        "card_ablation_full": full,
        "card_ablation_without_type": without(type_features),
        "card_ablation_without_role": without(role_features),
        "card_ablation_without_quality": without(quality_features),
        "card_ablation_without_rank": without(rank_features),
        "card_ablation_without_set": without(NARY_SET_FEATURES),
        "card_ablation_without_extractor": without(extractor_features),
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
        paired[f"{variant_name}_vs_candidate_ce"] = cv.paired_compare(
            aggregate_by_question, variant_name, "candidate_ce_input_order"
        )
        baseline = next(
            name
            for name in (
                "base_completion_no_nary_features",
                "card_ablation_base",
                "card_additive_base",
            )
            if name in variants
        )
        paired[f"{variant_name}_vs_baseline"] = cv.paired_compare(
            aggregate_by_question, variant_name, baseline
        )
    if {
        "base_completion_nary_point_set_features",
        "base_completion_nary_point_features",
    }.issubset(variants):
        paired["point_set_vs_point"] = cv.paired_compare(
            aggregate_by_question,
            "base_completion_nary_point_set_features",
            "base_completion_nary_point_features",
        )
    return {
        "method": "V3.6C n-ary completion selector CV",
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
    k = summary["k"]
    lines = [
        "# V3.6C N-ary Completion Selector CV",
        "",
        f"Folds: {summary['folds']}",
        f"K: {k}",
        f"Candidates: `{summary['candidate_file']}`",
        "",
        "## Aggregate",
        "",
        f"| Method | Questions | Hit@{k} | Recall@{k} | FullCover@{k} | AvgTokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method_name, metrics in summary["aggregate"].items():
        lines.append(
            f"| {method_name} | {metrics['num_questions']} | {metrics['hit']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['full_cover']:.4f} | {metrics['avg_tokens']:.1f} |"
        )
    lines.extend(["", "## Paired Compare", ""])
    lines.append("| Compare | Compared | LeftOnly | RightOnly | Net |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, metrics in summary["paired_compare"].items():
        lines.append(
            f"| {name} | {metrics['compared']} | {metrics['left_only']} | "
            f"{metrics['right_only']} | {metrics['net_left_minus_right']} |"
        )
    lines.extend(["", "## Importance Top10", ""])
    for method_name, items in summary["importance_top10_mean"].items():
        lines.append(f"### {method_name}")
        lines.append("")
        lines.append("| Feature | Mean Importance |")
        lines.append("| --- | ---: |")
        for item in items:
            lines.append(f"| {item['feature']} | {item['mean_importance']:.2f} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
