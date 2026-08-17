from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict

from common import read_json, resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topology_features import project_feature_rows
from hytopomem.retrieval.topology_selector import train_lightgbm_ranker


BASE_24 = [
    "ce_score",
    "base_score",
    "bm25_norm",
    "ce_rank",
    "ce_reciprocal_rank",
    "query_term_overlap",
    "text_token_count",
    "query_token_count",
    "has_topdown_route",
    "has_bottom_up_route",
    "route_from_both",
    "is_eu_route",
    "is_hyp_route",
    "route_source_count",
    "route_overlap_score",
    "eu_hyp_agreement",
    "bottom_up_eu_agreement",
    "bottom_up_hyp_agreement",
    "route_consistency_entropy",
    "is_nary_completion",
    "nary_hyperedge_size",
    "nary_hyperedge_confidence",
    "nary_same_hyperedge_count_in_candidate_pool",
    "nary_role_coverage_potential",
]


RRF_5 = [
    "eu_rrf_score",
    "hyp_rrf_score",
    "bottom_up_rrf_score",
    "topdown_rrf_score",
    "route_rrf_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add route RRF features before the V3.9 24-feature LightGBM selector."
    )
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--candidates", default="outputs/v3_9_query_cards/qwen3_card_guided_expand120.json")
    parser.add_argument("--cardce-paths", default="outputs/eval/v3_9_cardce_guided_ctx50/cardce_guided_topk_paths.json")
    parser.add_argument("--output-dir", default="outputs/eval/cv/v3_9_24_feature_plus_rrf_card_guided_expand120")
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()

    cv = load_module("30_run_loco_cv_selector.py", "loco_cv_helpers_v3_9_rrf")
    quota = load_module("108_run_v3_9_24_feature_card_quota_cv.py", "v3_9_quota_helpers_rrf")
    graph = JsonGraphStore().load(resolve_path(args.graph))
    candidates = read_json(resolve_path(args.candidates))
    card_index = quota.build_cardce_index(read_json(resolve_path(args.cardce_paths)))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "base24": BASE_24,
        "base24_plus_rrf5": BASE_24 + RRF_5,
    }
    feature_cache = cv.build_feature_cache(graph, candidates)
    conversations = cv.ordered_conversations(candidates)
    if args.max_folds:
        conversations = conversations[: args.max_folds]
    aggregate: dict[str, list] = defaultdict(list)
    folds = []
    full_paths: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    importances: dict[str, list] = defaultdict(list)

    for fold_index, test_conversation in enumerate(conversations):
        train_items = [
            item
            for item in candidates
            if cv.conversation_id_from_question(item["question_id"]) != test_conversation
        ]
        test_items = [
            item
            for item in candidates
            if cv.conversation_id_from_question(item["question_id"]) == test_conversation
        ]
        train_rows_all, train_labels, train_groups, _ = cv.flatten_feature_cache(
            feature_cache,
            [item["question_id"] for item in train_items],
        )
        fold = {"fold": fold_index, "test_conversation": test_conversation, "methods": {}}
        fold_dir = output_dir / f"fold_{fold_index:02d}_{test_conversation}"
        for variant_name, feature_names in variants.items():
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
            importances[variant_name].append(
                {
                    "fold": fold_index,
                    "test_conversation": test_conversation,
                    "feature_importance": cv.feature_importance_for_names(model, feature_names),
                }
            )
            for topk in args.topk:
                lgbm_method = f"{variant_name}_lgbm_top{topk}"
                quota_method = f"{variant_name}_card_quota_light_top{topk}"
                lgbm_items = cv.rerank_items_with_scores(test_items, scores, topk)
                quota_items = quota.card_quota_items(
                    test_items,
                    scores,
                    card_index,
                    topk=topk,
                    card_quota=2 if topk <= 5 else 5,
                    method=quota_method,
                )
                for method, items in [(lgbm_method, lgbm_items), (quota_method, quota_items)]:
                    evaluation = cv.evaluate_items(graph, items, topk, method)
                    write_json(items, fold_dir / f"{method}_paths.json")
                    write_json(evaluation, fold_dir / f"{method}_eval.json")
                    fold["methods"][method] = evaluation["summary"]
                    aggregate[method].extend(evaluation["per_question"])
                    full_paths[method][topk].extend(items)
        write_json(fold, fold_dir / "fold_summary.json")
        folds.append(fold)
        print(f"finished fold={fold_index} test={test_conversation}", flush=True)

    summary = {
        "method": "V3.9 24-feature selector with optional route RRF features",
        "graph": str(resolve_path(args.graph)),
        "candidates": str(resolve_path(args.candidates)),
        "cardce_paths": str(resolve_path(args.cardce_paths)),
        "variants": variants,
        "aggregate": {method: cv.summarize_rows(rows) for method, rows in aggregate.items()},
        "rrf_features": RRF_5,
        "folds": folds,
        "mean_importance": mean_importance(importances),
    }
    write_json(summary, output_dir / "rrf_feature_card_quota_summary.json")
    write_json(importances, output_dir / "feature_importance_by_fold.json")
    (output_dir / "rrf_feature_card_quota_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    for method, by_topk in full_paths.items():
        for topk, items in by_topk.items():
            write_json(items, output_dir / f"full_{method}_paths.json")
    print(render_markdown(summary), flush=True)


def load_module(filename: str, module_name: str):
    path = resolve_path("scripts") / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mean_importance(importances: dict[str, list[dict]]) -> dict[str, list[dict]]:
    output = {}
    for method, rows in importances.items():
        totals: dict[str, float] = defaultdict(float)
        for row in rows:
            for item in row["feature_importance"]:
                totals[item["feature"]] += float(item["importance"])
        denom = max(len(rows), 1)
        output[method] = [
            {"feature": feature, "mean_importance": value / denom}
            for feature, value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
        ]
    return output


def render_markdown(summary: dict) -> str:
    lines = [
        "# V3.9 RRF Features Before LightGBM",
        "",
        f"Candidates: `{summary['candidates']}`",
        "",
        "## Features",
        "",
        f"- Base features: {len(summary['variants']['base24'])}",
        f"- RRF-added features: `{', '.join(summary['rrf_features'])}`",
        "",
        "## Aggregate",
        "",
        "| Method | N | Hit | Recall | FullCover | AvgTokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in sorted(summary["aggregate"].items()):
        lines.append(
            f"| {method} | {metrics['num_questions']} | {metrics['hit']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['full_cover']:.4f} | {metrics['avg_tokens']:.1f} |"
        )
    for method, rows in summary.get("mean_importance", {}).items():
        lines.extend(["", f"## Top Importances: {method}", "", "| Feature | Mean Importance |", "|---|---:|"])
        for row in rows[:12]:
            lines.append(f"| {row['feature']} | {row['mean_importance']:.2f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
