from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

from common import read_json, resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topology_features import project_feature_rows
from hytopomem.retrieval.topology_selector import train_lightgbm_ranker


CARD_FIXED = [
    "is_nary_completion",
    "nary_hyperedge_size",
    "nary_hyperedge_confidence",
    "nary_same_hyperedge_count_in_candidate_pool",
    "nary_role_coverage_potential",
]


RETRIEVAL_CORE = [
    "ce_score",
    "base_score",
    "bm25_norm",
    "ce_rank",
    "ce_reciprocal_rank",
    "query_term_overlap",
    "text_token_count",
    "query_token_count",
]


HIERARCHY_COMPACT = [
    "is_seed",
    "hop",
    "candidate_source_seed",
    "candidate_source_same_event",
    "candidate_source_same_topic",
    "has_event",
    "log_event_degree",
    "event_confidence",
    "event_coherence",
    "edge_confidence_to_event",
    "same_event_as_seed",
    "has_topic",
    "log_topic_degree",
    "topic_confidence",
    "topic_coherence",
    "event_topic_confidence",
    "same_topic_as_seed",
]


ROUTE_COMPACT = [
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
]


LOW_REDUNDANCY_DROP = {
    "is_hop0",
    "is_hop2",
    "event_degree",
    "topic_degree",
    "route_source_eu_event",
    "route_source_eu_topic",
    "route_source_hyp_event",
    "route_source_hyp_topic",
}


def main() -> None:
    candidates_path = resolve_path("outputs/v3_9_query_cards/qwen3_card_annotated_base100_paths_v3_clean.json")
    graph_path = resolve_path("outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    output_dir = resolve_path("outputs/eval/cv/v3_9_base_feature_selection_top5")
    graph_v2 = load_module("41_run_graph_v2_selector_cv.py", "graph_v2_selector_for_v3_9_select")
    cv = load_cv_helpers()
    base44 = graph_v2.dedupe(
        graph_v2.RETRIEVAL_FEATURES
        + graph_v2.GRAPH_V2_FEATURES
        + graph_v2.ROUTE_ORIGIN_V1_FEATURES
        + graph_v2.ROUTE_AGREEMENT_V1_FEATURES
    )
    card5 = list(CARD_FIXED)
    variants = {
        "base44_card5_current": graph_v2.dedupe(base44 + card5),
        "base36_drop_redundant_card5": graph_v2.dedupe([name for name in base44 if name not in LOW_REDUNDANCY_DROP] + card5),
        "base30_compact_card5": graph_v2.dedupe(RETRIEVAL_CORE + HIERARCHY_COMPACT + ROUTE_COMPACT + card5),
        "base25_no_source_onehot_card5": graph_v2.dedupe(
            RETRIEVAL_CORE
            + [
                "is_seed",
                "hop",
                "has_event",
                "log_event_degree",
                "event_confidence",
                "edge_confidence_to_event",
                "same_event_as_seed",
                "has_topic",
                "log_topic_degree",
                "topic_confidence",
                "event_topic_confidence",
                "same_topic_as_seed",
            ]
            + [
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
            ]
            + card5
        ),
        "base20_minimal_card5": graph_v2.dedupe(
            [
                "ce_score",
                "base_score",
                "ce_rank",
                "ce_reciprocal_rank",
                "query_term_overlap",
                "is_seed",
                "hop",
                "has_event",
                "log_event_degree",
                "event_confidence",
                "same_event_as_seed",
                "has_topic",
                "log_topic_degree",
                "topic_confidence",
                "same_topic_as_seed",
                "has_topdown_route",
                "route_from_both",
                "route_source_count",
                "route_overlap_score",
                "eu_hyp_agreement",
            ]
            + card5
        ),
        "retrieval_route_card5": graph_v2.dedupe(
            RETRIEVAL_CORE
            + [
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
            ]
            + card5
        ),
    }

    graph = JsonGraphStore().load(graph_path)
    candidates = read_json(candidates_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = cv.build_feature_cache(graph, candidates)
    conversations = cv.ordered_conversations(candidates)
    aggregate = defaultdict(list)
    importances = defaultdict(list)
    folds = []
    for fold_index, test_conversation in enumerate(conversations):
        train_items = [item for item in candidates if cv.conversation_id_from_question(item["question_id"]) != test_conversation]
        test_items = [item for item in candidates if cv.conversation_id_from_question(item["question_id"]) == test_conversation]
        train_rows_all, train_labels, train_groups, _ = cv.flatten_feature_cache(
            feature_cache,
            [item["question_id"] for item in train_items],
        )
        fold = {
            "fold": fold_index,
            "test_conversation": test_conversation,
            "methods": {},
        }
        for method, feature_names in variants.items():
            model = train_lightgbm_ranker(
                project_feature_rows(train_rows_all, feature_names),
                train_labels,
                train_groups,
                n_estimators=80,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                n_jobs=8,
                random_state=13 + fold_index,
            )
            scores = cv.score_items_with_cache(test_items, model, feature_cache, feature_names)
            reranked = cv.rerank_items_with_scores(test_items, scores, 5)
            evaluation = cv.evaluate_items(graph, reranked, 5, method)
            fold["methods"][method] = evaluation["summary"]
            aggregate[method].extend(evaluation["per_question"])
            importances[method].append(
                {
                    "fold": fold_index,
                    "feature_importance": cv.feature_importance_for_names(model, feature_names),
                }
            )
            fold_dir = output_dir / f"fold_{fold_index:02d}_{test_conversation}"
            write_json(reranked, fold_dir / f"{method}_paths.json")
            write_json(evaluation, fold_dir / f"{method}_eval.json")
        write_json(fold, output_dir / f"fold_{fold_index:02d}_{test_conversation}" / "fold_summary.json")
        folds.append(fold)
        print(f"feature selection fold={fold_index} test={test_conversation}", flush=True)

    summary = {
        "method": "V3.9 base-feature selection with fixed card5",
        "candidate_file": str(candidates_path),
        "graph": str(graph_path),
        "fixed_card_features": card5,
        "variants": variants,
        "aggregate": {method: cv.summarize_rows(rows) for method, rows in aggregate.items()},
        "folds": folds,
        "mean_importance": mean_importance(importances),
    }
    write_json(summary, output_dir / "base_feature_selection_summary.json")
    write_json(importances, output_dir / "feature_importance_by_fold.json")
    (output_dir / "BASE_FEATURE_SELECTION_SUMMARY.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


def mean_importance(importances: dict) -> dict[str, list[dict]]:
    out = {}
    for method, fold_rows in importances.items():
        acc = defaultdict(list)
        for fold in fold_rows:
            for row in fold["feature_importance"]:
                acc[row["feature"]].append(float(row["importance"]))
        out[method] = [
            {"feature": feature, "mean_importance": sum(values) / len(values)}
            for feature, values in sorted(acc.items(), key=lambda item: -sum(item[1]) / len(item[1]))
        ]
    return out


def render_markdown(summary: dict) -> str:
    lines = [
        "# V3.9 Base Feature Selection with Fixed Card5",
        "",
        "Card features are fixed:",
        "",
        "```text",
        *summary["fixed_card_features"],
        "```",
        "",
        "| Method | Features | Hit@5 | Recall@5 | FullCover@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, row in summary["aggregate"].items():
        lines.append(
            f"| {method} | {len(summary['variants'][method])} | "
            f"{row['hit']:.4f} | {row['recall']:.4f} | {row['full_cover']:.4f} |"
        )
    best = max(summary["aggregate"], key=lambda name: summary["aggregate"][name]["hit"])
    lines.extend(
        [
            "",
            f"Best by Hit@5: `{best}`",
            "",
            "## Top Feature Importance",
            "",
        ]
    )
    for method in summary["aggregate"]:
        lines.append(f"### {method}")
        lines.append("")
        for row in summary["mean_importance"][method][:20]:
            lines.append(f"- `{row['feature']}`: {row['mean_importance']:.2f}")
        lines.append("")
    return "\n".join(lines)


def load_module(filename: str, name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cv_helpers():
    return load_module("30_run_loco_cv_selector.py", "loco_cv_helpers_v3_9_base_select")


if __name__ == "__main__":
    main()
