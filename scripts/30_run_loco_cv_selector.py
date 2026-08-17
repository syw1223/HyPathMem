from __future__ import annotations

import argparse
import time
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import conversation_id_from_question, evaluate_item, summarize
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topology_features import (
    FEATURE_NAMES,
    TopologyFeatureIndex,
    build_example,
    feature_indices,
    project_feature_rows,
    rerank_items_with_scores,
    select_feature_names,
    with_cached_query_terms,
)
from hytopomem.retrieval.topology_selector import train_lightgbm_ranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--candidates", default="outputs/paths/full_filtered_sibling_top100_deg20_ce_top100.json")
    parser.add_argument("--bm25-paths", default="outputs/paths/full_bm25_fact50_ce_top5.json")
    parser.add_argument("--graph-ce-paths", default="outputs/paths/full_filtered_sibling_top100_deg20_ce_top5.json")
    parser.add_argument("--output-dir", default="outputs/eval/cv/loco_selector_lgbm")
    parser.add_argument("--candidate-topn", type=int, default=0, help="Use only the first N candidates per query.")
    parser.add_argument("--final-topk", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument(
        "--selector-variants",
        choices=["both", "with_category", "without_category"],
        default="both",
    )
    parser.add_argument("--max-conversations", type=int, default=0, help="Debug only: keep the first N conversations.")
    parser.add_argument("--max-folds", type=int, default=0, help="Debug only: run the first N folds.")
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph or config["graph"]["graph_path"]))
    candidates = read_json(resolve_path(args.candidates))
    if args.candidate_topn:
        candidates = truncate_candidates(candidates, args.candidate_topn)
    bm25_items = {item["question_id"]: item for item in read_json(resolve_path(args.bm25_paths))}
    graph_ce_items = {item["question_id"]: item for item in read_json(resolve_path(args.graph_ce_paths))}
    output_dir = resolve_path(args.output_dir)

    conversation_ids = ordered_conversations(candidates)
    if args.max_conversations:
        conversation_ids = conversation_ids[: args.max_conversations]
        keep = set(conversation_ids)
        candidates = [item for item in candidates if conversation_id_from_question(item["question_id"]) in keep]
    if args.max_folds:
        conversation_ids = conversation_ids[: args.max_folds]
    variants = selector_variants(args.selector_variants)
    print(
        f"loaded graph and candidates: conversations={len(ordered_conversations(candidates))} "
        f"folds={len(conversation_ids)} questions={len(candidates)} "
        f"candidate_topn={args.candidate_topn or 'all'} elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    feature_cache = build_feature_cache(graph, candidates)
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
        train_rows_all, train_labels, train_groups, train_question_ids = flatten_feature_cache(
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
            ("graph_ce", graph_ce_items),
        ]:
            items = [item_map[qid] for qid in sorted(test_qids) if qid in item_map]
            payload = evaluate_items(graph, items, args.final_topk, method_name)
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
            scores = score_items_with_cache(test_items, model, feature_cache, feature_names)
            reranked = rerank_items_with_scores(test_items, scores, args.final_topk)
            for item in reranked:
                metadata = dict(item.get("metadata", {}))
                metadata["method"] = variant_name
                metadata["test_conversation"] = test_conversation
                metadata["feature_count"] = len(feature_names)
                item["metadata"] = metadata
            paths_output = fold_dir / f"{variant_name}_paths.json"
            eval_output = fold_dir / f"{variant_name}_eval.json"
            write_json(reranked, paths_output)
            payload = evaluate_items(graph, reranked, args.final_topk, variant_name)
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
                    "feature_importance": feature_importance_for_names(model, feature_names),
                }
            )
            print(
                f"finished fold={fold_index} variant={variant_name} "
                f"elapsed={time.perf_counter() - variant_started:.1f}s",
                flush=True,
            )

        write_json(fold, fold_dir / "fold_summary.json")
        fold_payloads.append(fold)
        print_fold(fold)
        print(f"finished fold={fold_index} elapsed={time.perf_counter() - fold_started:.1f}s", flush=True)

    summary = build_cv_summary(
        folds=fold_payloads,
        aggregate_results=aggregate_results,
        aggregate_by_question=aggregate_by_question,
        importances=importances,
        output_dir=output_dir,
        args=args,
    )
    write_json(summary, output_dir / "cv_summary.json")
    write_json(importances, output_dir / "feature_importance_by_fold.json")
    write_markdown_summary(summary, output_dir / "cv_summary.md")
    print("\nLOCO-CV aggregate:")
    for method_name, metrics in summary["aggregate"].items():
        print(
            f"{method_name}: questions={metrics['num_questions']} "
            f"hit={metrics['hit']:.4f} recall={metrics['recall']:.4f} "
            f"full_cover={metrics['full_cover']:.4f}"
        )
    print(f"wrote {output_dir / 'cv_summary.json'}")
    print(f"wrote {output_dir / 'cv_summary.md'}")


def ordered_conversations(items: list[dict]) -> list[str]:
    ordered = []
    seen = set()
    for item in items:
        conversation_id = conversation_id_from_question(item["question_id"])
        if conversation_id not in seen:
            seen.add(conversation_id)
            ordered.append(conversation_id)
    return ordered


def truncate_candidates(items: list[dict], topn: int) -> list[dict]:
    truncated = []
    for item in items:
        copied = dict(item)
        copied["paths"] = list(item.get("paths", []))[:topn]
        metadata = dict(copied.get("metadata", {}))
        metadata["candidate_topn_for_selector"] = topn
        copied["metadata"] = metadata
        truncated.append(copied)
    return truncated


def selector_variants(which: str) -> dict[str, list[str]]:
    variants = {
        "topology_selector_with_category": select_feature_names(),
        "topology_selector_without_category": select_feature_names(exclude=["question_category"]),
    }
    if which == "with_category":
        return {"topology_selector_with_category": variants["topology_selector_with_category"]}
    if which == "without_category":
        return {"topology_selector_without_category": variants["topology_selector_without_category"]}
    return variants


def build_feature_cache(graph, items: list[dict]) -> dict[str, dict]:
    started = time.perf_counter()
    index = TopologyFeatureIndex.from_graph(graph)
    cache = {}
    total_paths = 0
    for item_index, item in enumerate(items, start=1):
        item = with_cached_query_terms(item)
        rows = []
        labels = []
        for rank, path in enumerate(item.get("paths", []), start=1):
            example = build_example(graph, item, path, rank, index)
            rows.append(example.features)
            labels.append(example.label)
        total_paths += len(rows)
        cache[item["question_id"]] = {
            "rows": rows,
            "labels": labels,
            "group": len(rows),
            "conversation_id": conversation_id_from_question(item["question_id"]),
        }
        if item_index % 250 == 0 or item_index == len(items):
            print(
                f"  feature cache {item_index}/{len(items)} questions "
                f"paths={total_paths} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return cache


def flatten_feature_cache(cache: dict[str, dict], question_ids: list[str]) -> tuple[list[list[float]], list[int], list[int], list[str]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    kept_qids: list[str] = []
    for question_id in question_ids:
        entry = cache[question_id]
        if not entry["rows"]:
            continue
        rows.extend(entry["rows"])
        labels.extend(entry["labels"])
        groups.append(entry["group"])
        kept_qids.append(question_id)
    return rows, labels, groups, kept_qids


def score_items_with_cache(items: list[dict], model, cache: dict[str, dict], feature_names: list[str]) -> dict[str, list[float]]:
    selected_indices = feature_indices(feature_names)
    scores_by_question = {}
    for item in items:
        rows = cache[item["question_id"]]["rows"]
        projected = [[float(row[index]) for index in selected_indices] for row in rows]
        if projected:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                scores_by_question[item["question_id"]] = [float(score) for score in model.predict(projected)]
        else:
            scores_by_question[item["question_id"]] = []
    return scores_by_question


def evaluate_items(graph, items: list[dict], k: int, method: str) -> dict:
    results = [evaluate_item(graph, item, k) for item in items]
    return {
        "method": method,
        "k": k,
        "summary": summarize(results),
        "per_question": [result.__dict__ for result in results],
    }


def print_fold(fold: dict) -> None:
    print(f"fold={fold['fold']} test={fold['test_conversation']} questions={fold['num_test_questions']}")
    for method_name, metrics in fold["methods"].items():
        print(
            f"  {method_name}: hit={metrics['hit']:.4f} "
            f"recall={metrics['recall']:.4f} full_cover={metrics['full_cover']:.4f}"
        )


def build_cv_summary(
    *,
    folds: list[dict],
    aggregate_results: dict[str, list],
    aggregate_by_question: dict[str, dict[str, bool]],
    importances: dict[str, list[dict]],
    output_dir: Path,
    args,
) -> dict:
    aggregate = {}
    fold_stats = {}
    for method_name, rows in aggregate_results.items():
        aggregate[method_name] = summarize_rows(rows)
        per_fold = [fold["methods"][method_name] for fold in folds]
        fold_stats[method_name] = {
            "mean_hit": mean(metric["hit"] for metric in per_fold),
            "std_hit": pstdev(metric["hit"] for metric in per_fold),
            "mean_recall": mean(metric["recall"] for metric in per_fold),
            "std_recall": pstdev(metric["recall"] for metric in per_fold),
            "mean_full_cover": mean(metric["full_cover"] for metric in per_fold),
            "std_full_cover": pstdev(metric["full_cover"] for metric in per_fold),
        }
    paired = {
        "topology_selector_with_category_vs_bm25_fact50_ce": paired_compare(
            aggregate_by_question, "topology_selector_with_category", "bm25_fact50_ce"
        ),
        "topology_selector_without_category_vs_bm25_fact50_ce": paired_compare(
            aggregate_by_question, "topology_selector_without_category", "bm25_fact50_ce"
        ),
        "topology_selector_with_category_vs_graph_ce": paired_compare(
            aggregate_by_question, "topology_selector_with_category", "graph_ce"
        ),
        "topology_selector_without_category_vs_graph_ce": paired_compare(
            aggregate_by_question, "topology_selector_without_category", "graph_ce"
        ),
        "without_category_vs_with_category": paired_compare(
            aggregate_by_question, "topology_selector_without_category", "topology_selector_with_category"
        ),
    }
    return {
        "method": "LOCO-CV Topology Selector LightGBM",
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
        "feature_sets": selector_variants(args.selector_variants),
        "aggregate": aggregate,
        "fold_mean_std": fold_stats,
        "paired_compare": paired,
        "folds_detail": folds,
        "importance_top10_mean": {
            name: average_importance_topk(items, top_k=10) for name, items in importances.items()
        },
    }


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "num_questions": 0,
            "hit": 0.0,
            "recall": 0.0,
            "full_cover": 0.0,
            "avg_tokens": 0.0,
            "avg_path_len": 0.0,
        }
    n = len(rows)
    return {
        "num_questions": n,
        "hit": sum(float(row["hit"]) for row in rows) / n,
        "recall": sum(float(row["recall"]) for row in rows) / n,
        "full_cover": sum(float(row["full_cover"]) for row in rows) / n,
        "avg_tokens": sum(float(row["tokens"]) for row in rows) / n,
        "avg_path_len": sum(float(row["path_len"]) for row in rows) / n,
    }


def paired_compare(by_question: dict[str, dict[str, bool]], left: str, right: str) -> dict:
    both_hit = left_only = right_only = both_miss = compared = 0
    for methods in by_question.values():
        if left not in methods or right not in methods:
            continue
        compared += 1
        left_hit = methods[left]
        right_hit = methods[right]
        if left_hit and right_hit:
            both_hit += 1
        elif left_hit:
            left_only += 1
        elif right_hit:
            right_only += 1
        else:
            both_miss += 1
    return {
        "left": left,
        "right": right,
        "compared": compared,
        "both_hit": both_hit,
        "left_only": left_only,
        "right_only": right_only,
        "both_miss": both_miss,
        "net_left_minus_right": left_only - right_only,
    }


def feature_importance_for_names(model, feature_names: list[str]) -> list[dict]:
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        return []
    ranked = sorted(
        zip(feature_names, [float(value) for value in raw]),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"feature": name, "importance": value} for name, value in ranked]


def average_importance_topk(folds: list[dict], top_k: int) -> list[dict]:
    totals = defaultdict(float)
    for fold in folds:
        for item in fold.get("feature_importance", []):
            totals[item["feature"]] += float(item["importance"])
    averaged = [
        {"feature": feature, "mean_importance": value / max(len(folds), 1)}
        for feature, value in totals.items()
    ]
    return sorted(averaged, key=lambda item: item["mean_importance"], reverse=True)[:top_k]


def write_markdown_summary(summary: dict, path: Path) -> None:
    lines = [
        "# LOCO-CV Topology Selector",
        "",
        f"Folds: {summary['folds']}",
        f"K: {summary['k']}",
        "",
        "## Aggregate",
        "",
        "| Method | Questions | Hit@5 | Recall@5 | FullCover@5 | AvgTokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method_name, metrics in summary["aggregate"].items():
        lines.append(
            f"| {method_name} | {metrics['num_questions']} | {metrics['hit']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['full_cover']:.4f} | {metrics['avg_tokens']:.1f} |"
        )
    lines.extend(["", "## Fold Mean/Std", ""])
    lines.append("| Method | Mean Hit | Std Hit | Mean Recall | Std Recall |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for method_name, metrics in summary["fold_mean_std"].items():
        lines.append(
            f"| {method_name} | {metrics['mean_hit']:.4f} | {metrics['std_hit']:.4f} | "
            f"{metrics['mean_recall']:.4f} | {metrics['std_recall']:.4f} |"
        )
    lines.extend(["", "## Paired Compare", ""])
    lines.append("| Compare | Compared | LeftOnly | RightOnly | Net |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, metrics in summary["paired_compare"].items():
        lines.append(
            f"| {name} | {metrics['compared']} | {metrics['left_only']} | "
            f"{metrics['right_only']} | {metrics['net_left_minus_right']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
