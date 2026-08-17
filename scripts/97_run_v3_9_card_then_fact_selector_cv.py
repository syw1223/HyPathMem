from __future__ import annotations

import argparse
import importlib.util
import time
from collections import defaultdict
from pathlib import Path

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import conversation_id_from_question
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topology_features import project_feature_rows
from hytopomem.retrieval.topology_selector import train_lightgbm_ranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--candidates", default="outputs/v3_9_query_cards/qwen3_card_annotated_base100_paths_v3_clean.json")
    parser.add_argument(
        "--cardce-paths",
        default="outputs/eval/v3_9_cardce_guided_ctx50/cardce_guided_topk_paths.json",
    )
    parser.add_argument("--output-dir", default="outputs/eval/cv/v3_9_card_then_fact_ctx50")
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--card-quota-top5", type=int, default=3)
    parser.add_argument("--card-quota-top20", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args()

    cv = load_module("30_run_loco_cv_selector.py", "loco_cv_helpers")
    selector = load_module("41_run_graph_v2_selector_cv.py", "graph_v2_selector_helpers")
    nary = load_module("88_run_nary_completion_selector_cv_v3_6c.py", "nary_selector_helpers")

    graph = JsonGraphStore().load(resolve_path(args.graph))
    candidates = read_json(resolve_path(args.candidates))
    cardce_items = read_json(resolve_path(args.cardce_paths))
    card_index = build_cardce_index(cardce_items)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval = selector.dedupe(selector.RETRIEVAL_FEATURES)
    graph_features = selector.dedupe(retrieval + selector.GRAPH_V2_FEATURES)
    base_features = selector.dedupe(
        graph_features + selector.ROUTE_ORIGIN_V1_FEATURES + selector.ROUTE_AGREEMENT_V1_FEATURES
    )
    card_features = selector.dedupe(base_features + nary.NARY_POINT_FEATURES)
    feature_sets = {
        "card_then_fact_base_lgbm": base_features,
        "card_then_fact_card_lgbm": card_features,
    }

    started = time.perf_counter()
    feature_cache = cv.build_feature_cache(graph, candidates)
    print(f"built feature cache elapsed={time.perf_counter() - started:.1f}s", flush=True)

    conversations = cv.ordered_conversations(candidates)
    aggregate = {k: defaultdict(list) for k in args.topk}
    fold_rows = []
    for fold_index, test_conversation in enumerate(conversations):
        train_items = [
            item for item in candidates if conversation_id_from_question(item["question_id"]) != test_conversation
        ]
        test_items = [
            item for item in candidates if conversation_id_from_question(item["question_id"]) == test_conversation
        ]
        train_rows_all, train_labels, train_groups, _ = cv.flatten_feature_cache(
            feature_cache, [item["question_id"] for item in train_items]
        )
        fold = {"fold": fold_index, "test_conversation": test_conversation, "methods": {}}

        for method, feature_names in feature_sets.items():
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
            for k in args.topk:
                quota = args.card_quota_top5 if k == 5 else args.card_quota_top20
                selected = card_then_fact_items(test_items, scores, card_index, k, quota, method)
                write_json(selected, output_dir / f"fold_{fold_index:02d}_{test_conversation}" / f"{method}_top{k}_paths.json")
                evaluation = cv.evaluate_items(graph, selected, k, method)
                write_json(
                    evaluation,
                    output_dir / f"fold_{fold_index:02d}_{test_conversation}" / f"{method}_top{k}_eval.json",
                )
                fold["methods"][f"{method}_top{k}"] = evaluation["summary"]
                aggregate[k][method].extend(evaluation["per_question"])
        fold_rows.append(fold)
        write_json(fold, output_dir / f"fold_{fold_index:02d}_{test_conversation}" / "fold_summary.json")
        print(f"finished fold={fold_index} test={test_conversation}", flush=True)

    summary = {
        "method": "V3.9 Card-then-Fact Selector CV",
        "candidates": str(resolve_path(args.candidates)),
        "cardce_paths": str(resolve_path(args.cardce_paths)),
        "card_quota_top5": args.card_quota_top5,
        "card_quota_top20": args.card_quota_top20,
        "feature_sets": feature_sets,
        "aggregate": {
            f"top{k}": {method: cv.summarize_rows(rows) for method, rows in methods.items()}
            for k, methods in aggregate.items()
        },
        "folds": fold_rows,
    }
    write_json(summary, output_dir / "card_then_fact_summary.json")
    (output_dir / "card_then_fact_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


def build_cardce_index(items: list[dict]) -> dict[str, dict[str, float]]:
    index = {}
    for item in items:
        facts = {}
        for path in item.get("paths", []):
            metadata = path.get("metadata", {})
            if str(metadata.get("v3_9_cardce_selected", "")).lower() != "true":
                continue
            fact_id = evidence_node_id(path)
            if not fact_id:
                continue
            score = float(path.get("scores", {}).get("v3_9_card_ce", metadata.get("v3_9_cardce_score", 0.0)) or 0.0)
            facts[fact_id] = max(facts.get(fact_id, float("-inf")), score)
        index[item["question_id"]] = facts
    return index


def card_then_fact_items(
    items: list[dict],
    scores_by_qid: dict[str, list[float]],
    card_index: dict[str, dict[str, float]],
    topk: int,
    card_quota: int,
    method: str,
) -> list[dict]:
    output = []
    for item in items:
        scores = scores_by_qid[item["question_id"]]
        ranked = sorted(zip(item.get("paths", []), scores), key=lambda row: row[1], reverse=True)
        card_scores = card_index.get(item["question_id"], {})
        card_ranked = [row for row in ranked if evidence_node_id(row[0]) in card_scores]
        card_ranked.sort(key=lambda row: (card_scores[evidence_node_id(row[0])], row[1]), reverse=True)

        selected = []
        seen = set()
        for path, lgbm_score in card_ranked[:card_quota]:
            selected.append(with_selector_scores(path, lgbm_score, card_scores[evidence_node_id(path)], method))
            seen.add(evidence_node_id(path))
        for path, lgbm_score in ranked:
            fact_id = evidence_node_id(path)
            if fact_id in seen:
                continue
            selected.append(with_selector_scores(path, lgbm_score, card_scores.get(fact_id, 0.0), method))
            seen.add(fact_id)
            if len(selected) >= topk:
                break

        copied = dict(item)
        copied["paths"] = selected[:topk]
        metadata = dict(copied.get("metadata", {}))
        metadata.update({"method": method, "card_quota": card_quota, "final_topk": topk})
        copied["metadata"] = metadata
        output.append(copied)
    return output


def with_selector_scores(path: dict, lgbm_score: float, cardce_score: float, method: str) -> dict:
    copied = dict(path)
    scores = dict(copied.get("scores", {}))
    scores["topology_selector"] = float(lgbm_score)
    scores["v3_9_card_ce"] = float(cardce_score)
    copied["scores"] = scores
    metadata = dict(copied.get("metadata", {}))
    metadata["card_then_fact_method"] = method
    metadata["card_then_fact_is_card"] = str(cardce_score != 0.0).lower()
    copied["metadata"] = metadata
    return copied


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def render_markdown(summary: dict) -> str:
    lines = [
        "# V3.9 Card-then-Fact Selector CV",
        "",
        f"- Card quota Top5: {summary['card_quota_top5']}",
        f"- Card quota Top20: {summary['card_quota_top20']}",
        "",
        "| K | Method | Hit | Recall | FullCover |",
        "|---:|---|---:|---:|---:|",
    ]
    for k, methods in summary["aggregate"].items():
        for method, row in methods.items():
            lines.append(f"| {k} | {method} | {row['hit']:.4f} | {row['recall']:.4f} | {row['full_cover']:.4f} |")
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


if __name__ == "__main__":
    main()
