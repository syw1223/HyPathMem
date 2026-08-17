from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topology_features import project_feature_rows
from hytopomem.retrieval.topology_selector import train_lightgbm_ranker


RETRIEVAL_ROUTE_CARD5 = [
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--candidates", default="outputs/v3_9_query_cards/qwen3_card_annotated_base100_paths_v3_clean.json")
    parser.add_argument("--cardce-paths", default="outputs/eval/v3_9_cardce_guided_ctx50/cardce_guided_topk_paths.json")
    parser.add_argument("--output-dir", default="outputs/eval/cv/v3_9_24_feature_card_quota")
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args()

    cv = load_module("30_run_loco_cv_selector.py", "loco_cv_helpers_v3_9_24_quota")
    graph = JsonGraphStore().load(resolve_path(args.graph))
    candidates = read_json(resolve_path(args.candidates))
    card_index = build_cardce_index(read_json(resolve_path(args.cardce_paths)))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cache = cv.build_feature_cache(graph, candidates)
    conversations = cv.ordered_conversations(candidates)
    aggregate: dict[str, list] = defaultdict(list)
    selected_card_stats: dict[str, list] = defaultdict(list)
    folds = []

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
        model = train_lightgbm_ranker(
            project_feature_rows(train_rows_all, RETRIEVAL_ROUTE_CARD5),
            train_labels,
            train_groups,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            n_jobs=args.n_jobs,
            random_state=args.random_state + fold_index,
        )
        scores = cv.score_items_with_cache(test_items, model, feature_cache, RETRIEVAL_ROUTE_CARD5)
        fold = {"fold": fold_index, "test_conversation": test_conversation, "methods": {}}
        fold_dir = output_dir / f"fold_{fold_index:02d}_{test_conversation}"

        for topk in args.topk:
            method_items = {
                f"lgbm_top{topk}": cv.rerank_items_with_scores(test_items, scores, topk),
                f"card_quota_light_top{topk}": card_quota_items(
                    test_items,
                    scores,
                    card_index,
                    topk=topk,
                    card_quota=2 if topk <= 5 else 5,
                    method=f"card_quota_light_top{topk}",
                ),
                f"card_quota_medium_top{topk}": card_quota_items(
                    test_items,
                    scores,
                    card_index,
                    topk=topk,
                    card_quota=3 if topk <= 5 else 10,
                    method=f"card_quota_medium_top{topk}",
                ),
            }
            for method, items in method_items.items():
                evaluation = cv.evaluate_items(graph, items, topk, method)
                write_json(items, fold_dir / f"{method}_paths.json")
                write_json(evaluation, fold_dir / f"{method}_eval.json")
                fold["methods"][method] = evaluation["summary"]
                aggregate[method].extend(evaluation["per_question"])
                selected_card_stats[method].append(selected_card_summary(graph, items, topk))

        write_json(fold, fold_dir / "fold_summary.json")
        folds.append(fold)
        print(f"finished fold={fold_index} test={test_conversation}", flush=True)

    summary = {
        "method": "V3.9 24-feature LightGBM with card-quota selection",
        "features": RETRIEVAL_ROUTE_CARD5,
        "graph": str(resolve_path(args.graph)),
        "candidates": str(resolve_path(args.candidates)),
        "cardce_paths": str(resolve_path(args.cardce_paths)),
        "aggregate": {method: cv.summarize_rows(rows) for method, rows in aggregate.items()},
        "selected_card_summary": {
            method: summarize_selected_card_rows(rows)
            for method, rows in selected_card_stats.items()
        },
        "folds": folds,
    }
    write_json(summary, output_dir / "card_quota_24_feature_summary.json")
    (output_dir / "card_quota_24_feature_summary.md").write_text(render_markdown(summary), encoding="utf-8")
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
        index[str(item["question_id"])] = facts
    return index


def card_quota_items(
    items: list[dict],
    scores_by_qid: dict[str, list[float]],
    card_index: dict[str, dict[str, float]],
    *,
    topk: int,
    card_quota: int,
    method: str,
) -> list[dict]:
    output = []
    for item in items:
        qid = str(item["question_id"])
        scores = scores_by_qid[item["question_id"]]
        ranked = sorted(zip(item.get("paths", []), scores), key=lambda row: row[1], reverse=True)
        card_scores = card_index.get(qid, {})
        card_ranked = [
            (path, lgbm_score, card_scores[evidence_node_id(path)])
            for path, lgbm_score in ranked
            if evidence_node_id(path) in card_scores
        ]
        card_ranked.sort(key=lambda row: (row[2], row[1]), reverse=True)

        selected = []
        seen = set()
        for path, lgbm_score, cardce_score in card_ranked[:card_quota]:
            selected.append(with_scores(path, lgbm_score, cardce_score, method, True))
            seen.add(evidence_node_id(path))
        for path, lgbm_score in ranked:
            fact_id = evidence_node_id(path)
            if not fact_id or fact_id in seen:
                continue
            selected.append(with_scores(path, lgbm_score, card_scores.get(fact_id, 0.0), method, False))
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


def with_scores(path: dict, lgbm_score: float, cardce_score: float, method: str, quota_selected: bool) -> dict:
    copied = dict(path)
    scores = dict(copied.get("scores", {}))
    scores["topology_selector"] = float(lgbm_score)
    scores["v3_9_card_ce"] = float(cardce_score)
    copied["scores"] = scores
    metadata = dict(copied.get("metadata", {}))
    metadata["method"] = method
    metadata["card_quota_selected"] = str(quota_selected).lower()
    metadata["card_quota_cardce_score"] = f"{cardce_score:.8f}"
    copied["metadata"] = metadata
    return copied


def selected_card_summary(graph, items: list[dict], topk: int) -> dict:
    selected = 0
    selected_gold = 0
    quota_selected = 0
    quota_selected_gold = 0
    questions_with_card_gold = 0
    for item in items:
        gold = gold_set(item)
        q_card_gold = 0
        for path in item.get("paths", [])[:topk]:
            cardce_score = float(path.get("scores", {}).get("v3_9_card_ce", 0.0) or 0.0)
            if cardce_score <= 0.0:
                continue
            selected += 1
            is_gold = bool(evidence_ids_for_node(graph, evidence_node_id(path)) & gold)
            selected_gold += int(is_gold)
            if str(path.get("metadata", {}).get("card_quota_selected", "")).lower() == "true":
                quota_selected += 1
                quota_selected_gold += int(is_gold)
            q_card_gold += int(is_gold)
        questions_with_card_gold += int(q_card_gold > 0)
    return {
        "selected_card_facts": selected,
        "selected_card_gold_facts": selected_gold,
        "selected_card_gold_rate": selected_gold / max(selected, 1),
        "quota_selected_card_facts": quota_selected,
        "quota_selected_card_gold_facts": quota_selected_gold,
        "quota_selected_card_gold_rate": quota_selected_gold / max(quota_selected, 1),
        "questions_with_card_gold": questions_with_card_gold,
    }


def summarize_selected_card_rows(rows: list[dict]) -> dict:
    keys = [
        "selected_card_facts",
        "selected_card_gold_facts",
        "quota_selected_card_facts",
        "quota_selected_card_gold_facts",
        "questions_with_card_gold",
    ]
    total = {key: sum(row[key] for row in rows) for key in keys}
    total["selected_card_gold_rate"] = total["selected_card_gold_facts"] / max(total["selected_card_facts"], 1)
    total["quota_selected_card_gold_rate"] = total["quota_selected_card_gold_facts"] / max(total["quota_selected_card_facts"], 1)
    return total


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(value) for value in item.get("gold_evidence", []) or item.get("gold_supports", []) or []}


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def render_markdown(summary: dict) -> str:
    lines = [
        "# V3.9 24-feature Card-Quota Selection",
        "",
        "| Method | Hit | Recall | FullCover | CardFacts | CardGold | CardGoldRate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in summary["aggregate"].items():
        card = summary["selected_card_summary"].get(method, {})
        lines.append(
            f"| {method} | {metrics['hit']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['full_cover']:.4f} | {card.get('selected_card_facts', 0)} | "
            f"{card.get('selected_card_gold_facts', 0)} | {card.get('selected_card_gold_rate', 0.0):.4f} |"
        )
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
