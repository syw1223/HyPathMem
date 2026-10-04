from __future__ import annotations

import argparse
import importlib.util
import random
import warnings
from collections import defaultdict
from pathlib import Path

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evidence_ids_for_node, evaluate_item, normalize_evidence_id, summarize
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topology_features import feature_indices, project_feature_rows
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
    parser.add_argument("--graph", default="outputs/longmemeval_s/graph_semantic_hierarchy_v3.json")
    parser.add_argument("--candidates", default="outputs/longmemeval_s/cards/qwen3_card_annotated_top100_paths.json")
    parser.add_argument("--cardce-paths", default="")
    parser.add_argument("--output-dir", default="outputs/eval/longmemeval_v3_9_24_feature_card_selector")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--candidate-topn", type=int, default=100)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-questions", type=int, default=0)
    args = parser.parse_args()

    cv = load_module("30_run_loco_cv_selector.py", "loco_cv_helpers_longmemeval_v3_9")
    graph = JsonGraphStore().load(resolve_path(args.graph))
    candidates = canonicalize_longmemeval_candidates(read_json(resolve_path(args.candidates)))
    cardce_index = build_cardce_index(read_json(resolve_path(args.cardce_paths))) if args.cardce_paths else {}
    if args.max_questions:
        candidates = candidates[: args.max_questions]
    if args.candidate_topn:
        candidates = truncate_candidates(candidates, args.candidate_topn)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cache = cv.build_feature_cache(graph, candidates)
    qids = [str(item["question_id"]) for item in candidates]
    folds = make_folds(qids, args.folds, args.random_state)

    aggregate: dict[str, list[dict]] = defaultdict(list)
    aggregate_items: dict[str, list[dict]] = defaultdict(list)
    selected_card_stats: dict[str, list[dict]] = defaultdict(list)
    importances: list[dict] = []
    fold_payloads = []

    for fold_idx, test_qids in enumerate(folds):
        test_set = set(test_qids)
        train_qids = [qid for qid in qids if qid not in test_set and has_gold(candidates_by_qid(candidates)[qid])]
        test_items = [item for item in candidates if str(item["question_id"]) in test_set]
        train_rows_all, train_labels, train_groups, kept_train_qids = cv.flatten_feature_cache(feature_cache, train_qids)

        model = train_lightgbm_ranker(
            project_feature_rows(train_rows_all, RETRIEVAL_ROUTE_CARD5),
            train_labels,
            train_groups,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            n_jobs=args.n_jobs,
            random_state=args.random_state + fold_idx,
        )
        scores = score_items_with_cache(test_items, model, feature_cache, RETRIEVAL_ROUTE_CARD5)
        fold = {
            "fold": fold_idx,
            "num_train_questions": len(kept_train_qids),
            "num_test_questions": len(test_items),
            "methods": {},
        }
        if fold_idx == 0:
            importances = feature_importance_for_names(model, RETRIEVAL_ROUTE_CARD5)

        for topk in args.topk:
            methods = {
                f"base_ce_top{topk}": base_ce_items(test_items, topk),
                f"lgbm_top{topk}": rerank_items_with_scores(
                    test_items, scores, topk, method=f"lgbm_top{topk}", cardce_index=cardce_index
                ),
                f"card_quota_light_top{topk}": card_quota_items(
                    test_items,
                    scores,
                    cardce_index,
                    topk=topk,
                    card_quota=2 if topk <= 5 else 5,
                    method=f"card_quota_light_top{topk}",
                ),
                f"card_quota_medium_top{topk}": card_quota_items(
                    test_items,
                    scores,
                    cardce_index,
                    topk=topk,
                    card_quota=3 if topk <= 5 else 10,
                    method=f"card_quota_medium_top{topk}",
                ),
            }
            for method, items in methods.items():
                payload = evaluate_items(graph, items, topk, method)
                fold["methods"][method] = payload["summary_with_gold"]
                aggregate[method].extend(payload["per_question_with_gold"])
                aggregate_items[method].extend(items)
                selected_card_stats[method].append(selected_card_summary(graph, items, topk))

        fold_payloads.append(fold)
        print(f"finished fold={fold_idx} train={len(kept_train_qids)} test={len(test_items)}", flush=True)

    for method, items in aggregate_items.items():
        write_json(sorted(items, key=lambda item: str(item["question_id"])), output_dir / f"{method}_paths.json")

    summary = {
        "method": "LongMemEval V3.9 24-feature card-aware selector",
        "features": RETRIEVAL_ROUTE_CARD5,
        "graph": str(resolve_path(args.graph)),
        "candidates": str(resolve_path(args.candidates)),
        "cardce_paths": str(resolve_path(args.cardce_paths)) if args.cardce_paths else "",
        "folds": len(folds),
        "candidate_topn": args.candidate_topn,
        "lgbm": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_child_samples": args.min_child_samples,
            "n_jobs": args.n_jobs,
            "random_state": args.random_state,
        },
        "aggregate_with_gold": {method: summarize_rows(rows) for method, rows in aggregate.items()},
        "selected_card_summary": {
            method: summarize_selected_card_rows(rows) for method, rows in selected_card_stats.items()
        },
        "folds_detail": fold_payloads,
        "feature_importance_fold0": importances,
    }
    write_json(summary, output_dir / "summary.json")
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


def canonicalize_longmemeval_candidates(items: list[dict]) -> list[dict]:
    output = []
    for item in items:
        copied_item = dict(item)
        paths = []
        for path in item.get("paths", []):
            copied_path = dict(path)
            metadata = dict(copied_path.get("metadata", {}))
            route_source = str(metadata.get("route_source") or metadata.get("candidate_source") or "")
            route_tokens = [token for token in route_source.replace(",", "+").split("+") if token]
            canonical = canonical_route_tokens(route_tokens)
            metadata["longmemeval_original_route_source"] = route_source
            metadata["route_source"] = "+".join(canonical)
            metadata["candidate_source"] = "+".join(canonical)
            metadata["has_bottom_up_route"] = str("bottom_up" in canonical).lower()
            metadata["has_topdown_route"] = str(
                any(token in canonical for token in ["eu_event", "eu_topic", "hyp_event", "hyp_topic"])
            ).lower()
            metadata["is_eu_route"] = str(any(token.startswith("eu_") for token in canonical)).lower()
            metadata["is_hyp_route"] = str(any(token.startswith("hyp_") for token in canonical)).lower()
            metadata["route_source_count"] = str(
                sum(token in canonical for token in ["bottom_up", "eu_event", "eu_topic", "hyp_bottom", "hyp_event", "hyp_topic"])
            )
            if "bu_hyp" in route_tokens and not metadata.get("hyp_bottom_seed_rank"):
                metadata["hyp_bottom_seed_rank"] = metadata.get("bottom_up_rank", "0")
                metadata["hyp_bottom_score"] = metadata.get("bm25_norm", "0.0")
            copied_path["metadata"] = metadata
            paths.append(copied_path)
        copied_item["paths"] = paths
        output.append(copied_item)
    return output


def canonical_route_tokens(route_tokens: list[str]) -> list[str]:
    canonical: list[str] = []

    def add(token: str) -> None:
        if token not in canonical:
            canonical.append(token)

    for token in route_tokens:
        if token == "bu_eu":
            add("bottom_up")
            add("eu_event")
        elif token == "bu_hyp":
            add("bottom_up")
            add("hyp_bottom")
        elif token == "td_eu":
            add("eu_event")
            add("eu_topic")
        elif token == "td_hyp":
            add("hyp_event")
            add("hyp_topic")
        else:
            add(token)
    return canonical


def truncate_candidates(items: list[dict], topn: int) -> list[dict]:
    output = []
    for item in items:
        copied = dict(item)
        copied["paths"] = list(item.get("paths", []))[:topn]
        output.append(copied)
    return output


def build_cardce_index(items: list[dict]) -> dict[str, dict[str, float]]:
    index: dict[str, dict[str, float]] = {}
    for item in items:
        qid = str(item["question_id"])
        scores = index.setdefault(qid, {})
        for path in item.get("paths", []):
            metadata = path.get("metadata", {})
            if str(metadata.get("v3_9_cardce_selected", "")).lower() != "true":
                continue
            fact_id = evidence_node_id(path)
            if not fact_id:
                continue
            score = card_score(path)
            scores[fact_id] = max(scores.get(fact_id, float("-inf")), score)
    return index


def candidates_by_qid(items: list[dict]) -> dict[str, dict]:
    return {str(item["question_id"]): item for item in items}


def make_folds(qids: list[str], folds: int, seed: int) -> list[list[str]]:
    shuffled = list(qids)
    random.Random(seed).shuffle(shuffled)
    buckets = [[] for _ in range(max(folds, 1))]
    for index, qid in enumerate(shuffled):
        buckets[index % len(buckets)].append(qid)
    return [bucket for bucket in buckets if bucket]


def has_gold(item: dict) -> bool:
    return bool(item.get("gold_evidence") or item.get("gold_supports"))


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


def base_ce_items(items: list[dict], topk: int) -> list[dict]:
    output = []
    for item in items:
        copied = dict(item)
        copied["paths"] = list(item.get("paths", []))[:topk]
        metadata = dict(copied.get("metadata", {}))
        metadata["method"] = f"base_ce_top{topk}"
        metadata["final_topk"] = topk
        copied["metadata"] = metadata
        output.append(copied)
    return output


def rerank_items_with_scores(
    items: list[dict],
    scores_by_qid: dict[str, list[float]],
    topk: int,
    method: str,
    cardce_index: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    output = []
    for item in items:
        qid = str(item["question_id"])
        ranked = sorted(zip(item.get("paths", []), scores_by_qid.get(qid, [])), key=lambda row: row[1], reverse=True)
        paths = [
            with_scores(path, lgbm_score, card_score(path, cardce_index.get(qid, {}) if cardce_index else {}), method, False)
            for path, lgbm_score in ranked[:topk]
        ]
        copied = dict(item)
        copied["paths"] = paths
        metadata = dict(copied.get("metadata", {}))
        metadata["method"] = method
        metadata["final_topk"] = topk
        copied["metadata"] = metadata
        output.append(copied)
    return output


def card_quota_items(
    items: list[dict],
    scores_by_qid: dict[str, list[float]],
    cardce_index: dict[str, dict[str, float]] | None = None,
    *,
    topk: int,
    card_quota: int,
    method: str,
) -> list[dict]:
    output = []
    for item in items:
        qid = str(item["question_id"])
        ranked = sorted(zip(item.get("paths", []), scores_by_qid.get(qid, [])), key=lambda row: row[1], reverse=True)
        q_cardce = cardce_index.get(qid, {}) if cardce_index else {}
        card_ranked = [
            (path, lgbm_score, card_score(path, q_cardce))
            for path, lgbm_score in ranked
            if is_card_fact(path) and card_score(path, q_cardce) > 0.0
        ]
        card_ranked.sort(key=lambda row: (row[2], row[1]), reverse=True)
        selected = []
        seen = set()
        for path, lgbm_score, cscore in card_ranked[:card_quota]:
            selected.append(with_scores(path, lgbm_score, cscore, method, True))
            seen.add(evidence_node_id(path))
        for path, lgbm_score in ranked:
            fact_id = evidence_node_id(path)
            if not fact_id or fact_id in seen:
                continue
            selected.append(with_scores(path, lgbm_score, card_score(path, q_cardce), method, False))
            seen.add(fact_id)
            if len(selected) >= topk:
                break
        copied = dict(item)
        copied["paths"] = selected[:topk]
        metadata = dict(copied.get("metadata", {}))
        metadata["method"] = method
        metadata["card_quota"] = card_quota
        metadata["final_topk"] = topk
        copied["metadata"] = metadata
        output.append(copied)
    return output


def with_scores(path: dict, lgbm_score: float, cscore: float, method: str, quota_selected: bool) -> dict:
    copied = dict(path)
    scores = dict(copied.get("scores", {}))
    scores["topology_selector"] = float(lgbm_score)
    scores["v3_9_card_ce"] = float(cscore)
    copied["scores"] = scores
    copied["score"] = float(lgbm_score)
    metadata = dict(copied.get("metadata", {}))
    metadata["method"] = method
    metadata["card_quota_selected"] = str(quota_selected).lower()
    metadata["card_quota_cardce_score"] = f"{cscore:.8f}"
    metadata["v39_card_ce_score"] = f"{cscore:.8f}"
    copied["metadata"] = metadata
    return copied


def card_score(path: dict, cardce_scores: dict[str, float] | None = None) -> float:
    fact_id = evidence_node_id(path)
    if cardce_scores and fact_id in cardce_scores:
        return float(cardce_scores[fact_id])
    metadata = path.get("metadata", {})
    scores = path.get("scores", {})
    for value in [
        scores.get("v3_9_card_ce"),
        metadata.get("v39_card_ce_score"),
        metadata.get("v3_9_cardce_score"),
        scores.get("nary_card_confidence"),
        metadata.get("nary_hyperedge_confidence"),
    ]:
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score > 0.0:
            return score
    return 0.0


def is_card_fact(path: dict) -> bool:
    metadata = path.get("metadata", {})
    return str(metadata.get("is_nary_completion", "")).lower() == "true" or str(
        metadata.get("v3_9_query_card", "")
    ).lower() == "true"


def evaluate_items(graph, items: list[dict], k: int, method: str) -> dict:
    all_results = [evaluate_item(graph, item, k) for item in items]
    with_gold = [result for result in all_results if result.gold_evidence_ids]
    return {
        "method": method,
        "k": k,
        "summary_all": summarize(all_results),
        "summary_with_gold": summarize(with_gold),
        "per_question_all": [result.__dict__ for result in all_results],
        "per_question_with_gold": [result.__dict__ for result in with_gold],
    }


def selected_card_summary(graph, items: list[dict], topk: int) -> dict:
    selected = selected_gold = quota_selected = quota_selected_gold = questions_with_card_gold = 0
    for item in items:
        gold = gold_set(item)
        q_card_gold = 0
        for path in item.get("paths", [])[:topk]:
            if not is_card_fact(path):
                continue
            selected += 1
            is_gold = bool(evidence_ids_for_node(graph, evidence_node_id(path)) & gold)
            selected_gold += int(is_gold)
            q_card_gold += int(is_gold)
            if str(path.get("metadata", {}).get("card_quota_selected", "")).lower() == "true":
                quota_selected += 1
                quota_selected_gold += int(is_gold)
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
    total["quota_selected_card_gold_rate"] = total["quota_selected_card_gold_facts"] / max(
        total["quota_selected_card_facts"], 1
    )
    return total


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


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(value) for value in item.get("gold_evidence", []) or item.get("gold_supports", []) or []}


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def feature_importance_for_names(model, feature_names: list[str]) -> list[dict]:
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        return []
    return [
        {"feature": feature, "importance": value}
        for feature, value in sorted(zip(feature_names, [float(value) for value in raw]), key=lambda item: item[1], reverse=True)
    ]


def render_markdown(summary: dict) -> str:
    lines = [
        "# LongMemEval V3.9 24-feature Card Selector",
        "",
        f"Folds: {summary['folds']}",
        f"Candidate topN: {summary['candidate_topn']}",
        "",
        "## Aggregate With Gold",
        "",
        "| Method | Questions | Hit | Recall | FullCover | CardFacts | CardGold | CardGoldRate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in summary["aggregate_with_gold"].items():
        card = summary["selected_card_summary"].get(method, {})
        lines.append(
            f"| {method} | {metrics['num_questions']} | {metrics['hit']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['full_cover']:.4f} | {card.get('selected_card_facts', 0)} | "
            f"{card.get('selected_card_gold_facts', 0)} | {card.get('selected_card_gold_rate', 0.0):.4f} |"
        )
    lines.extend(["", "## Feature Importance Fold0", "", "| Feature | Importance |", "|---|---:|"])
    for row in summary.get("feature_importance_fold0", [])[:20]:
        lines.append(f"| {row['feature']} | {row['importance']:.1f} |")
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
