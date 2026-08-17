from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import (
    evidence_ids_for_node,
    normalize_evidence_id,
    summarize,
    RetrievalEvalResult,
)
from hytopomem.memory.graph_store import JsonGraphStore


DEFAULT_CANDIDATES = "outputs/v3_9_query_cards/qwen3_card_guided_expand120.json"
DEFAULT_GRAPH = "outputs/graphs/locomo_graph_v3_6b_qwen_all.json"
DEFAULT_OUTPUT_DIR = "outputs/eval/v3_9_route_stability"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--topn", type=int, default=120)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--write-paths", action="store_true")
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    items = read_json(resolve_path(args.candidates))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_cache = build_evidence_cache(graph, items, args.topn)
    diagnostics = route_diagnostics(graph, items, args.topn, evidence_cache)
    methods = build_rerank_methods(args.rrf_k)
    aggregate = {}
    selected = {}
    for method_name, method in methods.items():
        ranked_by_qid = rank_items(items, method, args.topn)
        aggregate[method_name] = {}
        selected[method_name] = {}
        for topk in args.topk:
            results = evaluate_ranked_items(items, ranked_by_qid, topk, evidence_cache)
            aggregate[method_name][f"top{topk}"] = summarize(results)
            selected[method_name][f"top{topk}"] = selected_route_summary(items, ranked_by_qid, topk, evidence_cache)
        if args.write_paths:
            write_json(materialize_ranked_items(items, ranked_by_qid), output_dir / f"{method_name}_paths.json")
        print(f"finished {method_name}", flush=True)

    summary = {
        "method": "V3.9 route stability diagnostics and rank-normalized fusion",
        "graph": str(resolve_path(args.graph)),
        "candidates": str(resolve_path(args.candidates)),
        "topn": args.topn,
        "rrf_k": args.rrf_k,
        "write_paths": args.write_paths,
        "diagnostics": diagnostics,
        "aggregate": aggregate,
        "selected_route_summary": selected,
    }
    write_json(summary, output_dir / "route_stability_fusion_summary.json")
    (output_dir / "route_stability_fusion_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


def build_evidence_cache(graph, items: list[dict], topn: int) -> dict[str, set[str]]:
    cache = {}
    for item in items:
        for path in item.get("paths", [])[:topn]:
            fact_id = evidence_node_id(path)
            if fact_id and fact_id not in cache:
                cache[fact_id] = evidence_ids_for_node(graph, fact_id)
    return cache


def route_diagnostics(graph, items: list[dict], topn: int, evidence_cache: dict[str, set[str]]) -> dict:
    per_question = []
    route_gold = defaultdict(lambda: {"available": 0, "gold_facts": 0, "questions_with_gold": 0})
    route_counts = Counter()
    entropy_rows = []
    card_gold = {"card_member_gold": 0, "card_expansion_gold": 0, "questions_with_card_gold": 0}

    for item in items:
        gold = gold_set(item)
        paths = list(item.get("paths", []))[:topn]
        route_seen_gold = defaultdict(bool)
        hyp_events = Counter()
        hyp_topics = Counter()
        q_card_gold = 0
        q_patterns = Counter()
        for path in paths:
            meta = path.get("metadata", {})
            routes = route_flags(meta)
            pattern = route_pattern(routes)
            route_counts[pattern] += 1
            q_patterns[pattern] += 1
            is_gold = bool(path_gold_ids(None, path, evidence_cache) & gold)
            if is_gold:
                for route in route_names(routes):
                    route_gold[route]["gold_facts"] += 1
                    route_seen_gold[route] = True
                if is_card_member(meta):
                    card_gold["card_member_gold"] += 1
                    q_card_gold += 1
                if is_card_expansion(meta):
                    card_gold["card_expansion_gold"] += 1
                    q_card_gold += 1
            for route in route_names(routes):
                route_gold[route]["available"] += 1
            if routes["top_hyp"] or routes["bottom_hyp"]:
                event_id = str(meta.get("event_node_id", "") or "")
                topic_id = str(meta.get("topic_node_id", "") or "")
                if event_id:
                    hyp_events[event_id] += 1
                if topic_id:
                    hyp_topics[topic_id] += 1
        for route, hit in route_seen_gold.items():
            if hit:
                route_gold[route]["questions_with_gold"] += 1
        card_gold["questions_with_card_gold"] += int(q_card_gold > 0)
        entropy_rows.append(
            {
                "question_id": item["question_id"],
                "hyp_event_entropy": normalized_entropy(hyp_events),
                "hyp_topic_entropy": normalized_entropy(hyp_topics),
                "hyp_unique_events": len(hyp_events),
                "hyp_unique_topics": len(hyp_topics),
            }
        )
        per_question.append(
            {
                "question_id": item["question_id"],
                "route_patterns": dict(q_patterns),
                "hyp_event_entropy": entropy_rows[-1]["hyp_event_entropy"],
                "hyp_topic_entropy": entropy_rows[-1]["hyp_topic_entropy"],
            }
        )

    route_gold_summary = {
        route: {
            **stats,
            "gold_rate": stats["gold_facts"] / max(stats["available"], 1),
            "question_gold_rate": stats["questions_with_gold"] / max(len(items), 1),
        }
        for route, stats in sorted(route_gold.items())
    }
    return {
        "num_questions": len(items),
        "route_pattern_counts": dict(route_counts.most_common()),
        "route_gold_summary": route_gold_summary,
        "card_gold_summary": card_gold,
        "hyp_entropy": {
            "avg_event_entropy": mean(row["hyp_event_entropy"] for row in entropy_rows),
            "avg_topic_entropy": mean(row["hyp_topic_entropy"] for row in entropy_rows),
            "avg_unique_hyp_events": mean(row["hyp_unique_events"] for row in entropy_rows),
            "avg_unique_hyp_topics": mean(row["hyp_unique_topics"] for row in entropy_rows),
        },
        "per_question_sample": per_question[:20],
    }


def build_rerank_methods(rrf_k: float):
    return {
        "ce_only": lambda paths: score_paths(paths, rrf_k, route_weight=0.0, card_weight=0.0),
        "ce_route_prior_w02": lambda paths: score_paths(paths, rrf_k, route_weight=0.2, card_weight=0.0),
        "ce_route_prior_w04": lambda paths: score_paths(paths, rrf_k, route_weight=0.4, card_weight=0.0),
        "ce_route_card_w02_c02": lambda paths: score_paths(paths, rrf_k, route_weight=0.2, card_weight=0.2),
        "ce_route_card_w04_c02": lambda paths: score_paths(paths, rrf_k, route_weight=0.4, card_weight=0.2),
        "ce_route_card_w04_c04": lambda paths: score_paths(paths, rrf_k, route_weight=0.4, card_weight=0.4),
        "ce_route_card_w04_c02_mmr": lambda paths: score_paths(
            paths, rrf_k, route_weight=0.4, card_weight=0.2, diversity_penalty=0.05
        ),
    }


def score_paths(
    paths: list[dict],
    rrf_k: float,
    *,
    route_weight: float,
    card_weight: float,
    diversity_penalty: float = 0.0,
) -> list[tuple[dict, float]]:
    ce_values = [ce_score(path) for path in paths]
    ce_norm = minmax(ce_values)
    route_values = [route_prior(path.get("metadata", {}), rrf_k) for path in paths]
    route_norm = minmax(route_values)
    card_values = [card_signal(path.get("metadata", {}), path.get("scores", {})) for path in paths]
    card_norm = minmax(card_values)
    scored = []
    for path, ce, route, card in zip(paths, ce_norm, route_norm, card_norm):
        score = ce + route_weight * route + card_weight * card
        scored.append((path, score))
    scored.sort(key=lambda row: row[1], reverse=True)
    if diversity_penalty <= 0:
        return scored
    return greedy_diversify(scored, diversity_penalty)


def rank_items(items: list[dict], scorer, topn: int) -> dict[str, list[dict]]:
    output = {}
    for item in items:
        paths = list(item.get("paths", []))[:topn]
        scored = scorer(paths)
        reranked = []
        for rank, (path, score) in enumerate(scored, start=1):
            reranked.append({"path": path, "score": float(score), "rank": rank})
        output[item["question_id"]] = reranked
    return output


def materialize_ranked_items(items: list[dict], ranked_by_qid: dict[str, list[dict]]) -> list[dict]:
    output = []
    for item in items:
        copied = dict(item)
        paths = []
        for row in ranked_by_qid.get(item["question_id"], []):
            path = dict(row["path"])
            scores = dict(path.get("scores", {}))
            scores["v3_9_route_fusion_score"] = row["score"]
            path["scores"] = scores
            meta = dict(path.get("metadata", {}))
            meta["v3_9_route_fusion_rank"] = row["rank"]
            path["metadata"] = meta
            paths.append(path)
        copied["paths"] = paths
        output.append(copied)
    return output


def evaluate_ranked_items(
    items: list[dict],
    ranked_by_qid: dict[str, list[dict]],
    topk: int,
    evidence_cache: dict[str, set[str]],
) -> list[RetrievalEvalResult]:
    results = []
    for item in items:
        gold = gold_set(item)
        predicted = set()
        selected_node_ids = []
        rows = ranked_by_qid.get(item["question_id"], [])[:topk]
        for row in rows:
            path = row["path"]
            fact_id = evidence_node_id(path)
            if fact_id and fact_id not in selected_node_ids:
                selected_node_ids.append(fact_id)
            predicted.update(path_gold_ids(None, path, evidence_cache))
        matched = sorted(gold & predicted)
        recall = len(matched) / len(gold) if gold else 0.0
        results.append(
            RetrievalEvalResult(
                question_id=item["question_id"],
                gold_evidence_ids=sorted(gold),
                selected_path_node_ids=selected_node_ids,
                matched_evidence_ids=matched,
                hit=bool(matched),
                recall=recall,
                full_cover=bool(gold) and gold.issubset(predicted),
                tokens=0,
                path_len=0.0,
            )
        )
    return results


def route_prior(meta: dict, rrf_k: float) -> float:
    # Rank-based normalization keeps Euclidean and hyperbolic scores on the same scale.
    score = 0.0
    score += 1.00 * rrf(meta, ["bottom_up_rank"], rrf_k)
    score += 0.80 * rrf(meta, ["bottom_hyp_rank", "hyp_bottom_seed_rank"], rrf_k)
    score += 1.00 * rrf(meta, ["eu_route_rank", "eu_event_rank", "eu_topic_rank"], rrf_k)
    score += 1.20 * rrf(meta, ["hyp_route_rank", "hyp_event_rank", "hyp_topic_rank"], rrf_k)
    routes = route_names(route_flags(meta))
    if len(routes) >= 3:
        score += 0.02 * len(routes)
    return score


def rrf(meta: dict, keys: list[str], k: float) -> float:
    values = []
    for key in keys:
        value = parse_float(meta.get(key))
        if value is not None and value >= 0:
            values.append(value)
    if not values:
        return 0.0
    rank = min(values)
    return 1.0 / (k + rank)


def card_signal(meta: dict, scores: dict) -> float:
    signal = 0.0
    if is_card_member(meta):
        signal += 1.0
    if is_card_expansion(meta):
        signal += 0.5
    signal += 0.5 * (parse_float(meta.get("nary_hyperedge_confidence")) or 0.0)
    signal += 0.5 * (parse_float(scores.get("nary_card_confidence")) or 0.0)
    signal += 0.25 * (parse_float(scores.get("nary_needed_role_score")) or 0.0)
    return signal


def greedy_diversify(scored: list[tuple[dict, float]], penalty: float) -> list[tuple[dict, float]]:
    remaining = list(scored)
    selected = []
    used_events = Counter()
    used_topics = Counter()
    while remaining:
        best_index = 0
        best_adjusted = float("-inf")
        for index, (path, score) in enumerate(remaining):
            meta = path.get("metadata", {})
            event_id = str(meta.get("event_node_id", "") or "")
            topic_id = str(meta.get("topic_node_id", "") or "")
            adjusted = score
            if event_id:
                adjusted -= penalty * used_events[event_id]
            if topic_id:
                adjusted -= 0.5 * penalty * used_topics[topic_id]
            if adjusted > best_adjusted:
                best_index = index
                best_adjusted = adjusted
        path, score = remaining.pop(best_index)
        selected.append((path, score))
        meta = path.get("metadata", {})
        event_id = str(meta.get("event_node_id", "") or "")
        topic_id = str(meta.get("topic_node_id", "") or "")
        if event_id:
            used_events[event_id] += 1
        if topic_id:
            used_topics[topic_id] += 1
    return selected


def selected_route_summary(
    items: list[dict],
    ranked_by_qid: dict[str, list[dict]],
    topk: int,
    evidence_cache: dict[str, set[str]],
) -> dict:
    rows = Counter()
    gold_rows = Counter()
    for item in items:
        gold = gold_set(item)
        for row in ranked_by_qid.get(item["question_id"], [])[:topk]:
            path = row["path"]
            meta = path.get("metadata", {})
            pattern = route_pattern(route_flags(meta))
            rows[pattern] += 1
            is_gold = bool(path_gold_ids(None, path, evidence_cache) & gold)
            if is_gold:
                gold_rows[pattern] += 1
            if is_card_member(meta):
                rows["card_member"] += 1
                gold_rows["card_member"] += int(is_gold)
            if is_card_expansion(meta):
                rows["card_expansion"] += 1
                gold_rows["card_expansion"] += int(is_gold)
    return {
        "selected": dict(rows),
        "selected_gold": dict(gold_rows),
        "gold_rate": {key: gold_rows[key] / max(value, 1) for key, value in rows.items()},
    }


def route_flags(meta: dict) -> dict[str, bool]:
    route_source = str(meta.get("route_source", "") or "")
    return {
        "bottom_up": bool(meta.get("bottom_up_rank") is not None or "bottom_up" in route_source),
        "bottom_hyp": bool(
            meta.get("bottom_hyp_rank") is not None
            or meta.get("hyp_bottom_score") is not None
            or "hyp_bottom" in route_source
        ),
        "top_eu": bool(
            meta.get("eu_event_rank") is not None
            or meta.get("eu_topic_rank") is not None
            or "eu_event" in route_source
            or "eu_topic" in route_source
        ),
        "top_hyp": bool(
            meta.get("hyp_event_rank") is not None
            or meta.get("hyp_topic_rank") is not None
            or "hyp_event" in route_source
            or "hyp_topic" in route_source
        ),
    }


def route_names(flags: dict[str, bool]) -> list[str]:
    return [name for name, value in flags.items() if value]


def route_pattern(flags: dict[str, bool]) -> str:
    names = route_names(flags)
    return "+".join(names) if names else "none"


def is_card_member(meta: dict) -> bool:
    return str(meta.get("v3_9_query_card", "")).lower() == "true" or str(
        meta.get("is_nary_completion", "")
    ).lower() == "true"


def is_card_expansion(meta: dict) -> bool:
    return str(meta.get("v3_9_card_guided_expansion", "")).lower() == "true"


def ce_score(path: dict) -> float:
    scores = path.get("scores", {})
    value = parse_float(scores.get("cross_encoder"))
    if value is not None:
        return value
    return parse_float(path.get("score")) or 0.0


def parse_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def normalized_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(len(counter))


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(value) for value in item.get("gold_evidence", []) or item.get("gold_supports", []) or []}


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def path_gold_ids(graph, path: dict, evidence_cache: dict[str, set[str]]) -> set[str]:
    fact_id = evidence_node_id(path)
    if fact_id:
        return evidence_cache.get(fact_id, set())
    predicted = set()
    for node_id in path.get("node_ids", []):
        predicted.update(evidence_ids_for_node(graph, node_id))
    return predicted


def render_markdown(summary: dict) -> str:
    lines = [
        "# V3.9 Route Stability And Fusion",
        "",
        "## Rerank Results",
        "",
        "| Method | Hit@5 | Recall@5 | FullCover@5 | Hit@20 | Recall@20 | FullCover@20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, by_k in summary["aggregate"].items():
        top5 = by_k.get("top5", {})
        top20 = by_k.get("top20", {})
        lines.append(
            f"| {method} | {top5.get('hit', 0.0):.4f} | {top5.get('recall', 0.0):.4f} | "
            f"{top5.get('full_cover', 0.0):.4f} | {top20.get('hit', 0.0):.4f} | "
            f"{top20.get('recall', 0.0):.4f} | {top20.get('full_cover', 0.0):.4f} |"
        )
    diag = summary["diagnostics"]
    lines.extend(
        [
            "",
            "## Hyp Route Diversity",
            "",
            f"- Avg hyp event entropy: {diag['hyp_entropy']['avg_event_entropy']:.4f}",
            f"- Avg hyp topic entropy: {diag['hyp_entropy']['avg_topic_entropy']:.4f}",
            f"- Avg unique hyp events: {diag['hyp_entropy']['avg_unique_hyp_events']:.2f}",
            f"- Avg unique hyp topics: {diag['hyp_entropy']['avg_unique_hyp_topics']:.2f}",
            "",
            "## Route Gold Summary",
            "",
            "| Route | Available Facts | Gold Facts | Gold Rate | Questions With Gold |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for route, stats in diag["route_gold_summary"].items():
        lines.append(
            f"| {route} | {stats['available']} | {stats['gold_facts']} | "
            f"{stats['gold_rate']:.4f} | {stats['questions_with_gold']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
