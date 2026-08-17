from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evaluate_item, summarize
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topdown_semantic_retriever import DEFAULT_LOCAL_EMBEDDER, SentenceTransformerEncoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose level-preserving propagation over V3.9 query relation-card members."
    )
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--candidates", default="outputs/v3_9_query_cards/qwen3_card_guided_expand120.json")
    parser.add_argument("--baseline", default="outputs/paths/full_v3_9_card_guided_expand120_light_quota_top20.json")
    parser.add_argument("--embeddings", default="outputs/embeddings/graph_v4_1_card_hybrid_fact_card_event_topic.npz")
    parser.add_argument("--embedder", default=DEFAULT_LOCAL_EMBEDDER)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4, 0.8])
    parser.add_argument("--levels", nargs="+", choices=["fact", "event", "topic"], default=["fact", "event", "topic"])
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/eval/v3_9_card_fact_hprop")
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    candidates = read_json(resolve_path(args.candidates))
    baseline = read_json(resolve_path(args.baseline)) if args.baseline else []
    if args.max_questions:
        candidates = candidates[: args.max_questions]
        baseline = baseline[: args.max_questions]
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_ids, matrix = load_embedding_cache(resolve_path(args.embeddings))
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    query_matrix = encode_queries(candidates, args.embedder, args.device, args.batch_size)

    summary = {
        "method": "V3.9 relation-card level-preserving hyperedge propagation diagnostic",
        "graph": str(resolve_path(args.graph)),
        "candidates": str(resolve_path(args.candidates)),
        "baseline": str(resolve_path(args.baseline)) if args.baseline else "",
        "embeddings": str(resolve_path(args.embeddings)),
        "num_questions": len(candidates),
        "card_stats": card_stats(candidates),
        "aggregate": {},
        "paired_hit": {},
        "best_by_full_cover_top20": None,
    }

    method_items: dict[str, list[dict]] = {}
    method_items["candidate_ce"] = rerank_by_ce(candidates)
    method_items["dense_original"] = rerank_by_original_dense(candidates, matrix, node_to_idx, query_matrix)
    for level in args.levels:
        for lam in args.lambdas:
            propagated_scores = compute_hprop_scores(candidates, matrix, node_to_idx, query_matrix, lam, level)
            method_items[f"hprop_{level}_only_l{lam:g}"] = rerank_by_vector_score(
                candidates, propagated_scores, f"hprop_{level}_only_l{lam:g}"
            )
            for alpha in args.alphas:
                name = f"ce_plus_hprop_{level}_l{lam:g}_a{alpha:g}"
                method_items[name] = rerank_by_ce_plus_hprop(candidates, propagated_scores, alpha, name)

    eval_rows_by_method: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    if baseline:
        for k in args.topk:
            payload = evaluate_items(graph, baseline, k, f"baseline_lgbm_cardquota_top{k}")
            summary["aggregate"][f"baseline_lgbm_cardquota_top{k}"] = payload["summary"]
            eval_rows_by_method["baseline_lgbm_cardquota"][k] = payload["per_question"]
            write_json(payload, out_dir / f"baseline_lgbm_cardquota_top{k}_eval.json")

    for method, items in method_items.items():
        for k in args.topk:
            payload = evaluate_items(graph, items, k, f"{method}_top{k}")
            summary["aggregate"][f"{method}_top{k}"] = payload["summary"]
            eval_rows_by_method[method][k] = payload["per_question"]
            write_json(payload, out_dir / f"{method}_top{k}_eval.json")

    if baseline:
        for method in method_items:
            for k in args.topk:
                summary["paired_hit"][f"{method}_vs_baseline_top{k}"] = paired_hit(
                    eval_rows_by_method[method][k],
                    eval_rows_by_method["baseline_lgbm_cardquota"][k],
                )

    best_name, best_metrics = best_method(summary["aggregate"], suffix="_top20")
    summary["best_by_full_cover_top20"] = {"method": best_name, "metrics": best_metrics}
    write_json(summary, out_dir / "summary.json")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary), flush=True)


def load_embedding_cache(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    node_ids = [str(x) for x in data["node_ids"]]
    matrix = np.asarray(data["embeddings"], dtype=np.float32)
    matrix = l2_normalize(matrix)
    return node_ids, matrix


def encode_queries(items: list[dict], embedder: str, device: str | None, batch_size: int) -> np.ndarray:
    encoder = SentenceTransformerEncoder(embedder, device=device, batch_size=batch_size)
    queries = [str(item.get("question") or "") for item in items]
    return l2_normalize(encoder.encode(queries))


def compute_hprop_scores(
    items: list[dict],
    matrix: np.ndarray,
    node_to_idx: dict[str, int],
    query_matrix: np.ndarray,
    lam: float,
    level: str,
) -> list[list[float]]:
    all_scores: list[list[float]] = []
    for qidx, item in enumerate(items):
        paths = item.get("paths", [])
        card_groups = build_card_groups(paths, node_to_idx, level)
        card_vectors = {}
        for key, members in card_groups.items():
            vecs = []
            weights = []
            for path_idx, node_id, weight in members:
                emb_idx = node_to_idx.get(node_id)
                if emb_idx is None:
                    continue
                vecs.append(matrix[emb_idx])
                weights.append(weight)
            if vecs:
                card_vectors[key] = weighted_average(np.asarray(vecs, dtype=np.float32), np.asarray(weights, dtype=np.float32))

        q = query_matrix[qidx]
        scores = []
        for path_idx, path in enumerate(paths):
            node_id = level_node_id(path, level)
            emb_idx = node_to_idx.get(node_id)
            if emb_idx is None:
                scores.append(0.0)
                continue
            vec = matrix[emb_idx]
            key = card_key(path)
            if key and key in card_vectors:
                vec = l2_normalize(((1.0 - lam) * vec + lam * card_vectors[key])[None, :])[0]
            scores.append(float(np.dot(q, vec)))
        all_scores.append(scores)
    return all_scores


def rerank_by_ce(items: list[dict]) -> list[dict]:
    output = []
    for item in items:
        indexed = list(enumerate(item.get("paths", [])))
        ranked = sorted(indexed, key=lambda row: ce_score(row[1]), reverse=True)
        output.append(copy_with_ranked_paths(item, ranked, "candidate_ce", None))
    return output


def rerank_by_original_dense(
    items: list[dict], matrix: np.ndarray, node_to_idx: dict[str, int], query_matrix: np.ndarray
) -> list[dict]:
    output = []
    for qidx, item in enumerate(items):
        q = query_matrix[qidx]
        scores = []
        for path in item.get("paths", []):
            emb_idx = node_to_idx.get(evidence_node_id(path))
            scores.append(float(np.dot(q, matrix[emb_idx])) if emb_idx is not None else 0.0)
        ranked = sorted(enumerate(item.get("paths", [])), key=lambda row: scores[row[0]], reverse=True)
        output.append(copy_with_ranked_paths(item, ranked, "dense_original", scores))
    return output


def rerank_by_vector_score(items: list[dict], vector_scores: list[list[float]], method: str) -> list[dict]:
    output = []
    for item, scores in zip(items, vector_scores):
        ranked = sorted(enumerate(item.get("paths", [])), key=lambda row: scores[row[0]], reverse=True)
        output.append(copy_with_ranked_paths(item, ranked, method, scores))
    return output


def rerank_by_ce_plus_hprop(
    items: list[dict], hprop_scores: list[list[float]], alpha: float, method: str
) -> list[dict]:
    output = []
    for item, hp_scores in zip(items, hprop_scores):
        ces = [ce_score(path) for path in item.get("paths", [])]
        ce_norm = minmax(ces)
        hp_norm = minmax(hp_scores)
        blend = [c + alpha * h for c, h in zip(ce_norm, hp_norm)]
        ranked = sorted(enumerate(item.get("paths", [])), key=lambda row: blend[row[0]], reverse=True)
        output.append(copy_with_ranked_paths(item, ranked, method, blend, hprop_scores=hp_scores))
    return output


def copy_with_ranked_paths(
    item: dict,
    ranked: list[tuple[int, dict]],
    method: str,
    scores: list[float] | None,
    *,
    hprop_scores: list[float] | None = None,
) -> dict:
    copied = dict(item)
    paths = []
    for rank, (idx, path) in enumerate(ranked, start=1):
        new_path = dict(path)
        score_value = float(scores[idx]) if scores is not None else ce_score(path)
        new_path["score"] = score_value
        scores_map = dict(new_path.get("scores", {}))
        scores_map[method] = score_value
        if hprop_scores is not None:
            scores_map["card_fact_hprop"] = float(hprop_scores[idx])
        new_path["scores"] = scores_map
        metadata = dict(new_path.get("metadata", {}))
        metadata[f"{method}_rank"] = str(rank)
        metadata["method"] = method
        new_path["metadata"] = metadata
        paths.append(new_path)
    copied["paths"] = paths
    metadata = dict(copied.get("metadata", {}))
    metadata["method"] = method
    copied["metadata"] = metadata
    return copied


def build_card_groups(paths: list[dict], node_to_idx: dict[str, int], level: str) -> dict[str, list[tuple[int, str, float]]]:
    groups: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for idx, path in enumerate(paths):
        key = card_key(path)
        node_id = level_node_id(path, level)
        if not key or node_id not in node_to_idx:
            continue
        groups[key].append((idx, node_id, card_member_weight(path)))
    return {key: members for key, members in groups.items() if len({node_id for _, node_id, _ in members}) >= 2}


def card_key(path: dict) -> str:
    md = path.get("metadata", {})
    if str(md.get("v3_9_query_card", "")).lower() != "true":
        return ""
    hyperedge = str(md.get("nary_hyperedge_id") or "")
    if not hyperedge:
        return ""
    return "|".join(
        [
            str(path.get("query_id") or ""),
            hyperedge,
            str(md.get("v3_9_card_rank") or ""),
            str(md.get("v3_9_card_summary") or ""),
        ]
    )


def card_member_weight(path: dict) -> float:
    md = path.get("metadata", {})
    return safe_float(md.get("nary_hyperedge_confidence"), 1.0) * safe_float(md.get("nary_role_confidence"), 1.0)


def weighted_average(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(weights, 1e-6)
    vec = (matrix * weights[:, None]).sum(axis=0) / weights.sum()
    return l2_normalize(vec[None, :])[0]


def evidence_node_id(path: dict) -> str:
    md = path.get("metadata", {})
    evidence_id = md.get("evidence_node_id")
    if evidence_id:
        return str(evidence_id)
    node_ids = path.get("node_ids") or []
    return str(node_ids[-1]) if node_ids else ""


def level_node_id(path: dict, level: str) -> str:
    md = path.get("metadata", {})
    if level == "fact":
        return evidence_node_id(path)
    if level == "event":
        return str(md.get("event_node_id") or "")
    if level == "topic":
        return str(md.get("topic_node_id") or "")
    raise ValueError(f"unsupported level: {level}")


def ce_score(path: dict) -> float:
    scores = path.get("scores", {})
    if "cross_encoder" in scores:
        return safe_float(scores["cross_encoder"], 0.0)
    return safe_float(path.get("score"), 0.0)


def safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isfinite(number):
        return number
    return default


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return matrix / denom


def evaluate_items(graph, items: list[dict], k: int, method: str) -> dict:
    results = [evaluate_item(graph, item, k) for item in items]
    return {
        "method": method,
        "k": k,
        "summary": summarize(results),
        "per_question": [result.__dict__ for result in results],
    }


def card_stats(items: list[dict]) -> dict:
    card_paths = 0
    card_groups = set()
    questions_with_card = 0
    for item in items:
        has_card = False
        for path in item.get("paths", []):
            key = card_key(path)
            if key:
                card_paths += 1
                card_groups.add(key)
                has_card = True
        questions_with_card += int(has_card)
    return {
        "questions": len(items),
        "questions_with_card_paths": questions_with_card,
        "card_member_paths": card_paths,
        "unique_query_cards": len(card_groups),
        "avg_card_member_paths_per_question": card_paths / len(items) if items else 0.0,
    }


def paired_hit(left_rows: list[dict], right_rows: list[dict]) -> dict:
    right = {row["question_id"]: row for row in right_rows}
    left_only = right_only = both_hit = both_miss = compared = 0
    for row in left_rows:
        other = right.get(row["question_id"])
        if other is None:
            continue
        compared += 1
        lh = bool(row["hit"])
        rh = bool(other["hit"])
        if lh and rh:
            both_hit += 1
        elif lh:
            left_only += 1
        elif rh:
            right_only += 1
        else:
            both_miss += 1
    return {
        "compared": compared,
        "both_hit": both_hit,
        "left_only": left_only,
        "right_only": right_only,
        "both_miss": both_miss,
        "net_left_minus_right": left_only - right_only,
    }


def best_method(aggregate: dict, suffix: str) -> tuple[str, dict]:
    candidates = [(name, metrics) for name, metrics in aggregate.items() if name.endswith(suffix)]
    if not candidates:
        return "", {}
    return max(candidates, key=lambda row: (row[1]["full_cover"], row[1]["recall"], row[1]["hit"]))


def render_markdown(summary: dict) -> str:
    lines = [
        "# V3.9 Card Level-Preserving Propagation Diagnostic",
        "",
        f"Candidates: `{summary['candidates']}`",
        f"Embeddings: `{summary['embeddings']}`",
        "",
        "## Card Scope",
        "",
        "| Questions | With Card Paths | Card Member Paths | Unique Query Cards | Avg Card Paths/Q |",
        "|---:|---:|---:|---:|---:|",
    ]
    stats = summary["card_stats"]
    lines.append(
        f"| {stats['questions']} | {stats['questions_with_card_paths']} | {stats['card_member_paths']} | "
        f"{stats['unique_query_cards']} | {stats['avg_card_member_paths_per_question']:.2f} |"
    )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Method | N | Hit | Recall | FullCover | AvgTokens |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, metrics in sorted(summary["aggregate"].items()):
        lines.append(
            f"| {method} | {metrics['num_questions']} | {metrics['hit']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['full_cover']:.4f} | {metrics['avg_tokens']:.1f} |"
        )
    best = summary.get("best_by_full_cover_top20") or {}
    if best.get("method"):
        metrics = best["metrics"]
        lines.extend(
            [
                "",
                "## Best Top20",
                "",
                f"`{best['method']}`: Hit {metrics['hit']:.4f}, Recall {metrics['recall']:.4f}, FullCover {metrics['full_cover']:.4f}.",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
