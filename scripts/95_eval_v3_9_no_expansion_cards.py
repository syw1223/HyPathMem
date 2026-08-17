from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evaluate_item, normalize_evidence_id, summarize
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topology_features import rerank_items_with_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--cards", required=True)
    parser.add_argument("--base-cv-dir", default="outputs/eval/cv/nary_v3_6c_selector_base100_top20")
    parser.add_argument("--base-method", default="base_completion_no_nary_features")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--card-alpha", type=float, default=0.18)
    parser.add_argument("--role-beta", type=float, default=0.05)
    parser.add_argument("--size-gamma", type=float, default=0.02)
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    card_items = read_json(resolve_path(args.cards))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods: dict[str, list[dict]] = {
        "base_ce_topk": [truncate_item(item, max(args.topk), rank_by_ce(item.get("paths", []))) for item in card_items],
        "card_guided_topk": [card_guided_item(item, max(args.topk)) for item in card_items],
        "card_bonus_topk": [
            truncate_item(item, max(args.topk), rank_by_card_bonus(item.get("paths", []), args)) for item in card_items
        ],
    }

    base_lgbm = load_cv_paths(resolve_path(args.base_cv_dir), args.base_method)
    if base_lgbm:
        methods["base_lgbm_topk"] = base_lgbm

    payload = {
        "metadata": {
            "graph": str(resolve_path(args.graph)),
            "cards": str(resolve_path(args.cards)),
            "base_cv_dir": str(resolve_path(args.base_cv_dir)),
            "base_method": args.base_method,
            "topk": args.topk,
            "card_alpha": args.card_alpha,
            "role_beta": args.role_beta,
            "size_gamma": args.size_gamma,
        },
        "retrieval": {},
        "card_pool": card_pool_summary(graph, card_items),
        "selected_card_analysis": {},
        "fixes_vs_base_lgbm": {},
    }

    for method_name, items in methods.items():
        write_json(items, output_dir / f"{method_name}_paths.json")
        payload["retrieval"][method_name] = {}
        payload["selected_card_analysis"][method_name] = {}
        for k in args.topk:
            eval_payload = evaluate_items(graph, items, k, method_name)
            write_json(eval_payload, output_dir / f"{method_name}_top{k}_eval.json")
            payload["retrieval"][method_name][f"top{k}"] = eval_payload["summary"]
            payload["selected_card_analysis"][method_name][f"top{k}"] = selected_card_summary(graph, items, k)

    if "base_lgbm_topk" in methods:
        base_items = methods["base_lgbm_topk"]
        for method_name, items in methods.items():
            if method_name == "base_lgbm_topk":
                continue
            payload["fixes_vs_base_lgbm"][method_name] = {}
            for k in args.topk:
                payload["fixes_vs_base_lgbm"][method_name][f"top{k}"] = compare_to_base(
                    graph, base_items, items, k
                )

    write_json(payload, output_dir / "v3_9_no_expansion_card_eval.json")
    (output_dir / "v3_9_no_expansion_card_eval.md").write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    print(f"wrote {output_dir / 'v3_9_no_expansion_card_eval.md'}")


def evaluate_items(graph, items: list[dict], k: int, method: str) -> dict:
    rows = [evaluate_item(graph, item, k) for item in items]
    return {"method": method, "k": k, "summary": summarize(rows), "per_question": [row.__dict__ for row in rows]}


def truncate_item(item: dict, topk: int, paths: list[dict]) -> dict:
    copied = dict(item)
    copied["paths"] = paths[:topk]
    return copied


def rank_by_ce(paths: list[dict]) -> list[dict]:
    return sorted((dict(path) for path in paths), key=ce_score, reverse=True)


def rank_by_card_bonus(paths: list[dict], args) -> list[dict]:
    ranked = []
    for path in paths:
        metadata = path.get("metadata", {})
        bonus = 0.0
        if is_card_fact(path):
            bonus += args.card_alpha * float_meta(metadata, "nary_hyperedge_confidence")
            bonus += args.role_beta * float_meta(metadata, "nary_role_confidence")
            bonus += args.size_gamma * min(float_meta(metadata, "nary_same_hyperedge_count_in_candidate_pool"), 5.0)
        copied = dict(path)
        scores = dict(copied.get("scores", {}))
        scores["v3_9_card_bonus_score"] = ce_score(path) + bonus
        copied["scores"] = scores
        ranked.append(copied)
    return sorted(ranked, key=lambda path: path["scores"]["v3_9_card_bonus_score"], reverse=True)


def card_guided_item(item: dict, topk: int) -> dict:
    paths = [dict(path) for path in item.get("paths", [])]
    grouped = defaultdict(list)
    for path in paths:
        metadata = path.get("metadata", {})
        card_id = metadata.get("nary_hyperedge_id")
        if is_card_fact(path) and card_id:
            grouped[str(card_id)].append(path)

    selected = []
    seen = set()
    for _, card_paths in sorted(grouped.items(), key=lambda row: card_group_score(row[1]), reverse=True):
        for path in sorted(card_paths, key=card_fact_score, reverse=True):
            fact_id = evidence_node_id(path)
            if fact_id and fact_id not in seen:
                selected.append(path)
                seen.add(fact_id)
            if len(selected) >= topk:
                break
        if len(selected) >= topk:
            break

    for path in rank_by_ce(paths):
        fact_id = evidence_node_id(path)
        if fact_id and fact_id not in seen:
            selected.append(path)
            seen.add(fact_id)
        if len(selected) >= topk:
            break

    copied = dict(item)
    copied["paths"] = selected
    metadata = dict(copied.get("metadata", {}))
    metadata["method"] = "v3_9_card_guided_no_expansion"
    copied["metadata"] = metadata
    return copied


def card_group_score(paths: list[dict]) -> float:
    if not paths:
        return 0.0
    confidences = [float_meta(path.get("metadata", {}), "nary_hyperedge_confidence") for path in paths]
    role_count = max(float_meta(path.get("metadata", {}), "nary_pool_covered_roles_count") for path in paths)
    return max(ce_score(path) for path in paths) + 0.18 * max(confidences) + 0.03 * min(role_count, 5.0)


def card_fact_score(path: dict) -> float:
    metadata = path.get("metadata", {})
    return (
        ce_score(path)
        + 0.12 * float_meta(metadata, "nary_hyperedge_confidence")
        + 0.05 * float_meta(metadata, "nary_role_confidence")
    )


def card_pool_summary(graph, items: list[dict]) -> dict:
    total_card_facts = 0
    total_card_gold = 0
    questions_with_card = 0
    questions_with_gold_card = 0
    type_counts = Counter()
    type_gold = Counter()
    for item in items:
        gold = gold_set(item)
        question_card = 0
        question_gold = 0
        seen_facts = set()
        for path in item.get("paths", []):
            if not is_card_fact(path):
                continue
            fact_id = evidence_node_id(path)
            if not fact_id or fact_id in seen_facts:
                continue
            seen_facts.add(fact_id)
            question_card += 1
            total_card_facts += 1
            relation_type = str(path.get("metadata", {}).get("nary_hyperedge_type") or "none")
            type_counts[relation_type] += 1
            if fact_evidence(graph, fact_id) & gold:
                question_gold += 1
                total_card_gold += 1
                type_gold[relation_type] += 1
        if question_card:
            questions_with_card += 1
        if question_gold:
            questions_with_gold_card += 1
    return {
        "card_facts": total_card_facts,
        "card_gold_facts": total_card_gold,
        "card_fact_gold_rate": total_card_gold / max(total_card_facts, 1),
        "questions_with_card": questions_with_card,
        "questions_with_gold_card": questions_with_gold_card,
        "type_counts": dict(type_counts),
        "type_gold": dict(type_gold),
    }


def selected_card_summary(graph, items: list[dict], k: int) -> dict:
    total_selected = 0
    selected_card = 0
    selected_card_gold = 0
    questions_with_card = 0
    questions_with_gold_card = 0
    type_counts = Counter()
    type_gold = Counter()
    for item in items:
        gold = gold_set(item)
        question_card = 0
        question_gold = 0
        for path in item.get("paths", [])[:k]:
            total_selected += 1
            if not is_card_fact(path):
                continue
            question_card += 1
            selected_card += 1
            relation_type = str(path.get("metadata", {}).get("nary_hyperedge_type") or "none")
            type_counts[relation_type] += 1
            if fact_evidence(graph, evidence_node_id(path)) & gold:
                question_gold += 1
                selected_card_gold += 1
                type_gold[relation_type] += 1
        if question_card:
            questions_with_card += 1
        if question_gold:
            questions_with_gold_card += 1
    return {
        "avg_selected_card_facts": selected_card / max(len(items), 1),
        "selected_card_share": selected_card / max(total_selected, 1),
        "selected_card_facts": selected_card,
        "selected_card_gold_facts": selected_card_gold,
        "selected_card_gold_rate": selected_card_gold / max(selected_card, 1),
        "questions_with_card": questions_with_card,
        "questions_with_gold_card": questions_with_gold_card,
        "type_counts": dict(type_counts),
        "type_gold": dict(type_gold),
    }


def compare_to_base(graph, base_items: list[dict], method_items: list[dict], k: int) -> dict:
    base_by_qid = {item["question_id"]: item for item in base_items}
    hit_fixed = 0
    full_fixed = 0
    hit_regressed = 0
    full_regressed = 0
    fixed_with_card_gold = 0
    examples = []
    for item in method_items:
        base = base_by_qid.get(item["question_id"])
        if base is None:
            continue
        base_eval = evaluate_item(graph, base, k)
        method_eval = evaluate_item(graph, item, k)
        card_gold = [
            path
            for path in item.get("paths", [])[:k]
            if is_card_fact(path) and (fact_evidence(graph, evidence_node_id(path)) & gold_set(item))
        ]
        if not base_eval.hit and method_eval.hit:
            hit_fixed += 1
            if card_gold:
                fixed_with_card_gold += 1
        if base_eval.hit and not method_eval.hit:
            hit_regressed += 1
        if not base_eval.full_cover and method_eval.full_cover:
            full_fixed += 1
        if base_eval.full_cover and not method_eval.full_cover:
            full_regressed += 1
        if card_gold and len(examples) < 20 and ((not base_eval.hit and method_eval.hit) or (not base_eval.full_cover and method_eval.full_cover)):
            examples.append(
                {
                    "question_id": item["question_id"],
                    "question": item.get("question", ""),
                    "base_hit": base_eval.hit,
                    "method_hit": method_eval.hit,
                    "base_full_cover": base_eval.full_cover,
                    "method_full_cover": method_eval.full_cover,
                    "card_gold": [path_payload(graph, path) for path in card_gold],
                }
            )
    return {
        "hit_fixed": hit_fixed,
        "hit_regressed": hit_regressed,
        "full_cover_fixed": full_fixed,
        "full_cover_regressed": full_regressed,
        "fixed_with_card_gold": fixed_with_card_gold,
        "examples": examples,
    }


def load_cv_paths(cv_dir: Path, method: str) -> list[dict]:
    paths = []
    for path in sorted(cv_dir.glob(f"fold_*/*{method}_paths.json")):
        paths.extend(read_json(path))
    if paths:
        return sorted(paths, key=lambda item: item["question_id"])
    return []


def path_payload(graph, path: dict) -> dict:
    fact_id = evidence_node_id(path)
    node = graph.nodes.get(fact_id)
    metadata = path.get("metadata", {})
    return {
        "fact_id": fact_id,
        "text": node.text if node else "",
        "type": metadata.get("nary_hyperedge_type", ""),
        "role": metadata.get("nary_role", ""),
        "card_id": metadata.get("nary_hyperedge_id", ""),
        "summary": metadata.get("v3_9_card_summary", ""),
        "ce": ce_score(path),
    }


def is_card_fact(path: dict) -> bool:
    metadata = path.get("metadata", {})
    return (
        str(metadata.get("v3_9_query_card", "")).lower() == "true"
        or "v3_9_query_card" in str(metadata.get("nary_extractor_type", "")).lower()
    )


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def ce_score(path: dict) -> float:
    scores = path.get("scores", {})
    return float(scores.get("cross_encoder", path.get("score", 0.0)) or 0.0)


def float_meta(metadata: dict, key: str) -> float:
    try:
        return float(metadata.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def fact_evidence(graph, fact_id: str) -> set[str]:
    node = graph.nodes.get(fact_id)
    if node is None:
        return set()
    out = {normalize_evidence_id(eid) for eid in node.support_ids}
    if not out:
        out.add(normalize_evidence_id(fact_id))
    return out


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}


def render_markdown(payload: dict) -> str:
    lines = ["# V3.9 No-expansion Relation Card Evaluation", ""]
    lines.extend(
        [
            "## Retrieval",
            "",
            "| Method | K | Hit | Recall | FullCover |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method, by_k in payload["retrieval"].items():
        for k, row in by_k.items():
            lines.append(f"| {method} | {k} | {row['hit']:.4f} | {row['recall']:.4f} | {row['full_cover']:.4f} |")
    pool = payload["card_pool"]
    lines.extend(
        [
            "",
            "## Card Pool",
            "",
            f"- card_facts: {pool['card_facts']}",
            f"- card_gold_facts: {pool['card_gold_facts']}",
            f"- card_fact_gold_rate: {pool['card_fact_gold_rate']:.4f}",
            f"- questions_with_card: {pool['questions_with_card']}",
            f"- questions_with_gold_card: {pool['questions_with_gold_card']}",
            "",
            "## Selected Card Facts",
            "",
            "| Method | K | AvgCardFacts | CardShare | CardGoldRate | QuestionsWithGoldCard |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, by_k in payload["selected_card_analysis"].items():
        for k, row in by_k.items():
            lines.append(
                f"| {method} | {k} | {row['avg_selected_card_facts']:.3f} | "
                f"{row['selected_card_share']:.3f} | {row['selected_card_gold_rate']:.4f} | "
                f"{row['questions_with_gold_card']} |"
            )
    if payload["fixes_vs_base_lgbm"]:
        lines.extend(
            [
                "",
                "## Fixes vs Base LightGBM",
                "",
                "| Method | K | HitFixed | HitRegressed | FullFixed | FullRegressed | FixedWithCardGold |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, by_k in payload["fixes_vs_base_lgbm"].items():
            for k, row in by_k.items():
                lines.append(
                    f"| {method} | {k} | {row['hit_fixed']} | {row['hit_regressed']} | "
                    f"{row['full_cover_fixed']} | {row['full_cover_regressed']} | {row['fixed_with_card_gold']} |"
                )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
