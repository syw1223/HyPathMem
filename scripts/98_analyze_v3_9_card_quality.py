from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evaluate_item, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument(
        "--cardce-paths",
        default="outputs/eval/v3_9_cardce_guided_ctx50/cardce_guided_topk_paths.json",
    )
    parser.add_argument("--base-cv-dir", default="outputs/eval/cv/nary_v3_6c_selector_base100_top20")
    parser.add_argument("--base-method", default="base_completion_no_nary_features")
    parser.add_argument("--output-json", default="outputs/eval/V3_9_CARD_QUALITY_ANALYSIS.json")
    parser.add_argument("--output-md", default="outputs/eval/V3_9_CARD_QUALITY_ANALYSIS.md")
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    card_items = read_json(resolve_path(args.cardce_paths))
    base_items = load_cv_paths(resolve_path(args.base_cv_dir), args.base_method)
    base_by_qid = {item["question_id"]: item for item in base_items}

    payload = {
        "card_retrieval": analyze_card_retrieval(graph, card_items),
        "fix_regress": {
            f"top{k}": analyze_fix_regress(graph, base_by_qid, card_items, k)
            for k in (5, 20)
        },
    }
    write_json(payload, resolve_path(args.output_json))
    resolve_path(args.output_md).write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))


def analyze_card_retrieval(graph, items: list[dict]) -> dict:
    top1_hit = top3_hit = 0
    questions_with_cards = 0
    total_card_facts = total_card_gold = 0
    type_facts = Counter()
    type_gold = Counter()
    for item in items:
        gold = gold_set(item)
        cards = grouped_cards(item)
        if cards:
            questions_with_cards += 1
        ranked = sorted(cards.values(), key=lambda row: row["score"], reverse=True)
        if ranked and card_has_gold(graph, ranked[0], gold):
            top1_hit += 1
        if ranked and any(card_has_gold(graph, card, gold) for card in ranked[:3]):
            top3_hit += 1
        for card in ranked:
            for fact_id in card["facts"]:
                total_card_facts += 1
                type_facts[card["type"]] += 1
                if fact_evidence(graph, fact_id) & gold:
                    total_card_gold += 1
                    type_gold[card["type"]] += 1
    return {
        "questions": len(items),
        "questions_with_cards": questions_with_cards,
        "cardce_top1_contains_gold": top1_hit,
        "cardce_top1_contains_gold_rate": top1_hit / max(len(items), 1),
        "cardce_top3_contains_gold": top3_hit,
        "cardce_top3_contains_gold_rate": top3_hit / max(len(items), 1),
        "card_support_facts": total_card_facts,
        "card_support_gold_facts": total_card_gold,
        "card_support_gold_rate": total_card_gold / max(total_card_facts, 1),
        "type_fact_counts": dict(type_facts),
        "type_gold_counts": dict(type_gold),
        "type_gold_rates": {
            relation_type: type_gold[relation_type] / max(count, 1)
            for relation_type, count in type_facts.items()
        },
    }


def analyze_fix_regress(graph, base_by_qid: dict[str, dict], card_items: list[dict], k: int) -> dict:
    hit_fixed_types = Counter()
    hit_regressed_types = Counter()
    full_fixed_types = Counter()
    full_regressed_types = Counter()
    counts = Counter()
    for item in card_items:
        base = base_by_qid.get(item["question_id"])
        if base is None:
            continue
        base_eval = evaluate_item(graph, base, k)
        card_eval = evaluate_item(graph, item, k)
        selected_types = selected_card_types(item, k)
        gold_types = selected_gold_card_types(graph, item, k)
        if not base_eval.hit and card_eval.hit:
            counts["hit_fixed"] += 1
            hit_fixed_types.update(gold_types or selected_types)
        if base_eval.hit and not card_eval.hit:
            counts["hit_regressed"] += 1
            hit_regressed_types.update(selected_types)
        if not base_eval.full_cover and card_eval.full_cover:
            counts["full_fixed"] += 1
            full_fixed_types.update(gold_types or selected_types)
        if base_eval.full_cover and not card_eval.full_cover:
            counts["full_regressed"] += 1
            full_regressed_types.update(selected_types)
    return {
        **dict(counts),
        "hit_fixed_types": dict(hit_fixed_types),
        "hit_regressed_types": dict(hit_regressed_types),
        "full_fixed_types": dict(full_fixed_types),
        "full_regressed_types": dict(full_regressed_types),
    }


def grouped_cards(item: dict) -> dict[str, dict]:
    cards = {}
    for path in item.get("paths", []):
        metadata = path.get("metadata", {})
        if str(metadata.get("v3_9_cardce_selected", "")).lower() != "true":
            continue
        card_id = str(metadata.get("nary_hyperedge_id") or "")
        if not card_id:
            continue
        row = cards.setdefault(
            card_id,
            {
                "score": float(path.get("scores", {}).get("v3_9_card_ce", 0.0) or 0.0),
                "type": str(metadata.get("v3_9_cardce_type") or metadata.get("nary_hyperedge_type") or "none"),
                "facts": set(),
            },
        )
        row["score"] = max(row["score"], float(path.get("scores", {}).get("v3_9_card_ce", 0.0) or 0.0))
        row["facts"].add(evidence_node_id(path))
    return cards


def card_has_gold(graph, card: dict, gold: set[str]) -> bool:
    return any(fact_evidence(graph, fact_id) & gold for fact_id in card["facts"])


def selected_card_types(item: dict, k: int) -> list[str]:
    return [
        str(path.get("metadata", {}).get("v3_9_cardce_type") or path.get("metadata", {}).get("nary_hyperedge_type") or "none")
        for path in item.get("paths", [])[:k]
        if str(path.get("metadata", {}).get("v3_9_cardce_selected", "")).lower() == "true"
    ]


def selected_gold_card_types(graph, item: dict, k: int) -> list[str]:
    gold = gold_set(item)
    return [
        str(path.get("metadata", {}).get("v3_9_cardce_type") or path.get("metadata", {}).get("nary_hyperedge_type") or "none")
        for path in item.get("paths", [])[:k]
        if str(path.get("metadata", {}).get("v3_9_cardce_selected", "")).lower() == "true"
        and fact_evidence(graph, evidence_node_id(path)) & gold
    ]


def load_cv_paths(cv_dir: Path, method: str) -> list[dict]:
    items = []
    for path in sorted(cv_dir.glob(f"fold_*/*{method}_paths.json")):
        items.extend(read_json(path))
    return items


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def fact_evidence(graph, fact_id: str) -> set[str]:
    node = graph.nodes.get(fact_id)
    if node is None:
        return set()
    evidence = {normalize_evidence_id(eid) for eid in node.support_ids}
    return evidence or {normalize_evidence_id(fact_id)}


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}


def render_markdown(payload: dict) -> str:
    retrieval = payload["card_retrieval"]
    lines = [
        "# V3.9 Card Quality Analysis",
        "",
        f"- CardCE Top1 contains gold: {retrieval['cardce_top1_contains_gold']} "
        f"({retrieval['cardce_top1_contains_gold_rate']:.4f})",
        f"- CardCE Top3 contains gold: {retrieval['cardce_top3_contains_gold']} "
        f"({retrieval['cardce_top3_contains_gold_rate']:.4f})",
        f"- Card support fact gold rate: {retrieval['card_support_gold_rate']:.4f}",
        "",
        "## Per Type",
        "",
        "| Type | Facts | Gold | GoldRate |",
        "|---|---:|---:|---:|",
    ]
    for relation_type, count in sorted(retrieval["type_fact_counts"].items()):
        gold = retrieval["type_gold_counts"].get(relation_type, 0)
        rate = retrieval["type_gold_rates"].get(relation_type, 0.0)
        lines.append(f"| {relation_type} | {count} | {gold} | {rate:.4f} |")
    for k, row in payload["fix_regress"].items():
        lines.extend(
            [
                "",
                f"## {k} Fix/Regress",
                "",
                f"- Hit fixed/regressed: {row.get('hit_fixed', 0)} / {row.get('hit_regressed', 0)}",
                f"- FullCover fixed/regressed: {row.get('full_fixed', 0)} / {row.get('full_regressed', 0)}",
                f"- Hit fixed types: `{row.get('hit_fixed_types', {})}`",
                f"- Hit regressed types: `{row.get('hit_regressed_types', {})}`",
                f"- FullCover fixed types: `{row.get('full_fixed_types', {})}`",
                f"- FullCover regressed types: `{row.get('full_regressed_types', {})}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
