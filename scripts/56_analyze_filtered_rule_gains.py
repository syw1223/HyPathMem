from __future__ import annotations

import argparse
from collections import Counter

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evaluate_item, evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-graph", default="outputs/graphs/locomo_graph_semantic_hierarchy_v3.json")
    parser.add_argument("--new-graph", default="outputs/graphs/locomo_graph_semantic_hierarchy_v3_1_filtered_rule.json")
    parser.add_argument(
        "--base-paths",
        default="outputs/paths/full_graph_v3_event_first_selector_entity_session_loco_cv_top5.json",
    )
    parser.add_argument(
        "--new-paths",
        default="outputs/paths/full_graph_v3_1_filtered_rule_selector_all_without_category_loco_cv_top5.json",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--examples", type=int, default=30)
    parser.add_argument("--output", default="outputs/eval/graph_v3_1_filtered_rule_gain_analysis.json")
    args = parser.parse_args()

    base_graph = JsonGraphStore().load(resolve_path(args.base_graph))
    new_graph = JsonGraphStore().load(resolve_path(args.new_graph))
    base_items = item_map(read_json(resolve_path(args.base_paths)))
    new_items = item_map(read_json(resolve_path(args.new_paths)))
    question_ids = sorted(set(base_items) & set(new_items))

    counts = Counter()
    category_counts: dict[str, Counter] = {}
    examples: dict[str, list[dict]] = {
        "new_only_hit": [],
        "partial_to_full_cover": [],
        "filtered_rule_gold_support": [],
        "filtered_rule_non_gold": [],
    }
    for question_id in question_ids:
        base_item = base_items[question_id]
        new_item = new_items[question_id]
        base_eval = evaluate_item(base_graph, base_item, args.k)
        new_eval = evaluate_item(new_graph, new_item, args.k)
        category = str(new_item.get("category", "unknown"))
        category_counter = category_counts.setdefault(category, Counter())

        counts["questions"] += 1
        counts["base_hit"] += int(base_eval.hit)
        counts["new_hit"] += int(new_eval.hit)
        counts["base_full_cover"] += int(base_eval.full_cover)
        counts["new_full_cover"] += int(new_eval.full_cover)
        if new_eval.hit and not base_eval.hit:
            counts["new_only_hit"] += 1
            category_counter["new_only_hit"] += 1
            add_example(examples["new_only_hit"], base_item, base_eval, new_eval, args.examples)
        if base_eval.hit and not new_eval.hit:
            counts["base_only_hit"] += 1
            category_counter["base_only_hit"] += 1
        if 0.0 < base_eval.recall < 1.0 and new_eval.full_cover:
            counts["partial_to_full_cover"] += 1
            category_counter["partial_to_full_cover"] += 1
            add_example(examples["partial_to_full_cover"], base_item, base_eval, new_eval, args.examples)

        gold = {normalize_evidence_id(item) for item in new_item.get("gold_evidence", [])}
        filtered_nodes = selected_filtered_rule_nodes(new_graph, new_item, args.k)
        if filtered_nodes:
            counts["questions_with_filtered_rule_topk"] += 1
            category_counter["questions_with_filtered_rule_topk"] += 1
        for node_id in filtered_nodes:
            counts["filtered_rule_topk_occurrences"] += 1
            support = evidence_ids_for_node(new_graph, node_id)
            row = {
                "question_id": question_id,
                "question": new_item.get("question"),
                "category": new_item.get("category"),
                "gold_evidence": sorted(gold),
                "node_id": node_id,
                "node_text": new_graph.nodes[node_id].text,
                "support_evidence": sorted(support),
                "matched_gold": sorted(gold & support),
            }
            if gold & support:
                counts["filtered_rule_gold_support_occurrences"] += 1
                category_counter["filtered_rule_gold_support_occurrences"] += 1
                append_limited(examples["filtered_rule_gold_support"], row, args.examples)
            else:
                counts["filtered_rule_non_gold_occurrences"] += 1
                append_limited(examples["filtered_rule_non_gold"], row, args.examples)

    payload = {
        "metadata": {
            "base_graph": str(resolve_path(args.base_graph)),
            "new_graph": str(resolve_path(args.new_graph)),
            "base_paths": str(resolve_path(args.base_paths)),
            "new_paths": str(resolve_path(args.new_paths)),
            "k": args.k,
        },
        "summary": {
            **dict(counts),
            "net_hit_questions": counts["new_only_hit"] - counts["base_only_hit"],
            "filtered_rule_gold_support_precision": safe_ratio(
                counts["filtered_rule_gold_support_occurrences"],
                counts["filtered_rule_topk_occurrences"],
            ),
            "filtered_rule_question_rate": safe_ratio(
                counts["questions_with_filtered_rule_topk"],
                counts["questions"],
            ),
        },
        "by_category": {category: dict(values) for category, values in sorted(category_counts.items())},
        "examples": examples,
    }
    write_json(payload, resolve_path(args.output))
    print(f"summary={payload['summary']}")
    print(f"wrote {resolve_path(args.output)}")


def item_map(rows: list[dict]) -> dict[str, dict]:
    return {str(row["question_id"]): row for row in rows}


def selected_filtered_rule_nodes(graph, item: dict, k: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for path in item.get("paths", [])[:k]:
        for node_id in path.get("node_ids", []):
            node = graph.nodes.get(node_id)
            if node is None or node.source != "filtered_rule_statement" or node_id in seen:
                continue
            seen.add(node_id)
            output.append(node_id)
    return output


def add_example(target: list[dict], item: dict, base_eval, new_eval, limit: int) -> None:
    append_limited(
        target,
        {
            "question_id": item["question_id"],
            "question": item.get("question"),
            "answer": item.get("answer"),
            "category": item.get("category"),
            "gold_evidence": item.get("gold_evidence", []),
            "base": {
                "hit": base_eval.hit,
                "recall": base_eval.recall,
                "full_cover": base_eval.full_cover,
                "matched": base_eval.matched_evidence_ids,
            },
            "new": {
                "hit": new_eval.hit,
                "recall": new_eval.recall,
                "full_cover": new_eval.full_cover,
                "matched": new_eval.matched_evidence_ids,
            },
        },
        limit,
    )


def append_limited(target: list[dict], row: dict, limit: int) -> None:
    if len(target) < limit:
        target.append(row)


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
