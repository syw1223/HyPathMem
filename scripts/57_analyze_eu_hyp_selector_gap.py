from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean, median

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_semantic_hierarchy_v3_1_filtered_rule.json")
    parser.add_argument(
        "--candidates",
        default="outputs/topdown/full_graph_v3_1_filtered_rule_eu_hyp_union200_ce_selector_top20base_union_paths.json",
        help="CE-ranked Eu-Hyp union paths. Usually top100 after CE.",
    )
    parser.add_argument(
        "--final",
        default="outputs/paths/full_graph_v3_1_eu_hyp_union200_selector_entity_session_topdown_route_loco_cv_top5.json",
        help="Final selector top-k paths.",
    )
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--low-ce-rank", type=int, default=20)
    parser.add_argument("--examples", type=int, default=25)
    parser.add_argument("--output-json", default="outputs/eval/graph_v3_1_eu_hyp_selector_gap_analysis.json")
    parser.add_argument("--output-md", default="outputs/eval/graph_v3_1_eu_hyp_selector_gap_analysis.md")
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    candidate_items = item_map(read_json(resolve_path(args.candidates)))
    final_items = item_map(read_json(resolve_path(args.final)))
    question_ids = sorted(set(candidate_items) & set(final_items))

    counts = Counter()
    labels = Counter()
    by_category: dict[str, Counter] = defaultdict(Counter)
    route_counts = Counter()
    gold_rank_values: list[int] = []
    gold_ce_values: list[float] = []
    non_gold_above_values: list[int] = []
    examples: dict[str, list[dict]] = defaultdict(list)

    for question_id in question_ids:
        candidate_item = candidate_items[question_id]
        final_item = final_items[question_id]
        category = str(candidate_item.get("category", "unknown"))
        gold = {normalize_evidence_id(item) for item in candidate_item.get("gold_evidence", [])}
        candidate_paths = candidate_item.get("paths", [])[: args.candidate_k]
        final_paths = final_item.get("paths", [])[: args.final_k]

        candidate_gold_rows = gold_candidate_rows(graph, candidate_paths, gold)
        final_gold_rows = gold_candidate_rows(graph, final_paths, gold)
        candidate_hit = bool(candidate_gold_rows)
        final_hit = bool(final_gold_rows)
        counts["questions"] += 1
        counts["candidate_hit"] += int(candidate_hit)
        counts["final_hit"] += int(final_hit)
        by_category[category]["questions"] += 1
        by_category[category]["candidate_hit"] += int(candidate_hit)
        by_category[category]["final_hit"] += int(final_hit)

        if not candidate_hit or final_hit:
            continue

        counts["oracle_hit_final_miss"] += 1
        by_category[category]["oracle_hit_final_miss"] += 1
        best_gold = candidate_gold_rows[0]
        gold_rank_values.append(best_gold["rank"])
        gold_ce_values.append(best_gold["ce_score"])
        non_gold_above = best_gold["rank"] - 1
        non_gold_above_values.append(non_gold_above)
        route_counts[best_gold["route_class"]] += 1

        add_label("gold_in_top100_but_selector_missed", labels, by_category[category])
        if best_gold["rank"] <= args.final_k:
            add_label("gold_ce_top5_but_selector_missed", labels, by_category[category])
            add_example(examples, "gold_ce_top5_but_selector_missed", candidate_item, final_item, best_gold, args.examples)
        if best_gold["rank"] <= args.low_ce_rank:
            add_label("gold_ce_top20_but_selector_missed", labels, by_category[category])
        else:
            add_label("gold_candidate_ce_low_rank_gt20", labels, by_category[category])
            add_example(examples, "gold_candidate_ce_low_rank_gt20", candidate_item, final_item, best_gold, args.examples)
        if non_gold_above > 0:
            add_label("gold_pushed_below_non_gold_high_ce", labels, by_category[category])
        if non_gold_above >= args.low_ce_rank:
            add_label("gold_pushed_below_20plus_non_gold_ce", labels, by_category[category])
        if best_gold["has_filtered_rule_gold"]:
            add_label("gold_filtered_rule_candidate_missed", labels, by_category[category])
            add_example(examples, "gold_filtered_rule_candidate_missed", candidate_item, final_item, best_gold, args.examples)
        if best_gold["route_class"] == "hyp_only":
            add_label("gold_hyp_only_candidate_missed", labels, by_category[category])
            add_example(examples, "gold_hyp_only_candidate_missed", candidate_item, final_item, best_gold, args.examples)
        if best_gold["route_class"] == "eu_hyp_no_bottom":
            add_label("gold_eu_hyp_topdown_candidate_missed", labels, by_category[category])
        if best_gold["route_class"] == "bottom_only":
            add_label("gold_bottom_only_candidate_missed", labels, by_category[category])

        add_example(examples, "oracle_hit_final_miss", candidate_item, final_item, best_gold, args.examples)

    summary = {
        "questions": counts["questions"],
        "candidate_hit_at_k": ratio(counts["candidate_hit"], counts["questions"]),
        "final_hit_at_k": ratio(counts["final_hit"], counts["questions"]),
        "oracle_hit_final_miss": counts["oracle_hit_final_miss"],
        "selector_gap_rate_over_all": ratio(counts["oracle_hit_final_miss"], counts["questions"]),
        "selector_gap_rate_over_candidate_hit": ratio(counts["oracle_hit_final_miss"], counts["candidate_hit"]),
        "best_gold_ce_rank": describe(gold_rank_values),
        "best_gold_ce_score": describe(gold_ce_values),
        "non_gold_above_best_gold": describe(non_gold_above_values),
    }
    payload = {
        "metadata": {
            "graph": str(resolve_path(args.graph)),
            "candidates": str(resolve_path(args.candidates)),
            "final": str(resolve_path(args.final)),
            "candidate_k": args.candidate_k,
            "final_k": args.final_k,
            "low_ce_rank": args.low_ce_rank,
            "limitations": [
                "The saved selector output contains only final top-k paths, so non-selected candidates do not have LightGBM scores or selector ranks.",
                "This analysis can identify gold candidates in CE top-k that were absent from final top-k, but cannot recover their exact LightGBM rank without rerunning selector scoring with full-score export.",
            ],
        },
        "summary": summary,
        "failure_labels": dict(labels),
        "best_gold_route_class": dict(route_counts),
        "by_category": {key: dict(value) for key, value in sorted(by_category.items())},
        "examples": dict(examples),
    }
    write_json(payload, resolve_path(args.output_json))
    write_markdown(payload, resolve_path(args.output_md))
    print(f"summary={summary}")
    print(f"failure_labels={dict(labels)}")
    print(f"best_gold_route_class={dict(route_counts)}")
    print(f"wrote {resolve_path(args.output_json)}")
    print(f"wrote {resolve_path(args.output_md)}")


def item_map(rows: list[dict]) -> dict[str, dict]:
    return {str(row["question_id"]): row for row in rows}


def gold_candidate_rows(graph, paths: list[dict], gold: set[str]) -> list[dict]:
    rows = []
    for rank, path in enumerate(paths, start=1):
        matching_nodes = []
        matched_evidence = set()
        has_filtered_rule_gold = False
        for node_id in path.get("node_ids", []):
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            evidence = evidence_ids_for_node(graph, node_id)
            matched = gold & evidence
            if not matched:
                continue
            matched_evidence.update(matched)
            matching_nodes.append(
                {
                    "node_id": node_id,
                    "source": node.source,
                    "text": node.text,
                    "matched_evidence": sorted(matched),
                }
            )
            if node.source == "filtered_rule_statement":
                has_filtered_rule_gold = True
        if not matched_evidence:
            continue
        metadata = path.get("metadata", {})
        rows.append(
            {
                "rank": rank,
                "ce_score": path_score(path, "cross_encoder"),
                "selector_score": path_score(path, "topology_selector"),
                "base_score": path_score(path, "base"),
                "node_ids": path.get("node_ids", []),
                "matched_evidence": sorted(matched_evidence),
                "matching_nodes": matching_nodes,
                "candidate_source": metadata.get("candidate_source", ""),
                "route_source": metadata.get("route_source", ""),
                "route_class": route_class(metadata),
                "has_filtered_rule_gold": has_filtered_rule_gold,
                "fusion_agreement": truthy(metadata.get("fusion_agreement")),
                "metadata": compact_metadata(metadata),
            }
        )
    return rows


def path_score(path: dict, name: str) -> float:
    scores = path.get("scores", {})
    if name in scores:
        return safe_float(scores.get(name))
    if name == "cross_encoder":
        return safe_float(path.get("score"))
    return 0.0


def route_class(metadata: dict) -> str:
    text = " ".join(
        str(metadata.get(key, ""))
        for key in ["route_source", "candidate_source", "fusion_strategy"]
    ).lower()
    has_bottom = "bottom" in text
    has_eu = "eu_event" in text or "eu_topic" in text
    has_hyp = "hyp_event" in text or "hyp_topic" in text or "hyperbolic" in text
    if has_bottom and has_eu and has_hyp:
        return "bottom_eu_hyp"
    if has_bottom and has_eu:
        return "bottom_eu"
    if has_bottom and has_hyp:
        return "bottom_hyp"
    if has_eu and has_hyp:
        return "eu_hyp_no_bottom"
    if has_hyp:
        return "hyp_only"
    if has_eu:
        return "eu_only"
    if has_bottom:
        return "bottom_only"
    return "unknown"


def compact_metadata(metadata: dict) -> dict:
    keys = [
        "candidate_source",
        "route_source",
        "bottom_up_rank",
        "eu_route_rank",
        "hyp_route_rank",
        "eu_event_rank",
        "eu_topic_rank",
        "hyp_event_rank",
        "hyp_topic_rank",
        "fusion_agreement",
        "fusion_score",
        "fusion_eu_norm",
        "fusion_hyp_norm",
        "event_node_id",
        "topic_node_id",
        "event_degree",
        "topic_degree",
        "fact_offset",
    ]
    return {key: metadata[key] for key in keys if key in metadata}


def add_label(label: str, labels: Counter, category_counter: Counter) -> None:
    labels[label] += 1
    category_counter[label] += 1


def add_example(examples: dict[str, list[dict]], label: str, candidate_item: dict, final_item: dict, best_gold: dict, limit: int) -> None:
    if len(examples[label]) >= limit:
        return
    examples[label].append(
        {
            "question_id": candidate_item["question_id"],
            "question": candidate_item.get("question"),
            "answer": candidate_item.get("answer"),
            "category": candidate_item.get("category"),
            "gold_evidence": candidate_item.get("gold_evidence", []),
            "best_gold_candidate": best_gold,
            "final_top5": [
                {
                    "rank": index,
                    "score": path.get("score"),
                    "scores": path.get("scores", {}),
                    "node_ids": path.get("node_ids", []),
                    "candidate_source": path.get("metadata", {}).get("candidate_source", ""),
                    "route_source": path.get("metadata", {}).get("route_source", ""),
                    "route_class": route_class(path.get("metadata", {})),
                }
                for index, path in enumerate(final_item.get("paths", [])[:5], start=1)
            ],
        }
    )


def describe(values: list[float] | list[int]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_markdown(payload: dict, output_path) -> None:
    summary = payload["summary"]
    lines = [
        "# Eu-Hyp Selector Gap Analysis",
        "",
        "## Summary",
        "",
        f"- Questions: {summary['questions']}",
        f"- Candidate Hit@{payload['metadata']['candidate_k']}: {summary['candidate_hit_at_k']:.4f}",
        f"- Final Hit@{payload['metadata']['final_k']}: {summary['final_hit_at_k']:.4f}",
        f"- Oracle-hit but final-miss questions: {summary['oracle_hit_final_miss']}",
        f"- Gap rate over candidate-hit questions: {summary['selector_gap_rate_over_candidate_hit']:.4f}",
        "",
        "## Failure Labels",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(payload["failure_labels"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Best Gold Candidate Route",
            "",
            "| Route class | Count |",
            "| --- | ---: |",
        ]
    )
    for key, value in sorted(payload["best_gold_route_class"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## By Category",
            "",
            "| Category | Questions | Candidate Hit | Final Hit | Gap | CE Top20 Miss | CE Rank >20 | Hyp-only | Filtered Rule |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, row in sorted(payload["by_category"].items()):
        questions = int(row.get("questions", 0))
        candidate_hit = int(row.get("candidate_hit", 0))
        final_hit = int(row.get("final_hit", 0))
        gap = int(row.get("oracle_hit_final_miss", 0))
        lines.append(
            f"| {category} | {questions} | {candidate_hit} | {final_hit} | {gap} | "
            f"{int(row.get('gold_ce_top20_but_selector_missed', 0))} | "
            f"{int(row.get('gold_candidate_ce_low_rank_gt20', 0))} | "
            f"{int(row.get('gold_hyp_only_candidate_missed', 0))} | "
            f"{int(row.get('gold_filtered_rule_candidate_missed', 0))} |"
        )
    rank = summary["best_gold_ce_rank"]
    pushed = summary["non_gold_above_best_gold"]
    lines.extend(
        [
            "",
            "## Rank Diagnostics",
            "",
            f"- Best gold CE rank: mean={rank['mean']:.2f}, median={rank['median']:.2f}, min={rank['min']:.0f}, max={rank['max']:.0f}",
            f"- Non-gold candidates above best gold: mean={pushed['mean']:.2f}, median={pushed['median']:.2f}, max={pushed['max']:.0f}",
            "",
            "## Limitations",
            "",
        ]
    )
    for item in payload["metadata"]["limitations"]:
        lines.append(f"- {item}")
    output_path = resolve_path(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
