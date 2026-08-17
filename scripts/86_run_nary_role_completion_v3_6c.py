from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, Node, NodeType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", required=True)
    parser.add_argument(
        "--base-paths",
        default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_notemporal_top150_D_true_bu_euhyp_td_euhyp_paths.json",
    )
    parser.add_argument("--questions", default=None)
    parser.add_argument("--base-topn", type=int, default=100)
    parser.add_argument("--completion-topn", default="5,10,20,50,100")
    parser.add_argument("--relation-types", default="change,plan_constraint,preference,state")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    questions = flatten_questions(read_json(resolve_path(args.questions or config["data"]["processed_path"])))
    base_items = {str(item["question_id"]): item for item in read_json(resolve_path(args.base_paths))}
    completion_topns = [int(value) for value in args.completion_topn.split(",") if value.strip()]
    relation_types = {value.strip() for value in args.relation_types.split(",") if value.strip()}
    hyperedges = nary_hyperedges(graph, relation_types)
    index = CompletionIndex.from_hyperedges(hyperedges)
    supported = supported_gold_by_conversation(hyperedges)

    rows_by_topn = {topn: [] for topn in completion_topns}
    examples_by_topn = {topn: [] for topn in completion_topns}
    for qa in questions:
        qid = str(qa["question_id"])
        gold = gold_set(qa)
        conv_id = conversation_id(qid)
        base_facts = base_fact_ids(base_items.get(qid, {}).get("paths", []))[: args.base_topn]
        base_predicted = predicted_for_facts(graph, base_facts)
        base_missing = gold - base_predicted
        completion_facts, completion_sources = index.complete(base_facts, conv_id)
        eligible = bool(gold & supported.get(conv_id, set()))
        eligible_base_miss = bool(base_missing & supported.get(conv_id, set()))
        for topn in completion_topns:
            selected = merge_append(base_facts, completion_facts[:topn])
            row = evaluate_selection(graph, qa, selected, base_predicted, base_missing, eligible, eligible_base_miss)
            row.update(
                {
                    "question_id": qid,
                    "base_facts": len(base_facts),
                    "completion_budget": topn,
                    "completion_available": len(completion_facts),
                    "completion_used": len([fact_id for fact_id in completion_facts[:topn] if fact_id not in set(base_facts)]),
                    "selected_facts": len(selected),
                    "triggered_hyperedges": len(completion_sources),
                }
            )
            rows_by_topn[topn].append(row)
            if row["delta_hit"] and len(examples_by_topn[topn]) < 10:
                examples_by_topn[topn].append(
                    {
                        "question_id": qid,
                        "question": qa.get("question", ""),
                        "gold": sorted(gold),
                        "base_matched": sorted(gold & base_predicted),
                        "after_matched": sorted(gold & predicted_for_facts(graph, selected)),
                        "completion_used": row["completion_used"],
                        "triggered_hyperedges": completion_sources[:5],
                    }
                )

    results = {}
    for topn in completion_topns:
        rows = rows_by_topn[topn]
        results[f"base{args.base_topn}_plus_completion{topn}"] = {
            "base_topn": args.base_topn,
            "completion_topn": topn,
            "summary": summarize_rows(rows),
            "eligible_summary": summarize_rows([row for row in rows if row["eligible"]]),
            "base_miss_summary": summarize_rows([row for row in rows if row["base_miss"]]),
            "eligible_base_miss_summary": summarize_rows([row for row in rows if row["eligible_base_miss"]]),
            "eligible_questions": sum(bool(row["eligible"]) for row in rows),
            "base_miss_questions": sum(bool(row["base_miss"]) for row in rows),
            "eligible_base_miss_questions": sum(bool(row["eligible_base_miss"]) for row in rows),
            "avg_selected_facts": mean([row["selected_facts"] for row in rows]) if rows else 0.0,
            "avg_completion_available": mean([row["completion_available"] for row in rows]) if rows else 0.0,
            "avg_completion_used": mean([row["completion_used"] for row in rows]) if rows else 0.0,
            "avg_triggered_hyperedges": mean([row["triggered_hyperedges"] for row in rows]) if rows else 0.0,
            "delta_hit_count": sum(bool(row["delta_hit"]) for row in rows),
            "delta_full_cover_count": sum(bool(row["delta_full_cover"]) for row in rows),
            "examples": examples_by_topn[topn],
        }

    payload = {
        "graph": str(resolve_path(args.graph)),
        "base_paths": str(resolve_path(args.base_paths)),
        "relation_types": sorted(relation_types),
        "base_topn": args.base_topn,
        "nary_hyperedges": len(hyperedges),
        "supported_eligible_questions": eligible_count(hyperedges, questions),
        "results": results,
    }
    out_json = resolve_path(args.output_json)
    write_json(payload, out_json)
    out_md = resolve_path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(compact_summary(payload), indent=2, ensure_ascii=False))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


class CompletionIndex:
    def __init__(self) -> None:
        self.fact_to_hyperedges: dict[str, list[Node]] = defaultdict(list)

    @classmethod
    def from_hyperedges(cls, hyperedges: list[Node]) -> "CompletionIndex":
        index = cls()
        for node in hyperedges:
            for fact_id in fact_ids_from_roles(node):
                index.fact_to_hyperedges[fact_id].append(node)
        return index

    def complete(self, seed_facts: list[str], conv_id: str) -> tuple[list[str], list[str]]:
        output = []
        seen = set(seed_facts)
        sources = []
        for seed_fact in seed_facts:
            for node in self.fact_to_hyperedges.get(seed_fact, []):
                if conversation_id(node.node_id) != conv_id:
                    continue
                sources.append(node.node_id)
                for fact_id in fact_ids_from_roles(node):
                    if fact_id not in seen:
                        seen.add(fact_id)
                        output.append(fact_id)
        return output, dedupe(sources)


def nary_hyperedges(graph: MemoryGraph, relation_types: set[str]) -> list[Node]:
    output = []
    for node in graph.iter_nodes(NodeType.EVIDENCE_PACK):
        if node.metadata.get("hierarchy_v3_6") != "typed_nary_hyperedge":
            continue
        if str(node.metadata.get("relation_type", "")) in relation_types:
            output.append(node)
    return output


def fact_ids_from_roles(node: Node) -> list[str]:
    output = []
    seen = set()
    for role in (node.metadata.get("roles") or {}).values():
        for fact_id in (role or {}).get("fact_ids", []):
            fact_id = str(fact_id)
            if fact_id and fact_id not in seen:
                seen.add(fact_id)
                output.append(fact_id)
    for fact_id in node.metadata.get("fact_ids", []) or node.support_ids:
        fact_id = str(fact_id)
        if fact_id and fact_id not in seen:
            seen.add(fact_id)
            output.append(fact_id)
    return output


def merge_append(base_facts: list[str], completion_facts: list[str]) -> list[str]:
    output = []
    seen = set()
    for fact_id in list(base_facts) + list(completion_facts):
        if fact_id not in seen:
            seen.add(fact_id)
            output.append(fact_id)
    return output


def evaluate_selection(
    graph: MemoryGraph,
    qa: dict,
    fact_ids: list[str],
    base_predicted: set[str],
    base_missing: set[str],
    eligible: bool,
    eligible_base_miss: bool,
) -> dict:
    gold = gold_set(qa)
    predicted = predicted_for_facts(graph, fact_ids)
    matched = gold & predicted
    base_matched = gold & base_predicted
    missing_matched = base_missing & predicted
    return {
        "hit": bool(matched),
        "recall": len(matched) / len(gold) if gold else 0.0,
        "full_cover": bool(gold) and gold.issubset(predicted),
        "base_hit": bool(base_matched),
        "base_recall": len(base_matched) / len(gold) if gold else 0.0,
        "base_full_cover": bool(gold) and gold.issubset(base_predicted),
        "delta_hit": bool(matched) and not bool(base_matched),
        "delta_full_cover": bool(gold) and gold.issubset(predicted) and not gold.issubset(base_predicted),
        "eligible": eligible,
        "base_miss": bool(base_missing),
        "eligible_base_miss": eligible_base_miss,
        "base_miss_hit": bool(missing_matched),
        "base_miss_recall": len(missing_matched) / len(base_missing) if base_missing else 0.0,
        "base_miss_full_cover": bool(base_missing) and base_missing.issubset(predicted),
    }


def predicted_for_facts(graph: MemoryGraph, fact_ids: list[str]) -> set[str]:
    output = set()
    for fact_id in fact_ids:
        output.update(evidence_ids_for_node(graph, fact_id))
    return output


def base_fact_ids(paths: list[dict]) -> list[str]:
    output = []
    seen = set()
    for path in paths:
        fact_id = str(path.get("metadata", {}).get("evidence_node_id", ""))
        if not fact_id:
            node_ids = path.get("node_ids", [])
            fact_id = str(node_ids[-1]) if node_ids else ""
        if fact_id and fact_id not in seen:
            seen.add(fact_id)
            output.append(fact_id)
    return output


def supported_gold_by_conversation(nodes: list[Node]) -> dict[str, set[str]]:
    output = defaultdict(set)
    for node in nodes:
        conv_id = conversation_id(node.node_id)
        for raw_id in node.metadata.get("raw_ids", []):
            output[conv_id].add(normalize_evidence_id(raw_id))
    return output


def eligible_count(nodes: list[Node], questions: list[dict]) -> int:
    supported = supported_gold_by_conversation(nodes)
    return sum(bool(gold_set(qa) & supported.get(conversation_id(qa["question_id"]), set())) for qa in questions)


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "num_questions": 0,
            "hit": 0.0,
            "recall": 0.0,
            "full_cover": 0.0,
            "base_miss_hit": 0.0,
            "base_miss_recall": 0.0,
            "base_miss_full_cover": 0.0,
            "delta_hit": 0.0,
            "delta_full_cover": 0.0,
        }
    return {
        "num_questions": len(rows),
        "hit": mean(float(row["hit"]) for row in rows),
        "recall": mean(float(row["recall"]) for row in rows),
        "full_cover": mean(float(row["full_cover"]) for row in rows),
        "base_miss_hit": mean(float(row["base_miss_hit"]) for row in rows),
        "base_miss_recall": mean(float(row["base_miss_recall"]) for row in rows),
        "base_miss_full_cover": mean(float(row["base_miss_full_cover"]) for row in rows),
        "delta_hit": mean(float(row["delta_hit"]) for row in rows),
        "delta_full_cover": mean(float(row["delta_full_cover"]) for row in rows),
    }


def flatten_questions(conversations: list[dict]) -> list[dict]:
    return [qa for conversation in conversations for qa in conversation.get("qa", [])]


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("evidence", item.get("gold_evidence", []))}


def conversation_id(node_id: str) -> str:
    return str(node_id).split(":", 1)[0]


def dedupe(values) -> list[str]:
    output = []
    seen = set()
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def compact_summary(payload: dict) -> dict:
    return {
        name: {
            "hit": item["summary"]["hit"],
            "recall": item["summary"]["recall"],
            "full_cover": item["summary"]["full_cover"],
            "delta_hit_count": item["delta_hit_count"],
            "delta_full_cover_count": item["delta_full_cover_count"],
            "eligible_base_miss_questions": item["eligible_base_miss_questions"],
            "eligible_base_miss_full_cover": item["eligible_base_miss_summary"]["base_miss_full_cover"],
            "avg_completion_used": item["avg_completion_used"],
            "avg_triggered_hyperedges": item["avg_triggered_hyperedges"],
        }
        for name, item in payload["results"].items()
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# V3.6C N-ary Role Completion",
        "",
        f"- Graph: `{payload['graph']}`",
        f"- Base paths: `{payload['base_paths']}`",
        f"- Relation types: `{', '.join(payload['relation_types'])}`",
        f"- Base topn: `{payload['base_topn']}`",
        f"- N-ary hyperedges: `{payload['nary_hyperedges']}`",
        f"- N-ary eligible questions: `{payload['supported_eligible_questions']}`",
        "",
        "| Setting | Hit | Recall | FullCover | Delta Hit Count | Delta FullCover Count | Base-miss Eligible | Base-miss Eligible FullCover | Avg Completion Used | Avg Triggered Hyperedges | Avg Selected Facts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in payload["results"].items():
        summary = item["summary"]
        eligible_miss = item["eligible_base_miss_summary"]
        lines.append(
            f"| {name} | {summary['hit']:.4f} | {summary['recall']:.4f} | {summary['full_cover']:.4f} | "
            f"{item['delta_hit_count']} | {item['delta_full_cover_count']} | {item['eligible_base_miss_questions']} | "
            f"{eligible_miss['base_miss_full_cover']:.4f} | {item['avg_completion_used']:.2f} | "
            f"{item['avg_triggered_hyperedges']:.2f} | {item['avg_selected_facts']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Base top100 is preserved; n-ary adds role facts through Fact -> TypedHyperedge -> RoleFact.",
            "- This is append-only candidate-pool diagnosis, not final CE/LightGBM reranking.",
            "- Delta counts measure questions fixed relative to the base top100 candidate pool.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
