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
from hytopomem.retrieval.bm25_retriever import BM25Retriever


DEFAULT_BUDGETS = ((100, 0), (0, 100), (90, 10), (80, 20), (70, 30), (50, 50))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", required=True)
    parser.add_argument(
        "--base-paths",
        default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_notemporal_top150_D_true_bu_euhyp_td_euhyp_paths.json",
    )
    parser.add_argument("--questions", default=None)
    parser.add_argument("--relation-types", default="change,plan_constraint,preference,state")
    parser.add_argument("--top-hyperedges", type=int, default=100)
    parser.add_argument("--final-facts", type=int, default=100)
    parser.add_argument("--budgets", default="100:0,0:100,90:10,80:20,70:30,50:50")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    questions = flatten_questions(read_json(resolve_path(args.questions or config["data"]["processed_path"])))
    base_items = {str(item["question_id"]): item for item in read_json(resolve_path(args.base_paths))}
    relation_types = {value.strip() for value in args.relation_types.split(",") if value.strip()}
    budgets = parse_budgets(args.budgets)
    hyperedges = nary_hyperedges(graph, relation_types)
    retriever = BM25Retriever(hyperedges)
    supported = supported_gold_by_conversation(hyperedges)

    rows_by_budget: dict[str, list[dict]] = {budget_name(b): [] for b in budgets}
    examples_by_budget: dict[str, list[dict]] = {budget_name(b): [] for b in budgets}
    for qa in questions:
        qid = str(qa["question_id"])
        gold = gold_set(qa)
        conv_id = conversation_id(qid)
        base_facts = base_fact_ids(base_items.get(qid, {}).get("paths", []))
        nary_facts = nary_fact_ids(
            graph,
            retriever,
            qa["question"],
            conv_id,
            top_hyperedges=args.top_hyperedges,
            final_facts=args.final_facts,
        )
        eligible = bool(gold & supported.get(conv_id, set()))
        base_missing = gold - predicted_for_facts(graph, base_facts[: args.final_facts])
        eligible_base_miss = bool(base_missing & supported.get(conv_id, set()))
        for budget in budgets:
            base_k, nary_k = budget
            selected = merge_budget(base_facts, nary_facts, base_k, nary_k, args.final_facts)
            row = evaluate_selection(graph, qa, selected, base_missing, eligible, eligible_base_miss)
            row.update(
                {
                    "question_id": qid,
                    "base_budget": base_k,
                    "nary_budget": nary_k,
                    "selected_facts": len(selected),
                    "base_overlap": len(set(selected) & set(base_facts[:base_k])),
                    "nary_overlap": len(set(selected) & set(nary_facts[:nary_k])),
                }
            )
            key = budget_name(budget)
            rows_by_budget[key].append(row)
            if row["hit"] and len(examples_by_budget[key]) < 5:
                examples_by_budget[key].append(
                    {
                        "question_id": qid,
                        "question": qa.get("question", ""),
                        "gold": sorted(gold),
                        "matched": sorted(gold & predicted_for_facts(graph, selected)),
                        "selected_facts": len(selected),
                    }
                )

    results = {}
    for budget in budgets:
        key = budget_name(budget)
        rows = rows_by_budget[key]
        results[key] = {
            "base_budget": budget[0],
            "nary_budget": budget[1],
            "summary": summarize_rows(rows),
            "eligible_summary": summarize_rows([row for row in rows if row["eligible"]]),
            "base_miss_summary": summarize_rows([row for row in rows if row["base_miss"]]),
            "eligible_base_miss_summary": summarize_rows([row for row in rows if row["eligible_base_miss"]]),
            "eligible_questions": sum(bool(row["eligible"]) for row in rows),
            "base_miss_questions": sum(bool(row["base_miss"]) for row in rows),
            "eligible_base_miss_questions": sum(bool(row["eligible_base_miss"]) for row in rows),
            "avg_selected_facts": mean([row["selected_facts"] for row in rows]) if rows else 0.0,
            "examples": examples_by_budget[key],
        }

    payload = {
        "graph": str(resolve_path(args.graph)),
        "base_paths": str(resolve_path(args.base_paths)),
        "relation_types": sorted(relation_types),
        "top_hyperedges": args.top_hyperedges,
        "final_facts": args.final_facts,
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


def nary_hyperedges(graph: MemoryGraph, relation_types: set[str]) -> list[Node]:
    output = []
    for node in graph.iter_nodes(NodeType.EVIDENCE_PACK):
        if node.metadata.get("hierarchy_v3_6") != "typed_nary_hyperedge":
            continue
        if str(node.metadata.get("relation_type", "")) in relation_types:
            output.append(node)
    return output


def nary_fact_ids(
    graph: MemoryGraph,
    retriever: BM25Retriever,
    question: str,
    conv_id: str,
    *,
    top_hyperedges: int,
    final_facts: int,
) -> list[str]:
    ranked = [
        node
        for node, _score in retriever.search(question, top_k=max(top_hyperedges * 5, top_hyperedges))
        if conversation_id(node.node_id) == conv_id
    ][:top_hyperedges]
    output = []
    seen = set()
    for node in ranked:
        for fact_id in fact_ids_from_roles(node):
            if fact_id in graph.nodes and fact_id not in seen:
                seen.add(fact_id)
                output.append(fact_id)
            if len(output) >= final_facts:
                return output
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


def merge_budget(base_facts: list[str], nary_facts: list[str], base_k: int, nary_k: int, final_k: int) -> list[str]:
    output = []
    seen = set()
    for fact_id in base_facts[:base_k]:
        if fact_id not in seen:
            seen.add(fact_id)
            output.append(fact_id)
    for fact_id in nary_facts[:nary_k]:
        if fact_id not in seen:
            seen.add(fact_id)
            output.append(fact_id)
    return output[:final_k]


def evaluate_selection(
    graph: MemoryGraph,
    qa: dict,
    fact_ids: list[str],
    base_missing: set[str],
    eligible: bool,
    eligible_base_miss: bool,
) -> dict:
    gold = gold_set(qa)
    predicted = predicted_for_facts(graph, fact_ids)
    matched = gold & predicted
    missing_matched = base_missing & predicted
    return {
        "hit": bool(matched),
        "recall": len(matched) / len(gold) if gold else 0.0,
        "full_cover": bool(gold) and gold.issubset(predicted),
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
        }
    return {
        "num_questions": len(rows),
        "hit": mean(float(row["hit"]) for row in rows),
        "recall": mean(float(row["recall"]) for row in rows),
        "full_cover": mean(float(row["full_cover"]) for row in rows),
        "base_miss_hit": mean(float(row["base_miss_hit"]) for row in rows),
        "base_miss_recall": mean(float(row["base_miss_recall"]) for row in rows),
        "base_miss_full_cover": mean(float(row["base_miss_full_cover"]) for row in rows),
    }


def parse_budgets(text: str) -> list[tuple[int, int]]:
    output = []
    for chunk in text.split(","):
        if not chunk.strip():
            continue
        left, right = chunk.split(":", 1)
        output.append((int(left), int(right)))
    return output


def budget_name(budget: tuple[int, int]) -> str:
    return f"base{budget[0]}_nary{budget[1]}"


def flatten_questions(conversations: list[dict]) -> list[dict]:
    return [qa for conversation in conversations for qa in conversation.get("qa", [])]


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("evidence", item.get("gold_evidence", []))}


def conversation_id(node_id: str) -> str:
    return str(node_id).split(":", 1)[0]


def compact_summary(payload: dict) -> dict:
    return {
        name: {
            "hit": item["summary"]["hit"],
            "recall": item["summary"]["recall"],
            "full_cover": item["summary"]["full_cover"],
            "eligible_hit": item["eligible_summary"]["hit"],
            "eligible_recall": item["eligible_summary"]["recall"],
            "eligible_full_cover": item["eligible_summary"]["full_cover"],
            "eligible_base_miss_questions": item["eligible_base_miss_questions"],
            "eligible_base_miss_full_cover": item["eligible_base_miss_summary"]["base_miss_full_cover"],
            "avg_selected_facts": item["avg_selected_facts"],
        }
        for name, item in payload["results"].items()
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# V3.6C Base + N-ary Controlled Budget Fusion",
        "",
        f"- Graph: `{payload['graph']}`",
        f"- Base paths: `{payload['base_paths']}`",
        f"- Relation types: `{', '.join(payload['relation_types'])}`",
        f"- Top hyperedges: `{payload['top_hyperedges']}`",
        f"- Final facts: `{payload['final_facts']}`",
        f"- N-ary hyperedges: `{payload['nary_hyperedges']}`",
        f"- N-ary eligible questions: `{payload['supported_eligible_questions']}`",
        "",
        "| Budget | Hit@100 | Recall@100 | FullCover@100 | Eligible Hit | Eligible Recall | Eligible FullCover | Base-miss Eligible | Base-miss Eligible FullCover | Avg Facts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in payload["results"].items():
        summary = item["summary"]
        eligible = item["eligible_summary"]
        eligible_miss = item["eligible_base_miss_summary"]
        lines.append(
            f"| {name} | {summary['hit']:.4f} | {summary['recall']:.4f} | {summary['full_cover']:.4f} | "
            f"{eligible['hit']:.4f} | {eligible['recall']:.4f} | {eligible['full_cover']:.4f} | "
            f"{item['eligible_base_miss_questions']} | {eligible_miss['base_miss_full_cover']:.4f} | "
            f"{item['avg_selected_facts']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Base candidates come first, then n-ary role facts fill the remaining controlled budget.",
            "- This is candidate-level/oracle-style fusion, not final CE/LightGBM reranking.",
            "- N-ary candidates are retrieved by query-to-hyperedge BM25 and expanded through role facts.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
