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


DEFAULT_TYPE_SETS = {
    "all_types": {"change", "plan_constraint", "preference", "state"},
    "no_change": {"plan_constraint", "preference", "state"},
    "preference_only": {"preference"},
    "plan_constraint_only": {"plan_constraint"},
    "state_only": {"state"},
    "change_only": {"change"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", required=True)
    parser.add_argument(
        "--base-candidates",
        default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_notemporal_top150_D_true_bu_euhyp_td_euhyp_paths.json",
    )
    parser.add_argument("--questions", default=None)
    parser.add_argument("--top-hyperedges", default="20,50,100")
    parser.add_argument("--final-facts", type=int, default=100)
    parser.add_argument("--max-facts-per-hyperedge", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    questions = flatten_questions(read_json(resolve_path(args.questions or config["data"]["processed_path"])))
    base_items = {str(item["question_id"]): item for item in read_json(resolve_path(args.base_candidates))}
    top_hyperedges = [int(value) for value in str(args.top_hyperedges).split(",") if value.strip()]

    payload = {
        "graph": str(resolve_path(args.graph)),
        "questions": len(questions),
        "top_hyperedges": top_hyperedges,
        "final_facts": args.final_facts,
        "max_facts_per_hyperedge": args.max_facts_per_hyperedge,
        "results": {},
    }
    for type_set_name, type_set in DEFAULT_TYPE_SETS.items():
        hyperedges = nary_hyperedges(graph, type_set)
        if not hyperedges:
            continue
        retriever = BM25Retriever(hyperedges)
        type_payload = {
            "relation_types": sorted(type_set),
            "hyperedge_count": len(hyperedges),
            "supported_eligible_questions": eligible_count(hyperedges, questions),
            "runs": {},
        }
        for topk in top_hyperedges:
            rows, examples = run_retrieval(
                graph,
                retriever,
                hyperedges,
                questions,
                base_items,
                topk,
                args.final_facts,
                args.max_facts_per_hyperedge,
            )
            type_payload["runs"][str(topk)] = {
                "top_hyperedges": topk,
                "summary": summarize_rows(rows),
                "eligible_summary": summarize_rows([row for row in rows if row["eligible"]]),
                "base_miss_summary": summarize_rows([row for row in rows if row["base_miss"]]),
                "eligible_base_miss_summary": summarize_rows(
                    [row for row in rows if row["eligible_base_miss"]]
                ),
                "eligible_questions": sum(bool(row["eligible"]) for row in rows),
                "base_miss_questions": sum(bool(row["base_miss"]) for row in rows),
                "eligible_base_miss_questions": sum(bool(row["eligible_base_miss"]) for row in rows),
                "avg_candidate_facts": mean([row["candidate_facts"] for row in rows]) if rows else 0.0,
                "examples": examples,
            }
        payload["results"][type_set_name] = type_payload

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
        relation_type = str(node.metadata.get("relation_type", ""))
        if relation_type in relation_types:
            output.append(node)
    return output


def run_retrieval(
    graph: MemoryGraph,
    retriever: BM25Retriever,
    hyperedges: list[Node],
    questions: list[dict],
    base_items: dict[str, dict],
    top_hyperedges: int,
    final_facts: int,
    max_facts_per_hyperedge: int,
) -> tuple[list[dict], list[dict]]:
    supported_by_conv = supported_gold_by_conversation(hyperedges)
    rows = []
    examples = []
    for qa in questions:
        qid = str(qa["question_id"])
        conv_id = conversation_id(qid)
        gold = gold_set(qa)
        ranked = [
            (node, score)
            for node, score in retriever.search(qa["question"], top_k=max(top_hyperedges * 5, top_hyperedges))
            if conversation_id(node.node_id) == conv_id
        ][:top_hyperedges]
        fact_ids = expand_facts(ranked, max_facts_per_hyperedge, final_facts)
        predicted = set()
        for fact_id in fact_ids:
            predicted.update(evidence_ids_for_node(graph, fact_id))
        matched = gold & predicted
        base_fact_ids = fact_ids_from_paths(base_items.get(qid, {}).get("paths", [])[:150])
        base_predicted = set()
        for fact_id in base_fact_ids:
            base_predicted.update(evidence_ids_for_node(graph, fact_id))
        base_missing = gold - base_predicted
        eligible = bool(gold & supported_by_conv.get(conv_id, set()))
        eligible_base_miss = bool(base_missing & supported_by_conv.get(conv_id, set()))
        row = {
            "question_id": qid,
            "hit": bool(matched),
            "recall": len(matched) / len(gold) if gold else 0.0,
            "full_cover": bool(gold) and gold.issubset(predicted),
            "eligible": eligible,
            "base_miss": bool(base_missing),
            "eligible_base_miss": eligible_base_miss,
            "base_miss_hit": bool(base_missing & predicted),
            "base_miss_recall": len(base_missing & predicted) / len(base_missing) if base_missing else 0.0,
            "base_miss_full_cover": bool(base_missing) and base_missing.issubset(predicted),
            "candidate_facts": len(fact_ids),
            "top_hyperedge_ids": [node.node_id for node, _ in ranked[:5]],
        }
        rows.append(row)
        if row["hit"] and len(examples) < 10:
            examples.append(
                {
                    "question_id": qid,
                    "question": qa.get("question", ""),
                    "gold": sorted(gold),
                    "matched": sorted(matched),
                    "candidate_facts": len(fact_ids),
                    "top_hyperedges": [
                        {
                            "node_id": node.node_id,
                            "score": score,
                            "relation_type": node.metadata.get("relation_type", ""),
                            "text": node.text[:240],
                        }
                        for node, score in ranked[:3]
                    ],
                }
            )
    return rows, examples


def expand_facts(
    ranked_hyperedges: list[tuple[Node, float]],
    max_facts_per_hyperedge: int,
    final_facts: int,
) -> list[str]:
    output = []
    seen = set()
    for node, _score in ranked_hyperedges:
        fact_ids = fact_ids_from_roles(node)
        if max_facts_per_hyperedge > 0:
            fact_ids = fact_ids[:max_facts_per_hyperedge]
        for fact_id in fact_ids:
            if fact_id not in seen:
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


def fact_ids_from_paths(paths: list[dict]) -> list[str]:
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


def flatten_questions(conversations: list[dict]) -> list[dict]:
    return [qa for conversation in conversations for qa in conversation.get("qa", [])]


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("evidence", item.get("gold_evidence", []))}


def conversation_id(node_id: str) -> str:
    return str(node_id).split(":", 1)[0]


def compact_summary(payload: dict) -> dict:
    output = {}
    for type_set_name, item in payload["results"].items():
        output[type_set_name] = {}
        for topk, run in item["runs"].items():
            output[type_set_name][topk] = {
                "hyperedges": item["hyperedge_count"],
                "eligible_questions": run["eligible_questions"],
                "hit": run["summary"]["hit"],
                "recall": run["summary"]["recall"],
                "full_cover": run["summary"]["full_cover"],
                "eligible_hit": run["eligible_summary"]["hit"],
                "eligible_recall": run["eligible_summary"]["recall"],
                "eligible_full_cover": run["eligible_summary"]["full_cover"],
                "eligible_base_miss_questions": run["eligible_base_miss_questions"],
                "eligible_base_miss_full_cover": run["eligible_base_miss_summary"]["base_miss_full_cover"],
                "avg_candidate_facts": run["avg_candidate_facts"],
            }
    return output


def render_markdown(payload: dict) -> str:
    lines = [
        "# V3.6C N-ary-only Relation Retrieval",
        "",
        f"- Graph: `{payload['graph']}`",
        f"- Questions: `{payload['questions']}`",
        f"- Final facts: `{payload['final_facts']}`",
        "",
        "## Main Table",
        "",
        "| Type Set | Top Hyperedges | Hyperedges | Eligible QA | Hit@100 | Recall@100 | FullCover@100 | Eligible Hit | Eligible Recall | Eligible FullCover | Base-Miss Eligible | Base-Miss Eligible FullCover | Avg Facts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for type_set_name, item in payload["results"].items():
        for topk, run in item["runs"].items():
            summary = run["summary"]
            eligible = run["eligible_summary"]
            eligible_miss = run["eligible_base_miss_summary"]
            lines.append(
                f"| {type_set_name} | {topk} | {item['hyperedge_count']} | {run['eligible_questions']} | "
                f"{summary['hit']:.4f} | {summary['recall']:.4f} | {summary['full_cover']:.4f} | "
                f"{eligible['hit']:.4f} | {eligible['recall']:.4f} | {eligible['full_cover']:.4f} | "
                f"{run['eligible_base_miss_questions']} | {eligible_miss['base_miss_full_cover']:.4f} | "
                f"{run['avg_candidate_facts']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is n-ary-only relation retrieval: query ranks typed hyperedges, then expands role facts.",
            "- No base seed, CE, LightGBM, or bottom-up completion is used.",
            "- Eligible QA means the gold evidence appears in at least one hyperedge in the selected type set.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
