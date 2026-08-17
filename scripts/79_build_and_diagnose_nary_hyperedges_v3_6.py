from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType
from hytopomem.retrieval.bm25_retriever import BM25Retriever


ROLE_NAMES = (
    "old_state",
    "new_state",
    "preference_value",
    "polarity",
    "state_value",
    "plan_goal",
    "constraint",
    "temporal_scope",
    "reason_or_constraint",
    "reason_or_trigger",
    "exception",
    "context",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_multiview_v3_5_packs.json")
    parser.add_argument("--annotations", default="outputs/nary_v3_6/nary_hyperedge_annotations.json")
    parser.add_argument(
        "--base-candidates",
        default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_notemporal_top150_D_true_bu_euhyp_td_euhyp_paths.json",
    )
    parser.add_argument("--questions", default=None)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--candidate-topn", type=int, default=150)
    parser.add_argument("--output-graph", default="outputs/graphs/locomo_graph_multiview_v3_6_nary.json")
    parser.add_argument("--output-json", default="outputs/eval/graph_v3_6_nary_hyperedge_diagnosis.json")
    parser.add_argument("--output-md", default="outputs/eval/graph_v3_6_nary_hyperedge_diagnosis.md")
    args = parser.parse_args()

    config = load_config(args.config)
    source_graph = JsonGraphStore().load(resolve_path(args.graph))
    annotation_payload = read_json(resolve_path(args.annotations))
    graph, build_stats = build_graph(source_graph, annotation_payload["records"], args.min_confidence)
    output_graph = resolve_path(args.output_graph)
    JsonGraphStore().save(graph, output_graph)

    questions = flatten_questions(read_json(resolve_path(args.questions or config["data"]["processed_path"])))
    base_items = {str(item["question_id"]): item for item in read_json(resolve_path(args.base_candidates))}
    diagnostics = diagnose(graph, questions, base_items, args.topk, args.candidate_topn)
    payload = {
        "source_graph": str(resolve_path(args.graph)),
        "output_graph": str(output_graph),
        "annotations": str(resolve_path(args.annotations)),
        "min_confidence": args.min_confidence,
        "topk": args.topk,
        "build_stats": build_stats,
        "diagnostics": diagnostics,
    }
    out_json = resolve_path(args.output_json)
    write_json(payload, out_json)
    out_md = resolve_path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"build_stats": build_stats, "diagnostics": diagnostics["summary"]}, indent=2, ensure_ascii=False))
    print(f"wrote {output_graph}")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


def build_graph(source: MemoryGraph, records: list[dict], min_confidence: float) -> tuple[MemoryGraph, dict]:
    graph = source.model_copy(deep=True)
    accepted = []
    role_edges = 0
    for record in records:
        annotation = record.get("annotation") or {}
        candidate = record.get("candidate") or {}
        confidence = float(annotation.get("confidence", 0.0))
        if not annotation.get("accept") or confidence < min_confidence:
            continue
        relation_type = str(annotation.get("relation_type", ""))
        node_id = str(candidate["candidate_id"]).replace(":nary_candidate:", ":naryv3_6:")
        role_values = {}
        support_fact_ids = []
        for role_name in ROLE_NAMES:
            role = (annotation.get("roles") or {}).get(role_name) or {}
            fact_ids = [str(fact_id) for fact_id in role.get("fact_ids", []) if str(fact_id) in graph.nodes]
            value = str(role.get("value", "")).strip()
            if value and fact_ids:
                role_values[role_name] = {"value": value, "fact_ids": fact_ids}
                support_fact_ids.extend(fact_ids)
        support_fact_ids = dedupe(support_fact_ids)
        raw_ids = dedupe(
            raw_id
            for fact_id in support_fact_ids
            for raw_id in graph.nodes[fact_id].metadata.get("support_raw_ids") or graph.nodes[fact_id].support_ids
        )
        node = Node(
            node_id=node_id,
            type=NodeType.EVIDENCE_PACK,
            text=nary_text(annotation),
            source="typed_nary_hyperedge_v3_6",
            confidence=confidence,
            support_ids=support_fact_ids,
            metadata={
                "hierarchy_v3_6": "typed_nary_hyperedge",
                "pack_type": f"nary_{relation_type}",
                "relation_type": relation_type,
                "entity": str(annotation.get("entity", "")),
                "aspect": str(annotation.get("aspect", "")),
                "roles": role_values,
                "fact_ids": support_fact_ids,
                "raw_ids": raw_ids,
                "num_facts": len(support_fact_ids),
                "prompt_version": record.get("prompt_version", ""),
                "model": record.get("model", ""),
            },
        )
        graph.add_node(node)
        for role_name, role in role_values.items():
            for fact_id in role["fact_ids"]:
                graph.add_edge(
                    Edge(
                        edge_id=f"{fact_id}->FILLS_ROLE:{role_name}->{node_id}",
                        src=fact_id,
                        dst=node_id,
                        relation=RelationType.FILLS_ROLE,
                        confidence=confidence,
                        metadata={
                            "hierarchy_v3_6": "fills_role",
                            "role": role_name,
                            "role_value": role["value"],
                            "role_confidence": confidence,
                        },
                    )
                )
                role_edges += 1
        accepted.append(node)
    metadata = dict(graph.metadata)
    metadata["typed_nary_hyperedges_v3_6"] = {
        "annotation_only": True,
        "uses_gold": False,
        "hyperedge_count": len(accepted),
        "role_edge_count": role_edges,
        "min_confidence": min_confidence,
    }
    graph.metadata = metadata
    graph.graph_id = f"{graph.graph_id}_typed_nary_v3_6"
    counts = defaultdict(int)
    for node in accepted:
        counts[str(node.metadata.get("relation_type", ""))] += 1
    return graph, {
        "hyperedge_count": len(accepted),
        "role_edge_count": role_edges,
        "type_counts": dict(sorted(counts.items())),
        "mean_facts_per_hyperedge": mean([len(node.support_ids) for node in accepted]) if accepted else 0.0,
        "raw_provenance_complete_ratio": (
            mean(float(bool(node.metadata.get("raw_ids"))) for node in accepted) if accepted else 0.0
        ),
        "mean_role_count": mean([len(node.metadata.get("roles", {})) for node in accepted]) if accepted else 0.0,
    }


def diagnose(
    graph: MemoryGraph,
    questions: list[dict],
    base_items: dict[str, dict],
    topk: int,
    candidate_topn: int,
) -> dict:
    by_type = defaultdict(list)
    for node in graph.iter_nodes(NodeType.EVIDENCE_PACK):
        if node.metadata.get("hierarchy_v3_6") == "typed_nary_hyperedge":
            by_type[str(node.metadata.get("relation_type", ""))].append(node)
    by_type["all"] = [node for values in by_type.values() for node in values]
    summary = {}
    for relation_type, nodes in by_type.items():
        retriever = BM25Retriever(nodes)
        rows = []
        base_miss_rows = []
        eligible_rows = []
        eligible_base_miss_rows = []
        supported_gold = {
            conversation_id(node.node_id): set().union(
                *[
                    {normalize_evidence_id(raw_id) for raw_id in candidate.metadata.get("raw_ids", [])}
                    for candidate in nodes
                    if conversation_id(candidate.node_id) == conversation_id(node.node_id)
                ]
            )
            for node in nodes
        }
        for qa in questions:
            gold = gold_set(qa)
            hits = [
                node
                for node, _score in retriever.search(qa["question"], top_k=max(topk * 10, topk))
                if conversation_id(node.node_id) == conversation_id(qa["question_id"])
            ][:topk]
            row = eval_hyperedges(hits, gold)
            rows.append(row)
            eligible = bool(gold & supported_gold.get(conversation_id(qa["question_id"]), set()))
            if eligible:
                eligible_rows.append(row)
            base_ids = fact_ids_from_paths(base_items.get(qa["question_id"], {}).get("paths", [])[:candidate_topn])
            missing = gold - predicted_gold(graph, base_ids)
            if missing:
                miss_row = eval_hyperedges(hits, missing)
                base_miss_rows.append(miss_row)
                if missing & supported_gold.get(conversation_id(qa["question_id"]), set()):
                    eligible_base_miss_rows.append(miss_row)
        summary[relation_type] = {
            "hyperedge_count": len(nodes),
            "all_questions": summarize_rows(rows),
            "eligible_questions": len(eligible_rows),
            "eligible": summarize_rows(eligible_rows),
            "base_miss_questions": len(base_miss_rows),
            "base_miss": summarize_rows(base_miss_rows),
            "eligible_base_miss_questions": len(eligible_base_miss_rows),
            "eligible_base_miss": summarize_rows(eligible_base_miss_rows),
        }
    return {"summary": summary}


def eval_hyperedges(nodes: list[Node], gold: set[str]) -> dict:
    predicted = set()
    best_purity = 0.0
    for node in nodes:
        evidence = {normalize_evidence_id(raw_id) for raw_id in node.metadata.get("raw_ids", [])}
        predicted.update(evidence)
        best_purity = max(best_purity, len(evidence & gold) / len(evidence) if evidence else 0.0)
    matched = predicted & gold
    return {
        "hit": bool(matched),
        "recall": len(matched) / len(gold) if gold else 0.0,
        "full_cover": bool(gold) and gold.issubset(predicted),
        "best_purity": best_purity,
    }


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"hit": 0.0, "recall": 0.0, "full_cover": 0.0, "best_purity": 0.0}
    return {key: mean(float(row[key]) for row in rows) for key in rows[0]}


def nary_text(annotation: dict) -> str:
    relation_type = str(annotation.get("relation_type", "")).title()
    entity = str(annotation.get("entity", "")).strip()
    aspect = str(annotation.get("aspect", "")).strip()
    values = []
    for role_name, role in (annotation.get("roles") or {}).items():
        value = str((role or {}).get("value", "")).strip()
        if value:
            values.append(f"{role_name}: {value}")
    return f"{relation_type} relation for {entity}, aspect {aspect}. " + "; ".join(values)


def predicted_gold(graph: MemoryGraph, fact_ids: list[str]) -> set[str]:
    output = set()
    for fact_id in fact_ids:
        output.update(evidence_ids_for_node(graph, fact_id))
    return output


def fact_ids_from_paths(paths: list[dict]) -> list[str]:
    output = []
    for path in paths:
        fact_id = str(path.get("metadata", {}).get("evidence_node_id", ""))
        if not fact_id:
            node_ids = path.get("node_ids", [])
            fact_id = str(node_ids[-1]) if node_ids else ""
        if fact_id:
            output.append(fact_id)
    return dedupe(output)


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


def render_markdown(payload: dict) -> str:
    lines = [
        "# Graph V3.6 Typed N-ary Hyperedge Diagnosis",
        "",
        f"- Source graph: `{payload['source_graph']}`",
        f"- Output graph: `{payload['output_graph']}`",
        f"- Min confidence: `{payload['min_confidence']}`",
        f"- Hyperedges: `{payload['build_stats']['hyperedge_count']}`",
        f"- FILLS_ROLE edges: `{payload['build_stats']['role_edge_count']}`",
        "",
        "## Retrieval Diagnosis",
        "",
        "| Type | Count | Eligible QA | All Recall | Eligible Recall | Eligible Purity | Base-Miss Recall | Eligible Base-Miss Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for relation_type, item in payload["diagnostics"]["summary"].items():
        all_rows = item["all_questions"]
        miss = item["base_miss"]
        eligible = item["eligible"]
        eligible_miss = item["eligible_base_miss"]
        lines.append(
            f"| {relation_type} | {item['hyperedge_count']} | {item['eligible_questions']} | "
            f"{all_rows['recall']:.4f} | {eligible['recall']:.4f} | "
            f"{eligible['best_purity']:.4f} | {miss['recall']:.4f} | {eligible_miss['recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Hyperedges are typed relations with role-bearing `FILLS_ROLE` edges.",
            "- `FILLS_ROLE` is not a partial-order relation and is excluded from hyperbolic hierarchy training.",
            "- No gold evidence is used during candidate generation or LLM extraction.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
