from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from common import load_config, read_json, resolve_path
from hytopomem.eval.oracle_metrics import evaluate_candidate_pool, summarize_oracle
from hytopomem.eval.retrieval_metrics import evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, NodeType, RelationType
from hytopomem.retrieval.bm25_retriever import BM25Retriever


TEMPORAL_QUERY_RE = re.compile(
    r"\b(when|date|time|day|before|after|earlier|later|previous|recent|recently|last|first|then|next|"
    r"多久|什么时候|哪天|之前|之后|最近|上次|第一次|后来|接着)\b",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_multiview_v3_4_ab.json")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--candidate-b", default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_B_true_bu_euhyp_td_hyp_paths.json")
    parser.add_argument("--candidate-d", default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_D_true_bu_euhyp_td_euhyp_paths.json")
    parser.add_argument("--seed-topn", type=int, default=20)
    parser.add_argument("--candidate-topn", type=int, default=100)
    parser.add_argument("--union-topn", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", default="outputs/eval/graph_v3_4_temporal_oracle_diagnosis.json")
    parser.add_argument("--output-md", default="outputs/eval/graph_v3_4_temporal_oracle_diagnosis.md")
    args = parser.parse_args()

    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    questions = flatten_questions(read_json(resolve_path(args.questions or config["data"]["processed_path"])), args.limit)
    facts = list(graph.iter_nodes(NodeType.FACT))
    bm25 = BM25Retriever(facts)
    temporal_index = TemporalIndex.from_graph(graph)

    b_items = {item["question_id"]: item for item in read_json(resolve_path(args.candidate_b))}
    d_items = {item["question_id"]: item for item in read_json(resolve_path(args.candidate_d))}

    pools: dict[str, dict[str, list[str]]] = {
        "BM25 Fact@50": {},
        f"Temporal Session@{args.candidate_topn}": {},
        f"B BU-EuHyp TD-Hyp@{args.candidate_topn}": {},
        f"D BU-EuHyp TD-EuHyp@{args.candidate_topn}": {},
        f"B@{args.candidate_topn} + Temporal@{args.candidate_topn} -> Union@{args.union_topn}": {},
        f"D@{args.candidate_topn} + Temporal@{args.candidate_topn} -> Union@{args.union_topn}": {},
    }

    for index, item in enumerate(questions, start=1):
        qid = item["question_id"]
        seed_hits = bm25.search(item["question"], top_k=max(args.seed_topn, 50))
        bm25_pool = [node.node_id for node, _score in seed_hits[:50]]
        temporal_pool = temporal_session_pool(
            [node.node_id for node, _score in seed_hits[: args.seed_topn]],
            temporal_index,
            topn=args.candidate_topn,
        )
        b_pool = candidate_ids_from_paths(b_items.get(qid, {}).get("paths", []), args.candidate_topn)
        d_pool = candidate_ids_from_paths(d_items.get(qid, {}).get("paths", []), args.candidate_topn)

        pools["BM25 Fact@50"][qid] = bm25_pool
        pools[f"Temporal Session@{args.candidate_topn}"][qid] = temporal_pool
        pools[f"B BU-EuHyp TD-Hyp@{args.candidate_topn}"][qid] = b_pool
        pools[f"D BU-EuHyp TD-EuHyp@{args.candidate_topn}"][qid] = d_pool
        pools[f"B@{args.candidate_topn} + Temporal@{args.candidate_topn} -> Union@{args.union_topn}"][qid] = dedupe(b_pool + temporal_pool)[: args.union_topn]
        pools[f"D@{args.candidate_topn} + Temporal@{args.candidate_topn} -> Union@{args.union_topn}"][qid] = dedupe(d_pool + temporal_pool)[: args.union_topn]

        if index % 250 == 0 or index == len(questions):
            print(f"built temporal pools {index}/{len(questions)}", flush=True)

    stage_results: dict[str, list] = {}
    stage_summaries: dict[str, dict] = {}
    for stage, by_qid in pools.items():
        results = [
            evaluate_candidate_pool(
                graph,
                question_id=item["question_id"],
                gold_evidence=item.get("evidence", item.get("gold_evidence", [])),
                candidate_node_ids=by_qid[item["question_id"]],
            )
            for item in questions
        ]
        stage_results[stage] = results
        summary = summarize_oracle(results)
        summary["gold_density"] = average_gold_density(graph, questions, by_qid)
        summary["positive_candidate_density"] = average_positive_candidate_density(graph, questions, by_qid)
        stage_summaries[stage] = summary

    group_summaries = {
        "temporal_queries": summarize_group(graph, questions, pools, lambda item: is_temporal_query(item.get("question", ""))),
        "non_temporal_queries": summarize_group(graph, questions, pools, lambda item: not is_temporal_query(item.get("question", ""))),
    }
    gains = paired_gain_summary(graph, questions, pools)

    payload = {
        "method": "Graph V3.4 temporal/session view oracle diagnosis",
        "graph": str(resolve_path(args.graph)),
        "seed_topn": args.seed_topn,
        "candidate_topn": args.candidate_topn,
        "union_topn": args.union_topn,
        "stages": stage_summaries,
        "groups": group_summaries,
        "paired_gains": gains,
        "per_stage": {stage: [result.__dict__ for result in results] for stage, results in stage_results.items()},
    }

    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, resolve_path(args.output_md))

    print(f"wrote {output_json}")
    print(f"wrote {resolve_path(args.output_md)}")
    for stage, summary in stage_summaries.items():
        print(
            f"{stage}: hit={summary['hit']:.4f} recall={summary['recall']:.4f} "
            f"full={summary['full_cover']:.4f} avg_cand={summary['avg_candidates']:.1f} "
            f"gold_density={summary['gold_density']:.5f}"
        )


class TemporalIndex:
    def __init__(self) -> None:
        self.fact_event: dict[str, str] = {}
        self.event_facts: dict[str, list[str]] = defaultdict(list)
        self.event_sessions: dict[str, list[str]] = defaultdict(list)
        self.session_events: dict[str, list[str]] = defaultdict(list)

    @classmethod
    def from_graph(cls, graph: MemoryGraph) -> "TemporalIndex":
        index = cls()
        for edge in graph.edges:
            if edge.relation != RelationType.IS_SPECIFIC_OF:
                continue
            src = graph.nodes.get(edge.src)
            dst = graph.nodes.get(edge.dst)
            if src is None or dst is None:
                continue
            if src.type == NodeType.FACT and dst.type == NodeType.EVENT:
                if edge.metadata.get("hierarchy_v3") != "lexical_alias_event":
                    index.fact_event[edge.src] = edge.dst
                    index.event_facts[edge.dst].append(edge.src)
            elif edge.metadata.get("hierarchy_v3_4_temporal") == "event_session":
                index.event_sessions[edge.src].append(edge.dst)
                index.session_events[edge.dst].append(edge.src)
        return index


def temporal_session_pool(seed_fact_ids: list[str], index: TemporalIndex, *, topn: int) -> list[str]:
    output: list[str] = []
    for fact_id in seed_fact_ids:
        output.append(fact_id)
        event_id = index.fact_event.get(fact_id)
        if not event_id:
            continue
        for sibling_fact_id in index.event_facts.get(event_id, []):
            output.append(sibling_fact_id)
        for session_id in index.event_sessions.get(event_id, []):
            for session_event_id in index.session_events.get(session_id, []):
                for sibling_fact_id in index.event_facts.get(session_event_id, []):
                    output.append(sibling_fact_id)
    return dedupe(output)[:topn]


def candidate_ids_from_paths(paths: list[dict], topn: int) -> list[str]:
    ids: list[str] = []
    for path in paths[:topn]:
        metadata = path.get("metadata", {})
        node_id = metadata.get("evidence_node_id")
        if not node_id:
            node_ids = path.get("node_ids", [])
            node_id = node_ids[-1] if node_ids else ""
        if node_id:
            ids.append(str(node_id))
    return dedupe(ids)[:topn]


def flatten_questions(conversations: list[dict], limit: int) -> list[dict]:
    questions = []
    for conversation in conversations:
        for qa in conversation.get("qa", []):
            questions.append(qa)
            if limit and len(questions) >= limit:
                return questions
    return questions


def is_temporal_query(question: str) -> bool:
    return bool(TEMPORAL_QUERY_RE.search(question or ""))


def summarize_group(graph: MemoryGraph, questions: list[dict], pools: dict[str, dict[str, list[str]]], predicate) -> dict:
    selected = [item for item in questions if predicate(item)]
    output = {"num_questions": len(selected)}
    for stage, by_qid in pools.items():
        results = [
            evaluate_candidate_pool(
                graph,
                question_id=item["question_id"],
                gold_evidence=item.get("evidence", item.get("gold_evidence", [])),
                candidate_node_ids=by_qid[item["question_id"]],
            )
            for item in selected
        ]
        output[stage] = summarize_oracle(results)
    return output


def paired_gain_summary(graph: MemoryGraph, questions: list[dict], pools: dict[str, dict[str, list[str]]]) -> dict:
    stages = list(pools)
    per_stage_hit: dict[str, dict[str, bool]] = {}
    per_stage_full: dict[str, dict[str, bool]] = {}
    for stage in stages:
        per_stage_hit[stage] = {}
        per_stage_full[stage] = {}
        for item in questions:
            result = evaluate_candidate_pool(
                graph,
                question_id=item["question_id"],
                gold_evidence=item.get("evidence", item.get("gold_evidence", [])),
                candidate_node_ids=pools[stage][item["question_id"]],
            )
            per_stage_hit[stage][item["question_id"]] = result.hit
            per_stage_full[stage][item["question_id"]] = result.full_cover
    output = {}
    pairs: list[tuple[str, str]] = [("Temporal Session@100", "BM25 Fact@50")]
    b_union = next((stage for stage in stages if stage.startswith("B@") and "Temporal" in stage), "")
    d_union = next((stage for stage in stages if stage.startswith("D@") and "Temporal" in stage), "")
    b_base = next((stage for stage in stages if stage.startswith("B BU-EuHyp")), "")
    d_base = next((stage for stage in stages if stage.startswith("D BU-EuHyp")), "")
    if b_union and b_base:
        pairs.insert(0, (b_union, b_base))
    if d_union and d_base:
        pairs.insert(1, (d_union, d_base))
    for left, right in pairs:
        if left not in pools or right not in pools:
            continue
        output[f"{left}_vs_{right}"] = {
            "hit_left_only": sum(per_stage_hit[left][item["question_id"]] and not per_stage_hit[right][item["question_id"]] for item in questions),
            "hit_right_only": sum(per_stage_hit[right][item["question_id"]] and not per_stage_hit[left][item["question_id"]] for item in questions),
            "full_left_only": sum(per_stage_full[left][item["question_id"]] and not per_stage_full[right][item["question_id"]] for item in questions),
            "full_right_only": sum(per_stage_full[right][item["question_id"]] and not per_stage_full[left][item["question_id"]] for item in questions),
        }
    return output


def average_gold_density(graph: MemoryGraph, questions: list[dict], pools: dict[str, list[str]]) -> float:
    values = []
    for item in questions:
        gold = {normalize_evidence_id(eid) for eid in item.get("evidence", item.get("gold_evidence", []))}
        matched = set()
        for node_id in pools[item["question_id"]]:
            matched.update(evidence_ids_for_node(graph, node_id) & gold)
        values.append(len(matched) / max(len(pools[item["question_id"]]), 1))
    return sum(values) / max(len(values), 1)


def average_positive_candidate_density(graph: MemoryGraph, questions: list[dict], pools: dict[str, list[str]]) -> float:
    values = []
    for item in questions:
        gold = {normalize_evidence_id(eid) for eid in item.get("evidence", item.get("gold_evidence", []))}
        positives = 0
        for node_id in pools[item["question_id"]]:
            if evidence_ids_for_node(graph, node_id) & gold:
                positives += 1
        values.append(positives / max(len(pools[item["question_id"]]), 1))
    return sum(values) / max(len(values), 1)


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Graph V3.4 Temporal View Oracle Diagnosis",
        "",
        f"Graph: `{payload['graph']}`",
        f"Seed topN: {payload['seed_topn']}",
        f"Candidate topN: {payload['candidate_topn']}",
        f"Union topN: {payload['union_topn']}",
        "",
        "## Overall",
        "",
        "| Stage | Questions | Hit | Recall | FullCover | Avg Cand | GoldDensity | PosCandDensity | Avg Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage, summary in payload["stages"].items():
        lines.append(
            f"| {stage} | {summary['num_questions']} | {summary['hit']:.4f} | {summary['recall']:.4f} | "
            f"{summary['full_cover']:.4f} | {summary['avg_candidates']:.1f} | {summary['gold_density']:.5f} | "
            f"{summary['positive_candidate_density']:.5f} | {summary['avg_tokens']:.1f} |"
        )
    lines.extend(["", "## Temporal Query Groups", ""])
    for group_name, group in payload["groups"].items():
        lines.append(f"### {group_name} ({group['num_questions']} questions)")
        lines.append("")
        lines.append("| Stage | Hit | Recall | FullCover | Avg Cand |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for stage, summary in group.items():
            if stage == "num_questions":
                continue
            lines.append(
                f"| {stage} | {summary['hit']:.4f} | {summary['recall']:.4f} | "
                f"{summary['full_cover']:.4f} | {summary['avg_candidates']:.1f} |"
            )
        lines.append("")
    lines.extend(["## Paired Gains", ""])
    lines.append("| Compare | Hit Left Only | Hit Right Only | Full Left Only | Full Right Only |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, values in payload["paired_gains"].items():
        lines.append(
            f"| {name} | {values['hit_left_only']} | {values['hit_right_only']} | "
            f"{values['full_left_only']} | {values['full_right_only']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dedupe(values: list[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


if __name__ == "__main__":
    main()
