from __future__ import annotations

import argparse
import json

from common import load_config, read_json, resolve_path
from hytopomem.eval.oracle_metrics import evaluate_candidate_pool, summarize_oracle
from hytopomem.eval.retrieval_metrics import evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import NodeType
from hytopomem.retrieval.bm25_retriever import BM25Retriever
from hytopomem.retrieval.hierarchy_v2_retriever import HierarchyV2Retriever, HierarchyV2RetrieverConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_hierarchy_v2.json")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed-topn", type=int, default=20)
    parser.add_argument("--preselect-topn", type=int, default=100)
    parser.add_argument("--topic-event-limits", nargs="+", type=int, default=[8])
    parser.add_argument("--output-json", default="outputs/eval/graph_v2_oracle_full.json")
    parser.add_argument("--output-md", default="outputs/eval/graph_v2_oracle_full.md")
    args = parser.parse_args()

    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    questions = flatten_questions(read_json(resolve_path(args.questions or config["data"]["processed_path"])), args.limit)
    facts = list(graph.iter_nodes(NodeType.FACT))
    bm25 = BM25Retriever(facts)
    retrievers = {
        "GraphV2 same_event": HierarchyV2Retriever(
            graph,
            HierarchyV2RetrieverConfig(
                mode="same_event",
                seed_topn=args.seed_topn,
                preselect_topn=args.preselect_topn,
            ),
        ),
        "GraphV2 same_topic_only": HierarchyV2Retriever(
            graph,
            HierarchyV2RetrieverConfig(
                mode="same_topic_only",
                seed_topn=args.seed_topn,
                preselect_topn=args.preselect_topn,
            ),
        ),
    }
    for limit in args.topic_event_limits:
        retrievers[f"GraphV2 event_topic_top{limit}"] = HierarchyV2Retriever(
            graph,
            HierarchyV2RetrieverConfig(
                mode="event_topic",
                seed_topn=args.seed_topn,
                preselect_topn=args.preselect_topn,
                topic_event_limit=limit,
            ),
        )

    pools = {"BM25 Fact@50": {}}
    pools.update({name: {} for name in retrievers})
    bm25_topk = max(args.preselect_topn * 4, 300, 50, args.seed_topn)
    for item_index, item in enumerate(questions, start=1):
        hits = bm25.search(item["question"], top_k=bm25_topk)
        max_score = max((score for _node, score in hits), default=1.0) or 1.0
        bm25_scores = {node.node_id: score / max_score for node, score in hits}
        seed_hits = hits[: args.seed_topn]
        pools["BM25 Fact@50"][item["question_id"]] = [node.node_id for node, _score in hits[:50]]
        for stage_name, retriever in retrievers.items():
            pools[stage_name][item["question_id"]] = retriever.candidate_node_ids_from_seed_hits(
                item["question"],
                seed_hits,
                bm25_scores,
            )
        if item_index % 250 == 0 or item_index == len(questions):
            print(f"built pools {item_index}/{len(questions)} questions", flush=True)

    stages = {}
    per_stage = {}
    for stage_name, stage_pools in pools.items():
        results = [
            evaluate_candidate_pool(
                graph,
                question_id=item["question_id"],
                gold_evidence=item.get("evidence", item.get("gold_evidence", [])),
                candidate_node_ids=stage_pools[item["question_id"]],
            )
            for item in questions
        ]
        stages[stage_name] = summarize_oracle(results)
        stages[stage_name]["gold_density"] = average_gold_density(graph, questions, stage_pools)
        stages[stage_name]["gold_candidate_density"] = average_gold_candidate_density(graph, questions, stage_pools)
        per_stage[stage_name] = [result.__dict__ for result in results]

    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"stages": stages, "per_stage": per_stage}, indent=2), encoding="utf-8")
    write_markdown(stages, resolve_path(args.output_md))
    print(resolve_path(args.output_md))
    for name, summary in stages.items():
        print(
            f"{name}: questions={summary['num_questions']} hit={summary['hit']:.4f} "
            f"recall={summary['recall']:.4f} full_cover={summary['full_cover']:.4f} "
            f"avg_candidates={summary['avg_candidates']:.1f} gold_density={summary['gold_density']:.5f}"
        )


def flatten_questions(conversations: list[dict], limit: int) -> list[dict]:
    questions = []
    for conversation in conversations:
        for qa in conversation.get("qa", []):
            questions.append(qa)
            if limit and len(questions) >= limit:
                return questions
    return questions


def write_markdown(stages: dict, path) -> None:
    lines = [
        "# Graph v2 Oracle",
        "",
        "| Stage | Questions | Hit | Recall | FullCover | Avg Cand | GoldDensity | GoldCandDensity | Avg Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage_name, summary in stages.items():
        lines.append(
            f"| {stage_name} | {summary['num_questions']} | {summary['hit']:.4f} | "
            f"{summary['recall']:.4f} | {summary['full_cover']:.4f} | "
            f"{summary['avg_candidates']:.1f} | {summary['gold_density']:.5f} | "
            f"{summary['gold_candidate_density']:.5f} | {summary['avg_tokens']:.1f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def average_gold_density(graph, questions: list[dict], pools: dict[str, list[str]]) -> float:
    values = []
    for item in questions:
        gold = {normalize_evidence_id(eid) for eid in item.get("evidence", item.get("gold_evidence", []))}
        candidates = pools[item["question_id"]]
        matched = set()
        for node_id in candidates:
            matched.update(evidence_ids_for_node(graph, node_id) & gold)
        values.append(len(matched) / max(len(candidates), 1))
    return sum(values) / max(len(values), 1)


def average_gold_candidate_density(graph, questions: list[dict], pools: dict[str, list[str]]) -> float:
    values = []
    for item in questions:
        gold = {normalize_evidence_id(eid) for eid in item.get("evidence", item.get("gold_evidence", []))}
        candidates = pools[item["question_id"]]
        positive_candidates = 0
        for node_id in candidates:
            if evidence_ids_for_node(graph, node_id) & gold:
                positive_candidates += 1
        values.append(positive_candidates / max(len(candidates), 1))
    return sum(values) / max(len(values), 1)


if __name__ == "__main__":
    main()
