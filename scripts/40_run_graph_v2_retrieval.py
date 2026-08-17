from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evaluate_item, summarize
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.cross_encoder_reranker import CrossEncoderReranker, reranked_paths
from hytopomem.retrieval.hierarchy_v2_retriever import HierarchyV2Retriever, HierarchyV2RetrieverConfig


DEFAULT_LOCAL_CE = "/home/sunyuwei/LightMem/LightMem/models/cross-encoder-ms-marco-MiniLM-L6-v2"


def default_model() -> str:
    return DEFAULT_LOCAL_CE if Path(DEFAULT_LOCAL_CE).exists() else "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_hierarchy_v2.json")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--mode", choices=["same_event", "same_topic", "same_topic_only", "event_topic"], default="event_topic")
    parser.add_argument("--seed-topn", type=int, default=20)
    parser.add_argument("--preselect-topn", type=int, default=100)
    parser.add_argument("--topic-event-limit", type=int, default=8)
    parser.add_argument("--topic-fact-limit-per-event", type=int, default=12)
    parser.add_argument("--final-topk", type=int, default=5)
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="outputs/paths/full_graph_v2_event_topic_ce_top5.json")
    parser.add_argument("--eval-output", default="outputs/eval/full_graph_v2_event_topic_ce_top5_k5.json")
    args = parser.parse_args()

    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    questions = flatten_questions(read_json(resolve_path(args.questions or config["data"]["processed_path"])), args.limit)
    retriever = HierarchyV2Retriever(
        graph,
        HierarchyV2RetrieverConfig(
            mode=args.mode,
            seed_topn=args.seed_topn,
            preselect_topn=args.preselect_topn,
            topic_event_limit=args.topic_event_limit,
            topic_fact_limit_per_event=args.topic_fact_limit_per_event,
        ),
    )
    reranker = CrossEncoderReranker(args.model, device=args.device, batch_size=args.batch_size)

    outputs = []
    total_candidates = 0
    for qa_index, qa in enumerate(questions, start=1):
        candidates = retriever.candidates(qa["question"])
        total_candidates += len(candidates)
        ranked = reranker.rerank(qa["question"], candidates, top_k=args.final_topk)
        paths = reranked_paths(qa["question_id"], ranked, retriever_name=f"graph_v2_{args.mode}_ce")
        outputs.append(
            {
                "question_id": qa["question_id"],
                "question": qa["question"],
                "answer": qa.get("answer"),
                "category": qa.get("category"),
                "gold_evidence": qa.get("evidence", []),
                "paths": [path.model_dump(mode="json") for path in paths],
                "metadata": {
                    "method": f"graph_v2_{args.mode}_ce",
                    "mode": args.mode,
                    "seed_topn": args.seed_topn,
                    "preselect_topn": args.preselect_topn,
                    "topic_event_limit": args.topic_event_limit,
                    "topic_fact_limit_per_event": args.topic_fact_limit_per_event,
                    "final_topk": args.final_topk,
                    "model": args.model,
                },
            }
        )
        if qa_index % 250 == 0 or qa_index == len(questions):
            print(f"processed {qa_index}/{len(questions)} questions", flush=True)

    output_path = resolve_path(args.output)
    write_json(outputs, output_path)
    results = [evaluate_item(graph, item, args.final_topk) for item in outputs]
    payload = {
        "summary": summarize(results),
        "per_question": [result.__dict__ for result in results],
        "metadata": {
            "paths": str(output_path),
            "mode": args.mode,
            "seed_topn": args.seed_topn,
            "preselect_topn": args.preselect_topn,
            "topic_event_limit": args.topic_event_limit,
            "topic_fact_limit_per_event": args.topic_fact_limit_per_event,
            "final_topk": args.final_topk,
        },
    }
    eval_path = resolve_path(args.eval_output)
    write_json(payload, eval_path)
    avg_candidates = total_candidates / max(len(outputs), 1)
    print(f"wrote paths to {output_path}; avg_candidates={avg_candidates:.1f}")
    print(f"wrote eval to {eval_path}; summary={payload['summary']}")


def flatten_questions(conversations: list[dict], limit: int) -> list[dict]:
    questions = []
    for conversation in conversations:
        for qa in conversation.get("qa", []):
            questions.append(qa)
            if limit and len(questions) >= limit:
                return questions
    return questions


if __name__ == "__main__":
    main()
