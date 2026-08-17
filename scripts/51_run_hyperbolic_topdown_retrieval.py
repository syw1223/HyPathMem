from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from common import load_config, read_json, resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.cross_encoder_reranker import CrossEncoderReranker
from hytopomem.retrieval.hyperbolic_topdown_retriever import HyperbolicTopDownRetriever
from hytopomem.retrieval.topdown_semantic_retriever import (
    SentenceTransformerEncoder,
    TopDownSemanticConfig,
    default_embedder,
)


DEFAULT_LOCAL_CE = "/home/sunyuwei/LightMem/LightMem/models/cross-encoder-ms-marco-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_hierarchy_v2.json")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--bottom-up-paths", default=None)
    parser.add_argument("--checkpoint", default="outputs/models/graph_v2_lorentz_router/minilm_structure_router.pt")
    parser.add_argument("--mode", choices=["event", "topic", "both"], default="both")
    parser.add_argument("--hierarchy-version", choices=["v2", "v3", "v3_3"], default="v2")
    parser.add_argument("--embedder", default=default_embedder())
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-cache", default="outputs/embeddings/graph_v2_minilm_event_topic.npz")
    parser.add_argument("--router-device", default="cuda")
    parser.add_argument("--router-batch-size", type=int, default=1024)
    parser.add_argument("--event-topk", type=int, default=20)
    parser.add_argument("--topic-topk", type=int, default=3)
    parser.add_argument("--events-per-topic", type=int, default=3)
    parser.add_argument("--facts-per-event", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--global-search", action="store_true")
    parser.add_argument("--ce-model", default=DEFAULT_LOCAL_CE)
    parser.add_argument("--ce-device", default=None)
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--skip-ce", action="store_true")
    parser.add_argument("--final-topk", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-prefix", default="outputs/topdown/hyp_minilm_both")
    args = parser.parse_args()

    helpers = load_topdown_helpers()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    question_path = resolve_path(args.questions or config["data"]["processed_path"])
    questions = helpers.flatten_questions(read_json(question_path), args.limit)
    bottom_items = helpers.load_item_map(args.bottom_up_paths)

    encoder = SentenceTransformerEncoder(
        args.embedder,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )
    retriever = HyperbolicTopDownRetriever(
        graph,
        encoder=encoder,
        checkpoint_path=resolve_path(args.checkpoint),
        config=TopDownSemanticConfig(
            event_topk=args.event_topk,
            topic_topk=args.topic_topk,
            events_per_topic=args.events_per_topic,
            facts_per_event=args.facts_per_event,
            max_candidates=args.max_candidates,
            mode=args.mode,
            restrict_conversation=not args.global_search,
            hierarchy_version=args.hierarchy_version,
        ),
        embedding_cache_path=resolve_path(args.embedding_cache),
        device=args.router_device,
        batch_size=args.router_batch_size,
    )
    query_vectors = encoder.encode([item["question"] for item in questions])
    topdown_groups = {}
    union_groups = {}
    topdown_pools = {}
    union_pools = {}
    for index, (qa, query_vector) in enumerate(zip(questions, query_vectors), start=1):
        topdown_candidates = retriever.candidates_from_vector(qa["question_id"], query_vector)
        topdown_groups[qa["question_id"]] = topdown_candidates
        topdown_pools[qa["question_id"]] = helpers.candidate_ids(topdown_candidates)
        if bottom_items:
            bottom_candidates = helpers.candidates_from_paths(graph, bottom_items.get(qa["question_id"], {}).get("paths", []))
            union_candidates = helpers.merge_candidates(bottom_candidates, topdown_candidates)
            union_groups[qa["question_id"]] = union_candidates
            union_pools[qa["question_id"]] = helpers.candidate_ids(union_candidates)
        if index % 500 == 0 or index == len(questions):
            print(f"generated candidates {index}/{len(questions)} questions", flush=True)

    score_map = {}
    if not args.skip_ce:
        reranker = CrossEncoderReranker(args.ce_model, device=args.ce_device, batch_size=args.ce_batch_size)
        score_map = helpers.score_candidate_groups(
            reranker,
            {item["question_id"]: item["question"] for item in questions},
            topdown_groups,
            union_groups,
        )

    topdown_outputs = []
    union_outputs = []
    for index, qa in enumerate(questions, start=1):
        topdown_paths = helpers.rank_candidates(
            qa["question_id"],
            qa["question"],
            topdown_groups.get(qa["question_id"], []),
            score_map=score_map,
            top_k=args.final_topk,
            retriever_name=f"hyp_topdown_{args.mode}",
        )
        topdown_outputs.append(output_item(qa, topdown_paths, f"hyp_topdown_{args.mode}", args, retriever))
        if bottom_items:
            union_paths = helpers.rank_candidates(
                qa["question_id"],
                qa["question"],
                union_groups.get(qa["question_id"], []),
                score_map=score_map,
                top_k=args.final_topk,
                retriever_name=f"bottom_union_hyp_topdown_{args.mode}",
            )
            union_outputs.append(output_item(qa, union_paths, f"bottom_union_hyp_topdown_{args.mode}", args, retriever))
        if index % 500 == 0 or index == len(questions):
            print(f"ranked {index}/{len(questions)} questions", flush=True)

    prefix = resolve_path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    topdown_paths_path = Path(f"{prefix}_paths.json")
    topdown_eval_path = Path(f"{prefix}_eval.json")
    write_json(topdown_outputs, topdown_paths_path)
    topdown_eval = helpers.evaluation_payload(
        graph,
        topdown_outputs,
        topdown_pools,
        args.final_topk,
        method=f"hyp_topdown_{args.mode}",
    )
    write_json(topdown_eval, topdown_eval_path)
    print(f"topdown summary={topdown_eval['summary']}")
    print(f"topdown candidate_pool={topdown_eval['candidate_pool']}")

    if bottom_items:
        union_paths_path = Path(f"{prefix}_union_paths.json")
        union_eval_path = Path(f"{prefix}_union_eval.json")
        diagnostics_path = Path(f"{prefix}_paired_diagnostics.json")
        write_json(union_outputs, union_paths_path)
        union_eval = helpers.evaluation_payload(
            graph,
            union_outputs,
            union_pools,
            args.final_topk,
            method=f"bottom_union_hyp_topdown_{args.mode}",
        )
        write_json(union_eval, union_eval_path)
        bottom_rows = [bottom_items[item["question_id"]] for item in questions if item["question_id"] in bottom_items]
        diagnostics = helpers.paired_diagnostics(
            graph,
            bottom_rows,
            topdown_outputs,
            union_outputs,
            topdown_pools,
            args.final_topk,
        )
        write_json(diagnostics, diagnostics_path)
        print(f"union summary={union_eval['summary']}")
        print(f"union candidate_pool={union_eval['candidate_pool']}")
        print(f"paired={diagnostics['summary']}")
        print(f"wrote {union_paths_path}, {union_eval_path}, {diagnostics_path}")
    print(f"wrote {topdown_paths_path}, {topdown_eval_path}")


def output_item(qa: dict, paths: list, method: str, args, retriever: HyperbolicTopDownRetriever) -> dict:
    return {
        "question_id": qa["question_id"],
        "question": qa["question"],
        "answer": qa.get("answer"),
        "category": qa.get("category"),
        "gold_evidence": qa.get("evidence", []),
        "paths": [path.model_dump(mode="json") for path in paths],
        "metadata": {
            "method": method,
            "mode": args.mode,
            "hierarchy_version": args.hierarchy_version,
            "embedder": args.embedder,
            "checkpoint": str(resolve_path(args.checkpoint)),
            "checkpoint_metadata": retriever.checkpoint_metadata,
            "event_topk": args.event_topk,
            "topic_topk": args.topic_topk,
            "events_per_topic": args.events_per_topic,
            "facts_per_event": args.facts_per_event,
            "max_candidates": args.max_candidates,
            "restrict_conversation": not args.global_search,
            "ce_model": None if args.skip_ce else args.ce_model,
            "final_topk": args.final_topk,
        },
    }


def load_topdown_helpers():
    path = Path(__file__).with_name("49_run_topdown_semantic_retrieval.py")
    spec = importlib.util.spec_from_file_location("topdown_semantic_retrieval_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
