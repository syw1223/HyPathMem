from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

from common import read_json, resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.hyperbolic_topdown_retriever import HyperbolicTopDownRetriever
from hytopomem.retrieval.topdown_semantic_retriever import (
    SentenceTransformerEncoder,
    TopDownSemanticConfig,
    default_embedder,
)


HELPERS_SPEC = importlib.util.spec_from_file_location(
    "longmemeval_retrieval_sanity_helpers",
    Path(__file__).resolve().parent / "112_run_longmemeval_retrieval_sanity.py",
)
helpers = importlib.util.module_from_spec(HELPERS_SPEC)
assert HELPERS_SPEC and HELPERS_SPEC.loader
HELPERS_SPEC.loader.exec_module(helpers)

SEM_SPEC = importlib.util.spec_from_file_location(
    "longmemeval_semantic_bottom_up_helpers",
    Path(__file__).resolve().parent / "113_run_longmemeval_semantic_bottom_up.py",
)
semantic = importlib.util.module_from_spec(SEM_SPEC)
assert SEM_SPEC and SEM_SPEC.loader
SEM_SPEC.loader.exec_module(semantic)

BU_HYP_SPEC = importlib.util.spec_from_file_location(
    "hyperbolic_bottom_up_helpers",
    Path(__file__).resolve().parent / "61_run_hyperbolic_bottom_up_retrieval.py",
)
bu_hyp = importlib.util.module_from_spec(BU_HYP_SPEC)
assert BU_HYP_SPEC and BU_HYP_SPEC.loader
sys.modules[BU_HYP_SPEC.name] = bu_hyp
BU_HYP_SPEC.loader.exec_module(bu_hyp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/longmemeval_s/graph_semantic_hierarchy_v3.json")
    parser.add_argument("--data", default="data/longmemeval/processed/longmemeval_s_mvp.json")
    parser.add_argument("--checkpoint", default="outputs/longmemeval_s/models/lorentz_router_v3_fact_event_topic_unit.pt")
    parser.add_argument("--output-json", default="outputs/eval/longmemeval_hyp_routes_ce.json")
    parser.add_argument("--output-md", default="outputs/eval/longmemeval_hyp_routes_ce.md")
    parser.add_argument("--output-predictions", default="outputs/longmemeval_s/predictions/hyp_routes_candidates.json")
    parser.add_argument("--topk", default="5,20,50,100")
    parser.add_argument("--embedder", default=default_embedder())
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--event-topic-cache", default="outputs/longmemeval_s/embeddings/semantic_v3_event_topic_minilm.npz")
    parser.add_argument("--fact-node-cache", default="outputs/longmemeval_s/embeddings/semantic_v3_fact_event_topic_minilm.npz")
    parser.add_argument("--router-device", default="cuda")
    parser.add_argument("--router-batch-size", type=int, default=512)
    parser.add_argument("--event-topk", type=int, default=20)
    parser.add_argument("--topic-topk", type=int, default=3)
    parser.add_argument("--events-per-topic", type=int, default=3)
    parser.add_argument("--facts-per-event", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--mode", choices=["event", "topic", "both"], default="both")
    parser.add_argument("--hierarchy-version", choices=["v3", "v3_3"], default="v3")
    parser.add_argument("--seed-topn", type=int, default=20)
    parser.add_argument("--preselect-topn", type=int, default=100)
    parser.add_argument("--same-event-limit", type=int, default=30)
    parser.add_argument("--topic-event-limit", type=int, default=8)
    parser.add_argument("--topic-fact-limit-per-event", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ce-model", default="")
    parser.add_argument("--ce-device", default=None)
    parser.add_argument("--ce-batch-size", type=int, default=64)
    args = parser.parse_args()

    topks = [int(item) for item in args.topk.split(",") if item.strip()]
    topn = max(topks)
    data = read_json(resolve_path(args.data))
    if args.limit:
        data = data[: args.limit]
    wanted_convs = {item["conversation_id"] for item in data}
    semantic_payload = semantic.load_graph(resolve_path(args.graph), wanted_convs)
    fact_index = semantic.build_semantic_fact_index(semantic_payload)
    items = helpers.qa_items(data)
    graph = JsonGraphStore().load(resolve_path(args.graph))

    encoder = SentenceTransformerEncoder(
        args.embedder,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )
    td_retriever = HyperbolicTopDownRetriever(
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
            restrict_conversation=True,
            hierarchy_version=args.hierarchy_version,
        ),
        embedding_cache_path=resolve_path(args.event_topic_cache),
        device=args.router_device,
        batch_size=args.router_batch_size,
    )
    bu_retriever = bu_hyp.HyperbolicBottomUpRetriever(
        graph,
        encoder=encoder,
        checkpoint_path=resolve_path(args.checkpoint),
        embedding_cache_path=resolve_path(args.fact_node_cache),
        config=bu_hyp.HypBottomConfig(
            seed_topn=args.seed_topn,
            preselect_topn=args.preselect_topn,
            same_event_limit=args.same_event_limit,
            topic_event_limit=args.topic_event_limit,
            topic_fact_limit_per_event=args.topic_fact_limit_per_event,
            hierarchy_version=args.hierarchy_version,
        ),
        device=args.router_device,
        batch_size=args.router_batch_size,
    )
    query_vectors = encoder.encode([item["question"] for item in items])

    predictions = {
        "bu_hyp": {},
        "td_hyp": {},
        "hh_bu_td_union": {},
    }
    for index, (item, query_vector) in enumerate(zip(items, query_vectors), start=1):
        qid = item["question_id"]
        bu_candidates = bu_retriever.candidates_from_vector(qid, item["question"], query_vector)
        td_candidates = td_retriever.candidates_from_vector(qid, query_vector)
        bu_paths = [candidate.node.node_id for candidate in bu_candidates[:topn]]
        td_paths = [candidate.node.node_id for candidate in td_candidates[:topn]]
        predictions["bu_hyp"][qid] = bu_paths
        predictions["td_hyp"][qid] = td_paths
        predictions["hh_bu_td_union"][qid] = union_ranked(bu_paths, td_paths, topn)
        if index % 50 == 0 or index == len(items):
            print(f"generated Hyp candidates {index}/{len(items)}", flush=True)

    results: dict[str, Any] = {
        "config": {
            "graph": args.graph,
            "data": args.data,
            "checkpoint": args.checkpoint,
            "topk": topks,
            "event_topk": args.event_topk,
            "topic_topk": args.topic_topk,
            "events_per_topic": args.events_per_topic,
            "facts_per_event": args.facts_per_event,
            "max_candidates": args.max_candidates,
            "mode": args.mode,
            "hierarchy_version": args.hierarchy_version,
            "seed_topn": args.seed_topn,
            "preselect_topn": args.preselect_topn,
            "embedder": args.embedder,
            "event_topic_cache": args.event_topic_cache,
            "fact_node_cache": args.fact_node_cache,
            "limit": args.limit,
        },
        "dataset": {
            "instances": len(data),
            "qa": len(items),
            "qa_with_gold": sum(bool(item["gold_raw_ids"]) for item in items),
            "abstention": sum(bool(item["is_abstention"]) for item in items),
        },
        "routes": {},
    }
    for route, route_predictions in predictions.items():
        results["routes"][route] = helpers.evaluate(items, route_predictions, fact_index, topks)
        if args.ce_model:
            ce_predictions = helpers.ce_rerank_predictions(
                items,
                route_predictions,
                fact_index,
                model_name_or_path=args.ce_model,
                device=args.ce_device,
                batch_size=args.ce_batch_size,
                topn=topn,
                route=route,
            )
            results["routes"][f"{route}_ce"] = helpers.evaluate(items, ce_predictions, fact_index, topks)

    write_json(results, resolve_path(args.output_json))
    write_json({"predictions": predictions, "config": results["config"]}, resolve_path(args.output_predictions))
    helpers.write_markdown(results, resolve_path(args.output_md))
    print(f"wrote {resolve_path(args.output_json)}")
    print(f"wrote {resolve_path(args.output_md)}")
    print(f"wrote {resolve_path(args.output_predictions)}")
    for route, payload in results["routes"].items():
        print(route, payload["overall_with_gold"], flush=True)


def union_ranked(left: list[str], right: list[str], topn: int) -> list[str]:
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for source_weight, values in ((1.0, left), (0.98, right)):
        for rank, fact_id in enumerate(values, start=1):
            if fact_id not in order:
                order[fact_id] = len(order)
            score = source_weight / (rank + 8.0)
            scores[fact_id] = max(scores.get(fact_id, float("-inf")), score)
    return [fact_id for fact_id, _ in sorted(scores.items(), key=lambda item: (-item[1], order[item[0]], item[0]))[:topn]]


if __name__ == "__main__":
    main()
