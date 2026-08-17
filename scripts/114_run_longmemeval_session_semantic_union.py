from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from common import read_json, resolve_path, write_json


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-graph", default="outputs/longmemeval_s/graph_session_hierarchy_v1.json")
    parser.add_argument("--semantic-graph", default="outputs/longmemeval_s/graph_semantic_hierarchy_v3.json")
    parser.add_argument("--data", default="data/longmemeval/processed/longmemeval_s_mvp.json")
    parser.add_argument("--output-json", default="outputs/eval/longmemeval_session_semantic_union_ce.json")
    parser.add_argument("--output-md", default="outputs/eval/longmemeval_session_semantic_union_ce.md")
    parser.add_argument("--topk", default="5,20,50,100")
    parser.add_argument("--seed-topk", type=int, default=20)
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

    session_nodes = helpers.load_graph_nodes(resolve_path(args.session_graph), wanted_convs)
    session_index = helpers.build_fact_index(session_nodes)
    semantic_graph = semantic.load_graph(resolve_path(args.semantic_graph), wanted_convs)
    semantic_index = semantic.build_semantic_fact_index(semantic_graph)
    items = helpers.qa_items(data)

    results = {
        "config": {
            "session_graph": args.session_graph,
            "semantic_graph": args.semantic_graph,
            "data": args.data,
            "topk": topks,
            "seed_topk": args.seed_topk,
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

    predictions = {route: {} for route in ("session_expand", "semantic_event_topic", "session_semantic_union")}
    for item in items:
        qid = item["question_id"]
        conv_id = item["conversation_id"]
        session_conv = session_index.get(conv_id)
        semantic_conv = semantic_index.get(conv_id)
        if session_conv is None or semantic_conv is None:
            for route in predictions:
                predictions[route][qid] = []
            continue
        session_ranked = helpers.rank_bm25(session_conv, item["question"], topn=max(topks + [args.seed_topk]))
        semantic_ranked = helpers.rank_bm25(semantic_conv, item["question"], topn=max(topks + [args.seed_topk]))
        session_paths = helpers.session_expand(session_conv, session_ranked, seed_topk=args.seed_topk, topn=topn)
        semantic_paths = semantic.semantic_expand(
            semantic_conv,
            semantic_ranked,
            seed_topk=args.seed_topk,
            topn=topn,
            include_topic=True,
        )
        predictions["session_expand"][qid] = session_paths
        predictions["semantic_event_topic"][qid] = semantic_paths
        predictions["session_semantic_union"][qid] = union_ranked(session_paths, semantic_paths, topn)

    for route, route_predictions in predictions.items():
        results["routes"][route] = helpers.evaluate(items, route_predictions, session_index, topks)
        if args.ce_model:
            ce_predictions = helpers.ce_rerank_predictions(
                items,
                route_predictions,
                session_index,
                model_name_or_path=args.ce_model,
                device=args.ce_device,
                batch_size=args.ce_batch_size,
                topn=topn,
                route=route,
            )
            results["routes"][f"{route}_ce"] = helpers.evaluate(items, ce_predictions, session_index, topks)

    write_json(results, resolve_path(args.output_json))
    helpers.write_markdown(results, resolve_path(args.output_md))
    print(f"wrote {resolve_path(args.output_json)}")
    print(f"wrote {resolve_path(args.output_md)}")
    for route, payload in results["routes"].items():
        print(route, payload["overall_with_gold"])


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
