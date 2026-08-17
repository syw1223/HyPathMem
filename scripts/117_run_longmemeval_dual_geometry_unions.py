from __future__ import annotations

import argparse
import importlib.util
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
    parser.add_argument("--graph", default="outputs/longmemeval_s/graph_semantic_hierarchy_v3.json")
    parser.add_argument("--data", default="data/longmemeval/processed/longmemeval_s_mvp.json")
    parser.add_argument("--eu-predictions", default="outputs/longmemeval_s/predictions/topdown_eu_ee_candidates.json")
    parser.add_argument("--hyp-predictions", default="outputs/longmemeval_s/predictions/hyp_routes_candidates.json")
    parser.add_argument("--output-json", default="outputs/eval/longmemeval_dual_geometry_unions_ce.json")
    parser.add_argument("--output-md", default="outputs/eval/longmemeval_dual_geometry_unions_ce.md")
    parser.add_argument("--output-predictions", default="outputs/longmemeval_s/predictions/dual_geometry_union_candidates.json")
    parser.add_argument("--topk", default="5,20,50,100")
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
    graph = semantic.load_graph(resolve_path(args.graph), wanted_convs)
    fact_index = semantic.build_semantic_fact_index(graph)
    items = helpers.qa_items(data)

    eu = read_json(resolve_path(args.eu_predictions))["predictions"]
    hyp = read_json(resolve_path(args.hyp_predictions))["predictions"]
    bu_eu = eu["bu_eu_semantic_event_topic"]
    td_eu = eu["td_eu_semantic"]
    bu_hyp = hyp["bu_hyp"]
    td_hyp = hyp["td_hyp"]

    route_inputs = {
        "EH_bu_eu_td_hyp": [("bu_eu", bu_eu), ("td_hyp", td_hyp)],
        "HE_bu_hyp_td_eu": [("bu_hyp", bu_hyp), ("td_eu", td_eu)],
        "E_EuHyp_bu_eu_td_euhyp": [("bu_eu", bu_eu), ("td_eu", td_eu), ("td_hyp", td_hyp)],
        "EuHyp_E_bu_euhyp_td_eu": [("bu_eu", bu_eu), ("bu_hyp", bu_hyp), ("td_eu", td_eu)],
        "EuHyp_EuHyp_all_four": [("bu_eu", bu_eu), ("bu_hyp", bu_hyp), ("td_eu", td_eu), ("td_hyp", td_hyp)],
    }

    predictions: dict[str, dict[str, list[str]]] = {route: {} for route in route_inputs}
    for item in items:
        qid = item["question_id"]
        for route, inputs in route_inputs.items():
            predictions[route][qid] = union_ranked([values.get(qid, []) for _name, values in inputs], topn)

    results: dict[str, Any] = {
        "config": {
            "graph": args.graph,
            "data": args.data,
            "eu_predictions": args.eu_predictions,
            "hyp_predictions": args.hyp_predictions,
            "topk": topks,
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


def union_ranked(groups: list[list[str]], topn: int) -> list[str]:
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    weights = [1.0, 0.99, 0.98, 0.97]
    for group_idx, values in enumerate(groups):
        weight = weights[group_idx] if group_idx < len(weights) else 0.95
        for rank, fact_id in enumerate(values, start=1):
            if fact_id not in order:
                order[fact_id] = len(order)
            score = weight / (rank + 8.0)
            scores[fact_id] = max(scores.get(fact_id, float("-inf")), score)
    return [fact_id for fact_id, _ in sorted(scores.items(), key=lambda item: (-item[1], order[item[0]], item[0]))[:topn]]


if __name__ == "__main__":
    main()
