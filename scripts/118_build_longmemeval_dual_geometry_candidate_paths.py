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
    parser.add_argument("--route", default="EuHyp_EuHyp_all_four")
    parser.add_argument("--candidate-topn", type=int, default=100)
    parser.add_argument("--output", default="outputs/longmemeval_s/paths/dual_geometry_euhyp_all_four_top100_ce_paths.json")
    parser.add_argument("--summary-json", default="outputs/eval/longmemeval_dual_geometry_euhyp_all_four_top100_ce_paths_summary.json")
    parser.add_argument("--topk", default="5,20,50,100")
    parser.add_argument("--ce-model", default="")
    parser.add_argument("--ce-device", default=None)
    parser.add_argument("--ce-batch-size", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    topks = [int(item) for item in args.topk.split(",") if item.strip()]
    data = read_json(resolve_path(args.data))
    if args.limit:
        data = data[: args.limit]
    wanted_convs = {item["conversation_id"] for item in data}
    graph = semantic.load_graph(resolve_path(args.graph), wanted_convs)
    fact_index = semantic.build_semantic_fact_index(graph)
    items = helpers.qa_items(data)
    eu = read_json(resolve_path(args.eu_predictions))["predictions"]
    hyp = read_json(resolve_path(args.hyp_predictions))["predictions"]

    route_maps = {
        "bu_eu": eu["bu_eu_semantic_event_topic"],
        "td_eu": eu["td_eu_semantic"],
        "bu_hyp": hyp["bu_hyp"],
        "td_hyp": hyp["td_hyp"],
    }
    route_groups = {
        "EH_bu_eu_td_hyp": ["bu_eu", "td_hyp"],
        "HE_bu_hyp_td_eu": ["bu_hyp", "td_eu"],
        "E_EuHyp_bu_eu_td_euhyp": ["bu_eu", "td_eu", "td_hyp"],
        "EuHyp_E_bu_euhyp_td_eu": ["bu_eu", "bu_hyp", "td_eu"],
        "EuHyp_EuHyp_all_four": ["bu_eu", "bu_hyp", "td_eu", "td_hyp"],
    }
    route_names = route_groups[args.route]

    raw_items = []
    for item in items:
        qid = item["question_id"]
        conv_index = fact_index.get(item["conversation_id"])
        if conv_index is None:
            raw_items.append(output_item(item, []))
            continue
        fact_ids = union_ranked([route_maps[name].get(qid, []) for name in route_names], args.candidate_topn)
        paths = [
            build_path(fact_id, rank, qid, route_maps, route_names, conv_index)
            for rank, fact_id in enumerate(fact_ids, start=1)
            if fact_id in conv_index["fact_by_id"]
        ]
        raw_items.append(output_item(item, paths))

    if args.ce_model:
        score_map = ce_score_paths(
            items,
            raw_items,
            fact_index,
            model_name_or_path=args.ce_model,
            device=args.ce_device,
            batch_size=args.ce_batch_size,
        )
        for item in raw_items:
            qid = item["question_id"]
            for path in item.get("paths", []):
                fact_id = evidence_node_id(path)
                score = score_map.get((qid, fact_id), float(path.get("score", 0.0)))
                path["score"] = score
                scores = dict(path.get("scores", {}))
                scores["cross_encoder"] = score
                path["scores"] = scores
            item["paths"].sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)

    summary = {
        "config": {
            "graph": args.graph,
            "data": args.data,
            "route": args.route,
            "route_names": route_names,
            "candidate_topn": args.candidate_topn,
            "ce_model": args.ce_model,
        },
        "dataset": {
            "instances": len(data),
            "qa": len(items),
            "qa_with_gold": sum(bool(item["gold_raw_ids"]) for item in items),
            "abstention": sum(bool(item["is_abstention"]) for item in items),
        },
        "metrics": helpers.evaluate(items, {item["question_id"]: [evidence_node_id(path) for path in item["paths"]] for item in raw_items}, fact_index, topks),
        "avg_candidates": sum(len(item.get("paths", [])) for item in raw_items) / max(len(raw_items), 1),
    }
    write_json(raw_items, resolve_path(args.output))
    write_json(summary, resolve_path(args.summary_json))
    print(f"wrote {resolve_path(args.output)}")
    print(f"wrote {resolve_path(args.summary_json)}")
    print(summary["metrics"]["overall_with_gold"])


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


def build_path(
    fact_id: str,
    rank: int,
    qid: str,
    route_maps: dict[str, dict[str, list[str]]],
    route_names: list[str],
    conv_index: dict[str, Any],
) -> dict:
    route_ranks = {}
    route_sources = []
    for name in route_names:
        values = route_maps[name].get(qid, [])
        if fact_id in values:
            route_ranks[name] = values.index(fact_id) + 1
            route_sources.append(name)
    fact = conv_index["fact_by_id"][fact_id]
    event_id = str(fact.get("event_id") or "")
    episode_id = str(fact.get("episode_id") or "")
    topic_id = str(fact.get("topic_id") or "")
    best_rank = min(route_ranks.values()) if route_ranks else rank
    metadata = {
        "evidence_node_id": fact_id,
        "candidate_source": "+".join(route_sources),
        "route_source": "+".join(route_sources),
        "event_node_id": event_id,
        "episode_node_id": episode_id,
        "topic_node_id": topic_id,
        "has_topdown_route": str(any(name.startswith("td_") for name in route_sources)).lower(),
        "has_bottom_up_route": str(any(name.startswith("bu_") for name in route_sources)).lower(),
        "is_eu_route": str(any(name.endswith("_eu") for name in route_sources)).lower(),
        "is_hyp_route": str(any(name.endswith("_hyp") for name in route_sources)).lower(),
        "route_source_count": str(len(route_sources)),
        "route_rank": str(best_rank),
        "bottom_up_rank": str(min([rank for name, rank in route_ranks.items() if name.startswith("bu_")] or [0])),
        "eu_event_rank": str(route_ranks.get("td_eu", 0)),
        "eu_topic_rank": str(route_ranks.get("td_eu", 0)),
        "hyp_event_rank": str(route_ranks.get("td_hyp", 0)),
        "hyp_topic_rank": str(route_ranks.get("td_hyp", 0)),
        "eu_event_score": f"{1.0 / (route_ranks.get('td_eu', 9999) + 8.0):.8f}" if "td_eu" in route_ranks else "0.00000000",
        "hyp_event_score": f"{1.0 / (route_ranks.get('td_hyp', 9999) + 8.0):.8f}" if "td_hyp" in route_ranks else "0.00000000",
        "bm25_norm": f"{1.0 / (rank + 8.0):.8f}",
    }
    base_score = max((1.0 / (value + 8.0) for value in route_ranks.values()), default=1.0 / (rank + 8.0))
    node_ids = [node_id for node_id in [topic_id, episode_id, event_id, fact_id] if node_id]
    return {
        "node_ids": node_ids,
        "edge_ids": [],
        "score": float(base_score),
        "scores": {"base": float(base_score)},
        "metadata": metadata,
    }


def output_item(item: dict[str, Any], paths: list[dict]) -> dict:
    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "answer": None,
        "question_type": item.get("question_type", "unknown"),
        "gold_evidence": list(item.get("gold_raw_ids", [])),
        "is_abstention": bool(item.get("is_abstention")),
        "paths": paths,
        "metadata": {"method": "longmemeval_dual_geometry_euhyp_all_four_top100_ce"},
    }


def ce_score_paths(
    qa_items: list[dict[str, Any]],
    path_items: list[dict[str, Any]],
    fact_index: dict[str, dict[str, Any]],
    *,
    model_name_or_path: str,
    device: str | None,
    batch_size: int,
) -> dict[tuple[str, str], float]:
    from sentence_transformers import CrossEncoder

    questions = {item["question_id"]: item for item in qa_items}
    model = CrossEncoder(model_name_or_path, device=device)
    pairs = []
    keys = []
    for item in path_items:
        qid = item["question_id"]
        qitem = questions[qid]
        conv_index = fact_index[qitem["conversation_id"]]
        for path in item.get("paths", []):
            fact_id = evidence_node_id(path)
            fact = conv_index["fact_by_id"].get(fact_id)
            if fact is None:
                continue
            keys.append((qid, fact_id))
            pairs.append((qitem["question"], fact["text"]))
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True) if pairs else []
    return {key: float(score) for key, score in zip(keys, scores)}


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


if __name__ == "__main__":
    main()
