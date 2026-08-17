from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from common import load_config, read_json, resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topdown_semantic_retriever import merge_route_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_semantic_hierarchy_v3_2_gpt4o_semantic.json")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--bottom-eu", default="outputs/paths/full_graph_v2_provenance_fix_event_topic_ce_top100.json")
    parser.add_argument("--bottom-hyp", default="outputs/dual_geometry/full_graph_v3_2_pure_hyp_bu_hyp_paths.json")
    parser.add_argument("--eu-topdown", default="outputs/_archive_not_mainline_20260616/topdown/full_graph_v3_2_gpt4o_eu_both_ce_selector_top20base_paths.json")
    parser.add_argument("--hyp-topdown", default="outputs/topdown/full_graph_v3_2_gpt4o_hyp_retrained_both_hardneg_ce_selector_top20base_paths.json")
    parser.add_argument("--candidate-topn", type=int, default=100)
    parser.add_argument("--output-prefix", default="outputs/dual_geometry/full_graph_v3_2_true_union")
    args = parser.parse_args()

    helpers = load_topdown_helpers()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    question_path = resolve_path(args.questions or config["data"]["processed_path"])
    questions = helpers.flatten_questions(read_json(question_path), 0)

    bottom_eu_items = load_item_map(args.bottom_eu)
    bottom_hyp_items = load_item_map(args.bottom_hyp)
    eu_items = load_item_map(args.eu_topdown)
    hyp_items = load_item_map(args.hyp_topdown)

    outputs = {
        "B_true_bu_euhyp_td_hyp": [],
        "D_true_bu_euhyp_td_euhyp": [],
    }
    pools = {key: {} for key in outputs}

    for index, qa in enumerate(questions, start=1):
        qid = qa["question_id"]
        bottom_eu_paths = [
            normalize_path(path, route="bottom_eu", rank=rank)
            for rank, path in enumerate(bottom_eu_items.get(qid, {}).get("paths", []), start=1)
        ]
        bottom_hyp_paths = [
            normalize_path(path, route="bottom_hyp", rank=rank)
            for rank, path in enumerate(bottom_hyp_items.get(qid, {}).get("paths", []), start=1)
        ]
        eu_paths = [
            normalize_path(path, route="eu", rank=rank)
            for rank, path in enumerate(eu_items.get(qid, {}).get("paths", []), start=1)
        ]
        hyp_paths = [
            normalize_path(path, route="hyp", rank=rank)
            for rank, path in enumerate(hyp_items.get(qid, {}).get("paths", []), start=1)
        ]

        b_paths = merge_path_lists([bottom_eu_paths, bottom_hyp_paths, hyp_paths], args.candidate_topn)
        d_paths = merge_path_lists([bottom_eu_paths, bottom_hyp_paths, eu_paths, hyp_paths], args.candidate_topn)

        outputs["B_true_bu_euhyp_td_hyp"].append(output_item(qa, b_paths, args, "B_true_bu_euhyp_td_hyp"))
        outputs["D_true_bu_euhyp_td_euhyp"].append(output_item(qa, d_paths, args, "D_true_bu_euhyp_td_euhyp"))
        pools["B_true_bu_euhyp_td_hyp"][qid] = [evidence_node_id(path) for path in b_paths]
        pools["D_true_bu_euhyp_td_euhyp"][qid] = [evidence_node_id(path) for path in d_paths]

        if index % 500 == 0 or index == len(questions):
            print(f"built true dual-geometry union candidates {index}/{len(questions)}", flush=True)

    prefix = resolve_path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for key, items in outputs.items():
        path = Path(f"{prefix}_{key}_paths.json")
        eval_path = Path(f"{prefix}_{key}_eval.json")
        write_json(items, path)
        payload = helpers.evaluation_payload(graph, items, pools[key], args.candidate_topn, method=key)
        write_json(payload, eval_path)
        print(f"{key} summary={payload['summary']}")
        print(f"{key} candidate_pool={payload['candidate_pool']}")
        print(f"wrote {path}")


def merge_path_lists(groups: list[list[dict]], topn: int) -> list[dict]:
    merged: dict[str, dict] = {}
    for group in groups:
        for path in group:
            node_id = evidence_node_id(path)
            if not node_id:
                continue
            previous = merged.get(node_id)
            if previous is None:
                merged[node_id] = dict(path)
            else:
                merged[node_id] = merge_paths(previous, path)
    rows = list(merged.values())
    rows.sort(key=lambda path: ce_score(path), reverse=True)
    return rows[:topn]


def merge_paths(left: dict, right: dict) -> dict:
    left_score = ce_score(left)
    right_score = ce_score(right)
    base = dict(left if left_score >= right_score else right)
    metadata = merge_route_metadata(dict(left.get("metadata", {})), dict(right.get("metadata", {})))
    scores = dict(left.get("scores", {}))
    for key, value in right.get("scores", {}).items():
        if key not in scores:
            scores[key] = value
        elif key in {"cross_encoder", "base"}:
            scores[key] = max(float(scores[key]), float(value))
    base["metadata"] = metadata
    base["scores"] = scores
    base["score"] = max(float(left.get("score", 0.0)), float(right.get("score", 0.0)), float(scores.get("cross_encoder", 0.0)))
    return base


def normalize_path(path: dict, *, route: str, rank: int) -> dict:
    copied = dict(path)
    copied["metadata"] = dict(path.get("metadata", {}))
    metadata = copied["metadata"]
    if route == "bottom_eu":
        previous_source = str(metadata.get("candidate_source", "bottom_up"))
        metadata["candidate_source"] = f"bottom_up:{previous_source}"
        metadata["route_source"] = merge_route_source(str(metadata.get("route_source", "")), "bottom_up")
        metadata["bottom_up_rank"] = str(rank)
    elif route == "bottom_hyp":
        metadata["route_source"] = merge_route_source(str(metadata.get("route_source", "")), "bottom_up+hyp_bottom")
        metadata["bottom_hyp_rank"] = str(rank)
        if not str(metadata.get("hyp_bottom_seed_rank", "")):
            metadata["hyp_bottom_seed_rank"] = str(rank)
    else:
        metadata[f"{route}_route_rank"] = str(rank)
    metadata["retriever"] = metadata.get("retriever", route)
    return copied


def merge_route_source(existing: str, addition: str) -> str:
    values = []
    for part in [*existing.split("+"), *addition.split("+")]:
        part = part.strip()
        if part and part not in values:
            values.append(part)
    return "+".join(values)


def output_item(qa: dict, paths: list[dict], args, method: str) -> dict:
    return {
        "question_id": qa["question_id"],
        "question": qa["question"],
        "answer": qa.get("answer"),
        "category": qa.get("category"),
        "gold_evidence": qa.get("evidence", []),
        "paths": paths,
        "metadata": {
            "method": method,
            "bottom_eu": str(resolve_path(args.bottom_eu)),
            "bottom_hyp": str(resolve_path(args.bottom_hyp)),
            "eu_topdown": str(resolve_path(args.eu_topdown)),
            "hyp_topdown": str(resolve_path(args.hyp_topdown)),
            "candidate_topn": args.candidate_topn,
        },
    }


def ce_score(path: dict) -> float:
    scores = path.get("scores", {})
    try:
        return float(scores.get("cross_encoder", path.get("score", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def evidence_node_id(path: dict) -> str:
    metadata_id = str(path.get("metadata", {}).get("evidence_node_id", ""))
    if metadata_id:
        return metadata_id
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def load_item_map(path: str) -> dict[str, dict]:
    return {str(item["question_id"]): item for item in read_json(resolve_path(path))}


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
