from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from common import read_json, resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, Node, NodeType
from hytopomem.retrieval.cross_encoder_reranker import CrossEncoderReranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument(
        "--base-paths",
        default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_notemporal_top150_D_true_bu_euhyp_td_euhyp_paths.json",
    )
    parser.add_argument("--base-topn", type=int, default=100)
    parser.add_argument("--completion-topn", type=int, default=20)
    parser.add_argument("--relation-types", default="change,plan_constraint,preference,state")
    parser.add_argument(
        "--ce-model",
        default="/home/sunyuwei/LightMem/LightMem/models/cross-encoder-ms-marco-MiniLM-L6-v2",
    )
    parser.add_argument("--ce-device", default="cpu")
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--skip-ce", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    base_items = read_json(resolve_path(args.base_paths))
    relation_types = {value.strip() for value in args.relation_types.split(",") if value.strip()}
    hyperedges = nary_hyperedges(graph, relation_types)
    index = CompletionIndex.from_hyperedges(hyperedges)

    output_items = []
    completion_paths_for_ce = []
    for item in base_items:
        base_paths = list(item.get("paths", []))[: args.base_topn]
        item_completion_paths = build_completion_paths(
            graph=graph,
            item=item,
            base_paths=base_paths,
            index=index,
            completion_topn=args.completion_topn,
        )
        completion_paths_for_ce.extend((item, path) for path in item_completion_paths)
        output_items.append((item, base_paths, item_completion_paths))

    if completion_paths_for_ce and not args.skip_ce:
        reranker = CrossEncoderReranker(args.ce_model, device=args.ce_device, batch_size=args.ce_batch_size)
        pairs = [
            (item.get("question", ""), graph.nodes[path["metadata"]["evidence_node_id"]].text)
            for item, path in completion_paths_for_ce
        ]
        scores = reranker.model.predict(pairs, batch_size=reranker.batch_size, show_progress_bar=True)
        for (_, path), score in zip(completion_paths_for_ce, scores):
            score = float(score)
            path["score"] = score
            path.setdefault("scores", {})["cross_encoder"] = score
            path.setdefault("scores", {})["base"] = float(path["metadata"].get("nary_seed_fact_score", 0.0))
    elif completion_paths_for_ce:
        for _, path in completion_paths_for_ce:
            seed_score = float(path["metadata"].get("nary_seed_fact_score", 0.0))
            path["score"] = seed_score
            path.setdefault("scores", {})["cross_encoder"] = seed_score
            path.setdefault("scores", {})["base"] = seed_score

    final_items = []
    stats = []
    for item, base_paths, completion_paths in output_items:
        paths = merge_paths(base_paths, completion_paths)
        annotate_pool_features(paths, index)
        paths.sort(key=path_score, reverse=True)
        copied = dict(item)
        copied["paths"] = paths
        metadata = dict(copied.get("metadata", {}))
        metadata.update(
            {
                "method": f"base{args.base_topn}_nary_completion{args.completion_topn}",
                "base_topn": args.base_topn,
                "completion_topn": args.completion_topn,
                "nary_graph": str(resolve_path(args.graph)),
                "nary_hyperedges": len(hyperedges),
                "relation_types": sorted(relation_types),
            }
        )
        copied["metadata"] = metadata
        final_items.append(copied)
        stats.append(
            {
                "base": len(base_paths),
                "completion": len(completion_paths),
                "paths": len(paths),
                "triggered_hyperedges": len({p["metadata"].get("nary_hyperedge_id") for p in completion_paths}),
            }
        )

    out = resolve_path(args.output)
    write_json(final_items, out)
    summary = {
        "output": str(out),
        "graph": str(resolve_path(args.graph)),
        "base_paths": str(resolve_path(args.base_paths)),
        "base_topn": args.base_topn,
        "completion_topn": args.completion_topn,
        "relation_types": sorted(relation_types),
        "nary_hyperedges": len(hyperedges),
        "questions": len(final_items),
        "avg_base": mean([row["base"] for row in stats]) if stats else 0.0,
        "avg_completion": mean([row["completion"] for row in stats]) if stats else 0.0,
        "avg_paths": mean([row["paths"] for row in stats]) if stats else 0.0,
        "avg_triggered_hyperedges": mean([row["triggered_hyperedges"] for row in stats]) if stats else 0.0,
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


class CompletionIndex:
    def __init__(self) -> None:
        self.fact_to_hyperedges: dict[str, list[Node]] = defaultdict(list)
        self.fact_role_by_hyperedge: dict[tuple[str, str], str] = {}
        self.roles_by_hyperedge: dict[str, set[str]] = {}
        self.fact_ids_by_hyperedge: dict[str, list[str]] = {}

    @classmethod
    def from_hyperedges(cls, hyperedges: list[Node]) -> "CompletionIndex":
        index = cls()
        for node in hyperedges:
            role_map = fact_roles(node)
            fact_ids = []
            seen = set()
            for fact_id, role in role_map.items():
                if fact_id not in seen:
                    seen.add(fact_id)
                    fact_ids.append(fact_id)
                index.fact_role_by_hyperedge[(node.node_id, fact_id)] = role
                index.fact_to_hyperedges[fact_id].append(node)
            index.roles_by_hyperedge[node.node_id] = set(role_map.values())
            index.fact_ids_by_hyperedge[node.node_id] = fact_ids
        return index


def build_completion_paths(
    *,
    graph: MemoryGraph,
    item: dict,
    base_paths: list[dict],
    index: CompletionIndex,
    completion_topn: int,
) -> list[dict]:
    if completion_topn <= 0:
        return []
    qid = str(item["question_id"])
    conv_id = conversation_id(qid)
    base_fact_ids = [evidence_node_id(path) for path in base_paths]
    seen_facts = {fact_id for fact_id in base_fact_ids if fact_id}
    outputs = []
    completion_seen = set()
    for seed_rank, seed_path in enumerate(base_paths, start=1):
        seed_fact = evidence_node_id(seed_path)
        if not seed_fact:
            continue
        for hyperedge in index.fact_to_hyperedges.get(seed_fact, []):
            if conversation_id(hyperedge.node_id) != conv_id:
                continue
            for role_fact_id in index.fact_ids_by_hyperedge.get(hyperedge.node_id, []):
                if role_fact_id in seen_facts or role_fact_id in completion_seen:
                    continue
                if graph.nodes.get(role_fact_id) is None:
                    continue
                completion_seen.add(role_fact_id)
                outputs.append(
                    completion_path(
                        qid=qid,
                        seed_path=seed_path,
                        seed_rank=seed_rank,
                        hyperedge=hyperedge,
                        role_fact_id=role_fact_id,
                        role=index.fact_role_by_hyperedge.get((hyperedge.node_id, role_fact_id), ""),
                        completion_rank=len(outputs) + 1,
                    )
                )
                if len(outputs) >= completion_topn:
                    return outputs
    return outputs


def completion_path(
    *,
    qid: str,
    seed_path: dict,
    seed_rank: int,
    hyperedge: Node,
    role_fact_id: str,
    role: str,
    completion_rank: int,
) -> dict:
    seed_score = path_score(seed_path)
    seed_meta = seed_path.get("metadata", {})
    relation_type = str(hyperedge.metadata.get("relation_type") or "")
    extractor_type = str(hyperedge.metadata.get("model") or hyperedge.metadata.get("extractor_type") or "")
    roles = hyperedge.metadata.get("roles") or {}
    role_payload = roles.get(role) if isinstance(roles, dict) else None
    role_confidence = float((role_payload or {}).get("confidence", hyperedge.confidence))
    metadata = {
        "retriever": "nary_role_completion_ce",
        "candidate_source": "nary_completion",
        "route_source": "nary_completion",
        "evidence_node_id": role_fact_id,
        "evidence_node_type": "FACT",
        "is_nary_completion": "true",
        "nary_hyperedge_id": hyperedge.node_id,
        "nary_hyperedge_type": relation_type,
        "nary_role": role,
        "nary_seed_fact_id": evidence_node_id(seed_path),
        "nary_seed_fact_rank": str(seed_rank),
        "nary_seed_fact_score": f"{seed_score:.6f}",
        "nary_seed_route_origin": str(seed_meta.get("route_source") or seed_meta.get("candidate_source") or ""),
        "nary_hyperedge_size": str(len(fact_ids_from_roles(hyperedge))),
        "nary_hyperedge_confidence": f"{float(hyperedge.confidence):.6f}",
        "nary_role_confidence": f"{role_confidence:.6f}",
        "nary_extractor_type": extractor_type,
        "nary_completion_rank": str(completion_rank),
    }
    return {
        "query_id": qid,
        "anchor_id": None,
        "node_ids": [evidence_node_id(seed_path), hyperedge.node_id, role_fact_id],
        "edge_ids": [],
        "score": seed_score,
        "scores": {"cross_encoder": seed_score, "base": seed_score, "nary_seed": seed_score},
        "metadata": metadata,
    }


def annotate_pool_features(paths: list[dict], index: CompletionIndex) -> None:
    fact_ids = {evidence_node_id(path) for path in paths}
    hyperedge_pool_facts: dict[str, set[str]] = defaultdict(set)
    hyperedge_pool_roles: dict[str, set[str]] = defaultdict(set)
    for hyperedge_id, hyperedge_fact_ids in index.fact_ids_by_hyperedge.items():
        overlap = fact_ids & set(hyperedge_fact_ids)
        if not overlap:
            continue
        hyperedge_pool_facts[hyperedge_id] = overlap
        for fact_id in overlap:
            role = index.fact_role_by_hyperedge.get((hyperedge_id, fact_id), "")
            if role:
                hyperedge_pool_roles[hyperedge_id].add(role)
    for path in paths:
        metadata = path.get("metadata", {})
        hyperedge_id = str(metadata.get("nary_hyperedge_id") or "")
        if not hyperedge_id:
            continue
        roles = hyperedge_pool_roles.get(hyperedge_id, set())
        required_covered = required_roles_covered(
            str(metadata.get("nary_hyperedge_type") or ""),
            roles,
        )
        metadata.update(
            {
                "nary_same_hyperedge_count_in_candidate_pool": str(len(hyperedge_pool_facts.get(hyperedge_id, set()))),
                "nary_role_coverage_potential": str(len(index.roles_by_hyperedge.get(hyperedge_id, set()))),
                "nary_pool_covered_roles_count": str(len(roles)),
                "nary_pool_required_roles_covered": str(required_covered),
                "nary_pool_has_preference_and_constraint": str(int("preference_value" in roles and "constraint" in roles)),
                "nary_pool_has_old_and_new_state": str(int("old_state" in roles and "new_state" in roles)),
                "nary_pool_has_reason": str(int("reason_or_trigger" in roles)),
                "nary_pool_has_time_scope": str(int("temporal_scope" in roles)),
            }
        )
        path["metadata"] = metadata


def required_roles_covered(relation_type: str, roles: set[str]) -> int:
    relation_type = relation_type.lower()
    if relation_type == "change":
        return int("old_state" in roles and "new_state" in roles)
    if relation_type == "preference":
        return int("preference_value" in roles or "polarity" in roles)
    if relation_type == "state":
        return int("state_value" in roles)
    if relation_type == "plan_constraint":
        return int("plan_goal" in roles and ("constraint" in roles or "temporal_scope" in roles))
    return 0


def nary_hyperedges(graph: MemoryGraph, relation_types: set[str]) -> list[Node]:
    output = []
    for node in graph.iter_nodes(NodeType.EVIDENCE_PACK):
        if node.metadata.get("hierarchy_v3_6") != "typed_nary_hyperedge":
            continue
        if str(node.metadata.get("relation_type", "")) in relation_types:
            output.append(node)
    return output


def fact_roles(node: Node) -> dict[str, str]:
    output = {}
    roles = node.metadata.get("roles") or {}
    if isinstance(roles, dict):
        for role, payload in roles.items():
            if not isinstance(payload, dict):
                continue
            for fact_id in payload.get("fact_ids", []) or []:
                fact_id = str(fact_id)
                if fact_id and fact_id not in output:
                    output[fact_id] = str(role)
    for fact_id in node.metadata.get("fact_ids", []) or node.support_ids:
        fact_id = str(fact_id)
        if fact_id and fact_id not in output:
            output[fact_id] = "support"
    return output


def fact_ids_from_roles(node: Node) -> list[str]:
    return list(fact_roles(node).keys())


def merge_paths(base_paths: list[dict], completion_paths: list[dict]) -> list[dict]:
    output = []
    seen = set()
    for path in list(base_paths) + list(completion_paths):
        fact_id = evidence_node_id(path)
        if not fact_id or fact_id in seen:
            continue
        seen.add(fact_id)
        output.append(path)
    return output


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def path_score(path: dict) -> float:
    scores = path.get("scores", {})
    try:
        return float(scores.get("cross_encoder", path.get("score", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def conversation_id(value: str) -> str:
    return str(value).split(":", 1)[0]


if __name__ == "__main__":
    main()
