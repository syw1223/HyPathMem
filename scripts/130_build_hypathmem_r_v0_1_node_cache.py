from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from common import read_json, resolve_path, write_json
from hytopomem.reconstruction.evidence_unit_builder import evidence_node_id
from hytopomem.reconstruction.support_closure import previous_turn_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a read-only HyPathMem-R candidate node cache.")
    parser.add_argument(
        "--paths",
        default="outputs/eval/longmemeval_v3_10_expand120_24_feature_cardce_selector_top50summary_top50/card_quota_light_top50_paths.json",
    )
    parser.add_argument(
        "--graph",
        default="outputs/longmemeval_s/v3_10_fine/graph_sentence_semantic_episode_qwen_pathnodes_top50_v3_10.json",
    )
    parser.add_argument("--output", default="outputs/reconstruction/hypathmem_r_v0_1/top50_node_cache.json")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="Allow replacing this cache artifact only.")
    args = parser.parse_args()

    paths_path = resolve_path(args.paths)
    graph_path = resolve_path(args.graph)
    output_path = resolve_path(args.output)
    refuse_existing(output_path, args.force)

    path_rows = list(read_json(paths_path))
    candidate_ids = {
        node_id
        for row in path_rows
        for path in (row.get("paths") or [])[: args.k]
        if (node_id := evidence_node_id(path))
    }
    print(f"loading read-only graph {graph_path}", flush=True)
    graph = read_json(graph_path)
    graph_nodes = graph.get("nodes") or {}

    selected_ids = set(candidate_ids)
    missing_candidates: list[str] = []
    direct_support_ids: set[str] = set()
    for node_id in sorted(candidate_ids):
        node = graph_nodes.get(node_id)
        if node is None:
            missing_candidates.append(node_id)
            continue
        metadata = node.get("metadata") or {}
        direct_support_ids.update(str(value) for value in metadata.get("support_raw_ids") or [])
        direct_support_ids.update(str(value) for value in node.get("support_ids") or [])
        if node.get("type") == "RAW":
            direct_support_ids.add(node_id)

    selected_ids.update(direct_support_ids)
    for raw_id in direct_support_ids:
        previous_id = previous_turn_id(raw_id)
        if previous_id:
            selected_ids.add(previous_id)

    nodes: dict[str, dict[str, Any]] = {
        node_id: graph_nodes[node_id] for node_id in sorted(selected_ids) if node_id in graph_nodes
    }
    payload = {
        "metadata": {
            "version": "hypathmem_r_v0_1",
            "stage": "candidate_node_cache_for_R1_R2",
            "paths": str(paths_path),
            "paths_sha256": sha256(paths_path),
            "graph": str(graph_path),
            "graph_sha256": sha256(graph_path),
            "graph_id": graph.get("graph_id"),
            "k": args.k,
            "candidate_node_count": len(candidate_ids),
            "direct_support_node_count": len(direct_support_ids),
            "cached_node_count": len(nodes),
            "missing_candidate_count": len(missing_candidates),
            "mutates_source_graph": False,
        },
        "missing_candidate_ids": missing_candidates,
        "nodes": nodes,
    }
    write_json(payload, output_path)
    print(f"cached {len(nodes)} nodes; missing candidates={len(missing_candidates)}", flush=True)
    print(f"wrote {output_path}", flush=True)


def refuse_existing(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {path}; pass --force explicitly")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
