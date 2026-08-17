from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from common import read_json, resolve_path, write_json
from hytopomem.reconstruction import AnswerPackCompiler, EvidenceUnitBuilder, HeuristicQueryRequirementCompiler


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HyPathMem-R R1/R2 packs from frozen candidate paths.")
    parser.add_argument(
        "--paths",
        default="outputs/eval/longmemeval_v3_10_expand120_24_feature_cardce_selector_top50summary_top50/card_quota_light_top50_paths.json",
    )
    parser.add_argument(
        "--graph",
        default="outputs/longmemeval_s/v3_10_fine/graph_sentence_semantic_episode_qwen_pathnodes_top50_v3_10.json",
    )
    parser.add_argument(
        "--node-cache",
        default="",
        help="Optional candidate-specific node cache produced by script 130; avoids loading the full graph.",
    )
    parser.add_argument("--output", default="outputs/reconstruction/hypathmem_r_v0_1/top50_structured_raw_packs.json")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=10_000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Allow replacing this new reconstruction artifact only.")
    args = parser.parse_args()

    paths_path = resolve_path(args.paths)
    graph_path = resolve_path(args.graph)
    node_cache_path = resolve_path(args.node_cache) if args.node_cache else None
    output_path = resolve_path(args.output)
    refuse_existing(output_path, args.force)

    items = list(read_json(paths_path))
    end = args.offset + args.limit if args.limit else None
    items = items[args.offset:end]
    if node_cache_path:
        print(f"loading read-only node cache {node_cache_path}", flush=True)
        node_cache = read_json(node_cache_path)
        nodes = node_cache.get("nodes") or {}
        graph_id = (node_cache.get("metadata") or {}).get("graph_id")
    else:
        print(f"loading read-only graph {graph_path}", flush=True)
        graph = read_json(graph_path)
        nodes = graph.get("nodes") or {}
        graph_id = graph.get("graph_id")

    query_compiler = HeuristicQueryRequirementCompiler()
    unit_builder = EvidenceUnitBuilder(nodes)
    pack_compiler = AnswerPackCompiler(token_budget=args.token_budget)
    rows = []
    for index, item in enumerate(items, start=1):
        contract = query_compiler.compile(
            str(item.get("question") or ""),
            question_date=str((item.get("metadata") or {}).get("question_date") or "") or None,
        )
        units = unit_builder.build(item.get("paths") or [], contract, k=args.k)
        pack = pack_compiler.compile(str(item.get("question") or ""), contract, units)
        rows.append(
            {
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "gold_answer": item.get("answer"),
                "is_abstention": item.get("is_abstention"),
                "pack": pack.model_dump(mode="json"),
                "answer_context": pack_compiler.render_json(pack),
            }
        )
        if index % 10 == 0 or index == len(items):
            print(f"built {index}/{len(items)} packs", flush=True)

    payload = {
        "metadata": {
            "version": "hypathmem_r_v0_1",
            "stage": "R1_structured_pack_plus_R2_raw_support_closure",
            "paths": str(paths_path),
            "paths_sha256": sha256(paths_path),
            "graph": str(graph_path),
            "graph_id": graph_id,
            "node_cache": str(node_cache_path) if node_cache_path else None,
            "node_cache_sha256": sha256(node_cache_path) if node_cache_path else None,
            "k": args.k,
            "token_budget": args.token_budget,
            "offset": args.offset,
            "limit": args.limit,
            "uses_benchmark_question_type": False,
            "mutates_source_graph": False,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    write_json(payload, output_path)
    print(f"wrote {output_path}")


def summarize(rows: list[dict]) -> dict:
    packs = [row["pack"] for row in rows]
    return {
        "num_questions": len(rows),
        "supported": sum(pack["answerability"] == "SUPPORTED" for pack in packs),
        "partially_supported": sum(pack["answerability"] == "PARTIALLY_SUPPORTED" for pack in packs),
        "unsupported": sum(pack["answerability"] == "UNSUPPORTED" for pack in packs),
        "avg_pack_tokens": sum(pack["token_cost"] for pack in packs) / len(packs) if packs else 0.0,
        "avg_selected_units": (
            sum(pack["diagnostics"]["selected_units"] for pack in packs) / len(packs) if packs else 0.0
        ),
        "raw_grounded_unit_ratio": ratio(
            sum(pack["diagnostics"]["raw_grounded_units"] for pack in packs),
            sum(pack["diagnostics"]["selected_units"] for pack in packs),
        ),
    }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


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
