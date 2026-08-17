from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evaluate_item, normalize_evidence_id, summarize
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, Node, NodeType
from hytopomem.retrieval.bm25_retriever import BM25Retriever


PACK_TYPES = ("episode", "entity_state", "bridge_entity_episode")


@dataclass(frozen=True)
class PackScore:
    pack: Node
    score: float
    rank: int
    pack_type: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_multiview_v3_5_packs.json")
    parser.add_argument(
        "--base-candidates",
        default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_notemporal_top150_D_true_bu_euhyp_td_euhyp_paths.json",
    )
    parser.add_argument("--questions", default=None)
    parser.add_argument("--candidate-topn", type=int, default=150)
    parser.add_argument("--final-topk", type=int, default=5)
    parser.add_argument("--pack-topk", type=int, default=20)
    parser.add_argument("--retrieval-multiplier", type=int, default=10)
    parser.add_argument("--lambdas", default="0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--context-weights", default="0.0,0.05,0.1")
    parser.add_argument("--redundancy-weights", default="0.0,0.05,0.1")
    parser.add_argument("--output-json", default="outputs/eval/graph_v3_5_pack_aware_selection_v1.json")
    parser.add_argument("--output-md", default="outputs/eval/graph_v3_5_pack_aware_selection_v1.md")
    args = parser.parse_args()

    helpers = load_topdown_helpers()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    question_path = resolve_path(args.questions or config["data"]["processed_path"])
    questions = helpers.flatten_questions(read_json(question_path), 0)
    base_items = {str(item["question_id"]): item for item in read_json(resolve_path(args.base_candidates))}

    fact_texts = {node.node_id: node.text for node in graph.iter_nodes(NodeType.FACT)}
    pack_nodes = list(graph.iter_nodes(NodeType.EVIDENCE_PACK))
    pack_retriever = BM25Retriever([retrieval_view(pack, graph, fact_texts) for pack in pack_nodes])
    fact_to_pack_edges = build_fact_to_pack_edges(graph)
    fact_event = fact_to_event_ids(graph)

    baseline_items = []
    per_question_context = {}
    for index, qa in enumerate(questions, start=1):
        qid = qa["question_id"]
        base_item = base_items.get(qid)
        if not base_item:
            continue
        base_paths = base_item.get("paths", [])[: args.candidate_topn]
        baseline_items.append(make_output_item(base_item, base_paths[: args.final_topk], "base_ce_top5"))
        base_scores = normalize_path_scores(base_paths)
        pack_scores = retrieve_pack_scores(
            pack_retriever,
            qid,
            qa["question"],
            base_scores,
            fact_to_pack_edges,
            args.pack_topk,
            args.retrieval_multiplier,
        )
        per_question_context[qid] = {
            "base_item": base_item,
            "base_paths": base_paths,
            "base_scores": base_scores,
            "pack_scores": pack_scores,
        }
        if index % 500 == 0 or index == len(questions):
            print(f"prepared pack-aware selection context {index}/{len(questions)}", flush=True)

    grid_results = []
    selected_outputs_by_name = {}
    for lambda_value in parse_floats(args.lambdas):
        for context_weight in parse_floats(args.context_weights):
            for redundancy_weight in parse_floats(args.redundancy_weights):
                name = f"pack_select_l{lambda_value:g}_c{context_weight:g}_r{redundancy_weight:g}"
                outputs = []
                for qid, ctx in per_question_context.items():
                    selected = greedy_select(
                        graph,
                        ctx["base_paths"],
                        ctx["base_scores"],
                        ctx["pack_scores"],
                        fact_to_pack_edges,
                        fact_event,
                        topk=args.final_topk,
                        lambda_value=lambda_value,
                        context_weight=context_weight,
                        redundancy_weight=redundancy_weight,
                    )
                    outputs.append(make_output_item(ctx["base_item"], selected, name))
                eval_payload = evaluate_outputs(graph, outputs, args.final_topk)
                grid_results.append(
                    {
                        "name": name,
                        "lambda": lambda_value,
                        "context_weight": context_weight,
                        "redundancy_weight": redundancy_weight,
                        **eval_payload["summary"],
                    }
                )
                selected_outputs_by_name[name] = outputs
                print(
                    f"{name}: hit={eval_payload['summary']['hit']:.4f} "
                    f"recall={eval_payload['summary']['recall']:.4f} "
                    f"full={eval_payload['summary']['full_cover']:.4f}",
                    flush=True,
                )

    baseline_eval = evaluate_outputs(graph, baseline_items, args.final_topk)
    grid_results.sort(key=lambda row: (row["hit"], row["recall"], row["full_cover"]), reverse=True)
    best_name = grid_results[0]["name"] if grid_results else ""
    best_outputs = selected_outputs_by_name.get(best_name, [])
    best_eval = evaluate_outputs(graph, best_outputs, args.final_topk) if best_outputs else {}
    payload = {
        "graph": str(resolve_path(args.graph)),
        "base_candidates": str(resolve_path(args.base_candidates)),
        "candidate_topn": args.candidate_topn,
        "final_topk": args.final_topk,
        "pack_topk": args.pack_topk,
        "baseline_ce_top5": baseline_eval,
        "best_name": best_name,
        "best_eval": best_eval,
        "grid_results": grid_results,
    }
    out_json = resolve_path(args.output_json)
    write_json(payload, out_json)
    out_md = resolve_path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"baseline": baseline_eval.get("summary"), "best": best_eval.get("summary"), "best_name": best_name}, indent=2, ensure_ascii=False))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


def greedy_select(
    graph: MemoryGraph,
    paths: list[dict],
    base_scores: dict[str, float],
    pack_scores: dict[str, PackScore],
    fact_to_pack_edges: dict[str, list[tuple[str, float]]],
    fact_event: dict[str, str],
    *,
    topk: int,
    lambda_value: float,
    context_weight: float,
    redundancy_weight: float,
) -> list[dict]:
    candidates = [path for path in paths if evidence_node_id(path)]
    selected = []
    selected_pack_ids: set[str] = set()
    selected_event_ids: set[str] = set()
    remaining = list(candidates)
    while remaining and len(selected) < topk:
        best_idx = 0
        best_score = float("-inf")
        best_metadata = {}
        for idx, path in enumerate(remaining):
            fact_id = evidence_node_id(path)
            base = base_scores.get(fact_id, 0.0)
            pack_rows = fact_to_pack_edges.get(fact_id, [])
            scored_pack_rows = [
                (pack_id, membership, pack_scores[pack_id])
                for pack_id, membership in pack_rows
                if pack_id in pack_scores
            ]
            best_context = max((score.score * membership for _pack_id, membership, score in scored_pack_rows), default=0.0)
            coverage_delta = max(
                (
                    score.score * membership
                    for pack_id, membership, score in scored_pack_rows
                    if pack_id not in selected_pack_ids
                ),
                default=0.0,
            )
            event_id = fact_event.get(fact_id, "")
            redundancy = 1.0 if event_id and event_id in selected_event_ids else 0.0
            total = base + context_weight * best_context + lambda_value * coverage_delta - redundancy_weight * redundancy
            if total > best_score:
                best_idx = idx
                best_score = total
                best_metadata = {
                    "pack_aware_score": f"{total:.6f}",
                    "pack_context_score": f"{best_context:.6f}",
                    "pack_coverage_delta": f"{coverage_delta:.6f}",
                    "pack_redundancy_penalty": f"{redundancy:.6f}",
                }
        chosen = dict(remaining.pop(best_idx))
        metadata = dict(chosen.get("metadata", {}))
        metadata.update(best_metadata)
        metadata["retriever"] = "pack_aware_selection_v1"
        chosen["metadata"] = metadata
        chosen["score"] = float(best_metadata.get("pack_aware_score", chosen.get("score", 0.0)))
        scores = dict(chosen.get("scores", {}))
        scores["pack_aware_selection"] = chosen["score"]
        chosen["scores"] = scores
        selected.append(chosen)
        fact_id = evidence_node_id(chosen)
        for pack_id, _membership in fact_to_pack_edges.get(fact_id, []):
            if pack_id in pack_scores:
                selected_pack_ids.add(pack_id)
        event_id = fact_event.get(fact_id, "")
        if event_id:
            selected_event_ids.add(event_id)
    return selected


def retrieve_pack_scores(
    pack_retriever: BM25Retriever,
    question_id: str,
    question: str,
    base_scores: dict[str, float],
    fact_to_pack_edges: dict[str, list[tuple[str, float]]],
    pack_topk: int,
    retrieval_multiplier: int,
) -> dict[str, PackScore]:
    fact_to_base = base_scores
    pack_to_base: dict[str, float] = {}
    for fact_id, rows in fact_to_pack_edges.items():
        fact_score = fact_to_base.get(fact_id, 0.0)
        if fact_score <= 0.0:
            continue
        for pack_id, membership in rows:
            pack_to_base[pack_id] = max(pack_to_base.get(pack_id, 0.0), fact_score * membership)
    raw_hits = [
        (pack, score)
        for pack, score in pack_retriever.search(question, top_k=max(pack_topk * retrieval_multiplier, pack_topk))
        if conversation_id(pack.node_id) == conversation_id(question_id)
    ]
    if not raw_hits:
        return {}
    max_bm25 = max(score for _pack, score in raw_hits) or 1.0
    scored = []
    for pack, bm25 in raw_hits:
        coherence = float(pack.metadata.get("coherence", 0.0))
        pack_base = pack_to_base.get(pack.node_id, 0.0)
        size_penalty = math.log1p(max(int(pack.metadata.get("num_facts", 1)), 1)) / math.log1p(24)
        score = 0.45 * (bm25 / max_bm25) + 0.35 * pack_base + 0.15 * coherence - 0.05 * size_penalty
        if pack.metadata.get("pack_type") == "bridge_entity_episode":
            score += 0.03
        scored.append(PackScore(pack=pack, score=score, rank=0, pack_type=str(pack.metadata.get("pack_type", ""))))
    scored.sort(key=lambda item: (-item.score, item.pack.node_id))
    return {item.pack.node_id: item.__class__(item.pack, item.score, rank, item.pack_type) for rank, item in enumerate(scored[:pack_topk], start=1)}


def retrieval_view(pack: Node, graph: MemoryGraph, fact_texts: dict[str, str]) -> Node:
    fact_ids = [str(fact_id) for fact_id in pack.metadata.get("fact_ids", [])]
    fact_snippets = " ".join(fact_texts.get(fact_id, "") for fact_id in fact_ids[:8])
    metadata_terms = []
    for key in ["entity", "aspect_keywords", "episode_keywords", "keywords", "entities"]:
        value = pack.metadata.get(key)
        if isinstance(value, list):
            metadata_terms.extend(str(item) for item in value)
        elif value:
            metadata_terms.append(str(value))
    return pack.model_copy(update={"text": " ".join([pack.text, " ".join(metadata_terms), fact_snippets]).strip()})


def build_fact_to_pack_edges(graph: MemoryGraph) -> dict[str, list[tuple[str, float]]]:
    mapping: dict[str, list[tuple[str, float]]] = {}
    for edge in graph.edges:
        if edge.metadata.get("hierarchy_v3_5_pack") != "pack_member":
            continue
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None or src.type != NodeType.FACT or dst.type != NodeType.EVIDENCE_PACK:
            continue
        weight = float(edge.metadata.get("membership_weight", edge.confidence))
        mapping.setdefault(edge.src, []).append((edge.dst, weight))
    return mapping


def fact_to_event_ids(graph: MemoryGraph) -> dict[str, str]:
    mapping = {}
    for edge in graph.edges:
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None:
            continue
        if src.type == NodeType.FACT and dst.type == NodeType.EVENT and edge.metadata.get("hierarchy_v3_3") == "fact_event":
            mapping[edge.src] = edge.dst
    return mapping


def normalize_path_scores(paths: list[dict]) -> dict[str, float]:
    rows = []
    for rank, path in enumerate(paths, start=1):
        fact_id = evidence_node_id(path)
        if not fact_id:
            continue
        score = path_score(path)
        rows.append((fact_id, score if score else 1.0 / rank))
    if not rows:
        return {}
    values = [score for _fact_id, score in rows]
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return {fact_id: 1.0 for fact_id, _score in rows}
    return {fact_id: (score - lo) / (hi - lo) for fact_id, score in rows}


def evaluate_outputs(graph: MemoryGraph, outputs: list[dict], k: int) -> dict:
    per_question = [evaluate_item(graph, item, k).__dict__ for item in outputs]
    summary = summarize([evaluate_item(graph, item, k) for item in outputs])
    return {"summary": summary, "per_question": per_question}


def make_output_item(base_item: dict, paths: list[dict], method: str) -> dict:
    item = dict(base_item)
    item["paths"] = paths
    metadata = dict(item.get("metadata", {}))
    metadata["method"] = method
    metadata["final_topk"] = len(paths)
    item["metadata"] = metadata
    return item


def evidence_node_id(path: dict) -> str:
    metadata_id = str(path.get("metadata", {}).get("evidence_node_id", ""))
    if metadata_id:
        return metadata_id
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def path_score(path: dict) -> float:
    scores = path.get("scores", {})
    try:
        return float(scores.get("cross_encoder", path.get("score", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def conversation_id(node_id: str) -> str:
    return node_id.split(":", 1)[0]


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def render_markdown(payload: dict) -> str:
    baseline = payload["baseline_ce_top5"]["summary"]
    best = payload["best_eval"]["summary"] if payload.get("best_eval") else {}
    lines = [
        "# Graph V3.5 Pack-Aware Selection V1",
        "",
        "Pack is used as a structure variable over base facts, not as a parallel candidate source.",
        "",
        f"- Graph: `{payload['graph']}`",
        f"- Base candidates: `{payload['base_candidates']}`",
        f"- Candidate topN: `{payload['candidate_topn']}`",
        f"- Final topK: `{payload['final_topk']}`",
        f"- Pack topK: `{payload['pack_topk']}`",
        "",
        "## Baseline vs Best",
        "",
        "| Method | Hit | Recall | FullCover |",
        "| --- | ---: | ---: | ---: |",
        f"| base_ce_top5 | {baseline['hit']:.4f} | {baseline['recall']:.4f} | {baseline['full_cover']:.4f} |",
    ]
    if best:
        lines.append(
            f"| {payload['best_name']} | {best['hit']:.4f} | {best['recall']:.4f} | {best['full_cover']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Grid",
            "",
            "| Variant | lambda | context | redundancy | Hit | Recall | FullCover |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["grid_results"][:20]:
        lines.append(
            f"| {row['name']} | {row['lambda']:.3g} | {row['context_weight']:.3g} | "
            f"{row['redundancy_weight']:.3g} | {row['hit']:.4f} | {row['recall']:.4f} | {row['full_cover']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Candidate set is unchanged: only base fact candidates are eligible.",
            "- Pack score provides context and coverage bonus for facts that already belong to retrieved evidence packs.",
            "- This is a heuristic diagnostic before adding pack-aware features into LightGBM.",
        ]
    )
    return "\n".join(lines) + "\n"


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
