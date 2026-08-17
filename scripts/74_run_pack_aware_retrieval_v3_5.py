from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, Node, NodeType
from hytopomem.retrieval.bm25_retriever import BM25Retriever


PACK_TYPES = ("episode", "entity_state", "bridge_entity_episode")


@dataclass(frozen=True)
class PackHit:
    pack: Node
    score: float
    bm25_score: float
    bm25_norm: float
    max_fact_score: float
    coherence: float
    avg_membership_weight: float
    size_penalty: float


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
    parser.add_argument("--pack-topk", type=int, default=20)
    parser.add_argument("--retrieval-multiplier", type=int, default=10)
    parser.add_argument("--reps-per-pack", type=int, default=2)
    parser.add_argument("--output-json", default="outputs/eval/graph_v3_5_pack_aware_retrieval_v1.json")
    parser.add_argument("--output-md", default="outputs/eval/graph_v3_5_pack_aware_retrieval_v1.md")
    args = parser.parse_args()

    helpers = load_topdown_helpers()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    questions = helpers.flatten_questions(read_json(resolve_path(args.questions or config["data"]["processed_path"])), 0)
    base_items = {str(item["question_id"]): item for item in read_json(resolve_path(args.base_candidates))}
    fact_texts = {node.node_id: node.text for node in graph.iter_nodes(NodeType.FACT)}
    fact_to_events = fact_to_event_ids(graph)
    membership_weights = pack_fact_membership_weights(graph)

    packs_by_type = {
        pack_type: [pack for pack in graph.iter_nodes(NodeType.EVIDENCE_PACK) if pack.metadata.get("pack_type") == pack_type]
        for pack_type in PACK_TYPES
    }
    retrievers = {
        pack_type: BM25Retriever([retrieval_view(pack, graph, fact_texts) for pack in packs])
        for pack_type, packs in packs_by_type.items()
    }

    pool_results: dict[str, dict[str, list[str]]] = {}
    for name in [
        f"base_fact_top{args.candidate_topn}",
        f"pack_rep_episode_top{args.candidate_topn}",
        f"pack_rep_entity_state_top{args.candidate_topn}",
        f"pack_rep_bridge_top{args.candidate_topn}",
        f"pack_rep_all_top{args.candidate_topn}",
        f"base_plus_episode_matched_top{args.candidate_topn}",
        f"base_plus_entity_state_matched_top{args.candidate_topn}",
        f"base_plus_bridge_matched_top{args.candidate_topn}",
        f"base_plus_all_pack_matched_top{args.candidate_topn}",
    ]:
        pool_results[name] = {}
    pack_cover_rows: dict[str, list[dict]] = {pack_type: [] for pack_type in (*PACK_TYPES, "all")}
    per_question = []

    for index, qa in enumerate(questions, start=1):
        qid = qa["question_id"]
        gold = gold_set(qa)
        base_paths = base_items.get(qid, {}).get("paths", [])[: args.candidate_topn]
        base_ranked = ranked_fact_ids_from_paths(base_paths)
        base_scores = fact_score_map(base_paths)
        base_ids = [fact_id for fact_id, _score in base_ranked]
        by_type_hits: dict[str, list[PackHit]] = {}
        by_type_reps: dict[str, list[tuple[str, float]]] = {}
        for pack_type in PACK_TYPES:
            pack_hits = retrieve_scored_packs(
                retrievers[pack_type],
                qid,
                qa["question"],
                base_scores,
                membership_weights,
                args.pack_topk,
                args.retrieval_multiplier,
            )
            by_type_hits[pack_type] = pack_hits
            by_type_reps[pack_type] = representative_fact_ids(
                graph,
                pack_hits,
                base_scores,
                membership_weights,
                fact_to_events,
                args.reps_per_pack,
            )
            pack_cover_rows[pack_type].append(pack_coverage_for_question(pack_hits, gold, k_values=(1, 5, 10, args.pack_topk)))

        all_hits = merge_pack_hits([hit for hits in by_type_hits.values() for hit in hits], args.pack_topk)
        all_reps = representative_fact_ids(
            graph,
            all_hits,
            base_scores,
            membership_weights,
            fact_to_events,
            args.reps_per_pack,
        )
        pack_cover_rows["all"].append(pack_coverage_for_question(all_hits, gold, k_values=(1, 5, 10, args.pack_topk)))

        pool_results[f"base_fact_top{args.candidate_topn}"][qid] = base_ids
        pool_results[f"pack_rep_episode_top{args.candidate_topn}"][qid] = ids_from_scored(by_type_reps["episode"], args.candidate_topn)
        pool_results[f"pack_rep_entity_state_top{args.candidate_topn}"][qid] = ids_from_scored(by_type_reps["entity_state"], args.candidate_topn)
        pool_results[f"pack_rep_bridge_top{args.candidate_topn}"][qid] = ids_from_scored(by_type_reps["bridge_entity_episode"], args.candidate_topn)
        pool_results[f"pack_rep_all_top{args.candidate_topn}"][qid] = ids_from_scored(all_reps, args.candidate_topn)
        pool_results[f"base_plus_episode_matched_top{args.candidate_topn}"][qid] = ids_from_scored(
            merge_scored(base_ranked, by_type_reps["episode"], args.candidate_topn), args.candidate_topn
        )
        pool_results[f"base_plus_entity_state_matched_top{args.candidate_topn}"][qid] = ids_from_scored(
            merge_scored(base_ranked, by_type_reps["entity_state"], args.candidate_topn), args.candidate_topn
        )
        pool_results[f"base_plus_bridge_matched_top{args.candidate_topn}"][qid] = ids_from_scored(
            merge_scored(base_ranked, by_type_reps["bridge_entity_episode"], args.candidate_topn), args.candidate_topn
        )
        pool_results[f"base_plus_all_pack_matched_top{args.candidate_topn}"][qid] = ids_from_scored(
            merge_scored(base_ranked, all_reps, args.candidate_topn), args.candidate_topn
        )
        per_question.append(
            {
                "question_id": qid,
                "category": qa.get("category"),
                "gold_count": len(gold),
                "base": pool_eval(graph, base_ids, gold),
                "episode": pool_eval(graph, pool_results[f"pack_rep_episode_top{args.candidate_topn}"][qid], gold),
                "entity_state": pool_eval(graph, pool_results[f"pack_rep_entity_state_top{args.candidate_topn}"][qid], gold),
                "bridge": pool_eval(graph, pool_results[f"pack_rep_bridge_top{args.candidate_topn}"][qid], gold),
                "all_pack": pool_eval(graph, pool_results[f"pack_rep_all_top{args.candidate_topn}"][qid], gold),
            }
        )
        if index % 500 == 0 or index == len(questions):
            print(f"pack-aware retrieval v1 {index}/{len(questions)}", flush=True)

    summary = {name: summarize_pool(graph, questions, pools) for name, pools in pool_results.items()}
    pack_summary = {name: summarize_pack_coverage(rows) for name, rows in pack_cover_rows.items()}
    payload = {
        "graph": str(resolve_path(args.graph)),
        "base_candidates": str(resolve_path(args.base_candidates)),
        "candidate_topn": args.candidate_topn,
        "pack_topk": args.pack_topk,
        "reps_per_pack": args.reps_per_pack,
        "summary": summary,
        "pack_summary": pack_summary,
        "per_question": per_question,
    }
    out_json = resolve_path(args.output_json)
    write_json(payload, out_json)
    out_md = resolve_path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"summary": summary, "pack_summary": pack_summary}, indent=2, ensure_ascii=False))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


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
    text = " ".join([pack.text, " ".join(metadata_terms), fact_snippets]).strip()
    return pack.model_copy(update={"text": text})


def retrieve_scored_packs(
    retriever: BM25Retriever,
    question_id: str,
    question: str,
    base_scores: dict[str, float],
    membership_weights: dict[tuple[str, str], float],
    pack_topk: int,
    retrieval_multiplier: int,
) -> list[PackHit]:
    raw_hits = [
        (pack, score)
        for pack, score in retriever.search(question, top_k=max(pack_topk * retrieval_multiplier, pack_topk))
        if conversation_id(pack.node_id) == conversation_id(question_id)
    ]
    if not raw_hits:
        return []
    max_bm25 = max(score for _pack, score in raw_hits) or 1.0
    scored = []
    for pack, bm25_score in raw_hits:
        fact_ids = [str(fact_id) for fact_id in pack.metadata.get("fact_ids", [])]
        max_fact_score = max((base_scores.get(fact_id, 0.0) for fact_id in fact_ids), default=0.0)
        coherence = float(pack.metadata.get("coherence", 0.0))
        avg_weight = average([membership_weights.get((fact_id, pack.node_id), 0.5) for fact_id in fact_ids])
        size_penalty = math.log1p(max(len(fact_ids), 1)) / math.log1p(24)
        bm25_norm = bm25_score / max_bm25
        score = 0.45 * bm25_norm + 0.30 * max_fact_score + 0.15 * coherence + 0.15 * avg_weight - 0.05 * size_penalty
        if pack.metadata.get("pack_type") == "bridge_entity_episode":
            score += 0.03
        scored.append(
            PackHit(
                pack=pack,
                score=score,
                bm25_score=bm25_score,
                bm25_norm=bm25_norm,
                max_fact_score=max_fact_score,
                coherence=coherence,
                avg_membership_weight=avg_weight,
                size_penalty=size_penalty,
            )
        )
    return sorted(scored, key=lambda item: (-item.score, item.pack.node_id))[:pack_topk]


def representative_fact_ids(
    graph: MemoryGraph,
    pack_hits: list[PackHit],
    base_scores: dict[str, float],
    membership_weights: dict[tuple[str, str], float],
    fact_to_events: dict[str, str],
    reps_per_pack: int,
) -> list[tuple[str, float]]:
    rows = []
    for pack_rank, hit in enumerate(pack_hits, start=1):
        used_events = set()
        fact_rows = []
        for offset, fact_id in enumerate(str(item) for item in hit.pack.metadata.get("fact_ids", [])):
            if graph.nodes.get(fact_id) is None:
                continue
            membership = membership_weights.get((fact_id, hit.pack.node_id), 0.5)
            base = base_scores.get(fact_id, 0.0)
            score = 0.55 * hit.score + 0.25 * base + 0.18 * membership + 0.02 / (offset + 1)
            fact_rows.append((fact_id, score, fact_to_events.get(fact_id, "")))
        fact_rows.sort(key=lambda item: (-item[1], item[0]))
        selected = 0
        delayed = []
        for fact_id, score, event_id in fact_rows:
            if event_id and event_id in used_events:
                delayed.append((fact_id, score, event_id))
                continue
            rows.append((fact_id, score + 0.001 / pack_rank))
            if event_id:
                used_events.add(event_id)
            selected += 1
            if selected >= reps_per_pack:
                break
        for fact_id, score, _event_id in delayed:
            if selected >= reps_per_pack:
                break
            rows.append((fact_id, score + 0.001 / pack_rank))
            selected += 1
    return dedupe_scored(sorted(rows, key=lambda item: (-item[1], item[0])))


def merge_pack_hits(hits: list[PackHit], topk: int) -> list[PackHit]:
    best: dict[str, PackHit] = {}
    for hit in hits:
        previous = best.get(hit.pack.node_id)
        if previous is None or hit.score > previous.score:
            best[hit.pack.node_id] = hit
    return sorted(best.values(), key=lambda item: (-item.score, item.pack.node_id))[:topk]


def ranked_fact_ids_from_paths(paths: list[dict]) -> list[tuple[str, float]]:
    rows = []
    for rank, path in enumerate(paths, start=1):
        fact_id = evidence_node_id(path)
        if not fact_id:
            continue
        score = path_score(path)
        rows.append((fact_id, score if score else 1.0 / rank))
    return dedupe_scored(rows)


def fact_score_map(paths: list[dict]) -> dict[str, float]:
    rows = ranked_fact_ids_from_paths(paths)
    if not rows:
        return {}
    max_score = max(abs(score) for _fact_id, score in rows) or 1.0
    return {fact_id: score / max_score for fact_id, score in rows}


def merge_scored(left: list[tuple[str, float]], right: list[tuple[str, float]], topn: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for fact_id, score in [*left, *right]:
        scores[fact_id] = max(scores.get(fact_id, float("-inf")), score)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:topn]


def ids_from_scored(rows: list[tuple[str, float]], topn: int) -> list[str]:
    return [fact_id for fact_id, _score in rows[:topn]]


def pack_coverage_for_question(pack_hits: list[PackHit], gold: set[str], k_values: tuple[int, ...]) -> dict:
    output = {}
    for k in k_values:
        selected = pack_hits[:k]
        predicted = set()
        best_purity = 0.0
        best_cover = 0.0
        for hit in selected:
            pack_gold = raw_gold_ids_for_pack(hit.pack)
            matched = pack_gold & gold
            predicted.update(pack_gold)
            best_cover = max(best_cover, len(matched) / len(gold) if gold else 0.0)
            best_purity = max(best_purity, len(matched) / len(pack_gold) if pack_gold else 0.0)
        matched_all = predicted & gold
        recall = len(matched_all) / len(gold) if gold else 0.0
        output[f"pack_hit@{k}"] = bool(matched_all)
        output[f"pack_recall@{k}"] = recall
        output[f"pack_full@{k}"] = bool(gold) and gold.issubset(predicted)
        output[f"best_pack_cover@{k}"] = best_cover
        output[f"best_pack_purity@{k}"] = best_purity
    return output


def summarize_pack_coverage(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = sorted(rows[0])
    return {key: average(float(row[key]) for row in rows) for key in keys}


def summarize_pool(graph: MemoryGraph, questions: list[dict], pools: dict[str, list[str]]) -> dict:
    rows = [pool_eval(graph, pools.get(item["question_id"], []), gold_set(item)) for item in questions]
    n = len(rows)
    return {
        "num_questions": n,
        "hit": sum(float(row["hit"]) for row in rows) / n if n else 0.0,
        "recall": sum(float(row["recall"]) for row in rows) / n if n else 0.0,
        "full_cover": sum(float(row["full_cover"]) for row in rows) / n if n else 0.0,
        "avg_candidates": sum(float(row["num_candidates"]) for row in rows) / n if n else 0.0,
        "gold_density": sum(float(row["gold_density"]) for row in rows) / n if n else 0.0,
    }


def pool_eval(graph: MemoryGraph, fact_ids: list[str], gold: set[str]) -> dict:
    predicted = set()
    positive = 0
    for fact_id in fact_ids:
        fact_gold = raw_gold_ids_for_fact(graph, fact_id)
        predicted.update(fact_gold)
        if fact_gold & gold:
            positive += 1
    matched = predicted & gold
    return {
        "hit": bool(matched),
        "recall": len(matched) / len(gold) if gold else 0.0,
        "full_cover": bool(gold) and gold.issubset(predicted),
        "num_candidates": len(fact_ids),
        "gold_density": positive / len(fact_ids) if fact_ids else 0.0,
    }


def pack_fact_membership_weights(graph: MemoryGraph) -> dict[tuple[str, str], float]:
    weights = {}
    for edge in graph.edges:
        if edge.metadata.get("hierarchy_v3_5_pack") != "pack_member":
            continue
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None or src.type != NodeType.FACT or dst.type != NodeType.EVIDENCE_PACK:
            continue
        weights[(edge.src, edge.dst)] = float(edge.metadata.get("membership_weight", edge.confidence))
    return weights


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


def raw_gold_ids_for_pack(pack: Node) -> set[str]:
    return {normalize_raw_id(raw_id) for raw_id in pack.metadata.get("raw_ids", [])}


def raw_gold_ids_for_fact(graph: MemoryGraph, fact_id: str) -> set[str]:
    node = graph.nodes.get(str(fact_id))
    if node is None:
        return set()
    values = list(node.metadata.get("support_raw_ids") or node.support_ids)
    return {normalize_raw_id(value) for value in values}


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("evidence", item.get("gold_evidence", []))}


def normalize_raw_id(value: object) -> str:
    text = str(value)
    if ":raw:" in text:
        text = text.rsplit(":raw:", 1)[-1]
    return normalize_evidence_id(text)


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


def dedupe_scored(rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
    output = {}
    for fact_id, score in rows:
        if fact_id not in output or score > output[fact_id]:
            output[fact_id] = score
    return sorted(output.items(), key=lambda item: (-item[1], item[0]))


def average(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def conversation_id(node_id: str) -> str:
    return node_id.split(":", 1)[0]


def render_markdown(payload: dict) -> str:
    lines = [
        "# Graph V3.5 Pack-Aware Retrieval V1",
        "",
        f"- Graph: `{payload['graph']}`",
        f"- Base candidates: `{payload['base_candidates']}`",
        f"- Candidate topN: `{payload['candidate_topn']}`",
        f"- Pack topK: `{payload['pack_topk']}`",
        f"- Representatives per pack: `{payload['reps_per_pack']}`",
        "",
        "## Fact Candidate Pools",
        "",
        "| Pool | Hit | Recall | FullCover | Avg Cand | Gold Density |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in payload["summary"].items():
        lines.append(
            f"| {name} | {item['hit']:.4f} | {item['recall']:.4f} | "
            f"{item['full_cover']:.4f} | {item['avg_candidates']:.2f} | {item['gold_density']:.5f} |"
        )
    lines.extend(["", "## Pack Coverage By Type", ""])
    for pack_type, summary in payload["pack_summary"].items():
        lines.extend([
            f"### {pack_type}",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ])
        for key, value in summary.items():
            lines.append(f"| {key} | {value:.4f} |")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- Pack score uses BM25 over enriched pack text, base candidate fact score, pack coherence, membership weight, and a size penalty.",
            "- BridgePack receives a small prior bonus because it is the intended semantic/entity cross-view intersection.",
            "- Each pack expands only representative facts with event diversification.",
            "- Matched-budget rows keep the final fact candidate budget fixed.",
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
