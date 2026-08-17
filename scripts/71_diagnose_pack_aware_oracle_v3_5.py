from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, Node, NodeType
from hytopomem.retrieval.bm25_retriever import BM25Retriever


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
    parser.add_argument("--reps-per-pack", type=int, default=3)
    parser.add_argument("--output-json", default="outputs/eval/graph_v3_5_pack_aware_oracle.json")
    parser.add_argument("--output-md", default="outputs/eval/graph_v3_5_pack_aware_oracle.md")
    args = parser.parse_args()

    helpers = load_topdown_helpers()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    question_path = resolve_path(args.questions or config["data"]["processed_path"])
    questions = helpers.flatten_questions(read_json(question_path), 0)
    base_items = {str(item["question_id"]): item for item in read_json(resolve_path(args.base_candidates))}

    packs = list(graph.iter_nodes(NodeType.EVIDENCE_PACK))
    pack_retriever = BM25Retriever(packs)

    per_question = []
    pool_results = {
        f"base_fact_top{args.candidate_topn}": {},
        f"pack_rep_top{args.candidate_topn}": {},
        f"base_plus_pack_matched_top{args.candidate_topn}": {},
    }
    pack_cover_rows = []
    for index, qa in enumerate(questions, start=1):
        qid = qa["question_id"]
        gold = gold_set(qa)
        base_paths = base_items.get(qid, {}).get("paths", [])[: args.candidate_topn]
        base_ranked = ranked_fact_ids_from_paths(base_paths)
        base_scores = fact_score_map(base_paths)
        pack_hits = pack_retriever.search(qa["question"], top_k=max(args.pack_topk * 10, 200))
        pack_ranked = [
            (pack, score)
            for pack, score in pack_hits
            if conversation_id(pack.node_id) == conversation_id(qid)
        ][: args.pack_topk]
        pack_rep_ranked = representative_fact_ids(pack_ranked, base_scores, args.reps_per_pack)
        merged_ranked = merge_scored(base_ranked, pack_rep_ranked, args.candidate_topn)

        base_ids = [fact_id for fact_id, _score in base_ranked[: args.candidate_topn]]
        pack_rep_ids = [fact_id for fact_id, _score in pack_rep_ranked[: args.candidate_topn]]
        merged_ids = [fact_id for fact_id, _score in merged_ranked]

        pool_results[f"base_fact_top{args.candidate_topn}"][qid] = base_ids
        pool_results[f"pack_rep_top{args.candidate_topn}"][qid] = pack_rep_ids
        pool_results[f"base_plus_pack_matched_top{args.candidate_topn}"][qid] = merged_ids

        pack_row = pack_coverage_for_question(pack_ranked, gold, k_values=(1, 5, 10, args.pack_topk))
        pack_cover_rows.append(pack_row)
        per_question.append(
            {
                "question_id": qid,
                "category": qa.get("category"),
                "gold_count": len(gold),
                "base": pool_eval(graph, base_ids, gold),
                "pack_rep": pool_eval(graph, pack_rep_ids, gold),
                "base_plus_pack": pool_eval(graph, merged_ids, gold),
                "pack_coverage": pack_row,
            }
        )
        if index % 500 == 0 or index == len(questions):
            print(f"diagnosed pack oracle {index}/{len(questions)}", flush=True)

    summary = {
        name: summarize_pool(graph, questions, pools)
        for name, pools in pool_results.items()
    }
    pack_summary = summarize_pack_coverage(pack_cover_rows)
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


def representative_fact_ids(
    pack_ranked: list[tuple[Node, float]],
    base_scores: dict[str, float],
    reps_per_pack: int,
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    max_pack_score = max((score for _pack, score in pack_ranked), default=1.0) or 1.0
    for pack_rank, (pack, pack_score) in enumerate(pack_ranked, start=1):
        fact_ids = list(pack.metadata.get("fact_ids", []))
        fact_rows = []
        for offset, fact_id in enumerate(fact_ids):
            score = (pack_score / max_pack_score) + 0.25 * base_scores.get(str(fact_id), 0.0) + 0.01 / (offset + 1)
            fact_rows.append((str(fact_id), score))
        fact_rows.sort(key=lambda item: (-item[1], item[0]))
        for fact_id, score in fact_rows[:reps_per_pack]:
            rows.append((fact_id, score + 0.001 / pack_rank))
    return dedupe_scored(sorted(rows, key=lambda item: (-item[1], item[0])))


def merge_scored(left: list[tuple[str, float]], right: list[tuple[str, float]], topn: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for fact_id, score in [*left, *right]:
        scores[fact_id] = max(scores.get(fact_id, float("-inf")), score)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:topn]


def pack_coverage_for_question(pack_ranked: list[tuple[Node, float]], gold: set[str], k_values: tuple[int, ...]) -> dict:
    output = {}
    for k in k_values:
        selected = pack_ranked[:k]
        predicted = set()
        best_purity = 0.0
        best_cover = 0.0
        for pack, _score in selected:
            pack_gold = raw_gold_ids_for_pack(pack)
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
    summary = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        summary[key] = sum(values) / len(values)
    return summary


def summarize_pool(graph: MemoryGraph, questions: list[dict], pools: dict[str, list[str]]) -> dict:
    rows = [pool_eval(graph, pools.get(item["question_id"], []), gold_set(item)) for item in questions]
    n = len(rows)
    return {
        "num_questions": n,
        "hit": sum(float(row["hit"]) for row in rows) / n if n else 0.0,
        "recall": sum(float(row["recall"]) for row in rows) / n if n else 0.0,
        "full_cover": sum(float(row["full_cover"]) for row in rows) / n if n else 0.0,
        "avg_candidates": sum(float(row["num_candidates"]) for row in rows) / n if n else 0.0,
    }


def pool_eval(graph: MemoryGraph, fact_ids: list[str], gold: set[str]) -> dict:
    predicted = set()
    for fact_id in fact_ids:
        predicted.update(raw_gold_ids_for_fact(graph, fact_id))
    matched = predicted & gold
    return {
        "hit": bool(matched),
        "recall": len(matched) / len(gold) if gold else 0.0,
        "full_cover": bool(gold) and gold.issubset(predicted),
        "num_candidates": len(fact_ids),
    }


def raw_gold_ids_for_pack(pack: Node) -> set[str]:
    raw_ids = set()
    for raw_id in pack.metadata.get("raw_ids", []):
        raw_ids.add(normalize_raw_id(raw_id))
    return raw_ids


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


def conversation_id(node_id: str) -> str:
    return node_id.split(":", 1)[0]


def render_markdown(payload: dict) -> str:
    lines = [
        "# Graph V3.5-C Pack-Aware Oracle Diagnosis",
        "",
        f"- Graph: `{payload['graph']}`",
        f"- Base candidates: `{payload['base_candidates']}`",
        f"- Candidate topN: `{payload['candidate_topn']}`",
        f"- Pack topK: `{payload['pack_topk']}`",
        f"- Representatives per pack: `{payload['reps_per_pack']}`",
        "",
        "## Fact Candidate Pools",
        "",
        "| Pool | Hit | Recall | FullCover | Avg Cand |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in payload["summary"].items():
        lines.append(
            f"| {name} | {item['hit']:.4f} | {item['recall']:.4f} | "
            f"{item['full_cover']:.4f} | {item['avg_candidates']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Pack Coverage",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for key, value in payload["pack_summary"].items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `base_fact_top150` is the no-temporal D true union top150 pool.",
            "- `pack_rep_top150` retrieves packs by BM25 over pack text and expands representative facts.",
            "- `base_plus_pack_matched_top150` merges base and pack facts but keeps the same top150 budget.",
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
