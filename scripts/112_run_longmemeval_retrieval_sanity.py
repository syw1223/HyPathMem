from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from common import read_json, resolve_path, write_json
from hytopomem.memory.node_extractor import content_terms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/longmemeval_s/graph_session_hierarchy_v1.json")
    parser.add_argument("--data", default="data/longmemeval/processed/longmemeval_s_mvp.json")
    parser.add_argument("--output-json", default="outputs/eval/longmemeval_retrieval_sanity_session_v1.json")
    parser.add_argument("--output-md", default="outputs/eval/longmemeval_retrieval_sanity_session_v1.md")
    parser.add_argument("--topk", default="5,20,50,100")
    parser.add_argument("--seed-topk", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ce-model", default="")
    parser.add_argument("--ce-device", default=None)
    parser.add_argument("--ce-batch-size", type=int, default=64)
    args = parser.parse_args()

    topks = [int(item) for item in args.topk.split(",") if item.strip()]
    data = read_json(resolve_path(args.data))
    if args.limit:
        data = data[: args.limit]
    wanted_convs = {item["conversation_id"] for item in data}
    graph = load_graph_nodes(resolve_path(args.graph), wanted_convs)

    fact_index = build_fact_index(graph)
    items = qa_items(data)
    results = {
        "config": {
            "graph": args.graph,
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

    for route in ("bm25_fact", "bm25_session_expand"):
        predictions = {}
        for item in items:
            conv_index = fact_index.get(item["conversation_id"])
            if conv_index is None:
                predictions[item["question_id"]] = []
                continue
            ranked = rank_bm25(conv_index, item["question"], topn=max(topks + [args.seed_topk]))
            if route == "bm25_fact":
                selected = [fact_id for fact_id, _score in ranked[: max(topks)]]
            else:
                selected = session_expand(conv_index, ranked, seed_topk=args.seed_topk, topn=max(topks))
            predictions[item["question_id"]] = selected
        results["routes"][route] = evaluate(items, predictions, fact_index, topks)

        if args.ce_model:
            ce_predictions = ce_rerank_predictions(
                items,
                predictions,
                fact_index,
                model_name_or_path=args.ce_model,
                device=args.ce_device,
                batch_size=args.ce_batch_size,
                topn=max(topks),
                route=route,
            )
            results["routes"][f"{route}_ce"] = evaluate(items, ce_predictions, fact_index, topks)

    write_json(results, resolve_path(args.output_json))
    write_markdown(results, resolve_path(args.output_md))
    print(f"wrote {resolve_path(args.output_json)}")
    print(f"wrote {resolve_path(args.output_md)}")
    for route, payload in results["routes"].items():
        print(route, payload["overall_with_gold"])


def load_graph_nodes(path, wanted_convs: set[str]) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    nodes = {}
    for node_id, node in payload["nodes"].items():
        if conversation_id(node_id) in wanted_convs:
            nodes[node_id] = node
    return nodes


def build_fact_index(nodes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_conv: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_to_facts: dict[str, list[str]] = defaultdict(list)
    fact_to_raw: dict[str, list[str]] = {}
    for node_id, node in nodes.items():
        if node.get("type") != "FACT":
            continue
        conv_id = conversation_id(node_id)
        metadata = node.get("metadata", {})
        terms = content_terms(node.get("text", ""))
        event_id = str(metadata.get("event_id") or "")
        record = {
            "fact_id": node_id,
            "text": node.get("text", ""),
            "terms": terms,
            "length": len(terms),
            "event_id": event_id,
            "raw_ids": list(node.get("support_ids") or metadata.get("support_raw_ids") or []),
        }
        by_conv[conv_id].append(record)
        fact_to_raw[node_id] = record["raw_ids"]
        if event_id:
            event_to_facts[event_id].append(node_id)

    output = {}
    for conv_id, facts in by_conv.items():
        df = Counter()
        inverted: dict[str, list[tuple[str, int]]] = defaultdict(list)
        lengths = []
        fact_by_id = {}
        for fact in facts:
            fact_by_id[fact["fact_id"]] = fact
            tf = Counter(fact["terms"])
            lengths.append(fact["length"])
            for term, count in tf.items():
                df[term] += 1
                inverted[term].append((fact["fact_id"], count))
        output[conv_id] = {
            "facts": facts,
            "fact_by_id": fact_by_id,
            "fact_to_raw": {fact["fact_id"]: fact["raw_ids"] for fact in facts},
            "event_to_facts": event_to_facts,
            "df": df,
            "inverted": inverted,
            "num_docs": len(facts),
            "avgdl": mean(lengths) if lengths else 1.0,
        }
    return output


def qa_items(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for conv in data:
        conv_id = conv["conversation_id"]
        for qa in conv.get("qa", []):
            gold_raw_ids = [f"{conv_id}:raw:{turn_id}" for turn_id in qa.get("gold_evidence", [])]
            items.append(
                {
                    "conversation_id": conv_id,
                    "question_id": qa["question_id"],
                    "question": qa.get("question", ""),
                    "question_type": qa.get("question_type", "unknown"),
                    "gold_raw_ids": gold_raw_ids,
                    "is_abstention": bool(qa.get("is_abstention")),
                }
            )
    return items


def rank_bm25(index: dict[str, Any], query: str, topn: int) -> list[tuple[str, float]]:
    qterms = content_terms(query)
    if not qterms:
        return []
    qtf = Counter(qterms)
    scores: dict[str, float] = defaultdict(float)
    n = max(1, int(index["num_docs"]))
    avgdl = max(1.0, float(index["avgdl"]))
    k1 = 1.2
    b = 0.75
    for term, qcount in qtf.items():
        postings = index["inverted"].get(term, [])
        if not postings:
            continue
        df = int(index["df"].get(term, 0))
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for fact_id, tf in postings:
            doc_len = max(1, int(index["fact_by_id"][fact_id]["length"]))
            denom = tf + k1 * (1.0 - b + b * doc_len / avgdl)
            scores[fact_id] += qcount * idf * (tf * (k1 + 1.0) / denom)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:topn]


def session_expand(index: dict[str, Any], ranked: list[tuple[str, float]], *, seed_topk: int, topn: int) -> list[str]:
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for rank, (fact_id, score) in enumerate(ranked[:seed_topk], start=1):
        if fact_id not in scores:
            order[fact_id] = len(order)
        scores[fact_id] = max(scores.get(fact_id, float("-inf")), score + 1.0 / rank)
        event_id = index["fact_by_id"].get(fact_id, {}).get("event_id", "")
        if not event_id:
            continue
        for offset, neighbor_id in enumerate(index["event_to_facts"].get(event_id, [])):
            if neighbor_id not in index["fact_by_id"]:
                continue
            if neighbor_id not in order:
                order[neighbor_id] = len(order)
            neighbor_score = score * 0.92 - 0.0001 * offset
            scores[neighbor_id] = max(scores.get(neighbor_id, float("-inf")), neighbor_score)
    return [
        fact_id
        for fact_id, _score in sorted(scores.items(), key=lambda item: (-item[1], order[item[0]], item[0]))[:topn]
    ]


def evaluate(
    items: list[dict[str, Any]],
    predictions: dict[str, list[str]],
    fact_index: dict[str, dict[str, Any]],
    topks: list[int],
) -> dict[str, Any]:
    overall = metrics_for_subset(items, predictions, fact_index, topks, require_gold=False)
    overall_with_gold = metrics_for_subset(items, predictions, fact_index, topks, require_gold=True)
    by_type = {}
    for qtype in sorted({item["question_type"] for item in items}):
        subset = [item for item in items if item["question_type"] == qtype]
        by_type[qtype] = metrics_for_subset(subset, predictions, fact_index, topks, require_gold=True)
    return {
        "overall_all_questions": overall,
        "overall_with_gold": overall_with_gold,
        "by_question_type_with_gold": by_type,
    }


def ce_rerank_predictions(
    items: list[dict[str, Any]],
    predictions: dict[str, list[str]],
    fact_index: dict[str, dict[str, Any]],
    *,
    model_name_or_path: str,
    device: str | None,
    batch_size: int,
    topn: int,
    route: str,
) -> dict[str, list[str]]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name_or_path, device=device)
    output: dict[str, list[str]] = {}
    for idx, item in enumerate(items, start=1):
        conv_index = fact_index.get(item["conversation_id"])
        if conv_index is None:
            output[item["question_id"]] = []
            continue
        fact_ids = predictions.get(item["question_id"], [])[:topn]
        pairs = [(item["question"], conv_index["fact_by_id"][fact_id]["text"]) for fact_id in fact_ids]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False) if pairs else []
        ranked = sorted(zip(fact_ids, [float(score) for score in scores]), key=lambda row: (-row[1], row[0]))
        output[item["question_id"]] = [fact_id for fact_id, _score in ranked[:topn]]
        if idx % 50 == 0:
            print(f"ce rerank {route}: {idx}/{len(items)}", flush=True)
    return output


def metrics_for_subset(
    items: list[dict[str, Any]],
    predictions: dict[str, list[str]],
    fact_index: dict[str, dict[str, Any]],
    topks: list[int],
    *,
    require_gold: bool,
) -> dict[str, Any]:
    rows = [item for item in items if (item["gold_raw_ids"] or not require_gold)]
    metrics = {"count": len(rows)}
    for k in topks:
        hits = []
        recalls = []
        fullcovers = []
        cand_counts = []
        for item in rows:
            gold = set(item["gold_raw_ids"])
            pred_fact_ids = predictions.get(item["question_id"], [])[:k]
            pred_raw_ids = raw_ids_for_facts(item["conversation_id"], pred_fact_ids, fact_index)
            cand_counts.append(len(pred_fact_ids))
            if not gold:
                hits.append(0.0)
                recalls.append(0.0)
                fullcovers.append(0.0)
                continue
            matched = gold & pred_raw_ids
            hits.append(float(bool(matched)))
            recalls.append(len(matched) / len(gold))
            fullcovers.append(float(gold <= pred_raw_ids))
        metrics[f"top{k}"] = {
            "hit": mean(hits) if hits else 0.0,
            "recall": mean(recalls) if recalls else 0.0,
            "fullcover": mean(fullcovers) if fullcovers else 0.0,
            "avg_candidates": mean(cand_counts) if cand_counts else 0.0,
        }
    return metrics


def raw_ids_for_facts(conv_id: str, fact_ids: list[str], fact_index: dict[str, dict[str, Any]]) -> set[str]:
    index = fact_index.get(conv_id, {})
    fact_to_raw = index.get("fact_to_raw", {})
    raw_ids = set()
    for fact_id in fact_ids:
        raw_ids.update(fact_to_raw.get(fact_id, []))
    return raw_ids


def conversation_id(node_id: str) -> str:
    for marker in (":raw:", ":fact:", ":anchor:", ":event_", ":topic_"):
        if marker in node_id:
            return node_id.split(marker, 1)[0]
    return node_id.split(":", 1)[0]


def write_markdown(results: dict[str, Any], path) -> None:
    lines = [
        "# LongMemEval Retrieval Sanity",
        "",
        f"Instances: {results['dataset']['instances']}",
        f"QA: {results['dataset']['qa']}",
        f"QA with gold: {results['dataset']['qa_with_gold']}",
        f"Abstention: {results['dataset']['abstention']}",
        "",
    ]
    for route, payload in results["routes"].items():
        lines.extend([f"## {route}", "", "| Split | K | Hit | Recall | FullCover | AvgCand |", "|---|---:|---:|---:|---:|---:|"])
        for split_name in ("overall_with_gold", "overall_all_questions"):
            split = payload[split_name]
            for key, values in split.items():
                if not key.startswith("top"):
                    continue
                lines.append(
                    f"| {split_name} | {key[3:]} | {values['hit']:.4f} | {values['recall']:.4f} | "
                    f"{values['fullcover']:.4f} | {values['avg_candidates']:.1f} |"
                )
        lines.append("")
        lines.extend(["### By Question Type (With Gold)", "", "| Type | K | Hit | Recall | FullCover |", "|---|---:|---:|---:|---:|"])
        for qtype, split in payload["by_question_type_with_gold"].items():
            for key, values in split.items():
                if key.startswith("top"):
                    lines.append(
                        f"| {qtype} | {key[3:]} | {values['hit']:.4f} | {values['recall']:.4f} | {values['fullcover']:.4f} |"
                    )
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
