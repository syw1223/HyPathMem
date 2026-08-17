from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import read_json, resolve_path, write_json
from hytopomem.memory.node_extractor import content_terms


HELPERS = importlib.util.spec_from_file_location(
    "longmemeval_retrieval_sanity_helpers",
    Path(__file__).resolve().parent / "112_run_longmemeval_retrieval_sanity.py",
)
helpers = importlib.util.module_from_spec(HELPERS)
assert HELPERS and HELPERS.loader
HELPERS.loader.exec_module(helpers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/longmemeval_s/graph_semantic_hierarchy_v3.json")
    parser.add_argument("--data", default="data/longmemeval/processed/longmemeval_s_mvp.json")
    parser.add_argument("--output-json", default="outputs/eval/longmemeval_semantic_bottom_up_v3.json")
    parser.add_argument("--output-md", default="outputs/eval/longmemeval_semantic_bottom_up_v3.md")
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
    graph = load_graph(resolve_path(args.graph), wanted_convs)
    fact_index = build_semantic_fact_index(graph)
    items = helpers.qa_items(data)

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

    routes = (
        "bm25_fact",
        "bm25_semantic_event_expand",
        "bm25_semantic_event_topic_expand",
    )
    for route in routes:
        predictions = {}
        for item in items:
            conv_index = fact_index.get(item["conversation_id"])
            if conv_index is None:
                predictions[item["question_id"]] = []
                continue
            ranked = helpers.rank_bm25(conv_index, item["question"], topn=max(topks + [args.seed_topk]))
            if route == "bm25_fact":
                selected = [fact_id for fact_id, _score in ranked[: max(topks)]]
            elif route == "bm25_semantic_event_expand":
                selected = semantic_expand(conv_index, ranked, seed_topk=args.seed_topk, topn=max(topks), include_topic=False)
            else:
                selected = semantic_expand(conv_index, ranked, seed_topk=args.seed_topk, topn=max(topks), include_topic=True)
            predictions[item["question_id"]] = selected
        results["routes"][route] = helpers.evaluate(items, predictions, fact_index, topks)

        if args.ce_model:
            ce_predictions = helpers.ce_rerank_predictions(
                items,
                predictions,
                fact_index,
                model_name_or_path=args.ce_model,
                device=args.ce_device,
                batch_size=args.ce_batch_size,
                topn=max(topks),
                route=route,
            )
            results["routes"][f"{route}_ce"] = helpers.evaluate(items, ce_predictions, fact_index, topks)

    write_json(results, resolve_path(args.output_json))
    helpers.write_markdown(results, resolve_path(args.output_md))
    print(f"wrote {resolve_path(args.output_json)}")
    print(f"wrote {resolve_path(args.output_md)}")
    for route, payload in results["routes"].items():
        print(route, payload["overall_with_gold"])


def load_graph(path, wanted_convs: set[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    nodes = {
        node_id: node
        for node_id, node in payload["nodes"].items()
        if conversation_id(node_id) in wanted_convs
    }
    edges = [
        edge
        for edge in payload["edges"]
        if edge.get("src") in nodes and edge.get("dst") in nodes
    ]
    return {"nodes": nodes, "edges": edges}


def build_semantic_fact_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = graph["nodes"]
    fact_to_event: dict[str, str] = {}
    event_to_facts: dict[str, list[str]] = defaultdict(list)
    event_to_topic: dict[str, str] = {}
    event_to_episode: dict[str, str] = {}
    episode_to_topic: dict[str, str] = {}
    topic_to_events: dict[str, list[str]] = defaultdict(list)
    for edge in graph["edges"]:
        metadata = edge.get("metadata", {})
        role = metadata.get("hierarchy_v3")
        role_v3_3 = metadata.get("hierarchy_v3_3")
        src = edge.get("src", "")
        dst = edge.get("dst", "")
        if role == "fact_event":
            fact_to_event[src] = dst
            event_to_facts[dst].append(src)
        elif role == "event_topic":
            event_to_topic[src] = dst
            topic_to_events[dst].append(src)
        if role_v3_3 == "event_episode":
            event_to_episode[src] = dst
        elif role_v3_3 == "episode_topic":
            episode_to_topic[src] = dst

    by_conv: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node_id, node in nodes.items():
        if node.get("type") != "FACT":
            continue
        conv_id = conversation_id(node_id)
        metadata = node.get("metadata", {})
        event_id = fact_to_event.get(node_id, "")
        episode_id = event_to_episode.get(event_id, "")
        topic_id = event_to_topic.get(event_id, "") or episode_to_topic.get(episode_id, "")
        terms = content_terms(node.get("text", ""))
        by_conv[conv_id].append(
            {
                "fact_id": node_id,
                "text": node.get("text", ""),
                "terms": terms,
                "length": len(terms),
                "event_id": event_id,
                "episode_id": episode_id,
                "topic_id": topic_id,
                "raw_ids": list(node.get("support_ids") or metadata.get("support_raw_ids") or []),
            }
        )

    output = {}
    for conv_id, facts in by_conv.items():
        df = Counter()
        inverted: dict[str, list[tuple[str, int]]] = defaultdict(list)
        lengths = []
        fact_by_id = {}
        conv_events = defaultdict(list)
        conv_topics = defaultdict(list)
        conv_event_to_topic = {}
        conv_event_to_episode = {}
        conv_episode_to_topic = {}
        for fact in facts:
            fact_id = fact["fact_id"]
            fact_by_id[fact_id] = fact
            tf = Counter(fact["terms"])
            lengths.append(fact["length"])
            for term, count in tf.items():
                df[term] += 1
                inverted[term].append((fact_id, count))
            if fact["event_id"]:
                conv_events[fact["event_id"]].append(fact_id)
            if fact["event_id"] and fact["episode_id"]:
                conv_event_to_episode[fact["event_id"]] = fact["episode_id"]
            if fact["episode_id"] and fact["topic_id"]:
                conv_episode_to_topic[fact["episode_id"]] = fact["topic_id"]
            if fact["event_id"] and fact["topic_id"]:
                conv_event_to_topic[fact["event_id"]] = fact["topic_id"]
        for event_id, topic_id in conv_event_to_topic.items():
            conv_topics[topic_id].append(event_id)
        output[conv_id] = {
            "facts": facts,
            "fact_by_id": fact_by_id,
            "fact_to_raw": {fact["fact_id"]: fact["raw_ids"] for fact in facts},
            "event_to_facts": dict(conv_events),
            "event_to_episode": dict(conv_event_to_episode),
            "event_to_topic": dict(conv_event_to_topic),
            "episode_to_topic": dict(conv_episode_to_topic),
            "topic_to_events": dict(conv_topics),
            "df": df,
            "inverted": inverted,
            "num_docs": len(facts),
            "avgdl": sum(lengths) / len(lengths) if lengths else 1.0,
        }
    return output


def semantic_expand(
    index: dict[str, Any],
    ranked: list[tuple[str, float]],
    *,
    seed_topk: int,
    topn: int,
    include_topic: bool,
) -> list[str]:
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for rank, (fact_id, score) in enumerate(ranked[:seed_topk], start=1):
        add_score(scores, order, fact_id, score + 1.0 / rank)
        event_id = index["fact_by_id"].get(fact_id, {}).get("event_id", "")
        if not event_id:
            continue
        for offset, neighbor_id in enumerate(index["event_to_facts"].get(event_id, [])):
            if neighbor_id in index["fact_by_id"]:
                add_score(scores, order, neighbor_id, score * 0.94 - 0.0001 * offset)
        if not include_topic:
            continue
        topic_id = index["event_to_topic"].get(event_id, "")
        for event_offset, topic_event_id in enumerate(index["topic_to_events"].get(topic_id, [])):
            penalty = 0.002 * event_offset
            for fact_offset, neighbor_id in enumerate(index["event_to_facts"].get(topic_event_id, [])):
                if neighbor_id in index["fact_by_id"]:
                    add_score(scores, order, neighbor_id, score * 0.84 - penalty - 0.0001 * fact_offset)
    return [
        fact_id
        for fact_id, _score in sorted(scores.items(), key=lambda item: (-item[1], order[item[0]], item[0]))[:topn]
    ]


def add_score(scores: dict[str, float], order: dict[str, int], fact_id: str, score: float) -> None:
    if fact_id not in order:
        order[fact_id] = len(order)
    scores[fact_id] = max(scores.get(fact_id, float("-inf")), float(score))


def conversation_id(node_id: str) -> str:
    for marker in (":raw:", ":fact:", ":anchor:", ":event", ":topic"):
        if marker in node_id:
            return node_id.split(marker, 1)[0]
    return node_id.split(":", 1)[0]


if __name__ == "__main__":
    main()
