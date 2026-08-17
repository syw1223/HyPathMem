from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, resolve_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/v3_9_query_cards/qwen3_card_annotated_base150_paths_v3_clean.json",
    )
    parser.add_argument("--base-topn", type=int, default=100)
    parser.add_argument("--extra", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--min-card-confidence", type=float, default=0.75)
    parser.add_argument("--output-prefix", default="outputs/v3_9_query_cards/qwen3_card_guided_expand")
    args = parser.parse_args()

    items = read_json(resolve_path(args.input))
    outputs = {extra: [] for extra in args.extra}
    stats = {extra: {"questions": 0, "added": 0, "same_event": 0, "same_topic": 0} for extra in args.extra}

    for item in items:
        paths = list(item.get("paths", []))
        base = paths[: args.base_topn]
        tail = paths[args.base_topn :]
        card_events, card_topics = card_scopes(base, args.min_card_confidence)
        ranked_tail = []
        for path in tail:
            score, source = expansion_score(path, card_events, card_topics)
            if score <= 0.0:
                continue
            ranked_tail.append((score, ce_score(path), source, path))
        ranked_tail.sort(key=lambda row: (row[0], row[1]), reverse=True)

        for extra in args.extra:
            selected_extra = ranked_tail[:extra]
            copied = dict(item)
            copied_paths = [dict(path) for path in base]
            for _score, _ce, source, path in selected_extra:
                copied_path = dict(path)
                metadata = dict(copied_path.get("metadata", {}))
                metadata["v3_9_card_guided_expansion"] = "true"
                metadata["v3_9_card_guided_expansion_source"] = source
                copied_path["metadata"] = metadata
                copied_paths.append(copied_path)
                stats[extra]["added"] += 1
                stats[extra][source] += 1
            metadata = dict(copied.get("metadata", {}))
            metadata.update(
                {
                    "method": f"v3_9_card_guided_local_expansion_extra{extra}",
                    "base_topn": args.base_topn,
                    "extra_budget": extra,
                    "expanded_candidate_count": len(copied_paths),
                    "min_card_confidence": args.min_card_confidence,
                }
            )
            copied["metadata"] = metadata
            copied["paths"] = copied_paths
            outputs[extra].append(copied)
            stats[extra]["questions"] += 1

    for extra, output_items in outputs.items():
        output = resolve_path(f"{args.output_prefix}{args.base_topn + extra}.json")
        write_json(output_items, output)
        summary = {
            "input": str(resolve_path(args.input)),
            "output": str(output),
            "base_topn": args.base_topn,
            "extra_budget": extra,
            "min_card_confidence": args.min_card_confidence,
            **stats[extra],
            "avg_added_per_question": stats[extra]["added"] / max(stats[extra]["questions"], 1),
        }
        write_json(summary, output.with_suffix(".summary.json"))
        print(summary)


def card_scopes(paths: list[dict], min_confidence: float) -> tuple[set[str], set[str]]:
    events = set()
    topics = set()
    for path in paths:
        metadata = path.get("metadata", {})
        if str(metadata.get("v3_9_query_card", "")).lower() != "true":
            continue
        confidence = float(metadata.get("nary_hyperedge_confidence", 0.0) or 0.0)
        if confidence < min_confidence:
            continue
        event_id = str(metadata.get("event_node_id", "") or "")
        topic_id = str(metadata.get("topic_node_id", "") or "")
        if event_id:
            events.add(event_id)
        if topic_id:
            topics.add(topic_id)
    return events, topics


def expansion_score(path: dict, card_events: set[str], card_topics: set[str]) -> tuple[float, str]:
    metadata = path.get("metadata", {})
    event_id = str(metadata.get("event_node_id", "") or "")
    topic_id = str(metadata.get("topic_node_id", "") or "")
    if event_id and event_id in card_events:
        return 2.0 - degree_penalty(metadata, "event_degree"), "same_event"
    if topic_id and topic_id in card_topics:
        return 1.0 - degree_penalty(metadata, "topic_degree"), "same_topic"
    return 0.0, ""


def degree_penalty(metadata: dict, key: str) -> float:
    try:
        degree = float(metadata.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        degree = 0.0
    return min(degree / 100.0, 0.5)


def ce_score(path: dict) -> float:
    scores = path.get("scores", {})
    try:
        return float(scores.get("cross_encoder", path.get("score", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
