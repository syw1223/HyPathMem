from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from common import resolve_path
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import NodeType


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build support-centered message chunks for Mnemis-style memory-pack QA."
    )
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--output", default="outputs/graphs/v3_11_message_chunks.jsonl")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=1800)
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    raw_by_session = raw_session_index(graph)
    chunks = []
    seen = set()
    for (conv_id, session_id), raw_ids in sorted(raw_by_session.items()):
        for index, raw_id in enumerate(raw_ids):
            start = max(0, index - args.window)
            end = min(len(raw_ids), index + args.window + 1)
            chunk_ids = raw_ids[start:end]
            chunk_key = (conv_id, session_id, start, end)
            if chunk_key in seen:
                continue
            seen.add(chunk_key)
            chunk = build_chunk(graph, conv_id, session_id, chunk_ids, start, end, args.max_chars)
            chunks.append(chunk)

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            print(json.dumps(chunk, ensure_ascii=False), file=handle)

    summary = {
        "graph": str(resolve_path(args.graph)),
        "output": str(output),
        "window": args.window,
        "max_chars": args.max_chars,
        "chunks": len(chunks),
        "sessions": len(raw_by_session),
        "raw_turns": sum(len(items) for items in raw_by_session.values()),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def raw_session_index(graph) -> dict[tuple[str, str], list[str]]:
    by_session: dict[tuple[str, str], list[str]] = defaultdict(list)
    for node in graph.iter_nodes(NodeType.RAW):
        conv_id = conversation_id(node.node_id)
        turn_id = str(node.metadata.get("turn_id") or node.node_id.split(":raw:", 1)[-1])
        session_id = canonical_session_id(turn_id)
        by_session[(conv_id, session_id)].append(node.node_id)
    for key, raw_ids in by_session.items():
        raw_ids.sort(key=lambda raw_id: raw_turn_sort_key(graph.nodes[raw_id]))
    return dict(by_session)


def build_chunk(graph, conv_id: str, session_id: str, raw_ids: list[str], start: int, end: int, max_chars: int) -> dict:
    texts = []
    timestamps = []
    turn_ids = []
    speakers = []
    for raw_id in raw_ids:
        raw = graph.nodes[raw_id]
        turn_id = str(raw.metadata.get("turn_id") or raw_id.split(":raw:", 1)[-1])
        speaker = str(raw.metadata.get("speaker") or "")
        prefix = f"[{raw.time}] " if raw.time else ""
        speaker_part = f"{speaker}: " if speaker else ""
        texts.append(f"{prefix}{speaker_part}{one_line(raw.text, 420)}")
        if raw.time:
            timestamps.append(str(raw.time))
        turn_ids.append(turn_id)
        speakers.append(speaker)
    text = "\n".join(texts)
    if len(text) > max_chars:
        text = text[: max_chars - 22] + "\n[chunk truncated]"
    return {
        "chunk_id": f"{conv_id}:session:{session_id}:chunk:{start:04d}-{end - 1:04d}",
        "conversation_id": conv_id,
        "session_id": session_id,
        "raw_ids": raw_ids,
        "turn_ids": turn_ids,
        "speakers": speakers,
        "timestamp_start": timestamps[0] if timestamps else "",
        "timestamp_end": timestamps[-1] if timestamps else "",
        "text": text,
        "char_len": len(text),
    }


def conversation_id(node_id: str) -> str:
    return node_id.split(":", 1)[0]


def canonical_session_id(turn_id: str) -> str:
    if ":" in turn_id:
        return turn_id.split(":", 1)[0]
    match = re.match(r"([A-Za-z]+\d+)", turn_id)
    return match.group(1) if match else turn_id


def raw_turn_sort_key(raw_node) -> tuple[int, int, str]:
    turn_id = str(raw_node.metadata.get("turn_id") or raw_node.node_id.split(":raw:", 1)[-1])
    match = re.match(r"[A-Za-z]+(\d+):(\d+)", turn_id)
    if match:
        return (int(match.group(1)), int(match.group(2)), turn_id)
    match = re.search(r":t(\d+)$", turn_id)
    if match:
        session = re.search(r"s(\d+)", turn_id)
        return (int(session.group(1)) if session else 0, int(match.group(1)), turn_id)
    return (0, 0, turn_id)


def one_line(text: str, limit: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


if __name__ == "__main__":
    main()
