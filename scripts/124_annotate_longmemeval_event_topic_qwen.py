from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from common import resolve_path, write_json
from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient
from hytopomem.llm.semantic_prompts import (
    EVENT_SYSTEM_PROMPT,
    PROMPT_VERSION,
    TOPIC_SYSTEM_PROMPT,
    event_text,
    event_user_prompt,
    topic_text,
    topic_user_prompt,
)
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, Node, NodeType


DEFAULT_GRAPH = "outputs/longmemeval_s/graph_semantic_hierarchy_v3.json"
DEFAULT_OUTPUT = "outputs/longmemeval_s/v3_10_fine/graph_semantic_hierarchy_v3_qwen_summary.json"
DEFAULT_CACHE = "outputs/llm_annotations/longmemeval_v3_10_qwen_event_topic_summary.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate LongMemEval semantic hierarchy EVENT/TOPIC nodes with Qwen summaries. "
            "The hierarchy edges stay fixed; only node text/metadata is replaced."
        )
    )
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--paths", default="")
    parser.add_argument("--path-topn", type=int, default=20)
    parser.add_argument("--scope", choices=["all", "path_nodes"], default="path_nodes")
    parser.add_argument("--model", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-event-tokens", type=int, default=220)
    parser.add_argument("--max-topic-tokens", type=int, default=260)
    parser.add_argument("--max-facts-per-event", type=int, default=8)
    parser.add_argument("--max-events-per-topic", type=int, default=10)
    parser.add_argument("--limit-events", type=int, default=0)
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--limit-topics", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    graph_path = resolve_path(args.graph)
    output_path = resolve_path(args.output)
    cache_path = resolve_path(args.cache)
    graph = JsonGraphStore().load(graph_path)
    target_event_ids, target_episode_ids, target_topic_ids = select_targets(graph, args)

    event_nodes = [graph.nodes[node_id] for node_id in sorted(target_event_ids) if node_id in graph.nodes]
    episode_nodes = [graph.nodes[node_id] for node_id in sorted(target_episode_ids) if node_id in graph.nodes]
    topic_nodes = [graph.nodes[node_id] for node_id in sorted(target_topic_ids) if node_id in graph.nodes]
    if args.limit_events:
        event_nodes = event_nodes[: args.limit_events]
    if args.limit_episodes:
        episode_nodes = episode_nodes[: args.limit_episodes]
    if args.limit_topics:
        topic_nodes = topic_nodes[: args.limit_topics]

    print(
        f"graph={graph_path} scope={args.scope} events={len(event_nodes)} episodes={len(episode_nodes)} topics={len(topic_nodes)} "
        f"output={output_path}",
        flush=True,
    )
    if args.dry_run:
        print_preview(graph, event_nodes[:2], episode_nodes[:2], topic_nodes[:2], args)
        return

    disable_proxy_for_local_endpoint(args.base_url)
    client = OpenAICompatibleChatClient(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    cache = load_cache(cache_path) if args.resume or cache_path.exists() else {}
    started = time.perf_counter()
    processed = 0

    for node in event_nodes:
        key = cache_key("event", node.node_id, args.model)
        record = cache.get(key)
        if record is None:
            facts = event_fact_payload(graph, node, max_facts=args.max_facts_per_event)
            record = call_json_annotation(
                client,
                model=args.model,
                system_prompt=EVENT_SYSTEM_PROMPT,
                user_prompt=event_user_prompt(facts),
                temperature=args.temperature,
                max_tokens=args.max_event_tokens,
                node_id=node.node_id,
                annotation_type="event",
            )
            append_cache(cache_path, key, record)
            cache[key] = record
        apply_event_annotation(graph.nodes[node.node_id], record, args)
        processed += 1
        maybe_save(graph, output_path, processed, args.save_every, graph_path, args, started)
        if processed % 25 == 0:
            print(f"annotated {processed}/{len(event_nodes) + len(episode_nodes) + len(topic_nodes)} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    for node in episode_nodes:
        key = cache_key("episode", node.node_id, args.model)
        record = cache.get(key)
        if record is None:
            facts = event_fact_payload(graph, node, max_facts=args.max_facts_per_event)
            record = call_json_annotation(
                client,
                model=args.model,
                system_prompt=EVENT_SYSTEM_PROMPT,
                user_prompt=event_user_prompt(facts),
                temperature=args.temperature,
                max_tokens=args.max_event_tokens,
                node_id=node.node_id,
                annotation_type="episode",
            )
            append_cache(cache_path, key, record)
            cache[key] = record
        apply_episode_annotation(graph.nodes[node.node_id], record, args)
        processed += 1
        maybe_save(graph, output_path, processed, args.save_every, graph_path, args, started)
        if processed % 25 == 0:
            print(f"annotated {processed}/{len(event_nodes) + len(episode_nodes) + len(topic_nodes)} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    for node in topic_nodes:
        key = cache_key("topic", node.node_id, args.model)
        record = cache.get(key)
        if record is None:
            events = topic_event_payload(graph, node, max_events=args.max_events_per_topic)
            record = call_json_annotation(
                client,
                model=args.model,
                system_prompt=TOPIC_SYSTEM_PROMPT,
                user_prompt=topic_user_prompt(events),
                temperature=args.temperature,
                max_tokens=args.max_topic_tokens,
                node_id=node.node_id,
                annotation_type="topic",
            )
            append_cache(cache_path, key, record)
            cache[key] = record
        apply_topic_annotation(graph.nodes[node.node_id], record, args)
        processed += 1
        maybe_save(graph, output_path, processed, args.save_every, graph_path, args, started)
        if processed % 25 == 0:
            print(f"annotated {processed}/{len(event_nodes) + len(episode_nodes) + len(topic_nodes)} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    finalize_metadata(graph, graph_path, args, started, len(event_nodes), len(episode_nodes), len(topic_nodes))
    JsonGraphStore().save(graph, output_path)
    diag = diagnostics(graph, event_nodes, episode_nodes, topic_nodes, args)
    write_json(diag, output_path.with_suffix(".diagnostics.json"))
    print(f"wrote {output_path}")
    print(f"wrote {output_path.with_suffix('.diagnostics.json')}")
    print(diag["summary"])


def select_targets(graph: MemoryGraph, args: argparse.Namespace) -> tuple[set[str], set[str], set[str]]:
    if args.scope == "all":
        return (
            {
                node.node_id
                for node in graph.iter_nodes(NodeType.EVENT)
                if node.metadata.get("hierarchy_v3_3") != "episode"
            },
            {
                node.node_id
                for node in graph.iter_nodes(NodeType.EVENT)
                if node.metadata.get("hierarchy_v3_3") == "episode"
            },
            {node.node_id for node in graph.iter_nodes(NodeType.TOPIC)},
        )
    if not args.paths:
        raise ValueError("--paths is required when --scope path_nodes")
    payload = read_json(resolve_path(args.paths))
    items = payload if isinstance(payload, list) else payload.get("items", [])
    event_ids: set[str] = set()
    episode_ids: set[str] = set()
    topic_ids: set[str] = set()
    for item in items:
        for path in item.get("paths", [])[: args.path_topn]:
            metadata = path.get("metadata") or {}
            if metadata.get("event_node_id"):
                add_event_or_episode(graph, str(metadata["event_node_id"]), event_ids, episode_ids)
            if metadata.get("topic_node_id"):
                topic_ids.add(str(metadata["topic_node_id"]))
            if metadata.get("episode_node_id"):
                add_event_or_episode(graph, str(metadata["episode_node_id"]), event_ids, episode_ids)
            for node_id in path.get("node_ids", []):
                if node_id in graph.nodes and graph.nodes[node_id].type == NodeType.EVENT:
                    add_event_or_episode(graph, node_id, event_ids, episode_ids)
                elif node_id in graph.nodes and graph.nodes[node_id].type == NodeType.TOPIC:
                    topic_ids.add(node_id)
    return event_ids, episode_ids, topic_ids


def add_event_or_episode(graph: MemoryGraph, node_id: str, event_ids: set[str], episode_ids: set[str]) -> None:
    node = graph.nodes.get(node_id)
    if node is None:
        return
    if node.metadata.get("hierarchy_v3_3") == "episode":
        episode_ids.add(node_id)
    else:
        event_ids.add(node_id)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def disable_proxy_for_local_endpoint(base_url: str) -> None:
    if "127.0.0.1" not in base_url and "localhost" not in base_url:
        return
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(name, None)
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    additions = ["127.0.0.1", "localhost"]
    values = [item.strip() for item in existing.split(",") if item.strip()]
    for item in additions:
        if item not in values:
            values.append(item)
    os.environ["NO_PROXY"] = ",".join(values)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


def call_json_annotation(
    client: OpenAICompatibleChatClient,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    node_id: str,
    annotation_type: str,
) -> dict:
    result = client.chat_completion_with_metadata(
        model=model,
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    parse_error = ""
    try:
        annotation = parse_json_object(result.content)
    except Exception as exc:  # noqa: BLE001 - bad model JSON should not stop a long annotation run.
        parse_error = f"{type(exc).__name__}: {exc}"
        annotation = fallback_annotation(annotation_type, user_prompt)
    return {
        "node_id": node_id,
        "type": annotation_type,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "annotation": annotation,
        "parse_error": parse_error,
        "raw_response": result.content,
        "usage": result.usage,
        "elapsed_seconds": result.elapsed_seconds,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object, got {type(payload).__name__}")
    return payload


def fallback_annotation(annotation_type: str, user_prompt: str) -> dict:
    payload = safe_json_from_prompt(user_prompt)
    if annotation_type in {"event", "episode"}:
        facts = payload.get("facts") if isinstance(payload, dict) else []
        facts = facts if isinstance(facts, list) else []
        summary = " ".join(str(row.get("text", "")) for row in facts[:2] if isinstance(row, dict)).strip()
        return {
            "event_title": "Fallback event summary",
            "event_summary": summary[:360] if summary else "Related memory facts.",
            "key_entities": [],
            "key_actions": [],
            "time_hint": "",
        }
    events = payload.get("events") if isinstance(payload, dict) else []
    events = events if isinstance(events, list) else []
    summary = " ".join(str(row.get("event_summary", "")) for row in events[:2] if isinstance(row, dict)).strip()
    return {
        "topic_name": "Fallback topic",
        "topic_summary": summary[:480] if summary else "Related memory events.",
        "main_themes": [],
        "key_entities": [],
    }


def safe_json_from_prompt(user_prompt: str) -> dict:
    start = user_prompt.find("{")
    end = user_prompt.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(user_prompt[start : end + 1])
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def event_fact_payload(graph: MemoryGraph, event: Node, *, max_facts: int) -> list[dict]:
    rows = []
    fact_ids = list(event.metadata.get("fact_ids") or event.support_ids)
    for fact_id in fact_ids[:max_facts]:
        fact = graph.nodes.get(str(fact_id))
        if fact is None:
            continue
        rows.append(
            {
                "fact_id": fact.node_id,
                "text": fact.text,
                "source": fact.source,
                "speaker": fact.metadata.get("speaker", ""),
                "time": fact.time or fact.metadata.get("turn_id", ""),
            }
        )
    return rows


def topic_event_payload(graph: MemoryGraph, topic: Node, *, max_events: int) -> list[dict]:
    rows = []
    event_ids = list(topic.metadata.get("event_ids") or topic.support_ids)
    for event_id in event_ids[:max_events]:
        event = graph.nodes.get(str(event_id))
        if event is None:
            continue
        llm = event.metadata.get("llm_semantic") or {}
        annotation = llm.get("annotation") if isinstance(llm, dict) else {}
        if not isinstance(annotation, dict):
            annotation = {}
        rows.append(
            {
                "event_id": event.node_id,
                "event_summary": str(annotation.get("event_summary") or event.text),
                "event_title": str(annotation.get("event_title") or ""),
                "key_entities": annotation.get("key_entities") or event.metadata.get("entities", []),
                "time_hint": str(annotation.get("time_hint") or ""),
            }
        )
    return rows


def apply_event_annotation(node: Node, record: dict, args: argparse.Namespace) -> None:
    annotation = dict(record.get("annotation") or {})
    metadata = dict(node.metadata)
    metadata.setdefault("template_text", node.text)
    metadata["label_type"] = "qwen_semantic"
    metadata["llm_semantic"] = semantic_metadata(record, args)
    node.metadata = metadata
    node.text = event_text(annotation)
    node.source = "qwen_event_summary_v3_10"


def apply_episode_annotation(node: Node, record: dict, args: argparse.Namespace) -> None:
    annotation = dict(record.get("annotation") or {})
    metadata = dict(node.metadata)
    metadata.setdefault("template_text", node.text)
    metadata["label_type"] = "qwen_semantic"
    metadata["llm_semantic"] = semantic_metadata(record, args)
    node.metadata = metadata
    node.text = "Episode: " + event_text(annotation)
    node.source = "qwen_episode_summary_v3_10"


def apply_topic_annotation(node: Node, record: dict, args: argparse.Namespace) -> None:
    annotation = dict(record.get("annotation") or {})
    metadata = dict(node.metadata)
    metadata.setdefault("template_text", node.text)
    metadata["label_type"] = "qwen_semantic"
    metadata["llm_semantic"] = semantic_metadata(record, args)
    node.metadata = metadata
    node.text = topic_text(annotation)
    node.source = "qwen_topic_summary_v3_10"


def semantic_metadata(record: dict, args: argparse.Namespace) -> dict:
    return {
        "annotation": record.get("annotation", {}),
        "model": record.get("model", args.model),
        "prompt_version": record.get("prompt_version", PROMPT_VERSION),
        "temperature": args.temperature,
        "usage": record.get("usage", {}),
        "elapsed_seconds": record.get("elapsed_seconds", 0.0),
        "created_at": record.get("created_at", ""),
    }


def cache_key(annotation_type: str, node_id: str, model: str) -> str:
    return f"longmemeval_v3_10|{annotation_type}|{node_id}|{model}|{PROMPT_VERSION}"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    output = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            output[str(row["cache_key"])] = dict(row["record"])
    return output


def append_cache(path: Path, key: str, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"cache_key": key, "record": record}, ensure_ascii=False) + "\n")


def maybe_save(
    graph: MemoryGraph,
    output_path: Path,
    processed: int,
    save_every: int,
    graph_path: Path,
    args: argparse.Namespace,
    started: float,
) -> None:
    if not save_every or processed % save_every != 0:
        return
    finalize_metadata(graph, graph_path, args, started, -1, -1, -1)
    JsonGraphStore().save(graph, output_path)


def finalize_metadata(
    graph: MemoryGraph,
    graph_path: Path,
    args: argparse.Namespace,
    started: float,
    event_count: int,
    episode_count: int,
    topic_count: int,
) -> None:
    metadata = dict(graph.metadata)
    metadata["longmemeval_v3_10_qwen_semantic_summary"] = {
        "source_graph": str(graph_path),
        "structure_fixed": True,
        "annotation_only": True,
        "scope": args.scope,
        "paths": args.paths,
        "path_topn": args.path_topn,
        "event_nodes_annotated": event_count,
        "episode_nodes_annotated": episode_count,
        "topic_nodes_annotated": topic_count,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "prompt_version": PROMPT_VERSION,
        "elapsed_seconds": time.perf_counter() - started,
    }
    graph.metadata = metadata


def diagnostics(graph: MemoryGraph, events: list[Node], episodes: list[Node], topics: list[Node], args: argparse.Namespace) -> dict:
    event_text_lengths = [len(graph.nodes[node.node_id].text.split()) for node in events if node.node_id in graph.nodes]
    episode_text_lengths = [len(graph.nodes[node.node_id].text.split()) for node in episodes if node.node_id in graph.nodes]
    topic_text_lengths = [len(graph.nodes[node.node_id].text.split()) for node in topics if node.node_id in graph.nodes]
    return {
        "summary": {
            "events_annotated": len(events),
            "episodes_annotated": len(episodes),
            "topics_annotated": len(topics),
            "event_text_words_mean": sum(event_text_lengths) / max(len(event_text_lengths), 1),
            "episode_text_words_mean": sum(episode_text_lengths) / max(len(episode_text_lengths), 1),
            "topic_text_words_mean": sum(topic_text_lengths) / max(len(topic_text_lengths), 1),
            "scope": args.scope,
            "path_topn": args.path_topn,
            "model": args.model,
        }
    }


def print_preview(graph: MemoryGraph, events: list[Node], episodes: list[Node], topics: list[Node], args: argparse.Namespace) -> None:
    for event in events:
        print(f"\nEVENT {event.node_id}")
        print(event_user_prompt(event_fact_payload(graph, event, max_facts=args.max_facts_per_event)))
    for episode in episodes:
        print(f"\nEPISODE {episode.node_id}")
        print(event_user_prompt(event_fact_payload(graph, episode, max_facts=args.max_facts_per_event)))
    for topic in topics:
        print(f"\nTOPIC {topic.node_id}")
        print(topic_user_prompt(topic_event_payload(graph, topic, max_events=args.max_events_per_topic)))


if __name__ == "__main__":
    main()
