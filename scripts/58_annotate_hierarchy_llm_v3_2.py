from __future__ import annotations

import argparse
import json
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


DEFAULT_GRAPH = "outputs/graphs/locomo_graph_semantic_hierarchy_v3_1_filtered_rule.json"
DEFAULT_OUTPUT = "outputs/graphs/locomo_graph_semantic_hierarchy_v3_2_gpt4o_semantic.json"
DEFAULT_CACHE = "outputs/llm_annotations/graph_v3_2_gpt4o_semantic_annotations.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api-key-env", default="E_MEM_API_KEY")
    parser.add_argument("--base-url-env", default="E_MEM_BASE_URL")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-event-tokens", type=int, default=220)
    parser.add_argument("--max-topic-tokens", type=int, default=260)
    parser.add_argument("--max-facts-per-event", type=int, default=8)
    parser.add_argument("--max-events-per-topic", type=int, default=10)
    parser.add_argument("--limit-events", type=int, default=0)
    parser.add_argument("--limit-topics", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    graph_path = resolve_path(args.graph)
    output_path = resolve_path(args.output)
    cache_path = resolve_path(args.cache)
    graph = JsonGraphStore().load(graph_path)
    output = graph.model_copy(deep=True)
    cache = load_cache(cache_path) if args.resume or cache_path.exists() else {}

    event_nodes = sorted(output.iter_nodes(NodeType.EVENT), key=lambda node: node.node_id)
    topic_nodes = sorted(output.iter_nodes(NodeType.TOPIC), key=lambda node: node.node_id)
    if args.limit_events:
        event_nodes = event_nodes[: args.limit_events]
    if args.limit_topics:
        topic_nodes = topic_nodes[: args.limit_topics]

    if args.dry_run:
        print_preview(output, event_nodes[:2], topic_nodes[:2], args)
        return

    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    started = time.perf_counter()
    processed = 0

    for node in event_nodes:
        key = cache_key("event", node.node_id, args.model)
        record = cache.get(key)
        if record is None:
            facts = event_fact_payload(output, node, max_facts=args.max_facts_per_event)
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
        apply_event_annotation(output.nodes[node.node_id], record, args)
        processed += 1
        maybe_save(output, output_path, processed, args.save_every, graph_path, args, started)
        if processed % 25 == 0:
            print(f"annotated events/topics={processed} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    for node in topic_nodes:
        key = cache_key("topic", node.node_id, args.model)
        record = cache.get(key)
        if record is None:
            events = topic_event_payload(output, node, max_events=args.max_events_per_topic)
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
        apply_topic_annotation(output.nodes[node.node_id], record, args)
        processed += 1
        maybe_save(output, output_path, processed, args.save_every, graph_path, args, started)
        if processed % 25 == 0:
            print(f"annotated events/topics={processed} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    finalize_metadata(output, graph_path, args, started)
    JsonGraphStore().save(output, output_path)
    print(f"wrote {output_path}")
    print(f"cache {cache_path}")
    print(output.metadata.get("hierarchy_v3_2_llm_semantic", {}))


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
    annotation = parse_json_object(result.content)
    return {
        "node_id": node_id,
        "type": annotation_type,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "annotation": annotation,
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


def event_fact_payload(graph: MemoryGraph, event: Node, *, max_facts: int) -> list[dict]:
    rows = []
    for fact_id in list(event.metadata.get("fact_ids") or event.support_ids)[:max_facts]:
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
    for event_id in list(topic.metadata.get("event_ids") or topic.support_ids)[:max_events]:
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


def apply_event_annotation(node: Node, record: dict, args) -> None:
    annotation = dict(record.get("annotation") or {})
    metadata = dict(node.metadata)
    metadata.setdefault("template_text", node.text)
    metadata["label_type"] = "llm_semantic"
    metadata["llm_semantic"] = semantic_metadata(record, args)
    node.metadata = metadata
    node.text = event_text(annotation)
    node.source = "llm_semantic_event_summary_v1"


def apply_topic_annotation(node: Node, record: dict, args) -> None:
    annotation = dict(record.get("annotation") or {})
    metadata = dict(node.metadata)
    metadata.setdefault("template_text", node.text)
    metadata["label_type"] = "llm_semantic"
    metadata["llm_semantic"] = semantic_metadata(record, args)
    node.metadata = metadata
    node.text = topic_text(annotation)
    node.source = "llm_semantic_topic_summary_v1"


def semantic_metadata(record: dict, args) -> dict:
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
    return f"{annotation_type}|{node_id}|{model}|{PROMPT_VERSION}"


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
    args,
    started: float,
) -> None:
    if not save_every or processed % save_every != 0:
        return
    finalize_metadata(graph, graph_path, args, started)
    JsonGraphStore().save(graph, output_path)


def finalize_metadata(graph: MemoryGraph, graph_path: Path, args, started: float) -> None:
    metadata = dict(graph.metadata)
    metadata["hierarchy_v3_2_llm_semantic"] = {
        "source_graph": str(graph_path),
        "structure_fixed": True,
        "annotation_only": True,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "base_url_env": args.base_url_env,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "prompt_version": PROMPT_VERSION,
        "event_prompt": "EVENT_SYSTEM_PROMPT",
        "topic_prompt": "TOPIC_SYSTEM_PROMPT",
        "elapsed_seconds": time.perf_counter() - started,
    }
    graph.graph_id = f"{graph.graph_id}_v3_2_llm_semantic"
    graph.metadata = metadata


def print_preview(graph: MemoryGraph, events: list[Node], topics: list[Node], args) -> None:
    for event in events:
        print(f"\nEVENT {event.node_id}")
        print(event_user_prompt(event_fact_payload(graph, event, max_facts=args.max_facts_per_event)))
    for topic in topics:
        print(f"\nTOPIC {topic.node_id}")
        print(topic_user_prompt(topic_event_payload(graph, topic, max_events=args.max_events_per_topic)))


if __name__ == "__main__":
    main()
