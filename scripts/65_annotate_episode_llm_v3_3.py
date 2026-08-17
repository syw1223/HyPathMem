from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import resolve_path
from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, Node, NodeType


DEFAULT_GRAPH = "outputs/graphs/locomo_graph_semantic_hierarchy_v3_3_episode.json"
DEFAULT_OUTPUT = "outputs/graphs/locomo_graph_semantic_hierarchy_v3_3_episode_gpt4o_named.json"
DEFAULT_CACHE = "outputs/llm_annotations/graph_v3_3_episode_gpt4o_annotations.jsonl"
PROMPT_VERSION = "graph_v3_3_episode_semantic_json_v1"


EPISODE_SYSTEM_PROMPT = """You are a memory compression module for a hierarchical retrieval system.

You are given multiple EVENT summaries from the same higher-level TOPIC.

Your task:
- Name ONE coherent episode/subtopic covered by these events
- Summarize the common local situation or activity
- Do NOT invent new facts
- Keep abstraction minimal but meaningful

Return STRICT JSON only:
{
  "episode_title": "",
  "episode_summary": "",
  "main_entities": [],
  "main_actions": [],
  "time_hint": ""
}

Rules:
- episode_title: <= 10 words
- episode_summary: 1 sentence only
- Only use information present in EVENTS
- No hallucination
- If time is absent, set time_hint to an empty string
"""


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
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--max-events-per-episode", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
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

    episodes = episode_nodes(output)
    if args.limit:
        episodes = episodes[: args.limit]

    if args.dry_run:
        print_preview(output, episodes[:3], args)
        return

    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    started = time.perf_counter()
    processed = 0
    for node in episodes:
        key = cache_key(node.node_id, args.model)
        record = cache.get(key)
        if record is None:
            events = episode_event_payload(output, node, max_events=args.max_events_per_episode)
            record = call_json_annotation(
                client,
                model=args.model,
                user_prompt=episode_user_prompt(events),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                node_id=node.node_id,
            )
            append_cache(cache_path, key, record)
            cache[key] = record
        apply_episode_annotation(output.nodes[node.node_id], record, args)
        processed += 1
        maybe_save(output, output_path, processed, args.save_every, graph_path, args, started)
        if processed % 25 == 0:
            print(f"annotated episodes={processed}/{len(episodes)} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    finalize_metadata(output, graph_path, args, started, len(episodes))
    JsonGraphStore().save(output, output_path)
    print(f"wrote {output_path}")
    print(f"cache {cache_path}")
    print(output.metadata.get("hierarchy_v3_3_episode_llm_semantic", {}))


def episode_nodes(graph: MemoryGraph) -> list[Node]:
    return sorted(
        [
            node
            for node in graph.iter_nodes(NodeType.EVENT)
            if node.metadata.get("hierarchy_v3_3") == "episode"
        ],
        key=lambda node: node.node_id,
    )


def episode_event_payload(graph: MemoryGraph, episode: Node, *, max_events: int) -> list[dict]:
    rows = []
    for event_id in list(episode.support_ids)[:max_events]:
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
                "event_title": str(annotation.get("event_title") or ""),
                "event_summary": str(annotation.get("event_summary") or event.text),
                "key_entities": annotation.get("key_entities") or event.metadata.get("entities", []),
                "key_actions": annotation.get("key_actions") or event.metadata.get("keywords", []),
                "time_hint": str(annotation.get("time_hint") or ""),
            }
        )
    return rows


def episode_user_prompt(events: list[dict]) -> str:
    return "EVENTS:\n" + json.dumps(events, ensure_ascii=False, indent=2)


def call_json_annotation(
    client: OpenAICompatibleChatClient,
    *,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    node_id: str,
) -> dict:
    result = client.chat_completion_with_metadata(
        model=model,
        messages=[
            ChatMessage(role="system", content=EPISODE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    annotation = parse_json_object(result.content)
    return {
        "node_id": node_id,
        "type": "episode",
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


def apply_episode_annotation(node: Node, record: dict, args) -> None:
    annotation = dict(record.get("annotation") or {})
    metadata = dict(node.metadata)
    metadata.setdefault("template_text", node.text)
    metadata["label_type"] = "llm_semantic"
    metadata["llm_semantic"] = semantic_metadata(record, args)
    node.metadata = metadata
    node.text = episode_text(annotation)
    node.source = "llm_semantic_episode_summary_v1"


def episode_text(annotation: dict) -> str:
    title = str(annotation.get("episode_title") or "").strip()
    summary = str(annotation.get("episode_summary") or "").strip()
    if title and summary:
        return f"Episode: {title}: {summary}"
    return f"Episode: {summary or title or 'Episode summary unavailable'}"


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


def cache_key(node_id: str, model: str) -> str:
    return f"episode|{node_id}|{model}|{PROMPT_VERSION}"


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
    finalize_metadata(graph, graph_path, args, started, processed)
    JsonGraphStore().save(graph, output_path)


def finalize_metadata(graph: MemoryGraph, graph_path: Path, args, started: float, episode_count: int) -> None:
    metadata = dict(graph.metadata)
    metadata["hierarchy_v3_3_episode_llm_semantic"] = {
        "source_graph": str(graph_path),
        "structure_fixed": True,
        "annotation_only": True,
        "episode_count": episode_count,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "base_url_env": args.base_url_env,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "prompt_version": PROMPT_VERSION,
        "episode_prompt": "EPISODE_SYSTEM_PROMPT",
        "elapsed_seconds": time.perf_counter() - started,
    }
    graph.graph_id = f"{graph.graph_id}_v3_3_episode_llm_semantic"
    graph.metadata = metadata


def print_preview(graph: MemoryGraph, episodes: list[Node], args) -> None:
    for episode in episodes:
        print(f"\nEPISODE {episode.node_id}")
        print(episode_user_prompt(episode_event_payload(graph, episode, max_events=args.max_events_per_episode)))


if __name__ == "__main__":
    main()

