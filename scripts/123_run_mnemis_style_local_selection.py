from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import resolve_path, write_json
from hytopomem.eval.retrieval_metrics import normalize_evidence_id


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mnemis-style local hierarchy selection over existing HyTopoMem candidates. "
            "This is an adapter/diagnostic, not a Graphiti rebuild."
        )
    )
    parser.add_argument(
        "--candidates",
        default="outputs/longmemeval_s/v3_10_fine/cards/qwen3_card_guided_expand_top50summary120.json",
    )
    parser.add_argument("--graph", default="outputs/longmemeval_s/v3_10_fine/graph_sentence_semantic_episode_qwen_pathnodes_top50_v3_10.json")
    parser.add_argument("--output-dir", default="outputs/eval/longmemeval_v3_10_mnemis_style_local_selection")
    parser.add_argument("--candidate-topn", type=int, default=120)
    parser.add_argument("--topic-topk", type=int, default=4)
    parser.add_argument("--episode-topk", type=int, default=6)
    parser.add_argument("--event-topk", type=int, default=10)
    parser.add_argument("--fact-topk", type=int, default=20)
    parser.add_argument("--shortcut-threshold", type=float, default=0.42)
    parser.add_argument("--max-questions", type=int, default=0)
    args = parser.parse_args()

    candidates = read_json(resolve_path(args.candidates))
    if args.max_questions:
        candidates = candidates[: args.max_questions]
    graph_payload = read_json(resolve_path(args.graph))
    nodes = graph_payload.get("nodes", {})

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_items = []
    diagnostics = []
    for item in candidates:
        selected, diag = select_item(item, nodes, args)
        selected_items.append(selected)
        diagnostics.append(diag)

    eval_payloads = {}
    for k in [5, 20, 60]:
        eval_payloads[f"top{k}"] = evaluate_items(nodes, selected_items, k=k)
        write_json(
            {
                "method": f"mnemis_style_local_top{k}",
                "k": k,
                "summary": eval_payloads[f"top{k}"]["summary_with_gold"],
                "per_question": eval_payloads[f"top{k}"]["per_question_with_gold"],
            },
            output_dir / f"mnemis_style_local_top{k}_eval.json",
        )

    write_json(selected_items, output_dir / "mnemis_style_local_paths.json")
    write_json(diagnostics, output_dir / "mnemis_style_local_diagnostics.json")
    summary = {
        "method": "mnemis_style_local_selection_v1",
        "candidates": str(resolve_path(args.candidates)),
        "graph": str(resolve_path(args.graph)),
        "candidate_topn": args.candidate_topn,
        "topic_topk": args.topic_topk,
        "episode_topk": args.episode_topk,
        "event_topk": args.event_topk,
        "fact_topk": args.fact_topk,
        "shortcut_threshold": args.shortcut_threshold,
        "eval": {key: value["summary_with_gold"] for key, value in eval_payloads.items()},
        "diagnostics": summarize_diagnostics(diagnostics),
    }
    write_json(summary, output_dir / "summary.json")
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


def select_item(item: dict, nodes: dict[str, dict], args: argparse.Namespace) -> tuple[dict, dict]:
    paths = list(item.get("paths", []))[: args.candidate_topn]
    question = str(item.get("question") or "")
    q_terms = terms(question)
    graph = build_local_hierarchy(paths, nodes)

    selected_topics, shortcut_topics = select_nodes(
        graph["topics"].values(),
        q_terms=q_terms,
        topk=args.topic_topk,
        shortcut_threshold=args.shortcut_threshold,
    )
    topic_ids = {node["id"] for node in selected_topics}
    shortcut_topic_ids = {node["id"] for node in shortcut_topics}

    episode_pool = [
        ep
        for ep in graph["episodes"].values()
        if ep["topic_id"] in topic_ids or ep["topic_id"] in shortcut_topic_ids
    ]
    selected_episodes, shortcut_episodes = select_nodes(
        episode_pool,
        q_terms=q_terms,
        topk=args.episode_topk,
        shortcut_threshold=args.shortcut_threshold,
    )
    episode_ids = {node["id"] for node in selected_episodes}
    shortcut_episode_ids = {node["id"] for node in shortcut_episodes}

    event_pool = [
        ev
        for ev in graph["events"].values()
        if ev["episode_id"] in episode_ids
        or ev["episode_id"] in shortcut_episode_ids
        or graph["episodes"].get(ev["episode_id"], {}).get("topic_id") in shortcut_topic_ids
    ]
    selected_events, shortcut_events = select_nodes(
        event_pool,
        q_terms=q_terms,
        topk=args.event_topk,
        shortcut_threshold=args.shortcut_threshold,
    )
    event_ids = {node["id"] for node in selected_events}
    shortcut_event_ids = {node["id"] for node in shortcut_events}

    selected_paths = []
    for path in paths:
        metadata = path.get("metadata", {})
        event_id = str(metadata.get("event_node_id") or path_node_by_kind(path, ":event"))
        episode_id = str(metadata.get("episode_node_id") or "")
        topic_id = str(metadata.get("topic_node_id") or path_node_by_kind(path, ":topic"))
        keep = (
            event_id in event_ids
            or event_id in shortcut_event_ids
            or episode_id in shortcut_episode_ids
            or topic_id in shortcut_topic_ids
        )
        if keep:
            selected_paths.append(with_selection_metadata(path, "mnemis_style_local", len(selected_paths) + 1))
        if len(selected_paths) >= args.fact_topk:
            break

    if len(selected_paths) < args.fact_topk:
        seen = {evidence_node_id(path) for path in selected_paths}
        for path in sorted(paths, key=base_rank_key):
            fact_id = evidence_node_id(path)
            if fact_id in seen:
                continue
            selected_paths.append(with_selection_metadata(path, "mnemis_style_local_backfill", len(selected_paths) + 1))
            seen.add(fact_id)
            if len(selected_paths) >= args.fact_topk:
                break

    copied = dict(item)
    copied["paths"] = selected_paths
    metadata = dict(copied.get("metadata", {}))
    metadata["method"] = "mnemis_style_local_selection_v1"
    metadata["mnemis_style_topic_ids"] = sorted(topic_ids)
    metadata["mnemis_style_shortcut_topic_ids"] = sorted(shortcut_topic_ids)
    metadata["mnemis_style_episode_ids"] = sorted(episode_ids)
    metadata["mnemis_style_shortcut_episode_ids"] = sorted(shortcut_episode_ids)
    metadata["mnemis_style_event_ids"] = sorted(event_ids)
    metadata["mnemis_style_shortcut_event_ids"] = sorted(shortcut_event_ids)
    copied["metadata"] = metadata

    diag = {
        "question_id": item.get("question_id"),
        "num_candidate_paths": len(paths),
        "num_topics": len(graph["topics"]),
        "num_episodes": len(graph["episodes"]),
        "num_events": len(graph["events"]),
        "selected_topics": len(topic_ids),
        "shortcut_topics": len(shortcut_topic_ids),
        "selected_episodes": len(episode_ids),
        "shortcut_episodes": len(shortcut_episode_ids),
        "selected_events": len(event_ids),
        "shortcut_events": len(shortcut_event_ids),
        "selected_paths": len(selected_paths),
        "backfilled": sum(
            1
            for path in selected_paths
            if path.get("metadata", {}).get("mnemis_style_selection") == "mnemis_style_local_backfill"
        ),
    }
    return copied, diag


def build_local_hierarchy(paths: list[dict], nodes: dict[str, dict]) -> dict[str, dict[str, dict]]:
    topics: dict[str, dict] = {}
    episodes: dict[str, dict] = {}
    events: dict[str, dict] = {}
    for path in paths:
        metadata = path.get("metadata", {})
        fact_id = evidence_node_id(path)
        event_id = str(metadata.get("event_node_id") or path_node_by_kind(path, ":event"))
        episode_id = str(metadata.get("episode_node_id") or f"{metadata.get('topic_node_id', '')}::episode_missing")
        topic_id = str(metadata.get("topic_node_id") or path_node_by_kind(path, ":topic"))
        if topic_id:
            node = topics.setdefault(topic_id, make_selection_node(topic_id, nodes, level="topic", parent_id=""))
            update_selection_node(node, path, fact_id, nodes)
        if episode_id:
            node = episodes.setdefault(
                episode_id, make_selection_node(episode_id, nodes, level="episode", parent_id=topic_id)
            )
            node["topic_id"] = topic_id
            update_selection_node(node, path, fact_id, nodes)
        if event_id:
            node = events.setdefault(event_id, make_selection_node(event_id, nodes, level="event", parent_id=episode_id))
            node["episode_id"] = episode_id
            update_selection_node(node, path, fact_id, nodes)
    return {"topics": topics, "episodes": episodes, "events": events}


def make_selection_node(node_id: str, nodes: dict[str, dict], *, level: str, parent_id: str) -> dict:
    node = nodes.get(node_id, {})
    metadata = node.get("metadata", {}) if isinstance(node, dict) else {}
    text = str(node.get("text") or "")
    label_bits = []
    for key in ["qwen_title", "event_title", "topic_name", "episode_name", "label", "name"]:
        if metadata.get(key):
            label_bits.append(str(metadata[key]))
    label = label_bits[0] if label_bits else short_label(text, fallback=node_id)
    summary = " ".join(str(metadata.get(key) or "") for key in ["qwen_summary", "summary", "topic_summary", "event_summary"]).strip()
    return {
        "id": node_id,
        "level": level,
        "parent_id": parent_id,
        "label": label,
        "summary": summary,
        "text": text,
        "fact_ids": set(),
        "path_count": 0,
        "score_sum": 0.0,
        "best_score": float("-inf"),
        "best_rank": 10**9,
        "card_summaries": [],
        "terms": Counter(),
    }


def update_selection_node(node: dict, path: dict, fact_id: str, nodes: dict[str, dict]) -> None:
    node["path_count"] += 1
    if fact_id:
        node["fact_ids"].add(fact_id)
    score = float(path.get("score") or path.get("scores", {}).get("topology_selector") or 0.0)
    node["score_sum"] += score
    node["best_score"] = max(float(node["best_score"]), score)
    node["best_rank"] = min(int(node["best_rank"]), int(path.get("metadata", {}).get("route_rank") or 10**9))
    card_summary = path.get("metadata", {}).get("v3_9_card_summary")
    if card_summary and len(node["card_summaries"]) < 3:
        node["card_summaries"].append(str(card_summary))
    text = " ".join(
        [
            str(node.get("label") or ""),
            str(node.get("summary") or ""),
            str(node.get("text") or ""),
            str(card_summary or ""),
            str(nodes.get(fact_id, {}).get("text") or ""),
        ]
    )
    node["terms"].update(terms(text))


def select_nodes(nodes_iter, *, q_terms: set[str], topk: int, shortcut_threshold: float) -> tuple[list[dict], list[dict]]:
    scored = []
    for node in nodes_iter:
        score = selection_score(node, q_terms)
        if score > 0.0:
            scored.append((score, node))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected = [node for _, node in scored[:topk]]
    shortcuts = [node for score, node in scored[:topk] if score >= shortcut_threshold and int(node["path_count"]) <= 8]
    return selected, shortcuts


def selection_score(node: dict, q_terms: set[str]) -> float:
    if not q_terms:
        overlap = 0.0
    else:
        node_terms = set(node["terms"])
        overlap = len(q_terms & node_terms) / max(1, len(q_terms))
    density = math.log1p(int(node["path_count"])) / math.log(1 + 120)
    best_rank = int(node["best_rank"])
    rank_score = 1.0 / math.sqrt(max(best_rank, 1)) if best_rank < 10**9 else 0.0
    best_score = float(node["best_score"])
    score_prior = 1.0 / (1.0 + math.exp(-best_score)) if best_score > -50 else 0.0
    card_bonus = 0.08 if node.get("card_summaries") else 0.0
    return 0.52 * overlap + 0.18 * density + 0.18 * rank_score + 0.12 * score_prior + card_bonus


def evaluate_items(nodes: dict[str, dict], items: list[dict], *, k: int) -> dict:
    rows = [evaluate_item(nodes, item, k) for item in items]
    with_gold = [row for row in rows if row["gold_evidence_ids"]]
    return {
        "summary_all": summarize_rows(rows),
        "summary_with_gold": summarize_rows(with_gold),
        "per_question": rows,
        "per_question_with_gold": with_gold,
    }


def evaluate_item(nodes: dict[str, dict], item: dict, k: int) -> dict:
    gold = {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}
    selected = selected_node_ids(item.get("paths", []), k)
    predicted = set()
    for node_id in selected:
        predicted.update(evidence_ids_for_node(nodes, node_id))
    matched = sorted(gold & predicted)
    return {
        "question_id": item.get("question_id"),
        "gold_evidence_ids": sorted(gold),
        "selected_path_node_ids": selected,
        "matched_evidence_ids": matched,
        "hit": bool(matched),
        "recall": len(matched) / len(gold) if gold else 0.0,
        "full_cover": bool(gold) and gold.issubset(predicted),
        "tokens": count_tokens(nodes, selected),
        "path_len": average_path_len(item.get("paths", []), k),
    }


def selected_node_ids(paths: list[dict], k: int) -> list[str]:
    output = []
    seen = set()
    for path in paths[:k]:
        for node_id in path.get("node_ids", []):
            if node_id not in seen:
                seen.add(node_id)
                output.append(node_id)
    return output


def evidence_ids_for_node(nodes: dict[str, dict], node_id: str) -> set[str]:
    node = nodes.get(node_id)
    if not node:
        return set()
    node_type = str(node.get("type") or "")
    metadata = node.get("metadata", {}) or {}
    output = set()
    if node_type == "RAW":
        output.add(normalize_evidence_id(str(metadata.get("turn_id") or node_id.rsplit(":raw:", 1)[-1])))
    elif node_type == "FACT":
        if metadata.get("turn_id"):
            output.add(normalize_evidence_id(str(metadata["turn_id"])))
        for support_id in node.get("support_ids", []) or []:
            output.add(normalize_evidence_id(str(support_id)))
        for support_id in metadata.get("support_raw_ids", []) or []:
            output.add(normalize_evidence_id(str(support_id)))
    return output


def count_tokens(nodes: dict[str, dict], node_ids: list[str]) -> int:
    return sum(len(str(nodes.get(node_id, {}).get("text") or "").split()) for node_id in node_ids)


def average_path_len(paths: list[dict], k: int) -> float:
    selected = paths[:k]
    return sum(len(path.get("node_ids", [])) for path in selected) / len(selected) if selected else 0.0


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"num_questions": 0, "hit": 0.0, "recall": 0.0, "full_cover": 0.0, "avg_tokens": 0.0, "avg_path_len": 0.0}
    n = len(rows)
    return {
        "num_questions": n,
        "hit": sum(float(row["hit"]) for row in rows) / n,
        "recall": sum(float(row["recall"]) for row in rows) / n,
        "full_cover": sum(float(row["full_cover"]) for row in rows) / n,
        "avg_tokens": sum(float(row["tokens"]) for row in rows) / n,
        "avg_path_len": sum(float(row["path_len"]) for row in rows) / n,
    }


def summarize_diagnostics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [
        "num_candidate_paths",
        "num_topics",
        "num_episodes",
        "num_events",
        "selected_topics",
        "shortcut_topics",
        "selected_episodes",
        "shortcut_episodes",
        "selected_events",
        "shortcut_events",
        "selected_paths",
        "backfilled",
    ]
    return {f"avg_{key}": sum(float(row.get(key, 0)) for row in rows) / len(rows) for key in keys}


def render_markdown(summary: dict) -> str:
    lines = [
        "# Mnemis-style Local Selection Adapter",
        "",
        f"Candidates: `{summary['candidates']}`",
        f"Graph: `{summary['graph']}`",
        "",
        "| Eval | Questions | Hit | Recall | FullCover | AvgTokens | AvgPathLen |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in summary["eval"].items():
        lines.append(
            f"| {key} | {value['num_questions']} | {value['hit']:.4f} | {value['recall']:.4f} | "
            f"{value['full_cover']:.4f} | {value['avg_tokens']:.1f} | {value['avg_path_len']:.2f} |"
        )
    lines.extend(["", "## Diagnostics", ""])
    for key, value in summary["diagnostics"].items():
        lines.append(f"- `{key}`: {value:.4f}")
    return "\n".join(lines)


def terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_'-]*", str(text).lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def short_label(text: str, *, fallback: str) -> str:
    text = " ".join(str(text).split())
    if not text:
        return fallback
    return text[:120]


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def path_node_by_kind(path: dict, marker: str) -> str:
    for node_id in path.get("node_ids", []):
        if marker in str(node_id):
            return str(node_id)
    return ""


def base_rank_key(path: dict) -> tuple[float, float]:
    metadata = path.get("metadata", {})
    try:
        rank = float(metadata.get("route_rank") or metadata.get("bottom_up_rank") or 10**9)
    except (TypeError, ValueError):
        rank = 10**9
    try:
        score = float(path.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return (rank, -score)


def with_selection_metadata(path: dict, selection: str, rank: int) -> dict:
    copied = dict(path)
    metadata = dict(copied.get("metadata", {}))
    metadata["mnemis_style_selection"] = selection
    metadata["mnemis_style_rank"] = str(rank)
    copied["metadata"] = metadata
    return copied


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
