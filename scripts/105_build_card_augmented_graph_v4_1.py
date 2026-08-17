from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import read_json, resolve_path
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument(
        "--cards",
        default="outputs/v3_9_query_cards/qwen3_card_annotated_base100_paths_v3_clean.json",
    )
    parser.add_argument(
        "--cardce-paths",
        default="outputs/eval/v3_9_cardce_guided_ctx50/cardce_guided_topk_paths.json",
    )
    parser.add_argument("--output", default="outputs/graphs/locomo_graph_v4_1_relation_card.json")
    parser.add_argument("--min-card-facts", type=int, default=2)
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    items = read_json(resolve_path(args.cards))
    cardce_scores = load_cardce_scores(resolve_path(args.cardce_paths))
    fact_to_event, event_to_episode, event_to_topic, episode_to_topic = hierarchy_maps(graph)

    stats = Counter()
    for item in items:
        qid = str(item["question_id"])
        conv_id = qid.split(":q", 1)[0]
        grouped: dict[str, list[dict]] = defaultdict(list)
        for path in item.get("paths", []):
            metadata = path.get("metadata", {})
            card_id = str(metadata.get("nary_hyperedge_id") or "")
            if card_id and is_card_fact(path):
                grouped[card_id].append(path)
        for card_id, paths in sorted(grouped.items()):
            fact_ids = unique([evidence_node_id(path) for path in paths if evidence_node_id(path)])
            if len(fact_ids) < args.min_card_facts:
                stats["skipped_small_cards"] += 1
                continue
            first = paths[0]
            metadata = first.get("metadata", {})
            node_id = relation_card_node_id(qid, card_id)
            card_text = build_card_text(metadata, paths, graph)
            roles_by_fact = roles_by_fact_from_paths(paths)
            parent_id, parent_role = choose_card_parent(paths, fact_to_event, event_to_episode, event_to_topic, episode_to_topic)
            if not parent_id or parent_id not in graph.nodes:
                stats["skipped_no_parent"] += 1
                continue
            graph.add_node(
                Node(
                    node_id=node_id,
                    type=NodeType.RELATION_CARD,
                    text=card_text,
                    source="v4_1_query_conditioned_relation_card",
                    confidence=float_meta(metadata, "nary_hyperedge_confidence", default=1.0),
                    support_ids=fact_ids,
                    metadata={
                        "hierarchy_v4_1": "relation_card",
                        "question_id": qid,
                        "conversation_id": conv_id,
                        "query_card_id": card_id,
                        "card_rank": metadata.get("v3_9_card_rank", ""),
                        "card_type": metadata.get("nary_hyperedge_type", ""),
                        "entity": metadata.get("v3_9_card_entity", ""),
                        "aspect": metadata.get("v3_9_card_aspect", ""),
                        "summary": metadata.get("v3_9_card_summary", ""),
                        "card_ce": cardce_scores.get((qid, card_id), float_meta(metadata, "v3_9_cardce_score")),
                        "parent_node_id": parent_id,
                        "parent_role": parent_role,
                        "support_fact_ids": fact_ids,
                        "roles_by_fact": {fact_id: sorted(roles) for fact_id, roles in roles_by_fact.items()},
                    },
                )
            )
            for fact_id in fact_ids:
                if fact_id not in graph.nodes:
                    continue
                roles = roles_by_fact.get(fact_id, {"evidence"})
                graph.add_edge(
                    Edge(
                        src=fact_id,
                        dst=node_id,
                        relation=RelationType.IS_SPECIFIC_OF,
                        confidence=max_role_confidence(paths, fact_id),
                        metadata={
                            "hierarchy_v4_1": "fact_card",
                            "role": ",".join(sorted(roles)),
                            "query_card_id": card_id,
                            "question_id": qid,
                        },
                    )
                )
                graph.add_edge(
                    Edge(
                        src=fact_id,
                        dst=node_id,
                        relation=RelationType.FILLS_ROLE,
                        confidence=max_role_confidence(paths, fact_id),
                        metadata={
                            "hierarchy_v4_1": "fact_card_role",
                            "role": ",".join(sorted(roles)),
                            "query_card_id": card_id,
                            "question_id": qid,
                        },
                    )
                )
                stats["fact_card_edges"] += 1
            graph.add_edge(
                Edge(
                    src=node_id,
                    dst=parent_id,
                    relation=RelationType.IS_SPECIFIC_OF,
                    confidence=float_meta(metadata, "nary_hyperedge_confidence", default=1.0),
                    metadata={
                        "hierarchy_v4_1": parent_role,
                        "query_card_id": card_id,
                        "question_id": qid,
                    },
                )
            )
            stats["cards"] += 1
            stats[parent_role] += 1

    graph.metadata["v4_1_relation_card_graph"] = {
        "source_graph": str(resolve_path(args.graph)),
        "cards": str(resolve_path(args.cards)),
        "cardce_paths": str(resolve_path(args.cardce_paths)),
        "stats": dict(stats),
    }
    graph.graph_id = f"{graph.graph_id}:v4_1_relation_card"
    output = resolve_path(args.output)
    JsonGraphStore().save(graph, output)
    print(json.dumps(graph.metadata["v4_1_relation_card_graph"], indent=2, ensure_ascii=False))
    print(f"nodes={len(graph.nodes)} edges={len(graph.edges)}")
    print(f"wrote {output}")


def load_cardce_scores(path: Path) -> dict[tuple[str, str], float]:
    scores = {}
    for item in read_json(path):
        qid = str(item["question_id"])
        for path_item in item.get("paths", []):
            metadata = path_item.get("metadata", {})
            card_id = str(metadata.get("nary_hyperedge_id") or "")
            if not card_id:
                continue
            score = float(path_item.get("scores", {}).get("v3_9_card_ce", metadata.get("v3_9_cardce_score", 0.0)) or 0.0)
            scores[(qid, card_id)] = max(scores.get((qid, card_id), float("-inf")), score)
    return scores


def hierarchy_maps(graph: MemoryGraph) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    fact_to_event = {}
    event_to_episode = {}
    event_to_topic = {}
    episode_to_topic = {}
    for edge in graph.edges:
        if edge.relation != RelationType.IS_SPECIFIC_OF:
            continue
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None:
            continue
        role = edge.metadata.get("hierarchy_v3_3") or edge.metadata.get("hierarchy_v2")
        if src.type == NodeType.FACT and dst.type == NodeType.EVENT and role == "fact_event":
            fact_to_event[edge.src] = edge.dst
        elif src.type == NodeType.EVENT and dst.type == NodeType.EVENT and role == "event_episode":
            event_to_episode[edge.src] = edge.dst
        elif src.type == NodeType.EVENT and dst.type == NodeType.TOPIC and role == "event_topic":
            event_to_topic[edge.src] = edge.dst
        elif src.type == NodeType.EVENT and dst.type == NodeType.TOPIC and role == "episode_topic":
            episode_to_topic[edge.src] = edge.dst
    return fact_to_event, event_to_episode, event_to_topic, episode_to_topic


def choose_card_parent(
    paths: list[dict],
    fact_to_event: dict[str, str],
    event_to_episode: dict[str, str],
    event_to_topic: dict[str, str],
    episode_to_topic: dict[str, str],
) -> tuple[str, str]:
    events = []
    for path in paths:
        event = str(path.get("metadata", {}).get("event_node_id") or "")
        if not event:
            event = fact_to_event.get(evidence_node_id(path), "")
        events.append(event)
    events = [event for event in events if event]
    if not events:
        topics = [str(path.get("metadata", {}).get("topic_node_id") or "") for path in paths]
        topics = [topic for topic in topics if topic]
        if topics:
            topic, _count = Counter(topics).most_common(1)[0]
            return topic, "card_topic"
        return "", ""
    event_counts = Counter(events)
    episodes = [event_to_episode[event] for event in events if event in event_to_episode]
    if episodes:
        episode, count = Counter(episodes).most_common(1)[0]
        if count >= 2 or len(set(events)) > 1:
            return episode, "card_episode"
    event, _count = event_counts.most_common(1)[0]
    return event, "card_event"


def build_card_text(metadata: dict, paths: list[dict], graph: MemoryGraph) -> str:
    parts = [
        f"type: {metadata.get('nary_hyperedge_type', '')}",
        f"entity: {metadata.get('v3_9_card_entity', '')}",
        f"aspect: {metadata.get('v3_9_card_aspect', '')}",
        f"summary: {metadata.get('v3_9_card_summary', '')}",
    ]
    role_rows = []
    for path in paths[:8]:
        fact_id = evidence_node_id(path)
        node = graph.nodes.get(fact_id)
        role = path.get("metadata", {}).get("nary_role", "evidence")
        if node is not None:
            role_rows.append(f"{role}: {node.text}")
    if role_rows:
        parts.append("roles: " + " | ".join(role_rows))
    return "\n".join(parts)


def roles_by_fact_from_paths(paths: list[dict]) -> dict[str, set[str]]:
    output = defaultdict(set)
    for path in paths:
        fact_id = evidence_node_id(path)
        if not fact_id:
            continue
        output[fact_id].add(normalize_role(path.get("metadata", {}).get("nary_role", "evidence")))
    return output


def max_role_confidence(paths: list[dict], fact_id: str) -> float:
    values = [
        float_meta(path.get("metadata", {}), "nary_role_confidence", default=1.0)
        for path in paths
        if evidence_node_id(path) == fact_id
    ]
    return max(values) if values else 1.0


def is_card_fact(path: dict) -> bool:
    metadata = path.get("metadata", {})
    return str(metadata.get("v3_9_query_card", "")).lower() == "true" or "v3_9_query_card" in str(metadata.get("nary_extractor_type", "")).lower()


def relation_card_node_id(question_id: str, card_id: str) -> str:
    safe_card = card_id.replace(":", "_")
    return f"{question_id}:relation_card:{safe_card}"


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def normalize_role(value: object) -> str:
    return str(value or "evidence").strip().lower().replace("-", "_").replace(" ", "_")


def float_meta(metadata: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(metadata.get(key, default) or default)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
