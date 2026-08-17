from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict

from common import resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, Node, NodeType, RelationType


CHANGE_RE = re.compile(r"\b(chang|instead|no longer|anymore|now|later|switch|cancel|reschedul|move[sd]?|decid)\w*", re.I)
PREFERENCE_RE = re.compile(r"\b(prefer|favorite|favourite|like|love|enjoy|dislike|hate|avoid|want|wish)\w*", re.I)
PLAN_RE = re.compile(
    r"\b(plan|intend|schedule|appointment|meeting|trip|travel|goal|deadline|available|busy|cannot|can't|must|need|constraint)\w*",
    re.I,
)
STATE_RE = re.compile(
    r"\b(currently|working|preparing|studying|living|feeling|recovering|looking|trying|playing|learning|building)\w*",
    re.I,
)
REASON_RE = re.compile(r"\b(because|due to|so that|therefore|reason|since)\b", re.I)
BAD_ENTITY_HINTS = {
    "",
    "A",
    "An",
    "Anything",
    "Good",
    "Great",
    "Hi",
    "How",
    "It",
    "Let",
    "No",
    "Nothing",
    "Oh",
    "Ok",
    "Okay",
    "Saturday",
    "Sunday",
    "Thanks",
    "That",
    "The",
    "This",
    "Today",
    "Tomorrow",
    "What",
    "Wow",
    "Yes",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_multiview_v3_5_packs.json")
    parser.add_argument("--output", default="outputs/nary_v3_6b/high_recall_candidates.json")
    parser.add_argument("--target-count", type=int, default=800)
    parser.add_argument("--max-facts", type=int, default=5)
    parser.add_argument("--event-quota", type=int, default=250)
    parser.add_argument("--adjacent-event-quota", type=int, default=200)
    parser.add_argument("--entity-aspect-quota", type=int, default=250)
    parser.add_argument("--update-context-quota", type=int, default=200)
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    index = GraphIndex.from_graph(graph)
    source_rows = {
        "event": event_candidates(graph, index, args.max_facts),
        "adjacent_event": adjacent_event_candidates(graph, index, args.max_facts),
        "entity_aspect": entity_aspect_candidates(graph, index, args.max_facts),
        "update_context": update_context_candidates(graph, index, args.max_facts),
    }
    quotas = {
        "event": args.event_quota,
        "adjacent_event": args.adjacent_event_quota,
        "entity_aspect": args.entity_aspect_quota,
        "update_context": args.update_context_quota,
    }
    selected = []
    seen = set()
    for source, rows in source_rows.items():
        rows.sort(key=lambda row: (-float(row["candidate_score"]), row["candidate_id"]))
        count = 0
        for row in rows:
            key = tuple(sorted(row["fact_ids"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            count += 1
            if count >= quotas[source]:
                break
    if len(selected) < args.target_count:
        remaining = [
            row
            for rows in source_rows.values()
            for row in rows
            if tuple(sorted(row["fact_ids"])) not in seen
        ]
        remaining.sort(key=lambda row: (-float(row["candidate_score"]), row["candidate_id"]))
        for row in remaining:
            key = tuple(sorted(row["fact_ids"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= args.target_count:
                break
    selected = selected[: args.target_count]
    payload = {
        "graph": str(resolve_path(args.graph)),
        "method": "V3.6-B high-recall typed n-ary candidate generation",
        "uses_gold": False,
        "target_count": args.target_count,
        "source_counts_available": {source: len(rows) for source, rows in source_rows.items()},
        "source_counts_selected": {
            source: sum(row["candidate_source"] == source for row in selected)
            for source in source_rows
        },
        "hint_counts": hint_counts(selected),
        "candidates": selected,
    }
    output = resolve_path(args.output)
    write_json(payload, output)
    print(json.dumps({key: payload[key] for key in ["source_counts_available", "source_counts_selected", "hint_counts"]}, indent=2))
    print(f"total={len(selected)}")
    print(f"wrote {output}")


class GraphIndex:
    def __init__(self) -> None:
        self.fact_event: dict[str, str] = {}
        self.event_facts: dict[str, list[str]] = defaultdict(list)
        self.event_episode: dict[str, str] = {}
        self.episode_events: dict[str, list[str]] = defaultdict(list)
        self.fact_states: dict[str, list[str]] = defaultdict(list)
        self.state_facts: dict[str, list[str]] = defaultdict(list)
        self.fact_relations: dict[str, list[tuple[str, RelationType, float]]] = defaultdict(list)

    @classmethod
    def from_graph(cls, graph: MemoryGraph) -> "GraphIndex":
        index = cls()
        for edge in graph.edges:
            src = graph.nodes.get(edge.src)
            dst = graph.nodes.get(edge.dst)
            if src is None or dst is None:
                continue
            role = edge.metadata.get("hierarchy_v3_3") or edge.metadata.get("hierarchy_v3")
            if src.type == NodeType.FACT and dst.type == NodeType.EVENT and role == "fact_event":
                index.fact_event[src.node_id] = dst.node_id
                index.event_facts[dst.node_id].append(src.node_id)
            elif src.type == NodeType.EVENT and dst.type == NodeType.EVENT and edge.metadata.get("hierarchy_v3_3") == "event_episode":
                index.event_episode[src.node_id] = dst.node_id
                index.episode_events[dst.node_id].append(src.node_id)
            elif src.type == NodeType.FACT and dst.type == NodeType.ENTITY_STATE and edge.metadata.get("hierarchy_v3_5_entity") == "fact_entity_state":
                index.fact_states[src.node_id].append(dst.node_id)
                index.state_facts[dst.node_id].append(src.node_id)
            if edge.relation in {RelationType.UPDATES, RelationType.CONFLICTS_WITH}:
                if src.type == NodeType.FACT and dst.type == NodeType.FACT:
                    index.fact_relations[src.node_id].append((dst.node_id, edge.relation, float(edge.confidence)))
                    index.fact_relations[dst.node_id].append((src.node_id, edge.relation, float(edge.confidence)))
        return index


def event_candidates(graph: MemoryGraph, index: GraphIndex, max_facts: int) -> list[dict]:
    output = []
    for event_id, fact_ids in index.event_facts.items():
        facts = unique_observation_facts(graph, fact_ids)
        if len(facts) < 2:
            continue
        event = graph.nodes[event_id]
        selected = select_diverse_facts(graph, facts, max_facts)
        hints = relation_hints(graph, selected)
        score = 0.45 * float(event.confidence) + 0.25 * float(event.metadata.get("coherence", 0.0)) + 0.20 * signal_density(graph, selected) + 0.10 * min(len(selected) / 3.0, 1.0)
        output.append(candidate_record(graph, index, "event", event_id, selected, score, hints))
    return output


def adjacent_event_candidates(graph: MemoryGraph, index: GraphIndex, max_facts: int) -> list[dict]:
    output = []
    for episode_id, event_ids in index.episode_events.items():
        ordered = sorted(event_ids)
        for left_id, right_id in zip(ordered, ordered[1:]):
            facts = unique_observation_facts(graph, index.event_facts.get(left_id, []) + index.event_facts.get(right_id, []))
            if len(facts) < 2:
                continue
            selected = select_diverse_facts(graph, facts, max_facts)
            hints = relation_hints(graph, selected)
            if "change" not in hints and "plan_constraint" not in hints:
                hints = dedupe([*hints, "change", "plan_constraint"])
            episode = graph.nodes[episode_id]
            score = 0.40 * float(episode.confidence) + 0.25 * signal_density(graph, selected) + 0.20 * time_diversity(graph, selected) + 0.15 * min(len(selected) / 3.0, 1.0)
            output.append(candidate_record(graph, index, "adjacent_event", episode_id, selected, score, hints))
    return output


def entity_aspect_candidates(graph: MemoryGraph, index: GraphIndex, max_facts: int) -> list[dict]:
    output = []
    for state_id, fact_ids in index.state_facts.items():
        state = graph.nodes.get(state_id)
        if state is None or str(state.metadata.get("entity", "")) in BAD_ENTITY_HINTS:
            continue
        facts = unique_observation_facts(graph, fact_ids)
        if len(facts) < 2:
            continue
        selected = select_diverse_facts(graph, facts, max_facts)
        hints = relation_hints(graph, selected)
        score = 0.40 * float(state.confidence) + 0.30 * float(state.metadata.get("coherence", 0.0)) + 0.20 * signal_density(graph, selected) + 0.10 * min(len(selected) / 3.0, 1.0)
        output.append(candidate_record(graph, index, "entity_aspect", state_id, selected, score, hints))
    return output


def update_context_candidates(graph: MemoryGraph, index: GraphIndex, max_facts: int) -> list[dict]:
    output = []
    seen = set()
    for fact_id, relations in index.fact_relations.items():
        for other_id, relation, confidence in relations:
            pair = tuple(sorted([fact_id, other_id]))
            if pair in seen:
                continue
            seen.add(pair)
            pool = [fact_id, other_id]
            for center_id in pair:
                event_id = index.fact_event.get(center_id, "")
                pool.extend(index.event_facts.get(event_id, []))
                for state_id in index.fact_states.get(center_id, []):
                    pool.extend(index.state_facts.get(state_id, []))
            facts = unique_observation_facts(graph, pool)
            if len(facts) < 2:
                continue
            selected = select_relation_centered_facts(graph, pair, facts, max_facts)
            hints = dedupe(["change", *relation_hints(graph, selected)])
            score = 0.45 * confidence + 0.25 * signal_density(graph, selected) + 0.15 * time_diversity(graph, selected) + 0.15 * min(len(selected) / 3.0, 1.0)
            output.append(candidate_record(graph, index, "update_context", relation.value, selected, score, hints))
    return output


def candidate_record(
    graph: MemoryGraph,
    index: GraphIndex,
    source: str,
    anchor_id: str,
    fact_ids: list[str],
    score: float,
    hints: list[str],
) -> dict:
    state_ids = dedupe(
        state_id
        for fact_id in fact_ids
        for state_id in index.fact_states.get(fact_id, [])
    )
    states = [graph.nodes[state_id] for state_id in state_ids if state_id in graph.nodes]
    entity_hints = dedupe(str(state.metadata.get("entity", "")) for state in states if state.metadata.get("entity"))
    aspect_keywords = dedupe(
        keyword
        for state in states
        for keyword in state.metadata.get("aspect_keywords", [])
    )[:12]
    facts = []
    raw_ids = []
    for fact_id in fact_ids:
        fact = graph.nodes[fact_id]
        supports = [str(item) for item in fact.metadata.get("support_raw_ids") or fact.support_ids]
        raw_ids.extend(supports)
        facts.append(
            {
                "fact_id": fact_id,
                "text": fact.text,
                "time": fact.time or "",
                "status": fact.status.value,
                "support_raw_ids": supports,
                "support_texts": list(fact.metadata.get("support_texts", [])),
            }
        )
    digest = hashlib.sha1("|".join(sorted(fact_ids)).encode("utf-8")).hexdigest()[:12]
    conv_id = conversation_id(fact_ids[0])
    return {
        "candidate_id": f"{conv_id}:nary_candidate_v3_6b:{source}:{digest}",
        "conversation_id": conv_id,
        "candidate_source": source,
        "candidate_score": round(float(score), 6),
        "anchor_id": anchor_id,
        "relation_hints": hints,
        "entity_hints": entity_hints[:4],
        "aspect_keywords": aspect_keywords,
        "fact_ids": fact_ids,
        "support_raw_ids": dedupe(raw_ids),
        "facts": facts,
    }


def relation_hints(graph: MemoryGraph, fact_ids: list[str]) -> list[str]:
    text = " ".join(graph.nodes[fact_id].text for fact_id in fact_ids)
    hints = []
    if CHANGE_RE.search(text):
        hints.append("change")
    if PREFERENCE_RE.search(text):
        hints.append("preference")
    if PLAN_RE.search(text):
        hints.append("plan_constraint")
    if STATE_RE.search(text) or not hints:
        hints.append("state")
    return hints


def signal_density(graph: MemoryGraph, fact_ids: list[str]) -> float:
    signaled = 0
    for fact_id in fact_ids:
        text = graph.nodes[fact_id].text
        signaled += int(any(pattern.search(text) for pattern in [CHANGE_RE, PREFERENCE_RE, PLAN_RE, STATE_RE, REASON_RE]))
    return signaled / len(fact_ids) if fact_ids else 0.0


def time_diversity(graph: MemoryGraph, fact_ids: list[str]) -> float:
    times = {graph.nodes[fact_id].time for fact_id in fact_ids if graph.nodes[fact_id].time}
    return min(len(times) / 2.0, 1.0)


def select_diverse_facts(graph: MemoryGraph, fact_ids: list[str], max_facts: int) -> list[str]:
    ranked = sorted(
        fact_ids,
        key=lambda fact_id: (
            -int(any(pattern.search(graph.nodes[fact_id].text) for pattern in [CHANGE_RE, PREFERENCE_RE, PLAN_RE, STATE_RE, REASON_RE])),
            graph.nodes[fact_id].time or "",
            fact_id,
        ),
    )
    return ranked[:max_facts]


def select_relation_centered_facts(
    graph: MemoryGraph,
    pair: tuple[str, str],
    fact_ids: list[str],
    max_facts: int,
) -> list[str]:
    selected = [fact_id for fact_id in pair if fact_id in fact_ids]
    others = [fact_id for fact_id in select_diverse_facts(graph, fact_ids, max_facts) if fact_id not in selected]
    return (selected + others)[:max_facts]


def unique_observation_facts(graph: MemoryGraph, fact_ids: list[str]) -> list[str]:
    output = []
    seen_raw = set()
    seen_text = set()
    for fact_id in fact_ids:
        node = graph.nodes.get(fact_id)
        if node is None or node.type != NodeType.FACT or "observation" not in node.source:
            continue
        raw_key = tuple(sorted(str(item) for item in node.metadata.get("support_raw_ids") or node.support_ids))
        text_key = node.text.strip().lower()
        if (raw_key and raw_key in seen_raw) or text_key in seen_text:
            continue
        if raw_key:
            seen_raw.add(raw_key)
        seen_text.add(text_key)
        output.append(fact_id)
    return output


def hint_counts(rows: list[dict]) -> dict[str, int]:
    counts = defaultdict(int)
    for row in rows:
        for hint in row["relation_hints"]:
            counts[hint] += 1
    return dict(sorted(counts.items()))


def conversation_id(node_id: str) -> str:
    return str(node_id).split(":", 1)[0]


def dedupe(values) -> list[str]:
    output = []
    seen = set()
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


if __name__ == "__main__":
    main()
