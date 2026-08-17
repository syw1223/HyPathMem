from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict

from common import resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, Node, NodeType, RelationType


CHANGE_TERMS = re.compile(r"\b(chang|instead|no longer|anymore|now|later|switch|decid|cancel|reschedul|move[sd]?)\w*", re.I)
PREFERENCE_TERMS = re.compile(
    r"\b(prefer|favorite|favourite|dislike|hate|avoid|likes|loves|enjoys|wants|cannot|can't)\w*",
    re.I,
)
STATE_TERMS = re.compile(
    r"\b(is|are|was|were|working|preparing|planning|studying|living|feeling|recovering|looking|trying|currently)\b",
    re.I,
)
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
    parser.add_argument("--output", default="outputs/nary_v3_6/nary_hyperedge_candidates.json")
    parser.add_argument("--per-type-limit", type=int, default=60)
    parser.add_argument("--max-facts", type=int, default=5)
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    state_to_facts, fact_to_states = entity_state_index(graph)
    fact_relations = relation_index(graph)
    candidates = []
    candidates.extend(change_candidates(graph, state_to_facts, fact_to_states, fact_relations, args.max_facts))
    candidates.extend(state_change_candidates(graph, state_to_facts, args.max_facts))
    candidates.extend(preference_candidates(graph, state_to_facts, args.max_facts))
    candidates.extend(state_candidates(graph, state_to_facts, args.max_facts))

    candidates = dedupe_candidates(candidates)
    selected = []
    for relation_type in ["change", "preference", "state"]:
        rows = sorted(
            [row for row in candidates if row["candidate_type"] == relation_type],
            key=lambda row: (-float(row["candidate_score"]), row["candidate_id"]),
        )[: args.per_type_limit]
        selected.extend(rows)
    payload = {
        "graph": str(resolve_path(args.graph)),
        "method": "V3.6 typed n-ary candidate generation",
        "uses_gold": False,
        "config": {
            "per_type_limit": args.per_type_limit,
            "max_facts": args.max_facts,
        },
        "counts": {
            relation_type: sum(row["candidate_type"] == relation_type for row in selected)
            for relation_type in ["change", "preference", "state"]
        },
        "candidates": selected,
    }
    output = resolve_path(args.output)
    write_json(payload, output)
    print(json.dumps(payload["counts"], indent=2))
    print(f"wrote {output}")


def change_candidates(
    graph: MemoryGraph,
    state_to_facts: dict[str, list[str]],
    fact_to_states: dict[str, list[str]],
    fact_relations: dict[str, list[tuple[str, RelationType, float]]],
    max_facts: int,
) -> list[dict]:
    output = []
    seen = set()
    for fact_id, relations in fact_relations.items():
        for other_id, relation, confidence in relations:
            key = tuple(sorted([fact_id, other_id]))
            if key in seen:
                continue
            seen.add(key)
            left = graph.nodes[fact_id]
            right = graph.nodes[other_id]
            if not is_observation_fact(left) or not is_observation_fact(right):
                continue
            if shared_raw_ids(left, right):
                continue
            if not (change_signal(left) or change_signal(right)):
                continue
            shared_states = sorted(set(fact_to_states.get(fact_id, [])) & set(fact_to_states.get(other_id, [])))
            if not shared_states:
                continue
            state_id = shared_states[0] if shared_states else ""
            state = graph.nodes.get(state_id)
            if state is None or str(state.metadata.get("entity", "")) in BAD_ENTITY_HINTS:
                continue
            pool = unique_observation_facts(graph, state_to_facts.get(state_id, []))
            ranked = [fact_id, other_id]
            for candidate_id in pool:
                if candidate_id in ranked:
                    continue
                candidate = graph.nodes.get(candidate_id)
                if candidate is not None and reason_or_time_signal(candidate.text):
                    ranked.append(candidate_id)
            fact_ids = ranked[:max_facts]
            signal = max(change_signal(graph.nodes[fact_id]), change_signal(graph.nodes[other_id]))
            score = 0.55 * confidence + 0.25 * signal + 0.20 * float(bool(shared_states))
            output.append(candidate_record(graph, "change", state_id, fact_ids, score, relation.value))
    return output


def preference_candidates(graph: MemoryGraph, state_to_facts: dict[str, list[str]], max_facts: int) -> list[dict]:
    output = []
    for state_id, fact_ids in state_to_facts.items():
        state = graph.nodes[state_id]
        if str(state.metadata.get("entity", "")) in BAD_ENTITY_HINTS:
            continue
        unique_facts = unique_observation_facts(graph, fact_ids)
        matched = [
            fact_id
            for fact_id in unique_facts
            if PREFERENCE_TERMS.search(graph.nodes[fact_id].text)
            and not conversational_preference_false_positive(graph.nodes[fact_id].text)
        ]
        if not matched:
            continue
        supporting = [fact_id for fact_id in unique_facts if fact_id not in matched]
        selected = (matched + supporting)[:max_facts]
        if len(selected) < 2:
            continue
        density = len(matched) / max(len(fact_ids), 1)
        score = 0.45 * float(state.confidence) + 0.35 * density + 0.20 * min(len(selected) / 3.0, 1.0)
        output.append(candidate_record(graph, "preference", state_id, selected, score, "preference_signal"))
    return output


def state_change_candidates(graph: MemoryGraph, state_to_facts: dict[str, list[str]], max_facts: int) -> list[dict]:
    output = []
    for state_id, fact_ids in state_to_facts.items():
        state = graph.nodes.get(state_id)
        if state is None or str(state.metadata.get("entity", "")) in BAD_ENTITY_HINTS:
            continue
        unique_facts = unique_observation_facts(graph, fact_ids)
        if len(unique_facts) < 2:
            continue
        signal_facts = [fact_id for fact_id in unique_facts if change_signal(graph.nodes[fact_id])]
        if not signal_facts:
            continue
        selected = []
        for fact_id in unique_facts:
            if fact_id not in signal_facts:
                selected.append(fact_id)
        selected.extend(signal_facts)
        selected = selected[-max_facts:]
        score = (
            0.45 * float(state.confidence)
            + 0.30 * max(change_signal(graph.nodes[fact_id]) for fact_id in signal_facts)
            + 0.25 * min(len(selected) / 3.0, 1.0)
        )
        output.append(candidate_record(graph, "change", state_id, selected, score, "state_change_signal"))
    return output


def state_candidates(graph: MemoryGraph, state_to_facts: dict[str, list[str]], max_facts: int) -> list[dict]:
    output = []
    for state_id, fact_ids in state_to_facts.items():
        state = graph.nodes[state_id]
        entity = str(state.metadata.get("entity", ""))
        if entity in BAD_ENTITY_HINTS or float(state.metadata.get("coherence", 0.0)) < 0.25:
            continue
        unique_facts = unique_observation_facts(graph, fact_ids)
        matched = [fact_id for fact_id in unique_facts if STATE_TERMS.search(graph.nodes[fact_id].text)]
        selected = (matched + [fact_id for fact_id in unique_facts if fact_id not in matched])[:max_facts]
        if len(selected) < 2:
            continue
        coherence = float(state.metadata.get("coherence", 0.0))
        score = 0.50 * float(state.confidence) + 0.30 * coherence + 0.20 * min(len(selected) / 3.0, 1.0)
        output.append(candidate_record(graph, "state", state_id, selected, score, "entity_state"))
    return output


def candidate_record(
    graph: MemoryGraph,
    candidate_type: str,
    state_id: str,
    fact_ids: list[str],
    score: float,
    signal: str,
) -> dict:
    state = graph.nodes.get(state_id)
    entity = str(state.metadata.get("entity", "")) if state else ""
    aspect_keywords = list(state.metadata.get("aspect_keywords", [])) if state else []
    facts = []
    raw_ids = []
    seen_raw = set()
    for fact_id in fact_ids:
        fact = graph.nodes[fact_id]
        fact_raw_ids = [str(item) for item in fact.metadata.get("support_raw_ids") or fact.support_ids]
        for raw_id in fact_raw_ids:
            if raw_id not in seen_raw:
                seen_raw.add(raw_id)
                raw_ids.append(raw_id)
        facts.append(
            {
                "fact_id": fact_id,
                "text": fact.text,
                "time": fact.time or "",
                "status": fact.status.value,
                "support_raw_ids": fact_raw_ids,
                "support_texts": list(fact.metadata.get("support_texts", [])),
            }
        )
    conv_id = conversation_id(fact_ids[0]) if fact_ids else conversation_id(state_id)
    suffix = re.sub(r"[^a-zA-Z0-9]+", "_", state_id.rsplit(":", 2)[-1] if state_id else signal).strip("_")
    digest = hashlib.sha1("|".join(fact_ids).encode("utf-8")).hexdigest()[:10]
    return {
        "candidate_id": f"{conv_id}:nary_candidate:{candidate_type}:{suffix}:{digest}",
        "conversation_id": conv_id,
        "candidate_type": candidate_type,
        "candidate_score": round(float(score), 6),
        "signal": signal,
        "entity_state_id": state_id,
        "entity_hint": entity,
        "aspect_keywords": aspect_keywords,
        "fact_ids": fact_ids,
        "support_raw_ids": raw_ids,
        "facts": facts,
    }


def entity_state_index(graph: MemoryGraph) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    state_to_facts: dict[str, list[str]] = defaultdict(list)
    fact_to_states: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.metadata.get("hierarchy_v3_5_entity") != "fact_entity_state":
            continue
        state_to_facts[edge.dst].append(edge.src)
        fact_to_states[edge.src].append(edge.dst)
    return (
        {key: sorted(set(values)) for key, values in state_to_facts.items()},
        {key: sorted(set(values)) for key, values in fact_to_states.items()},
    )


def relation_index(graph: MemoryGraph) -> dict[str, list[tuple[str, RelationType, float]]]:
    output: dict[str, list[tuple[str, RelationType, float]]] = defaultdict(list)
    for edge in graph.edges:
        if edge.relation not in {RelationType.UPDATES, RelationType.CONFLICTS_WITH}:
            continue
        if graph.nodes.get(edge.src, None) is None or graph.nodes.get(edge.dst, None) is None:
            continue
        if graph.nodes[edge.src].type != NodeType.FACT or graph.nodes[edge.dst].type != NodeType.FACT:
            continue
        output[edge.src].append((edge.dst, edge.relation, float(edge.confidence)))
        output[edge.dst].append((edge.src, edge.relation, float(edge.confidence)))
    return output


def change_signal(node: Node) -> float:
    return float(bool(CHANGE_TERMS.search(node.text)) or node.status.value in {"outdated", "exception"})


def reason_or_time_signal(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["because", "due to", "after", "before", "later", "then", "so that"])


def is_observation_fact(node: Node) -> bool:
    return node.type == NodeType.FACT and "observation" in node.source


def raw_id_set(node: Node) -> set[str]:
    return {str(item) for item in node.metadata.get("support_raw_ids") or node.support_ids}


def shared_raw_ids(left: Node, right: Node) -> bool:
    return bool(raw_id_set(left) & raw_id_set(right))


def unique_observation_facts(graph: MemoryGraph, fact_ids: list[str]) -> list[str]:
    output = []
    seen_raw_sets = set()
    seen_text = set()
    for fact_id in fact_ids:
        node = graph.nodes.get(fact_id)
        if node is None or not is_observation_fact(node):
            continue
        raw_key = tuple(sorted(raw_id_set(node)))
        text_key = node.text.strip().lower()
        if (raw_key and raw_key in seen_raw_sets) or text_key in seen_text:
            continue
        if raw_key:
            seen_raw_sets.add(raw_key)
        seen_text.add(text_key)
        output.append(fact_id)
    return output


def conversational_preference_false_positive(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in [
            "let me know",
            "let you know",
            "let him know",
            "let her know",
            "need any help",
            "would love to see",
            "would love to check",
        ]
    )


def dedupe_candidates(candidates: list[dict]) -> list[dict]:
    best = {}
    for row in candidates:
        key = (row["candidate_type"], tuple(sorted(row["fact_ids"])))
        previous = best.get(key)
        if previous is None or float(row["candidate_score"]) > float(previous["candidate_score"]):
            best[key] = row
    return list(best.values())


def conversation_id(node_id: str) -> str:
    return str(node_id).split(":", 1)[0]


if __name__ == "__main__":
    main()
