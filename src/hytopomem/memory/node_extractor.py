from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from hytopomem.memory.schema import Node, NodeStatus, NodeType


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]+")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "but",
    "for",
    "from",
    "has",
    "have",
    "into",
    "that",
    "the",
    "their",
    "then",
    "there",
    "they",
    "this",
    "was",
    "were",
    "with",
    "would",
}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def content_terms(text: str) -> List[str]:
    return [
        token.lower()
        for token in _WORD_RE.findall(text)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    ]


def anchor_text_from_fact(text: str, max_terms: int = 6) -> str:
    terms = content_terms(text)
    if not terms:
        return "General memory"
    deduped = []
    for term in terms:
        if term not in deduped:
            deduped.append(term)
    return "Topic: " + ", ".join(deduped[:max_terms])


@dataclass(frozen=True)
class ExtractedNodes:
    raw_nodes: List[Node]
    fact_nodes: List[Node]
    anchor_nodes: List[Node]


class RuleBasedNodeExtractor:
    """Offline extractor used before wiring an LLM prompt into the same API."""

    def extract(self, conversation_id: str, turns: Sequence[Dict[str, str]]) -> ExtractedNodes:
        raw_nodes: List[Node] = []
        fact_nodes: List[Node] = []
        anchors_by_text: Dict[str, Node] = {}

        for idx, turn in enumerate(turns):
            turn_id = str(turn.get("turn_id") or f"t{idx + 1:04d}")
            speaker = str(turn.get("speaker") or "unknown")
            text = normalize_text(str(turn.get("text") or ""))
            if not text:
                continue

            timestamp = turn.get("timestamp")
            raw_id = f"{conversation_id}:raw:{turn_id}"
            fact_id = f"{conversation_id}:fact:{turn_id}"
            raw_nodes.append(
                Node(
                    node_id=raw_id,
                    type=NodeType.RAW,
                    text=f"{speaker}: {text}",
                    time=timestamp,
                    source="raw_dialogue",
                    confidence=1.0,
                    metadata={"turn_id": turn_id, "speaker": speaker},
                )
            )

            fact_text = f"{speaker} said: {text}"
            status = self._infer_status(fact_text)
            fact_nodes.append(
                Node(
                    node_id=fact_id,
                    type=NodeType.FACT,
                    text=fact_text,
                    time=timestamp,
                    source="rule_extracted",
                    status=status,
                    confidence=0.72,
                    support_ids=[raw_id],
                    metadata={
                        "turn_id": turn_id,
                        "speaker": speaker,
                        "support_raw_ids": [raw_id],
                        "support_timestamps": [timestamp] if timestamp else [],
                        "support_texts": [f"{speaker}: {text}"],
                    },
                )
            )

            anchor_text = anchor_text_from_fact(text)
            anchor_key = anchor_text.lower()
            if anchor_key not in anchors_by_text:
                anchors_by_text[anchor_key] = Node(
                    node_id=f"{conversation_id}:anchor:{len(anchors_by_text) + 1:04d}",
                    type=NodeType.ANCHOR,
                    text=anchor_text,
                    time=timestamp,
                    source="rule_inferred_anchor",
                    confidence=0.58,
                    support_ids=[fact_id],
                )
            else:
                anchors_by_text[anchor_key].support_ids.append(fact_id)

        return ExtractedNodes(raw_nodes, fact_nodes, list(anchors_by_text.values()))

    def _infer_status(self, text: str) -> NodeStatus:
        lowered = text.lower()
        if any(token in lowered for token in ["not anymore", "no longer", "changed", "instead"]):
            return NodeStatus.OUTDATED
        if any(token in lowered for token in ["except", "unless"]):
            return NodeStatus.EXCEPTION
        if any(token in lowered for token in ["maybe", "possibly", "unclear", "not sure"]):
            return NodeStatus.DISPUTED
        return NodeStatus.ACTIVE


def nodes_from_observations(
    conversation_id: str,
    observations: Dict[str, object],
    evidence_lookup: Dict[str, Dict[str, Any]] | None = None,
) -> List[Node]:
    nodes: List[Node] = []
    counter = 0
    evidence_lookup = evidence_lookup or {}
    for session_key, by_speaker in observations.items():
        if not isinstance(by_speaker, dict):
            continue
        for speaker, items in by_speaker.items():
            if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
                continue
            for item in items:
                if not isinstance(item, list) or not item:
                    continue
                counter += 1
                text = normalize_text(str(item[0]))
                support_turn_ids = parse_support_turn_ids(item[1] if len(item) > 1 else "")
                raw_ids = [f"{conversation_id}:raw:{turn_id}" for turn_id in support_turn_ids]
                support_turns = [evidence_lookup.get(turn_id, {}) for turn_id in support_turn_ids]
                timestamps = [turn.get("timestamp") for turn in support_turns if turn.get("timestamp")]
                support_texts = []
                for turn in support_turns:
                    support_text = normalize_text(str(turn.get("text", "")))
                    if support_text:
                        support_texts.append(f"{turn.get('speaker', speaker)}: {support_text}")
                nodes.append(
                    Node(
                        node_id=f"{conversation_id}:fact:obs:{counter:04d}",
                        type=NodeType.FACT,
                        text=text,
                        time=timestamps[0] if timestamps else None,
                        source="locomo_observation",
                        confidence=0.86,
                        support_ids=raw_ids,
                        metadata={
                            "session": session_key,
                            "speaker": speaker,
                            "support_turn_ids": support_turn_ids,
                            "support_raw_ids": raw_ids,
                            "support_timestamps": timestamps,
                            "support_texts": support_texts,
                        },
                    )
                )
    return nodes


def parse_support_turn_ids(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    if "," in text:
        return [part.strip().strip("'\"") for part in text.split(",") if part.strip().strip("'\"")]
    return [text.strip("'\"")]
