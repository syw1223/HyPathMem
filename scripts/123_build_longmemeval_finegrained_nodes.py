from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import load_config, read_json, resolve_path, write_json
from hytopomem.memory.node_extractor import anchor_text_from_fact, normalize_text
from hytopomem.memory.schema import Node, NodeStatus, NodeType


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
LOW_INFO_RE = re.compile(
    r"^(?:thanks|thank you|ok|okay|sure|yes|no|yeah|yep|cool|great|nice|awesome|haha)[!,. ]*$",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build LongMemEval-S fine-grained nodes. RAW remains one turn; FACT is split "
            "into sentence-level atomic claims with full RAW provenance."
        )
    )
    parser.add_argument("--config", default="configs/longmemeval_s.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default="outputs/longmemeval_s/v3_10_fine/nodes_sentence_facts.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-sentence-chars", type=int, default=420)
    parser.add_argument("--min-sentence-chars", type=int, default=6)
    parser.add_argument("--max-sentences-per-turn", type=int, default=6)
    parser.add_argument("--include-low-info", action="store_true")
    parser.add_argument("--anchors", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = resolve_path(args.input or config["data"]["processed_path"])
    output_path = resolve_path(args.output)
    conversations = read_json(input_path)
    if args.limit:
        conversations = conversations[: args.limit]

    nodes: list[Node] = []
    sentence_counts: list[int] = []
    skipped_low_info = 0
    skipped_short = 0
    for conversation in conversations:
        conversation_id = str(conversation["conversation_id"])
        anchors_by_text: dict[str, Node] = {}
        for turn_index, turn in enumerate(conversation.get("turns", [])):
            turn_id = str(turn.get("turn_id") or f"t{turn_index + 1:04d}")
            speaker = str(turn.get("speaker") or "unknown")
            text = normalize_text(str(turn.get("text") or ""))
            if not text:
                continue
            timestamp = turn.get("timestamp")
            session_id = turn.get("session_id") or session_id_from_turn_id(turn_id)
            raw_id = f"{conversation_id}:raw:{turn_id}"
            raw_text = f"{speaker}: {text}"
            nodes.append(
                Node(
                    node_id=raw_id,
                    type=NodeType.RAW,
                    text=raw_text,
                    time=timestamp,
                    source="raw_dialogue",
                    confidence=1.0,
                    metadata={
                        "turn_id": turn_id,
                        "speaker": speaker,
                        "session_id": session_id,
                        "question_date": turn.get("question_date") or conversation.get("question_date", ""),
                    },
                )
            )

            sentences = split_sentences(text, max_chars=args.max_sentence_chars)
            kept_sentences = []
            for sentence in sentences:
                sentence = normalize_text(sentence)
                if len(sentence) < args.min_sentence_chars:
                    skipped_short += 1
                    continue
                if not args.include_low_info and LOW_INFO_RE.match(sentence):
                    skipped_low_info += 1
                    continue
                kept_sentences.append(sentence)
            if args.max_sentences_per_turn > 0:
                kept_sentences = kept_sentences[: args.max_sentences_per_turn]
            if not kept_sentences:
                kept_sentences = [text[: args.max_sentence_chars]]
            sentence_counts.append(len(kept_sentences))

            for sent_index, sentence in enumerate(kept_sentences, start=1):
                fact_id = f"{conversation_id}:fact_sent:{turn_id}:s{sent_index:03d}"
                fact_text = f"{speaker} said: {sentence}"
                nodes.append(
                    Node(
                        node_id=fact_id,
                        type=NodeType.FACT,
                        text=fact_text,
                        time=timestamp,
                        source="sentence_fact",
                        status=infer_status(fact_text),
                        confidence=0.78,
                        support_ids=[raw_id],
                        metadata={
                            "turn_id": turn_id,
                            "speaker": speaker,
                            "session_id": session_id,
                            "sentence_index": sent_index,
                            "sentence_count": len(kept_sentences),
                            "sentence_text": sentence,
                            "granularity": "sentence",
                            "support_raw_ids": [raw_id],
                            "support_timestamps": [timestamp] if timestamp else [],
                            "support_texts": [raw_text],
                        },
                    )
                )
                if args.anchors:
                    anchor_text = anchor_text_from_fact(sentence)
                    anchor_key = anchor_text.lower()
                    if anchor_key not in anchors_by_text:
                        anchors_by_text[anchor_key] = Node(
                            node_id=f"{conversation_id}:anchor_sent:{len(anchors_by_text) + 1:04d}",
                            type=NodeType.ANCHOR,
                            text=anchor_text,
                            time=timestamp,
                            source="sentence_fact_anchor",
                            confidence=0.55,
                            support_ids=[fact_id],
                        )
                    else:
                        anchors_by_text[anchor_key].support_ids.append(fact_id)
        nodes.extend(anchors_by_text.values())

    write_json([node.model_dump(mode="json") for node in nodes], output_path)
    counts = Counter(node.type.value for node in nodes)
    diagnostics = {
        "input": str(input_path),
        "output": str(output_path),
        "conversations": len(conversations),
        "node_counts": dict(counts),
        "avg_sentence_facts_per_turn": sum(sentence_counts) / max(len(sentence_counts), 1),
        "max_sentence_facts_per_turn": max(sentence_counts, default=0),
        "skipped_low_info": skipped_low_info,
        "skipped_short": skipped_short,
        "design": "RAW turn -> sentence-level FACT with support_raw_ids to original RAW turn",
    }
    diag_path = output_path.with_suffix(".diagnostics.json")
    write_json(diagnostics, diag_path)
    print(f"wrote {len(nodes)} nodes to {output_path}")
    print(f"wrote {diag_path}")
    print(diagnostics)


def split_sentences(text: str, *, max_chars: int) -> list[str]:
    pieces: list[str] = []
    for part in SENTENCE_SPLIT_RE.split(text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            pieces.append(part)
            continue
        pieces.extend(split_long_sentence(part, max_chars=max_chars))
    return pieces


def split_long_sentence(text: str, *, max_chars: int) -> list[str]:
    chunks = []
    current = []
    current_len = 0
    for token in text.split():
        token_len = len(token) + 1
        if current and current_len + token_len > max_chars:
            chunks.append(" ".join(current))
            current = [token]
            current_len = token_len
        else:
            current.append(token)
            current_len += token_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def infer_status(text: str) -> NodeStatus:
    lowered = text.lower()
    if any(token in lowered for token in ["not anymore", "no longer", "changed", "instead", "used to"]):
        return NodeStatus.OUTDATED
    if any(token in lowered for token in ["except", "unless"]):
        return NodeStatus.EXCEPTION
    if any(token in lowered for token in ["maybe", "possibly", "unclear", "not sure"]):
        return NodeStatus.DISPUTED
    return NodeStatus.ACTIVE


def session_id_from_turn_id(turn_id: str) -> str:
    if ":t" in turn_id:
        return turn_id.split(":t", 1)[0]
    return turn_id.rsplit(":t", 1)[0]


if __name__ == "__main__":
    main()
