from __future__ import annotations

import argparse
from typing import Any

from common import load_config, read_json, resolve_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/longmemeval_s.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = resolve_path(args.input or config["data"]["raw_path"])
    output_path = resolve_path(args.output or config["data"]["processed_path"])
    raw = read_json(input_path)
    if not isinstance(raw, list):
        raise ValueError("LongMemEval-S expects a top-level list of evaluation instances")
    if args.limit:
        raw = raw[: args.limit]

    processed = [convert_instance(item, index) for index, item in enumerate(raw)]
    write_json(processed, output_path)
    print(f"wrote {len(processed)} LongMemEval-S instances to {output_path}")
    print(render_sanity(processed))


def convert_instance(item: dict[str, Any], index: int) -> dict[str, Any]:
    original_question_id = str(item.get("question_id") or f"longmemeval_{index + 1:04d}")
    conversation_id = f"lme_{index + 1:04d}_{safe_id(original_question_id)}"
    question_id = f"{conversation_id}:q0001"
    question_type = str(item.get("question_type") or "unknown")
    question_date = str(item.get("question_date") or "")
    answer_session_ids = [str(value) for value in item.get("answer_session_ids", [])]

    session_ids = [str(value) for value in item.get("haystack_session_ids", [])]
    dates = [str(value) for value in item.get("haystack_dates", [])]
    sessions = item.get("haystack_sessions", [])
    if not (len(session_ids) == len(dates) == len(sessions)):
        raise ValueError(
            f"mismatched LongMemEval haystack lengths for {original_question_id}: "
            f"session_ids={len(session_ids)} dates={len(dates)} sessions={len(sessions)}"
        )

    turns: list[dict[str, Any]] = []
    evidence_lookup: dict[str, dict[str, Any]] = {}
    gold_turn_ids: list[str] = []
    for session_idx, (session_id, timestamp, session) in enumerate(zip(session_ids, dates, sessions)):
        if not isinstance(session, list):
            continue
        for turn_idx, turn in enumerate(session):
            if not isinstance(turn, dict):
                continue
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            role = str(turn.get("role") or "unknown")
            turn_id = f"s{session_idx:03d}:{session_id}:t{turn_idx:03d}"
            converted = {
                "turn_id": turn_id,
                "speaker": role,
                "text": content,
                "timestamp": timestamp,
                "session_id": session_id,
                "original_question_id": original_question_id,
                "question_date": question_date,
            }
            turns.append(converted)
            evidence_lookup[turn_id] = converted
            if bool(turn.get("has_answer")):
                gold_turn_ids.append(turn_id)

    qa = [
        {
            "question_id": question_id,
            "original_question_id": original_question_id,
            "question_type": question_type,
            "question": str(item.get("question") or ""),
            "answer": str(item.get("answer") or ""),
            "question_date": question_date,
            "evidence": gold_turn_ids,
            "gold_evidence": gold_turn_ids,
            "answer_session_ids": answer_session_ids,
            "is_abstention": original_question_id.endswith("_abs"),
        }
    ]
    return {
        "conversation_id": conversation_id,
        "dataset": "longmemeval_s",
        "original_question_id": original_question_id,
        "question_type": question_type,
        "question_date": question_date,
        "turns": turns,
        "qa": qa,
        "evidence_lookup": evidence_lookup,
        "haystack_session_ids": session_ids,
        "answer_session_ids": answer_session_ids,
    }


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:80]


def render_sanity(conversations: list[dict[str, Any]]) -> str:
    total_turns = sum(len(conv.get("turns", [])) for conv in conversations)
    total_gold = sum(len(conv.get("qa", [{}])[0].get("gold_evidence", [])) for conv in conversations)
    abstentions = sum(1 for conv in conversations if conv.get("qa", [{}])[0].get("is_abstention"))
    qtypes: dict[str, int] = {}
    for conv in conversations:
        qtype = str(conv.get("question_type") or "unknown")
        qtypes[qtype] = qtypes.get(qtype, 0) + 1
    return "\n".join(
        [
            "LongMemEval-S sanity:",
            f"- instances: {len(conversations)}",
            f"- turns: {total_turns}",
            f"- gold_turn_ids: {total_gold}",
            f"- abstention_instances: {abstentions}",
            f"- question_types: {dict(sorted(qtypes.items()))}",
        ]
    )


if __name__ == "__main__":
    main()
