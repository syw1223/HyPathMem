from __future__ import annotations

import argparse
from typing import Any, Dict, List

from common import load_config, read_json, resolve_path, write_json


def convert_sample(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    conversation = sample.get("conversation", {})
    conversation_id = str(sample.get("sample_id") or f"conv_{index + 1:03d}")
    turns: List[Dict[str, Any]] = []
    evidence_lookup: Dict[str, Dict[str, Any]] = {}

    for session_idx in range(1, 100):
        session_key = f"session_{session_idx}"
        if session_key not in conversation:
            continue
        timestamp = conversation.get(f"{session_key}_date_time")
        for turn in conversation.get(session_key, []):
            dia_id = str(turn.get("dia_id") or f"D{session_idx}:{len(turns) + 1}")
            converted = {
                "turn_id": dia_id,
                "speaker": turn.get("speaker", ""),
                "text": turn.get("text", ""),
                "timestamp": timestamp,
                "session_id": session_key,
            }
            turns.append(converted)
            evidence_lookup[dia_id] = converted

    qa = []
    for q_idx, item in enumerate(sample.get("qa", [])):
        qa.append(
            {
                "question_id": f"{conversation_id}:q{q_idx + 1:04d}",
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category"),
                "evidence": item.get("evidence", []),
            }
        )

    return {
        "conversation_id": conversation_id,
        "turns": turns,
        "qa": qa,
        "evidence_lookup": evidence_lookup,
        "observation": sample.get("observation", {}),
        "event_summary": sample.get("event_summary", {}),
        "session_summary": sample.get("session_summary", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = resolve_path(args.input or config["data"]["raw_path"])
    output_path = resolve_path(args.output or config["data"]["processed_path"])
    raw = read_json(input_path)
    if not isinstance(raw, list):
        raise ValueError("LoCoMo MVP expects a top-level list of conversation samples")
    processed = [convert_sample(sample, idx) for idx, sample in enumerate(raw)]
    write_json(processed, output_path)
    print(f"wrote {len(processed)} conversations to {output_path}")


if __name__ == "__main__":
    main()
