#!/usr/bin/env python3
"""Build a leakage-safe temporal oracle diagnostic set.

This script uses LongMemEval gold turn IDs only to select and bind evidence. It
does not derive an answer or temporal operands from the gold answer. The output
is diagnostic-only and must not be used as a production retrieval result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


DEFAULT_SELECTED = {
    "full_cover_wrong": [
        "lme_0326_982b5123:q0001",
        "lme_0345_gpt4_cd90e484:q0001",
        "lme_0349_b29f3365:q0001",
        "lme_0357_gpt4_5438fa52:q0001",
        "lme_0366_gpt4_fe651585_abs:q0001",
    ],
    "d0_correct_temporal_regression": [
        "lme_0243_9a707b81:q0001",
        "lme_0267_gpt4_ec93e27f:q0001",
        "lme_0296_gpt4_4929293b:q0001",
        "lme_0360_8c18457d:q0001",
        "lme_0365_gpt4_c27434e8_abs:q0001",
    ],
    "stable_correct": [
        "lme_0235_gpt4_f49edff3:q0001",
        "lme_0245_gpt4_e072b769:q0001",
        "lme_0297_gpt4_468eb064:q0001",
        "lme_0320_gpt4_6ed717ea:q0001",
        "lme_0344_gpt4_d31cdae3:q0001",
    ],
}


def stream_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON array without loading the 618 MB file."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    with path.open(encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk and position >= len(buffer):
                return
            buffer = buffer[position:] + chunk
            position = 0
            if not started:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position >= len(buffer):
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                position += 1
                started = True
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                position = end
                yield value
            if not chunk:
                raise ValueError(f"Incomplete JSON value at end of {path}")


def canonical_turn_id(value: str) -> str:
    """Map graph-prefixed raw IDs back to LongMemEval's session/turn ID."""
    value = str(value or "")
    marker = ":raw:"
    if marker in value:
        return value.split(marker, 1)[1]
    return value


def load_rows(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list field {key!r} in {path}")
    return rows


def evidence_unit_turn_ids(unit: dict[str, Any]) -> set[str]:
    ids = {canonical_turn_id(value) for value in unit.get("raw_message_ids", [])}
    for quote in unit.get("raw_quotes", []):
        ids.add(canonical_turn_id(quote.get("message_id", "")))
    return {value for value in ids if value}


def bind_gold_units(
    units: list[dict[str, Any]], gold_ids: list[str]
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    gold_set = set(gold_ids)
    matches: dict[str, list[str]] = {gold_id: [] for gold_id in gold_ids}
    bound: list[dict[str, Any]] = []
    for unit in units:
        overlap = sorted(evidence_unit_turn_ids(unit) & gold_set)
        if not overlap:
            continue
        copy = dict(unit)
        copy["oracle_gold_turn_ids"] = overlap
        bound.append(copy)
        for gold_id in overlap:
            matches[gold_id].append(str(unit.get("unit_id", "")))
    return bound, matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("data/longmemeval/processed/longmemeval_s_mvp.json"),
    )
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_r_v0_1/top50_structured_raw_packs.json"),
    )
    parser.add_argument(
        "--d0",
        type=Path,
        default=Path(
            "outputs/qa/longmemeval_v3_10_cardquota_light_top50_c1_window_chunks_80k_"
            "gpt41mini_judge_gpt4omini.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_temporal_v0_2_oracle"),
    )
    args = parser.parse_args()

    selected_to_cohort = {
        question_id: cohort
        for cohort, question_ids in DEFAULT_SELECTED.items()
        for question_id in question_ids
    }
    selected_ids = set(selected_to_cohort)

    conversations: dict[str, dict[str, Any]] = {}
    for conversation in stream_json_array(args.processed):
        qa_rows = conversation.get("qa", [])
        if any(row.get("question_id") in selected_ids for row in qa_rows):
            conversations[str(conversation["conversation_id"])] = conversation
        if len(conversations) == len(selected_ids):
            break

    pack_by_id = {
        row["question_id"]: row for row in load_rows(args.packs, "rows") if row["question_id"] in selected_ids
    }
    d0_by_id = {
        row["question_id"]: row
        for row in load_rows(args.d0, "per_question")
        if row["question_id"] in selected_ids
    }

    rows: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for question_id in sorted(selected_ids):
        conversation_id = question_id.rsplit(":q", 1)[0]
        conversation = conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(f"Missing processed conversation for {question_id}")
        qa = next(row for row in conversation["qa"] if row["question_id"] == question_id)
        pack_row = pack_by_id[question_id]
        d0 = d0_by_id[question_id]
        gold_ids = list(qa.get("gold_evidence", qa.get("evidence", [])))
        lookup = conversation.get("evidence_lookup", {})
        gold_turns = [lookup[turn_id] for turn_id in gold_ids if turn_id in lookup]
        units = pack_row.get("pack", {}).get("evidence_units", [])
        bound_units, matches = bind_gold_units(units, gold_ids)
        missing = [turn_id for turn_id, unit_ids in matches.items() if not unit_ids]
        rows.append(
            {
                "question_id": question_id,
                "conversation_id": conversation_id,
                "cohort": selected_to_cohort[question_id],
                "question": qa["question"],
                "question_date": qa.get("question_date"),
                "question_type": qa.get("question_type"),
                "gold_answer": qa.get("answer"),
                "is_abstention": bool(qa.get("is_abstention", False)),
                "gold_evidence_ids": gold_ids,
                "gold_turns": gold_turns,
                "oracle_bound_d2_units": bound_units,
                "gold_to_d2_unit_ids": matches,
                "missing_gold_evidence_in_d2_top50": missing,
                "oracle_top50_full_cover": not missing,
                "frozen_d0": {
                    "prediction": d0.get("prediction"),
                    "judge_correct": d0.get("judge_correct"),
                    "retrieval_full_cover": d0.get("retrieval_full_cover"),
                    "retrieval_recall": d0.get("retrieval_recall"),
                },
            }
        )
        annotations.append(
            {
                "question_id": question_id,
                "annotation_status": "pending_manual_review",
                "operation": None,
                "operands": [],
                "anchor": None,
                "constraints": [],
                "expected_solver_output_type": None,
                "notes": "Derive only from question and gold_turns; do not use gold_answer.",
            }
        )

    full_cover = sum(bool(row["oracle_top50_full_cover"]) for row in rows)
    payload = {
        "metadata": {
            "version": "hypathmem_temporal_v0_2_oracle",
            "diagnostic_oracle_only": True,
            "gold_answer_must_not_feed_solver": True,
            "fallback": "frozen_D0",
            "cohorts": DEFAULT_SELECTED,
        },
        "summary": {
            "num_questions": len(rows),
            "top50_gold_full_cover": full_cover,
            "top50_gold_full_cover_rate": full_cover / len(rows),
            "missing_gold_evidence_questions": [
                row["question_id"] for row in rows if not row["oracle_top50_full_cover"]
            ],
        },
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oracle15_gold_binding.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "oracle15_temporal_annotations.template.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "diagnostic_oracle_only": True,
                    "annotation_rule": "Use question and gold_turns only; never inspect gold_answer.",
                },
                "annotations": annotations,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
