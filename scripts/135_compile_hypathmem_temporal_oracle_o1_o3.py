#!/usr/bin/env python3
"""Compile O1/O2 prompts and deterministic O3 outputs for oracle diagnostics."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def plural(value: int, unit: str) -> str:
    suffix = "" if value == 1 else "s"
    return f"{value} {unit.rstrip('s')}{suffix}"


def solve(annotation: dict[str, Any]) -> tuple[str | None, bool, str]:
    status = annotation["status"]
    operation = annotation["operation"]
    operands = annotation.get("operands", [])
    if status != "verified":
        return None, False, annotation.get("notes", "Oracle annotation is unresolved.")
    if operation == "ORDERING":
        ordered = sorted(operands, key=lambda operand: parse_date(operand["time"]))
        return ", then ".join(operand["label"] for operand in ordered), True, "sorted normalized event times"
    if operation == "MOST_RECENT":
        latest = max(operands, key=lambda operand: parse_date(operand["time"]))
        return latest["label"], True, "selected maximum normalized event time"
    if operation == "ATTRIBUTE_AT_TIME":
        return operands[0]["label"], True, "selected attribute bound to normalized anchor"
    if operation == "ELAPSED":
        if len(operands) == 1:
            start, end = parse_date(operands[0]["time"]), parse_date(annotation["anchor"]["time"])
        else:
            start, end = sorted(parse_date(operand["time"]) for operand in operands)
        return plural((end - start).days, annotation["unit"]), True, "date subtraction"
    if operation == "ELAPSED_ROUNDED":
        start = parse_date(operands[0]["time"])
        end = parse_date(annotation["anchor"]["time"])
        days = (end - start).days
        weeks = round(days / 7)
        return plural(weeks, annotation["unit"]), True, f"rounded {days} days to nearest week"
    if operation == "CHAINED_ELAPSED":
        trip = parse_date(operands[0]["time"])
        lead = int(operands[1]["duration"])
        anchor = parse_date(annotation["anchor"]["time"])
        month_delta = (anchor.year - trip.year) * 12 + anchor.month - trip.month
        return plural(month_delta + lead, annotation["unit"]), True, "trip age plus booking lead time"
    if operation == "DURATION_BETWEEN_RELATIVE_EVENTS":
        offsets = [int(operand["offset"]) for operand in operands]
        return plural(abs(offsets[0] - offsets[1]), annotation["unit"]), True, "difference between relative offsets"
    if operation == "DURATION_AT_EVENT":
        elapsed_now = int(operands[0]["duration"])
        event_offset = int(operands[1]["offset"])
        return plural(elapsed_now - event_offset, annotation["unit"]), True, "elapsed-now minus event offset"
    if operation == "REQUIRED_OPERAND_CHECK":
        missing = [operand["label"] for operand in operands if operand.get("missing")]
        if missing:
            return f"Insufficient information: no evidence for {', '.join(missing)}.", True, "required operand absent"
    return None, False, f"Unsupported operation: {operation}"


def quote_text(turn: dict[str, Any]) -> str:
    return (
        f"[{turn.get('timestamp', 'unknown time')}] {turn.get('speaker', 'unknown')}: "
        f"{turn.get('text', '')}"
    )


def o1_context(row: dict[str, Any]) -> str:
    return "\n".join(quote_text(turn) for turn in row["gold_turns"])


def o2_context(row: dict[str, Any], annotation: dict[str, Any]) -> str:
    packet = {
        "operation": annotation["operation"],
        "annotation_status": annotation["status"],
        "operands": annotation.get("operands", []),
        "anchor": annotation.get("anchor"),
        "constraints": annotation.get("constraints", []),
    }
    quotes = [quote_text(turn) for turn in row["gold_turns"]]
    return (
        "TEMPORAL OPERANDS (oracle diagnostic; normalized from the evidence below):\n"
        + json.dumps(packet, ensure_ascii=False, indent=2)
        + "\n\nSOURCE QUOTES:\n"
        + "\n".join(quotes)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binding",
        type=Path,
        default=Path(
            "outputs/reconstruction/hypathmem_temporal_v0_2_oracle/oracle15_gold_binding.json"
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "outputs/reconstruction/hypathmem_temporal_v0_2_oracle/oracle15_temporal_annotations.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/reconstruction/hypathmem_temporal_v0_2_oracle/oracle15_o1_o3_packets.json"
        ),
    )
    args = parser.parse_args()

    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    annotations_payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotation_by_id = {
        annotation["question_id"]: annotation for annotation in annotations_payload["annotations"]
    }
    rows = []
    for row in binding["rows"]:
        annotation = annotation_by_id[row["question_id"]]
        answer, verified, trace = solve(annotation)
        rows.append(
            {
                "question_id": row["question_id"],
                "cohort": row["cohort"],
                "question": row["question"],
                "question_date": row["question_date"],
                "gold_answer": row["gold_answer"],
                "is_abstention": row.get("is_abstention", False),
                "frozen_d0": row["frozen_d0"],
                "oracle_top50_full_cover": row["oracle_top50_full_cover"],
                "o1": {"context": o1_context(row)},
                "o2": {"context": o2_context(row, annotation), "annotation": annotation},
                "o3": {
                    "prediction": answer,
                    "solver_verified": verified,
                    "solver_trace": trace,
                    "fallback_prediction": None if verified else row["frozen_d0"]["prediction"],
                    "final_prediction": answer if verified else row["frozen_d0"]["prediction"],
                    "route": "oracle_solver" if verified else "frozen_D0_fallback",
                },
            }
        )
    verified_count = sum(row["o3"]["solver_verified"] for row in rows)
    output = {
        "metadata": {
            "version": "hypathmem_temporal_v0_2_oracle",
            "diagnostic_oracle_only": True,
            "o1": "gold evidence identity + raw quotes",
            "o2": "O1 + manually verified temporal operands and anchors",
            "o3": "O2 + deterministic solver; unresolved cases fall back to frozen D0",
            "gold_answer_used_by_solver": False,
        },
        "summary": {
            "num_questions": len(rows),
            "solver_verified": verified_count,
            "frozen_d0_fallback": len(rows) - verified_count,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
