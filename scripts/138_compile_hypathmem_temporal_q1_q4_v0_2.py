#!/usr/bin/env python3
"""Normalize Qwen-bound temporal operands and compile Q1-Q4 candidates."""

from __future__ import annotations

import argparse
import calendar
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def parse_message_time(value: str) -> datetime | None:
    cleaned = re.sub(r"\s*\([A-Za-z]{3}\)\s*", " ", str(value)).strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass
    return None


def shift_months(value: datetime, months: int) -> datetime:
    index = value.year * 12 + value.month - 1 + months
    year, month0 = divmod(index, 12)
    month = month0 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def shift(value: datetime, amount: int, unit: str) -> datetime:
    unit = unit.rstrip("s")
    if unit == "day":
        return value + timedelta(days=amount)
    if unit == "week":
        return value + timedelta(weeks=amount)
    if unit == "month":
        return shift_months(value, amount)
    if unit == "year":
        return shift_months(value, amount * 12)
    raise ValueError(unit)


def number(value: str) -> int:
    lowered = value.lower()
    if lowered in NUMBER_WORDS:
        return NUMBER_WORDS[lowered]
    return int(value) if value.isdigit() else 0


def normalize_binding(binding: dict[str, Any]) -> dict[str, Any]:
    expression = str(binding.get("time_expression") or "").strip().lower()
    mentioned = parse_message_time(str(binding.get("mentioned_at") or ""))
    result = {
        "role": binding.get("role"),
        "identity": binding.get("identity"),
        "fact_id": binding.get("fact_id"),
        "raw_id": binding.get("raw_id"),
        "evidence_span": binding.get("evidence_span"),
        "mentioned_at": mentioned.isoformat() if mentioned else None,
        "time_expression": binding.get("time_expression"),
        "anchor_type": binding.get("anchor_type"),
        "anchor_id": binding.get("anchor_id"),
        "occurred_at": None,
        "duration_value": None,
        "duration_unit": None,
        "precision": "unknown",
        "normalization_status": "unresolved",
        "normalization_trace": [],
    }
    if not mentioned:
        result["normalization_trace"].append("missing mentioned_at")
        return result
    if not expression:
        result["normalization_trace"].append("no explicit temporal expression")
        return result
    if expression in {"today", "just", "recently"} or expression.startswith("just recently"):
        result["occurred_at"] = mentioned.isoformat()
        result["precision"] = "day" if expression == "today" else "approximate_recent"
        result["normalization_status"] = "resolved"
        result["normalization_trace"].append("bound recent expression to mentioned_at")
        return result
    if "yesterday" in expression:
        result["occurred_at"] = shift(mentioned, -1, "day").isoformat()
        result["precision"] = "day"
        result["normalization_status"] = "resolved"
        result["normalization_trace"].append("mentioned_at - 1 day")
        return result
    match = re.search(
        r"(?:exactly\s+|about\s+|around\s+|for\s+(?:the\s+past\s+)?|past\s+)?"
        r"(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(day|week|month|year)s?(?:\s+ago|\s+now)?",
        expression,
    )
    if match:
        amount, unit = number(match.group(1)), match.group(2)
        result["occurred_at"] = shift(mentioned, -amount, unit).isoformat()
        result["duration_value"] = amount
        result["duration_unit"] = unit
        result["precision"] = "approximate" if any(token in expression for token in ("about", "around")) else unit
        result["normalization_status"] = "resolved"
        result["normalization_trace"].append(f"mentioned_at - {amount} {unit}(s)")
        return result
    if expression == "last week":
        result["occurred_at"] = shift(mentioned, -1, "week").isoformat()
        result["precision"] = "week"
        result["normalization_status"] = "resolved"
        return result
    if "last summer" in expression:
        result["occurred_at"] = datetime(mentioned.year - 1, 7, 1).isoformat()
        result["precision"] = "season"
        result["normalization_status"] = "resolved"
        return result
    if "few years ago" in expression:
        result["occurred_at"] = shift(mentioned, -3, "year").isoformat()
        result["precision"] = "coarse_multi_year"
        result["normalization_status"] = "resolved"
        return result
    slash = re.search(r"\b(?:on\s+the\s+)?(\d{1,2})/(\d{1,2})\b", expression)
    if slash:
        month, day = int(slash.group(1)), int(slash.group(2))
        result["occurred_at"] = mentioned.replace(month=month, day=day).isoformat()
        result["precision"] = "day"
        result["normalization_status"] = "resolved"
        return result
    month_name = re.search(
        r"(?:on\s+the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+of\s+"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        expression,
    )
    if month_name:
        months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
        result["occurred_at"] = mentioned.replace(
            month=months.index(month_name.group(2)) + 1, day=int(month_name.group(1))
        ).isoformat()
        result["precision"] = "day"
        result["normalization_status"] = "resolved"
        return result
    month_day = re.search(
        r"(?:on\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?",
        expression,
    )
    if month_day:
        months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
        result["occurred_at"] = mentioned.replace(
            month=months.index(month_day.group(1)) + 1, day=int(month_day.group(2))
        ).isoformat()
        result["precision"] = "day"
        result["normalization_status"] = "resolved"
        return result
    if re.search(r"\bin\s+january\b", expression):
        year = mentioned.year if mentioned.month > 1 else mentioned.year - 1
        result["occurred_at"] = datetime(year, 1, 1).isoformat()
        result["precision"] = "month"
        result["normalization_status"] = "resolved"
        return result
    result["normalization_trace"].append("unsupported expression")
    return result


def answer_unit(question: str) -> str:
    lowered = question.lower()
    for unit in ("day", "week", "month", "year"):
        if re.search(rf"\b{unit}s?\b", lowered):
            return unit
    return "day"


def format_amount(value: float, unit: str) -> str:
    rounded = round(value)
    text = str(rounded) if abs(value - rounded) < 1e-6 else f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{text} {unit if abs(value - 1) < 1e-6 else unit + 's'}"


def difference(left: datetime, right: datetime, unit: str) -> float:
    days = abs((right.date() - left.date()).days)
    if unit == "day":
        return float(days)
    if unit == "week":
        return days / 7
    months = abs((right.year - left.year) * 12 + right.month - left.month)
    if unit == "month":
        return float(months)
    return months / 12


def solve_hypothesis(question: str, query_type: str, operands: list[dict[str, Any]], question_date: str) -> dict[str, Any]:
    resolved = [operand for operand in operands if operand["occurred_at"]]
    result = {"success": False, "answer": None, "trace": [], "failure_reason": None}
    if len(resolved) != len(operands) or not operands:
        result["failure_reason"] = "not_all_operands_normalized"
        return result
    times = [datetime.fromisoformat(operand["occurred_at"]) for operand in operands]
    lowered = question.lower()
    if query_type == "ordering":
        ordered = sorted(zip(times, operands), key=lambda item: item[0])
        labels = [str(operand.get("identity") or operand.get("role")) for _, operand in ordered]
        first_only = (
            lowered.startswith("which")
            and "first" in lowered
            and "from first to last" not in lowered
            and "order" not in lowered
        )
        result["answer"] = f"{labels[0]} was first" if first_only else ", then ".join(labels)
        result["success"] = True
        result["trace"] = ["sorted normalized occurrence times ascending"]
        return result
    if query_type == "attribute_at_time":
        result["answer"] = str(operands[0].get("identity") or operands[0].get("role"))
        result["success"] = True
        result["trace"] = ["returned attribute identity bound to requested time"]
        return result
    if query_type == "recency" and len(operands) >= 2:
        latest = max(zip(times, operands), key=lambda item: item[0])[1]
        result["answer"] = str(latest.get("identity") or latest.get("role"))
        result["success"] = True
        result["trace"] = ["selected latest normalized occurrence"]
        return result
    if query_type == "recency" and len(operands) == 1 and not lowered.startswith("how many"):
        result["answer"] = str(operands[0].get("identity") or operands[0].get("role"))
        result["success"] = True
        result["trace"] = ["returned identity of event matching requested relative window"]
        return result
    if query_type in {"elapsed", "duration", "recency"}:
        unit = answer_unit(question)
        operand_units = {operand.get("duration_unit") for operand in operands if operand.get("duration_unit")}
        if (query_type == "duration" or lowered.startswith("how long")) and unit == "day" and len(operand_units) == 1:
            unit = str(next(iter(operand_units)))
        if len(operands) == 1:
            end = parse_message_time(question_date)
            if not end:
                result["failure_reason"] = "question_time_unparseable"
                return result
            value = difference(times[0], end, unit)
        else:
            value = difference(times[0], times[1], unit)
        if lowered.startswith("how many") and unit in {"week", "month", "year"}:
            value = float(round(value))
        result["answer"] = format_amount(value, unit)
        result["success"] = True
        result["trace"] = [f"computed temporal difference in {unit}s"]
        return result
    result["failure_reason"] = "unsupported_query_type"
    return result


def q1_context(hypothesis: dict[str, Any]) -> str:
    lines = []
    for binding in hypothesis.get("bindings", []):
        lines.append(
            f"[{binding.get('mentioned_at')}] role={binding.get('role')} raw_id={binding.get('raw_id')}\n"
            f"QUOTE: {binding.get('evidence_span')}"
        )
    return "\n\n".join(lines)


def restrictive_clause_grounded(question: str, operands: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    """Fail closed when a `when ...` condition has no lexical evidence support."""
    match = re.search(r"\bwhen\b\s+(.+?)(?:\?|$)", question.lower())
    if not match:
        return True, {"activated": False}
    stop = {
        "i", "my", "me", "the", "a", "an", "did", "had", "have", "been", "was", "were",
        "at", "in", "on", "for", "to", "of", "that", "this", "new",
    }
    clause_terms = {
        token.rstrip("s")
        for token in re.findall(r"[a-z][a-z'-]+", match.group(1))
        if token not in stop and len(token) > 2
    }
    evidence = " ".join(
        str(operand.get("identity") or "") + " " + str(operand.get("evidence_span") or "")
        for operand in operands
    ).lower()
    evidence_terms = {token.rstrip("s") for token in re.findall(r"[a-z][a-z'-]+", evidence)}
    matched = clause_terms & evidence_terms
    coverage = len(matched) / max(1, len(clause_terms))
    grounded = len(clause_terms) < 2 or coverage >= 0.35
    return grounded, {
        "activated": True,
        "clause": match.group(1),
        "terms": sorted(clause_terms),
        "matched_terms": sorted(matched),
        "coverage": coverage,
        "threshold": 0.35,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extractions",
        type=Path,
        default=Path(
            "outputs/reconstruction/hypathmem_temporal_v0_2_qwen/paired20_joint_extraction_gpu4_v2.json"
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "outputs/qa/longmemeval_v3_10_cardquota_light_top50_c1_window_chunks_80k_"
            "gpt41mini_judge_gpt4omini.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_temporal_v0_2_qwen/paired20_q1_q4_compiled.json"),
    )
    args = parser.parse_args()
    for name in ("extractions", "baseline", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    extraction_payload = json.loads(args.extractions.read_text(encoding="utf-8"))
    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_by_id = {row["question_id"]: row for row in baseline_payload["per_question"]}
    rows = []
    for source in extraction_payload["rows"]:
        extraction = source["extraction"]
        audit_by_id = {item["hypothesis_id"]: item for item in source["validation"]["hypotheses"]}
        hypotheses = []
        for hypothesis in extraction.get("hypotheses", []):
            audit = audit_by_id.get(hypothesis.get("hypothesis_id"), {})
            normalized = [normalize_binding(binding) for binding in hypothesis.get("bindings", [])]
            solution = solve_hypothesis(
                source["question"], extraction.get("query_type", "other"), normalized, source["question_date"]
            )
            hypotheses.append(
                {
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    "confidence": hypothesis.get("confidence", 0.0),
                    "source_validation": audit,
                    "bindings": hypothesis.get("bindings", []),
                    "normalized_operands": normalized,
                    "solution": solution,
                    "q1_context": q1_context(hypothesis),
                    "q2_context": json.dumps(
                        {
                            "query_type": extraction.get("query_type"),
                            "required_roles": extraction.get("required_roles"),
                            "operands": normalized,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        eligible = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis["source_validation"].get("eligible_for_normalization")
            and hypothesis["solution"]["success"]
        ]
        clause_checks = [restrictive_clause_grounded(source["question"], hypothesis["normalized_operands"])[1] for hypothesis in eligible]
        clause_grounded = all(
            restrictive_clause_grounded(source["question"], hypothesis["normalized_operands"])[0]
            for hypothesis in eligible
        )
        answers = {str(hypothesis["solution"]["answer"]).strip().lower() for hypothesis in eligible}
        robust = bool(eligible) and len(answers) == 1 and clause_grounded
        base = baseline_by_id[source["question_id"]]
        rows.append(
            {
                "question_id": source["question_id"],
                "question": source["question"],
                "question_date": source["question_date"],
                "question_type": base.get("question_type"),
                "gold_answer": base.get("gold_answer"),
                "is_abstention": base.get("is_abstention", False),
                "frozen_d0": {
                    "prediction": base.get("prediction"),
                    "judge_correct": base.get("judge_correct"),
                    "retrieval_full_cover": base.get("retrieval_full_cover"),
                },
                "query_type": extraction.get("query_type"),
                "required_roles": extraction.get("required_roles"),
                "ambiguities": extraction.get("ambiguities", []),
                "hypotheses": hypotheses,
                "pre_verifier": {
                    "eligible_solution_count": len(eligible),
                    "solution_answers": sorted(answers),
                    "multi_hypothesis_answer_consistent": robust,
                    "restrictive_clause_grounded": clause_grounded,
                    "restrictive_clause_checks": clause_checks,
                    "candidate_answer": eligible[0]["solution"]["answer"] if robust else None,
                    "eligible_to_call_solution_verifier": robust,
                },
            }
        )
    output = {
        "metadata": {
            "version": "hypathmem_temporal_v0_2_q1_q4",
            "gold_evidence_or_answer_used_for_compilation": False,
            "fallback": "frozen_D0",
        },
        "summary": {
            "num_questions": len(rows),
            "eligible_to_call_solution_verifier": sum(
                row["pre_verifier"]["eligible_to_call_solution_verifier"] for row in rows
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
