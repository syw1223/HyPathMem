"""LoCoMo-specific temporal compiler with conservative, typed takeover gates.

This module is intentionally separate from the LongMemEval and LoCoMo v1
implementations.  It never reads references or judge labels.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Callable

from hytopomem.locomo_time_v8_5 import (
    explicit_duration_answer,
    failure,
    format_temporal_answer,
    parse_locomo_datetime,
)


VERSION = "8.5time_locomo_v2"
WHEN_RE = re.compile(r"^\s*when\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
NAMED_WEEKDAY_RE = re.compile(
    r"\blast\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


def normalize_locomo_binding_v2(
    binding: dict[str, Any],
    base_normalize: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Normalize from the evidence-local message time, fixing compound offsets."""

    prepared = dict(binding)
    mentioned = parse_locomo_datetime(str(binding.get("mentioned_at") or ""))
    if mentioned:
        prepared["mentioned_at"] = mentioned.strftime("%Y-%m-%d %H:%M")
    result = base_normalize(prepared)
    result["local_anchor_valid"] = bool(
        mentioned
        and str(binding.get("anchor_type") or "") == "mentioned_at"
        and str(binding.get("anchor_id") or "") == str(binding.get("raw_id") or "")
    )
    if mentioned is None:
        return result

    expression = canonical_expression(binding.get("time_expression"))
    if "day after tomorrow" in expression:
        return resolved(result, mentioned + timedelta(days=2), "day", "mentioned_at+2_days")
    if "day before yesterday" in expression:
        return resolved(result, mentioned - timedelta(days=2), "day", "mentioned_at-2_days")
    if re.search(r"\byesterday\b", expression):
        return resolved(result, mentioned - timedelta(days=1), "day", "mentioned_at-1_day")
    if re.search(r"\btomorrow\b", expression):
        return resolved(result, mentioned + timedelta(days=1), "day", "mentioned_at+1_day")

    # LoCoMo references are not consistent enough about "last Friday/Sunday".
    # Keep the candidate for auditing but do not permit deterministic takeover.
    if NAMED_WEEKDAY_RE.search(expression):
        updated = dict(result)
        updated.update(
            {
                "occurred_at": None,
                "precision": "weekday_relative_ambiguous",
                "normalization_status": "unresolved",
                "normalization_trace": ["named_weekday_fail_closed"],
            }
        )
        return updated
    return result


def solve_locomo_hypothesis_v2(
    question: str,
    query_type: str,
    operands: list[dict[str, Any]],
    question_date: str,
    base_solver: Callable[[str, str, list[dict[str, Any]], str], dict[str, Any]],
) -> dict[str, Any]:
    """Solve according to the answer type requested by the question."""

    if WHEN_RE.search(question):
        return solve_when(operands)
    duration = explicit_duration_answer(question, query_type, operands)
    if duration is not None:
        duration["answer_type"] = "DURATION"
        return duration
    solution = base_solver(question, query_type, operands, question_date)
    if not solution.get("success"):
        return solution
    answer_type = infer_answer_type(str(solution.get("answer") or ""))
    if answer_type is None:
        return failure_v2("solver_answer_type_invalid")
    solution = dict(solution)
    solution["answer_type"] = answer_type
    return solution


def solve_when(operands: list[dict[str, Any]]) -> dict[str, Any]:
    if len(operands) != 1:
        return failure_v2("when_requires_one_event")
    operand = operands[0]
    if not operand.get("local_anchor_valid"):
        return failure_v2("evidence_local_anchor_invalid")
    if str(operand.get("normalization_status") or "") != "resolved":
        return failure_v2("when_time_unresolved")
    occurred = parse_locomo_datetime(str(operand.get("occurred_at") or ""))
    mentioned = parse_locomo_datetime(str(operand.get("mentioned_at") or ""))
    if occurred is None:
        return failure_v2("when_time_missing")

    expression = canonical_expression(operand.get("time_expression"))
    precision = str(operand.get("precision") or "unknown").lower()
    if NAMED_WEEKDAY_RE.search(expression):
        return failure_v2("named_weekday_ambiguous")
    if precision in {"week", "approximate_week"} or "last week" in expression:
        if mentioned is None:
            return failure_v2("week_anchor_missing")
        answer = f"the week before {format_temporal_answer(mentioned, 'day')}"
        return success(answer, "DATE_WINDOW", "preserved_relative_week_granularity")
    if precision == "season" or any(season in expression for season in ("spring", "summer", "autumn", "fall", "winter")):
        season = next((value for value in ("spring", "summer", "autumn", "fall", "winter") if value in expression), None)
        if season is None:
            return failure_v2("season_name_missing")
        rendered = "autumn" if season == "fall" else season
        return success(f"{rendered} {occurred.year}", "DATE_WINDOW", "preserved_season_granularity")
    if precision in {"approximate", "approximate_year"}:
        return success(f"around {occurred.year}", "YEAR", "preserved_approximate_year_granularity")
    if precision == "year":
        return success(str(occurred.year), "YEAR", "formatted_year")
    if precision == "month":
        return success(occurred.strftime("%B %Y"), "MONTH", "formatted_month")
    if precision == "day":
        return success(format_temporal_answer(occurred, "day"), "DATE_POINT", "formatted_exact_day")
    return failure_v2("when_precision_unsafe")


def ambiguity_safe(ambiguities: list[Any]) -> bool:
    """V2 does not override D0 when extraction reports any temporal ambiguity."""

    return not any(str(value).strip() for value in ambiguities)


def typed_solution_safe(question: str, solution: dict[str, Any]) -> bool:
    if not solution.get("success"):
        return False
    answer = str(solution.get("answer") or "").strip()
    answer_type = str(solution.get("answer_type") or "")
    if WHEN_RE.search(question):
        return answer_type in {"DATE_POINT", "DATE_WINDOW", "MONTH", "YEAR"} and infer_answer_type(answer) is not None
    return infer_answer_type(answer) is not None


def infer_answer_type(answer: str) -> str | None:
    lowered = answer.strip().lower()
    if not lowered:
        return None
    if re.fullmatch(r"(?:around\s+)?(?:18|19|20)\d{2}", lowered):
        return "YEAR"
    if re.fullmatch(
        r"\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:18|19|20)\d{2}",
        lowered,
    ):
        return "DATE_POINT"
    if re.fullmatch(
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:18|19|20)\d{2}",
        lowered,
    ):
        return "MONTH"
    if lowered.startswith("the week before ") and YEAR_RE.search(lowered):
        return "DATE_WINDOW"
    if re.fullmatch(r"(?:spring|summer|autumn|winter)\s+(?:18|19|20)\d{2}", lowered):
        return "DATE_WINDOW"
    if re.fullmatch(r"\d+(?:\.\d+)?\s+(?:days?|weeks?|months?|years?)", lowered):
        return "DURATION"
    return None


def canonical_expression(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def resolved(result: dict[str, Any], value: datetime, precision: str, trace: str) -> dict[str, Any]:
    updated = dict(result)
    updated.update(
        {
            "occurred_at": value.isoformat(),
            "precision": precision,
            "normalization_status": "resolved",
            "normalization_trace": [trace],
        }
    )
    return updated


def success(answer: str, answer_type: str, trace: str) -> dict[str, Any]:
    return {
        "success": True,
        "answer": answer,
        "answer_type": answer_type,
        "operation": "DATE",
        "trace": [trace],
    }


def failure_v2(reason: str) -> dict[str, Any]:
    value = failure(reason)
    value["answer_type"] = None
    return value
