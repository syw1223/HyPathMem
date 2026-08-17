"""LoCoMo adapters for fail-closed HyPathMem 8.5 temporal solving."""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta
from typing import Any, Callable


VERSION = "8.5time_locomo_v1"
WEEKDAYS = {name.lower(): index for index, name in enumerate(calendar.day_name)}
SAFE_POINT_PRECISIONS = {"day", "week", "month", "year"}


def parse_locomo_datetime(value: str | None) -> datetime | None:
    cleaned = re.sub(r"\s*\([A-Za-z]{3}\)\s*", " ", str(value or "")).strip()
    formats = (
        "%I:%M %p on %d %B, %Y",
        "%I:%M%p on %d %B, %Y",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y/%m/%d",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def latest_conversation_time(turns: list[dict[str, Any]]) -> datetime | None:
    parsed = [parse_locomo_datetime(str(turn.get("timestamp") or "")) for turn in turns]
    values = [value for value in parsed if value is not None]
    return max(values, default=None)


def normalize_locomo_binding(
    binding: dict[str, Any],
    base_normalize: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    prepared = dict(binding)
    mentioned = parse_locomo_datetime(str(binding.get("mentioned_at") or ""))
    if mentioned:
        prepared["mentioned_at"] = mentioned.strftime("%Y-%m-%d %H:%M")
    result = base_normalize(prepared)
    if result.get("occurred_at") or mentioned is None:
        return result

    expression = str(binding.get("time_expression") or "").strip().lower()
    weekday = re.search(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        expression,
    )
    if weekday:
        target = WEEKDAYS[weekday.group(1)]
        delta = (mentioned.weekday() - target) % 7 or 7
        occurred = mentioned - timedelta(days=delta)
        return resolved(result, occurred, "day", "previous_named_weekday@mentioned_at")
    if "tomorrow" in expression:
        return resolved(result, mentioned + timedelta(days=1), "day", "mentioned_at+1_day")
    if expression == "last year":
        return resolved(result, datetime(mentioned.year - 1, 1, 1), "year", "previous_calendar_year")
    if expression == "last month":
        month = mentioned.month - 1 or 12
        year = mentioned.year if mentioned.month > 1 else mentioned.year - 1
        return resolved(result, datetime(year, month, 1), "month", "previous_calendar_month")
    return result


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


def solve_locomo_hypothesis(
    question: str,
    query_type: str,
    operands: list[dict[str, Any]],
    question_date: str,
    base_solver: Callable[[str, str, list[dict[str, Any]], str], dict[str, Any]],
) -> dict[str, Any]:
    if query_type != "date":
        explicit_duration = explicit_duration_answer(question, query_type, operands)
        if explicit_duration is not None:
            return explicit_duration
        solution = base_solver(question, query_type, operands, question_date)
        return fail_closed_solution(solution, operands)
    if len(operands) != 1:
        return failure("date_requires_one_event")
    operand = operands[0]
    occurred = parse_locomo_datetime(str(operand.get("occurred_at") or ""))
    precision = str(operand.get("precision") or "unknown")
    status = str(operand.get("normalization_status") or "")
    if occurred is None or status != "resolved" or precision not in SAFE_POINT_PRECISIONS:
        return failure("date_not_safely_resolved")
    answer = format_temporal_answer(occurred, precision)
    return {
        "success": True,
        "answer": answer,
        "operation": "DATE",
        "trace": ["formatted deterministic occurrence time"],
    }


def explicit_duration_answer(
    question: str, query_type: str, operands: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if query_type not in {"elapsed", "duration", "recency"} or len(operands) != 1:
        return None
    lowered = question.lower()
    if not (lowered.startswith("how long") or lowered.startswith("how many")):
        return None
    operand = operands[0]
    value = operand.get("duration_value")
    unit = str(operand.get("duration_unit") or "").rstrip("s")
    precision = str(operand.get("precision") or "unknown")
    if not isinstance(value, (int, float)) or value <= 0 or unit not in {"day", "week", "month", "year"}:
        return None
    if precision in {"approximate", "approximate_recent", "coarse_multi_year", "unknown"}:
        return None
    rendered_value = str(int(value)) if float(value).is_integer() else str(value)
    rendered_unit = unit if float(value) == 1 else unit + "s"
    return {
        "success": True,
        "answer": f"{rendered_value} {rendered_unit}",
        "operation": "EXPLICIT_DURATION",
        "trace": ["returned duration explicitly stated in grounded raw quote"],
    }


def fail_closed_solution(solution: dict[str, Any], operands: list[dict[str, Any]]) -> dict[str, Any]:
    if not solution.get("success"):
        return solution
    unsafe = {
        "approximate_recent",
        "coarse_multi_year",
        "unknown",
    }
    if any(str(operand.get("precision") or "unknown") in unsafe for operand in operands):
        return failure("operand_time_precision_unsafe")
    return solution


def format_temporal_answer(value: datetime, precision: str) -> str:
    if precision == "year":
        return str(value.year)
    if precision == "month":
        return value.strftime("%B %Y")
    return f"{value.day} {value.strftime('%B %Y')}"


def failure(reason: str) -> dict[str, Any]:
    return {"success": False, "answer": None, "operation": "DATE", "failure_reason": reason}
