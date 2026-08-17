"""Fail-closed executable-constraint conversion for HyPathMem 8.5time_v.

8.5 never sends a non-executable packet to an answer reader. It diagnoses the
8.4 structural closure, accepts only raw-grounded temporal relations, computes
times deterministically, and emits a candidate for the existing Q4 verifier.
"""

from __future__ import annotations

import copy
import calendar
import re
from datetime import datetime, timedelta
from typing import Any, Sequence

from hytopomem.temporal_time_v8_4 import (
    normalize_absolute_date,
    normalize_relative,
    parse_datetime,
    shift_months,
    solve_constraint,
)


VERSION = "8.5time_v"
SUPPORTED_QUERY_TYPES = {"ordering", "elapsed", "duration", "recency"}
SUPPORTED_RELATIONS = {"SAME_TIME", "BEFORE_OFFSET", "AFTER_OFFSET", "ABSOLUTE", "INTERVAL"}


def diagnose_non_executable(row: dict[str, Any]) -> dict[str, Any]:
    closure = row.get("h2_h4_operand_closure") or {}
    candidate = row.get("h5_constraint_candidate") or {}
    operands = list(candidate.get("operands") or [])
    selected = closure.get("selected_by_role") or {}
    required_roles = list(candidate.get("required_roles") or [])
    query_type = str(candidate.get("query_type") or "other")
    selected_base_roles = {
        role.rsplit(":", 1)[0] if role.endswith((":start", ":end")) else role
        for role in selected
    }
    structural_complete = bool(closure.get("operand_full_coverage"))
    roles_present = not required_roles or all(role in selected_base_roles for role in required_roles)
    requires_pairwise = bool((closure.get("plan") or {}).get("requires_pairwise_operands"))
    pairwise_complete = roles_present and (not requires_pairwise or len(operands) >= 2)
    identity_grounded = bool(selected) and all(
        bool(item.get("trusted_q4_binding")) for item in selected.values()
    )
    raw_grounded = bool(operands) and all(
        bool(operand.get("raw_id")) and bool(str(operand.get("evidence_span") or "").strip())
        for operand in operands
    )
    occurred_time_grounded = bool(operands) and all(operand.get("occurred_at") for operand in operands)
    anchor_resolved = all(
        str(operand.get("normalization_status") or "") not in {"symbolic", "ambiguous"}
        for operand in operands
    )
    consistent = bool((closure.get("temporal_consistency") or {}).get("consistent", True))
    restrictive_clause_pass = bool(
        (row.get("audit") or {}).get("restrictive_clause_grounded", True)
    )
    solver_supported = query_type in SUPPORTED_QUERY_TYPES
    validated_coverage = all(
        (
            structural_complete,
            pairwise_complete,
            identity_grounded,
            raw_grounded,
            occurred_time_grounded,
            anchor_resolved,
            consistent,
            restrictive_clause_pass,
        )
    )
    if not structural_complete or not pairwise_complete:
        failure_type = "operand_missing_or_outside_top50"
    elif not identity_grounded:
        failure_type = "event_identity_unreliable"
    elif not raw_grounded:
        failure_type = "raw_grounding_missing"
    elif not solver_supported:
        failure_type = "solver_type_unsupported"
    elif not consistent:
        failure_type = "hypothesis_conflict"
    elif not restrictive_clause_pass:
        failure_type = "restrictive_clause_failed"
    elif not occurred_time_grounded:
        failure_type = "occurred_time_missing"
    elif not anchor_resolved:
        failure_type = "anchor_unresolved"
    else:
        failure_type = "safety_gate_rejected"
    repairable_by_time_enrichment = all(
        (
            structural_complete,
            pairwise_complete,
            identity_grounded,
            raw_grounded,
            solver_supported,
            consistent,
            restrictive_clause_pass,
            not occurred_time_grounded,
        )
    )
    return {
        "question_id": row.get("question_id"),
        "question_type": query_type,
        "structural_roles_complete": structural_complete,
        "pairwise_operands_complete": pairwise_complete,
        "identity_grounded": identity_grounded,
        "raw_grounded": raw_grounded,
        "occurred_time_grounded": occurred_time_grounded,
        "anchor_resolved": anchor_resolved,
        "solver_supported": solver_supported,
        "hypothesis_consistent": consistent,
        "restrictive_clause_pass": restrictive_clause_pass,
        "validated_operand_coverage": validated_coverage,
        "repairable_by_time_enrichment": repairable_by_time_enrichment,
        "failure_type": failure_type,
    }


def validate_enrichment(
    extraction: dict[str, Any],
    operand: dict[str, Any],
    allowed_event_ids: set[str],
) -> tuple[dict[str, Any] | None, str]:
    if not extraction.get("identity_supported"):
        return None, "identity_not_supported"
    if extraction.get("ambiguity"):
        return None, "ambiguity_not_empty"
    event_id = str(extraction.get("event_id") or "")
    if event_id != str(operand.get("fact_id") or ""):
        return None, "event_id_mismatch"
    expression = str(extraction.get("time_expression") or "").strip()
    evidence = str(operand.get("evidence_span") or "")
    if not expression or expression.lower() not in evidence.lower():
        return None, "time_expression_not_in_raw"
    mentioned = parse_datetime(operand.get("mentioned_at"))
    interval = resolve_grounded_interval(expression, mentioned)
    point, point_status, point_trace = resolve_grounded_point(expression, mentioned)
    confidence = float(extraction.get("confidence") or 0.0)
    if confidence < 0.90 and interval is None and point is None:
        return None, "confidence_below_0.90"
    relation = str(extraction.get("relation") or "")
    if interval is not None:
        relation = "INTERVAL"
    elif point is not None and relation == "NONE":
        relation = "ABSOLUTE"
    if relation not in SUPPORTED_RELATIONS:
        return None, "unsupported_relation"
    anchor_id = str(extraction.get("anchor_event_id") or "").strip() or None
    if interval is not None or point is not None:
        # Deterministic RAW+message-time normalization supersedes a model's
        # accidental self/raw-id anchor field.
        anchor_id = None
    if anchor_id and anchor_id not in allowed_event_ids:
        return None, "anchor_not_in_allowed_events"
    if relation in {"BEFORE_OFFSET", "AFTER_OFFSET", "SAME_TIME"} and not anchor_id:
        anchor_type = str(extraction.get("anchor_type") or "")
        if anchor_type != "mentioned_at":
            return None, "missing_anchor"
    value = int(extraction.get("offset_value") or 0)
    unit = str(extraction.get("offset_unit") or "day").lower()
    if value < 0 or value > 1200 or unit not in {"day", "week", "month", "year"}:
        return None, "invalid_offset"
    validated = {
        "event_id": event_id,
        "identity_supported": True,
        "evidence_quote": evidence,
        "time_expression": expression,
        "anchor_type": "mentioned_at" if interval is not None or point is not None else str(extraction.get("anchor_type") or "event"),
        "anchor_event_id": anchor_id,
        "relation": relation,
        "offset_value": value,
        "offset_unit": unit,
        "confidence": max(confidence, 0.90 if interval is not None or point is not None else confidence),
        "ambiguity": [],
    }
    if interval is not None:
        validated.update(
            {
                "resolved_start": interval[0].isoformat(),
                "resolved_end": interval[1].isoformat(),
                "normalization_status": "grounded_interval",
                "normalization_trace": interval[2],
            }
        )
        return validated, "accepted"
    if point is not None:
        validated.update(
            {
                "resolved_at": point.isoformat(),
                "normalization_status": point_status,
                "normalization_trace": point_trace,
            }
        )
        return validated, "accepted"
    if validated["anchor_type"] == "mentioned_at":
        resolved, status, trace = normalize_relative(expression, mentioned)
        if resolved is None:
            resolved, status, trace = normalize_absolute_date(expression, mentioned)
        if resolved is None:
            return None, "mentioned_at_expression_not_deterministically_resolved"
        validated.update(
            {
                "resolved_at": resolved.isoformat(),
                "normalization_status": status,
                "normalization_trace": trace,
            }
        )
    return validated, "accepted"


def resolve_grounded_point(
    expression: str, mentioned: datetime | None
) -> tuple[datetime | None, str, str]:
    text = expression.strip().lower()
    text = re.sub(r"^just\s+", "", text)
    text = re.sub(r"\ba couple of\s+days?\s+ago\b", "two days ago", text)
    value, status, trace = normalize_relative(text, mentioned)
    if value is not None:
        return value, status, trace
    return normalize_absolute_date(text, mentioned)


def resolve_grounded_interval(
    expression: str, mentioned: datetime | None
) -> tuple[datetime, datetime, str] | None:
    if mentioned is None:
        return None
    text = expression.strip().lower()
    text = re.sub(r"^just\s+", "", text)
    if text == "last weekend":
        # Most recent completed Saturday-Sunday interval before message day.
        days_since_sunday = (mentioned.weekday() - 6) % 7
        end_date = (mentioned - timedelta(days=days_since_sunday or 7)).date()
        start_date = end_date - timedelta(days=1)
        return (
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.max.time()),
            "last_completed_weekend@mentioned_at",
        )
    if text in {"last month", "last year"}:
        if text == "last month":
            anchor = shift_months(mentioned.replace(day=1), -1)
            last_day = calendar.monthrange(anchor.year, anchor.month)[1]
            return (
                anchor.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
                anchor.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999999),
                "previous_calendar_month@mentioned_at",
            )
        year = mentioned.year - 1
        return (
            datetime(year, 1, 1),
            datetime(year, 12, 31, 23, 59, 59, 999999),
            "previous_calendar_year@mentioned_at",
        )
    match = re.fullmatch(
        r"(?:in\s+)?mid[-\s](january|february|march|april|may|june|july|august|"
        r"september|october|november|december)(?:\s+(\d{4}))?",
        text,
    )
    if match:
        month = list(calendar.month_name).index(match.group(1).title())
        year = int(match.group(2)) if match.group(2) else mentioned.year
        return (
            datetime(year, month, 10),
            datetime(year, month, 20, 23, 59, 59, 999999),
            "mid_month_interval",
        )
    return None


def propagate_constraints(
    operands: Sequence[dict[str, Any]], constraints: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    known = {
        str(operand.get("fact_id")): parse_datetime(operand.get("occurred_at"))
        for operand in operands
        if operand.get("fact_id") and operand.get("occurred_at")
    }
    known = {key: value for key, value in known.items() if value is not None}
    conflicts: list[str] = []
    for item in constraints:
        if item.get("resolved_at"):
            _assign_time(known, str(item["event_id"]), parse_datetime(item["resolved_at"]), conflicts)
    for _ in range(len(constraints) + 1):
        changed = False
        for item in constraints:
            event_id = str(item.get("event_id") or "")
            anchor_id = str(item.get("anchor_event_id") or "")
            if not event_id or not anchor_id:
                continue
            offset = signed_offset(item)
            if anchor_id in known and event_id not in known:
                known[event_id] = apply_offset(known[anchor_id], offset)
                changed = True
            elif event_id in known and anchor_id not in known:
                known[anchor_id] = apply_offset(known[event_id], invert_offset(offset))
                changed = True
            elif event_id in known and anchor_id in known:
                expected = apply_offset(known[anchor_id], offset)
                if expected.date() != known[event_id].date():
                    conflicts.append(f"{event_id}!={anchor_id}{offset}")
        if not changed:
            break
    return {
        "resolved_times": {key: value.isoformat() for key, value in sorted(known.items())},
        "conflicts": sorted(set(conflicts)),
        "consistent": not conflicts,
    }


def compile_enriched_candidate(
    row: dict[str, Any], constraints: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    source = row.get("h5_constraint_candidate") or {}
    operands = copy.deepcopy(list(source.get("operands") or []))
    propagation = propagate_constraints(operands, constraints)
    for operand in operands:
        interval = next(
            (item for item in constraints if item.get("event_id") == operand.get("fact_id") and item.get("resolved_start")),
            None,
        )
        if interval and not operand.get("occurred_at"):
            operand["occurred_start"] = interval["resolved_start"]
            operand["occurred_end"] = interval["resolved_end"]
            operand["normalization_status"] = "8.5_grounded_interval"
            operand["confidence"] = interval["confidence"]
        resolved = propagation["resolved_times"].get(str(operand.get("fact_id") or ""))
        if resolved and not operand.get("occurred_at"):
            operand["occurred_at"] = resolved
            operand["normalization_status"] = "8.5_enriched"
            operand["confidence"] = min(
                [float(item.get("confidence") or 0.0) for item in constraints if item.get("event_id") == operand.get("fact_id")]
                or [0.0]
            )
    closure = row.get("h2_h4_operand_closure") or {}
    solution = solve_enriched_constraint(row, operands, closure)
    reverse = reverse_verify(row, operands, solution)
    all_grounded = bool(operands) and all(
        operand.get("occurred_at") or (operand.get("occurred_start") and operand.get("occurred_end"))
        for operand in operands
    )
    eligible = bool(
        propagation["consistent"]
        and all_grounded
        and solution.get("success")
        and reverse["passed"]
        and (closure.get("temporal_consistency") or {}).get("consistent", True)
    )
    return {
        "version": VERSION,
        "question_id": row.get("question_id"),
        "candidate_source": "8.5time_v_constraint_conversion",
        "operands": operands,
        "constraints": list(constraints),
        "propagation": propagation,
        "solution": solution,
        "reverse_verification": reverse,
        "eligible_for_q4_verifier": eligible,
    }


def solve_enriched_constraint(
    row: dict[str, Any], operands: list[dict[str, Any]], closure: dict[str, Any]
) -> dict[str, Any]:
    source = row.get("h5_constraint_candidate") or {}
    query_type = str(source.get("query_type") or "other")
    if query_type == "ordering" and len(operands) >= 2:
        intervals = []
        for operand in operands:
            point = parse_datetime(operand.get("occurred_at"))
            start = parse_datetime(operand.get("occurred_start")) or point
            end = parse_datetime(operand.get("occurred_end")) or point
            if start is None or end is None:
                break
            intervals.append((operand, start, end))
        if len(intervals) == len(operands):
            ordered = sorted(intervals, key=lambda item: item[1])
            if all(left[2] < right[1] for left, right in zip(ordered, ordered[1:])):
                return {
                    "success": True,
                    "answer": f"{ordered[0][0]['identity']} was first",
                    "operation": "ORDERING",
                    "interval_ordering": True,
                }
            return {
                "success": False,
                "answer": None,
                "operation": "ORDERING",
                "failure_reason": "temporal_intervals_overlap",
            }
    return solve_constraint(
        str(row.get("question") or ""),
        query_type,
        operands,
        parse_datetime(row.get("question_date")),
        row.get("h1_temporal_sidecar") or {},
        closure,
    )


def reverse_verify(
    row: dict[str, Any], operands: Sequence[dict[str, Any]], solution: dict[str, Any]
) -> dict[str, Any]:
    if not solution.get("success"):
        return {"passed": False, "reason": "solver_failed"}
    operation = str(solution.get("operation") or "")
    if operation == "ORDERING":
        intervals = []
        for item in operands:
            point = parse_datetime(item.get("occurred_at"))
            start = parse_datetime(item.get("occurred_start")) or point
            end = parse_datetime(item.get("occurred_end")) or point
            if start is None or end is None:
                return {"passed": False, "reason": "unresolved_operand_time"}
            intervals.append((item, start, end))
        ordered = sorted(intervals, key=lambda value: value[1])
        if not all(left[2] < right[1] for left, right in zip(ordered, ordered[1:])):
            return {"passed": False, "reason": "interval_order_not_strict"}
        first = ordered[0][0]
        passed = str(solution.get("answer") or "").startswith(str(first.get("identity") or ""))
        return {"passed": passed, "reason": "ordering_recomputed" if passed else "ordering_mismatch"}
    if operation in {"ELAPSED", "DURATION", "RECENCY"}:
        times = [parse_datetime(item.get("occurred_at")) for item in operands]
        if not times or any(value is None for value in times):
            return {"passed": False, "reason": "unresolved_operand_time"}
        answer = str(solution.get("answer") or "")
        passed = bool(re.fullmatch(r"\d+\s+(?:days|weeks|months|years)", answer))
        return {"passed": passed, "reason": "duration_recomputed" if passed else "duration_format_mismatch"}
    return {"passed": False, "reason": f"reverse_check_unsupported:{operation}"}


def signed_offset(item: dict[str, Any]) -> tuple[int, str]:
    value = int(item.get("offset_value") or 0)
    relation = str(item.get("relation") or "SAME_TIME")
    if relation == "BEFORE_OFFSET":
        value = -value
    elif relation == "SAME_TIME":
        value = 0
    return value, str(item.get("offset_unit") or "day")


def invert_offset(offset: tuple[int, str]) -> tuple[int, str]:
    return -offset[0], offset[1]


def apply_offset(value: datetime, offset: tuple[int, str]) -> datetime:
    amount, unit = offset
    if unit == "day":
        return value + timedelta(days=amount)
    if unit == "week":
        return value + timedelta(weeks=amount)
    if unit == "month":
        return shift_months(value, amount)
    if unit == "year":
        return shift_months(value, amount * 12)
    raise ValueError(f"unsupported offset unit: {unit}")


def _assign_time(
    known: dict[str, datetime], event_id: str, value: datetime | None, conflicts: list[str]
) -> None:
    if value is None:
        return
    if event_id in known and known[event_id].date() != value.date():
        conflicts.append(f"absolute_conflict:{event_id}")
    else:
        known[event_id] = value
