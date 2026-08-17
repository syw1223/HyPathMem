from __future__ import annotations

import re
from datetime import datetime

from hytopomem.temporal.schema import TemporalEventRecord, TemporalQueryPlan, TemporalSolution
from hytopomem.temporal.sidecar import parse_datetime


STOP = {"the", "a", "an", "i", "my", "did", "was", "were", "have", "had", "to", "in", "at", "on", "for"}


class TemporalSolver:
    def solve(self, plan: TemporalQueryPlan, records: list[TemporalEventRecord]) -> TemporalSolution:
        selected = select_operands(plan, records)
        coverage = len(selected) / len(plan.required_roles) if plan.required_roles else 0.0
        if coverage < 1.0:
            return TemporalSolution(
                operator=plan.operator,
                operand_coverage=coverage,
                failure_reason="required_temporal_operands_missing",
                selected_event_ids=[record.event_id for record in selected.values()],
            )
        if plan.operator == "ELAPSED_TO_QUESTION":
            return self._elapsed(plan, selected, records)
        if plan.operator == "CALENDAR_DIFF":
            return self._difference(plan, selected)
        if plan.operator == "SORT_EVENTS":
            return self._ordering(plan, selected)
        if plan.operator == "RESOLVE_DATE":
            return self._date(plan, selected)
        return TemporalSolution(
            operator=plan.operator,
            operand_coverage=coverage,
            failure_reason="operator_not_yet_deterministic",
            selected_event_ids=[record.event_id for record in selected.values()],
        )

    def _elapsed(
        self,
        plan: TemporalQueryPlan,
        selected: dict[str, TemporalEventRecord],
        records: list[TemporalEventRecord],
    ) -> TemporalSolution:
        target = selected["target_event"]
        trace: list[str] = []
        question_time = parse_datetime(plan.question_time or "")
        if target.occurred_start and question_time:
            occurred = datetime.fromisoformat(target.occurred_start)
            unit = plan.answer_unit or target.offset_unit or "day"
            value = datetime_difference(occurred, question_time, unit)
            trace.extend(
                [
                    f"normalized target occurrence to {target.occurred_start}",
                    f"computed elapsed time from target occurrence to question_time={plan.question_time}",
                ]
            )
        elif target.anchor == "mentioned_at" and target.offset_value is not None:
            value = target.offset_value
            unit = target.offset_unit
            trace.append(f"target is {format_value(value)} {unit}(s) before its mention anchor")
        elif target.anchor == "related_event" and target.offset_value is not None:
            anchors = [record for record in records if record.anchor == "mentioned_at" and record.offset_value is not None]
            anchors.sort(key=lambda record: role_score(record, " ".join(role.description for role in plan.required_roles)), reverse=True)
            if not anchors:
                return failed(plan, selected, "symbolic_anchor_not_found")
            anchor = anchors[0]
            value = convert(anchor.offset_value, anchor.offset_unit, plan.answer_unit or target.offset_unit)
            value += convert(target.offset_value, target.offset_unit, plan.answer_unit or target.offset_unit)
            unit = plan.answer_unit or target.offset_unit
            trace.extend(
                [
                    f"related event is {format_value(anchor.offset_value)} {anchor.offset_unit}(s) before question anchor",
                    f"target is {format_value(target.offset_value)} {target.offset_unit}(s) before related event",
                ]
            )
        else:
            return failed(plan, selected, "target_has_no_elapsed_constraint")
        return success(plan, selected, value=value, unit=unit, answer=f"{format_value(value)} {plural(unit, value)}", trace=trace)

    def _difference(self, plan: TemporalQueryPlan, selected: dict[str, TemporalEventRecord]) -> TemporalSolution:
        values = list(selected.values())
        if len(values) == 1 and values[0].duration_value is not None:
            value, unit = values[0].duration_value, values[0].duration_unit
        elif len(values) >= 2 and all(record.occurred_start for record in values[:2]):
            left = datetime.fromisoformat(values[0].occurred_start or "")
            right = datetime.fromisoformat(values[1].occurred_start or "")
            unit = plan.answer_unit or common_unit(values[:2])
            value = datetime_difference(left, right, unit)
        elif len(values) >= 2 and all(record.offset_value is not None for record in values[:2]):
            unit = plan.answer_unit or common_unit(values[:2])
            left = convert(values[0].offset_value or 0.0, values[0].offset_unit, unit)
            right = convert(values[1].offset_value or 0.0, values[1].offset_unit, unit)
            value = abs(left - right)
        else:
            return failed(plan, selected, "event_times_not_resolved")
        trace = [f"computed absolute difference between event_A and event_B in {unit}s"]
        return success(plan, selected, value=value, unit=unit, answer=f"{format_value(value)} {plural(unit, value)}", trace=trace)

    def _ordering(self, plan: TemporalQueryPlan, selected: dict[str, TemporalEventRecord]) -> TemporalSolution:
        values = list(selected.items())
        if not all(record.occurred_start for _, record in values):
            return failed(plan, selected, "event_times_not_resolved")
        ordered = sorted(values, key=lambda item: item[1].occurred_start or "")
        labels = [role for role, _ in ordered]
        descriptions = {role.role: role.description for role in plan.required_roles}
        ordered_events = [descriptions[label] for label in labels]
        answer = ordered_events[0] if plan.subtype == "first_of_candidates" else " then ".join(ordered_events)
        return success(
            plan,
            selected,
            value=ordered_events,
            unit=None,
            answer=answer,
            trace=["sorted resolved occurrence times ascending"],
        )

    def _date(self, plan: TemporalQueryPlan, selected: dict[str, TemporalEventRecord]) -> TemporalSolution:
        target = selected["target_event"]
        if not target.occurred_start:
            return failed(plan, selected, "target_event_time_not_resolved")
        value = datetime.fromisoformat(target.occurred_start).date().isoformat()
        return success(plan, selected, value=value, unit=None, answer=value, trace=["resolved target event occurrence date"])


def select_operands(plan: TemporalQueryPlan, records: list[TemporalEventRecord]) -> dict[str, TemporalEventRecord]:
    selected: dict[str, TemporalEventRecord] = {}
    used: set[str] = set()
    for role in plan.required_roles:
        ranked = sorted(records, key=lambda record: role_score(record, role.description), reverse=True)
        choice = next((record for record in ranked if record.event_id not in used and role_score(record, role.description) > 0), None)
        if choice:
            selected[role.role] = choice
            used.add(choice.event_id)
    return selected


def role_score(record: TemporalEventRecord, description: str) -> float:
    overlap = role_overlap(record, description)
    temporal = 0.35 if record.relative_expression else 0.0
    resolved = 0.15 if record.normalization_status == "resolved" else 0.0
    rank = float(record.provenance.get("rank") or 100)
    return overlap * 3.0 + temporal + resolved + 0.1 / rank


def role_overlap(record: TemporalEventRecord, description: str) -> float:
    query_terms = terms(description)
    event_terms = terms(record.event)
    return len(query_terms & event_terms) / max(1, len(query_terms))


def terms(text: str) -> set[str]:
    return {stem(token) for token in re.findall(r"[a-z][a-z0-9'-]+", text.lower()) if len(token) > 2 and token not in STOP}


def stem(token: str) -> str:
    irregular = {
        "got": "get", "bought": "buy", "saw": "see", "seen": "see",
        "went": "go", "gone": "go", "met": "meet", "took": "take",
    }
    if token in irregular:
        return irregular[token]
    for suffix in ("ing", "ied", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def convert(value: float, source: str | None, target: str | None) -> float:
    if not source or not target or source == target:
        return value
    days = {"hour": 1 / 24, "day": 1, "week": 7, "month": 30.436875, "year": 365.2425}
    return value * days[source] / days[target]


def common_unit(records: list[TemporalEventRecord]) -> str:
    units = [record.offset_unit for record in records if record.offset_unit]
    return units[0] if units and len(set(units)) == 1 else "day"


def datetime_difference(left: datetime, right: datetime, unit: str) -> float:
    if unit == "hour":
        return abs((right - left).total_seconds()) / 3600
    days = abs((right.date() - left.date()).days)
    if unit == "day":
        return float(days)
    if unit == "week":
        return days / 7
    earlier, later = sorted((left.date(), right.date()))
    months = (later.year - earlier.year) * 12 + later.month - earlier.month
    if later.day < earlier.day:
        months -= 1
    if unit == "month":
        return float(months)
    return months / 12


def success(
    plan: TemporalQueryPlan,
    selected: dict[str, TemporalEventRecord],
    *,
    value: float | str | list[str],
    unit: str | None,
    answer: str,
    trace: list[str],
) -> TemporalSolution:
    solution = TemporalSolution(
        success=True,
        operator=plan.operator,
        answer=answer,
        value=value,
        unit=unit,
        selected_event_ids=[record.event_id for record in selected.values()],
        constraint_satisfaction=1.0,
        operand_coverage=1.0,
        trace=trace,
    )
    descriptions = {role.role: role.description for role in plan.required_roles}
    identity_scores = [role_overlap(record, descriptions.get(role, "")) for role, record in selected.items()]
    solution.trace.append("operand identity overlap=" + ",".join(f"{score:.3f}" for score in identity_scores))
    solution.verified = verify(solution) and bool(identity_scores) and min(identity_scores) >= 0.60
    return solution


def failed(plan: TemporalQueryPlan, selected: dict[str, TemporalEventRecord], reason: str) -> TemporalSolution:
    return TemporalSolution(
        operator=plan.operator,
        operand_coverage=len(selected) / max(1, len(plan.required_roles)),
        selected_event_ids=[record.event_id for record in selected.values()],
        failure_reason=reason,
    )


def verify(solution: TemporalSolution) -> bool:
    if not solution.success or solution.operand_coverage < 1.0 or solution.constraint_satisfaction < 1.0:
        return False
    if isinstance(solution.value, float):
        return solution.value >= 0 and solution.value < 10000
    return bool(solution.value)


def format_value(value: float) -> str:
    rounded = round(value)
    return str(rounded) if abs(value - rounded) < 1e-6 else f"{value:.2f}".rstrip("0").rstrip(".")


def plural(unit: str | None, value: float) -> str:
    base = unit or "unit"
    return base if abs(value - 1) < 1e-9 else base + "s"
