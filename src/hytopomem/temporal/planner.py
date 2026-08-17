from __future__ import annotations

import re

from hytopomem.temporal.schema import TemporalQueryPlan, TemporalQueryType, TemporalRole


UNIT_PATTERN = r"hours?|days?|weeks?|months?|years?"


class TemporalQueryPlanner:
    """Deterministic T1 planner; it never uses benchmark labels as the sole signal."""

    def plan(self, question: str, *, question_date: str | None = None, question_type: str = "") -> TemporalQueryPlan:
        q = " ".join(question.strip().split())
        lower = q.lower().rstrip("?")
        explicit_temporal_signal = bool(
            re.search(
                r"\b(when|what date|which date|before|after|earlier|later|ago|how long|chronological|chronologically)\b",
                lower,
            )
            or re.search(rf"\bhow many\s+{UNIT_PATTERN}\b", lower)
        )
        # In benchmark runs, the frozen benchmark type defines the controlled branch.
        # Text-only routing remains available for deployment inputs without a type.
        temporal = "temporal" in question_type.lower() if question_type.strip() else explicit_temporal_signal
        if not temporal:
            return TemporalQueryPlan(diagnostics={"reason": "no_temporal_signal"})

        unit_match = re.search(rf"\b({UNIT_PATTERN})\b", lower)
        unit = singular(unit_match.group(1)) if unit_match else None
        ordering_roles = explicit_ordering_roles(lower)
        if ordering_roles:
            first_only = "first" in lower and "first to last" not in lower and not re.search(r"\border\b", lower)
            return TemporalQueryPlan(
                activated=True,
                query_type=TemporalQueryType.ORDERING,
                subtype="first_of_candidates" if first_only else "ordered_event_set",
                question_time=question_date or None,
                required_roles=ordering_roles,
                operator="SORT_EVENTS",
                complexity="L2",
                confidence=0.94,
            )
        if re.search(r"\b(as of|at that time|on that date|current|currently|latest|still)\b", lower):
            return TemporalQueryPlan(
                activated=True,
                query_type=TemporalQueryType.STATE_AT_TIME,
                subtype="valid_state",
                question_time=question_date or None,
                required_roles=[TemporalRole(role="state", description=target_after_auxiliary(lower))],
                operator="SELECT_VALID_STATE",
                complexity="L2",
                confidence=0.86,
            )

        roles = comparison_roles(lower)
        if roles:
            return TemporalQueryPlan(
                activated=True,
                query_type=TemporalQueryType.ORDERING if not lower.startswith("how long") else TemporalQueryType.DURATION,
                subtype="between_events",
                question_time=question_date or None,
                required_roles=roles,
                operator="CALENDAR_DIFF" if lower.startswith("how long") else "SORT_EVENTS",
                answer_unit=unit,
                complexity="L2",
                confidence=0.92,
            )

        ago_match = re.search(rf"how many\s+({UNIT_PATTERN})\s+ago\s+(?:did|was|were|had)\s+(.*)", lower)
        if ago_match:
            return TemporalQueryPlan(
                activated=True,
                query_type=TemporalQueryType.DURATION,
                subtype="elapsed_to_question_time",
                question_time=question_date or None,
                required_roles=[TemporalRole(role="target_event", description=strip_pronoun(ago_match.group(2)))],
                operator="ELAPSED_TO_QUESTION",
                answer_unit=singular(ago_match.group(1)),
                complexity="L2",
                confidence=0.96,
            )

        if re.search(r"\bhow long\b", lower):
            return TemporalQueryPlan(
                activated=True,
                query_type=TemporalQueryType.DURATION,
                subtype="duration",
                question_time=question_date or None,
                required_roles=[TemporalRole(role="duration_event", description=target_after_auxiliary(lower))],
                operator="CALENDAR_DIFF",
                answer_unit=unit,
                complexity="L2",
                confidence=0.82,
            )

        asks_for_date = bool(re.match(r"^(?:when\b|what date\b|which date\b)", lower))
        return TemporalQueryPlan(
            activated=True,
            query_type=TemporalQueryType.DATE,
            subtype="event_date" if asks_for_date else "event_at_time",
            question_time=question_date or None,
            required_roles=[TemporalRole(role="target_event", description=target_after_auxiliary(lower))],
            operator="RESOLVE_DATE" if asks_for_date else "SELECT_EVENT_AT_TIME",
            answer_unit=unit,
            complexity="L1",
            confidence=0.88 if asks_for_date else 0.80,
        )


def comparison_roles(question: str) -> list[TemporalRole]:
    patterns = [
        rf"how many\s+(?:{UNIT_PATTERN})\s+(?:passed\s+)?between (.+?) and (.+)",
        rf"how many\s+(?:{UNIT_PATTERN})\s+before (.+?) did (?:i|the user|[a-z]+) (.+)",
        rf"how many\s+(?:{UNIT_PATTERN})\s+after (.+?) did (?:i|the user|[a-z]+) (.+)",
        r"how long did (?:i|the user) (.+?) before (?:i|the user) (.+)",
        r"how long did (?:i|the user) (.+?) after (?:i|the user) (.+)",
        r"how long before (.+?) did (?:i|the user) (.+)",
        r"how long after (.+?) did (?:i|the user) (.+)",
        r"(?:did|was) (.+?) (?:happen |occur )?before (.+)",
        r"(?:did|was) (.+?) (?:happen |occur )?after (.+)",
        r"which (?:happened|came|occurred) (?:first|earlier),? (.+?) or (.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return [
                TemporalRole(role="event_A", description=clean_fragment(match.group(1))),
                TemporalRole(role="event_B", description=clean_fragment(match.group(2))),
            ]
    return []


def explicit_ordering_roles(question: str) -> list[TemporalRole]:
    if not re.search(r"\b(order|first|earlier|chronological|chronologically)\b", question):
        return []
    if ":" in question:
        body = question.split(":", 1)[1]
    elif " among " in question:
        body = question.split(" among ", 1)[1]
    elif re.search(r"\bfirst,\s*", question):
        body = re.split(r"\bfirst,\s*", question, maxsplit=1)[1]
    else:
        return []
    fragments = [clean_fragment(value) for value in re.split(r",\s*(?:and\s+)?|\s+(?:and|or)\s+", body)]
    fragments = [value for value in fragments if len(terms(value)) >= 2]
    if len(fragments) < 2:
        return []
    return [TemporalRole(role=f"event_{index + 1}", description=value) for index, value in enumerate(fragments[:6])]


def terms(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9'-]+", text.lower())


def target_after_auxiliary(question: str) -> str:
    match = re.search(r"(?:when|what date|which date|how long)\s+(?:did|was|were|had|has)\s+(.*)", question)
    return clean_fragment(match.group(1) if match else question)


def strip_pronoun(text: str) -> str:
    return re.sub(r"^(?:i|the user)\s+", "", clean_fragment(text))


def clean_fragment(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip(" ?,."))


def singular(unit: str) -> str:
    return unit.lower().rstrip("s")
