"""Query-conditioned temporal sidecar for HyPathMem 8.4time_v.

The semantic graph and its Lorentz representations remain immutable.  This
module builds a per-question Euclidean/symbolic temporal view over frozen
evidence packs, closes required operands as a set, and emits constraints for a
deterministic solver.  It deliberately does not inspect benchmark answers or
gold evidence.
"""

from __future__ import annotations

import calendar
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol, Sequence


VERSION = "8.4time_v"
TEMPORAL_RELATIONS = {
    "BEFORE",
    "AFTER",
    "OVERLAPS",
    "SAME_TIME",
    "ANCHOR_OF",
    "UPDATED_BY",
    "CONTRADICTS",
    "VALID_DURING",
    "SUPERSEDES",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RELATIVE_RE = re.compile(
    r"\b(?:today|yesterday|the day before yesterday|"
    r"(?:about\s+)?(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:day|week|month|year)s?\s+ago|last\s+(?:week|month|year))\b",
    re.IGNORECASE,
)
_MONTH_DATE_RE = re.compile(
    r"\b(?:on\s+)?(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,?\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_SAME_DAY_RETURN_RE = re.compile(
    r"\b(?:just\s+)?(?:got\s+back|returned)\s+from\s+(?:an?\s+)?"
    r"(?:guided\s+)?(?:tour|visit|trip|event|concert|exhibit)\b",
    re.IGNORECASE,
)
_START_MARKER_RE = re.compile(r"\b(?:start(?:ed|ing)?|began|beginning)\b", re.IGNORECASE)
_END_MARKER_RE = re.compile(
    r"\b(?:got\s+back|returned|finish(?:ed|ing)?|end(?:ed|ing)?|completed)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
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


class SupplementaryRetriever(Protocol):
    """Optional source of evidence outside the frozen Top50 pack."""

    def retrieve(self, question_id: str, role: str, top_k: int) -> list[dict[str, Any]]: ...


class NullSupplementaryRetriever:
    def retrieve(self, question_id: str, role: str, top_k: int) -> list[dict[str, Any]]:
        del question_id, role, top_k
        return []


@dataclass(frozen=True)
class TemporalPlan:
    query_type: str
    required_roles: tuple[str, ...]
    target_time: str | None
    requires_state_versions: bool
    requires_pairwise_operands: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "required_roles": list(self.required_roles),
            "target_time": self.target_time,
            "requires_state_versions": self.requires_state_versions,
            "requires_pairwise_operands": self.requires_pairwise_operands,
        }


def parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = re.sub(r"\s*\([A-Za-z]{3}\)\s*", " ", text).strip()
    normalized = normalized.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for pattern in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, pattern)
        except ValueError:
            continue
    return None


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def shift_months(value: datetime, months: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 + months
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def normalize_relative(expression: str, anchor: datetime | None) -> tuple[datetime | None, str, str]:
    text = re.sub(r"^about\s+", "", expression.strip().lower())
    if not text or anchor is None:
        return None, "symbolic" if text else "unknown", "missing_expression_or_anchor"
    if text == "today":
        return anchor, "exact", "today@mentioned_at"
    if text == "yesterday":
        return anchor - timedelta(days=1), "exact", "yesterday@mentioned_at"
    if text == "the day before yesterday":
        return anchor - timedelta(days=2), "exact", "day_before_yesterday@mentioned_at"
    if text.startswith("last "):
        amount, unit = 1, text.split()[-1]
    else:
        match = re.fullmatch(r"(\w+)\s+(day|week|month|year)s?\s+ago", text)
        if not match:
            return None, "symbolic", "unsupported_relative_expression"
        amount_text, unit = match.groups()
        amount = _NUMBER_WORDS.get(amount_text, int(amount_text) if amount_text.isdigit() else 0)
    if amount <= 0:
        return None, "ambiguous", "invalid_relative_amount"
    maximums = {"day": 36500, "week": 5200, "month": 1200, "year": 100}
    if amount > maximums.get(unit, 0):
        return None, "ambiguous", "relative_amount_out_of_range"
    if unit == "day":
        try:
            return anchor - timedelta(days=amount), "exact", "relative_day@mentioned_at"
        except (OverflowError, ValueError):
            return None, "ambiguous", "relative_date_out_of_range"
    if unit == "week":
        try:
            return anchor - timedelta(weeks=amount), "exact", "relative_week@mentioned_at"
        except (OverflowError, ValueError):
            return None, "ambiguous", "relative_date_out_of_range"
    if unit == "month":
        try:
            return shift_months(anchor, -amount), "exact", "relative_month@mentioned_at"
        except (OverflowError, ValueError):
            return None, "ambiguous", "relative_date_out_of_range"
    if unit == "year":
        try:
            return shift_months(anchor, -12 * amount), "exact", "relative_year@mentioned_at"
        except (OverflowError, ValueError):
            return None, "ambiguous", "relative_date_out_of_range"
    return None, "symbolic", "unsupported_relative_unit"


def normalize_absolute_date(expression: str, anchor: datetime | None) -> tuple[datetime | None, str, str]:
    """Resolve an explicit month/day, borrowing only the year from message time."""
    match = _MONTH_DATE_RE.search(expression)
    if not match:
        return None, "unknown", "missing_absolute_date"
    month_name, day_text, year_text = match.groups()
    if not year_text and anchor is None:
        return None, "symbolic", "missing_year_anchor"
    year = int(year_text) if year_text else anchor.year
    try:
        value = datetime.strptime(f"{month_name} {int(day_text)} {year}", "%B %d %Y")
    except ValueError:
        return None, "ambiguous", "invalid_absolute_date"
    if anchor is not None:
        value = value.replace(hour=anchor.hour, minute=anchor.minute, second=anchor.second)
    return value, "exact", "explicit_month_day"


def plan_query(compiled_row: dict[str, Any]) -> TemporalPlan:
    query_type = str(compiled_row.get("query_type") or "other").lower()
    roles = tuple(str(role) for role in compiled_row.get("required_roles") or [] if str(role).strip())
    target_time = infer_question_target_time(
        str(compiled_row.get("question") or ""),
        parse_datetime(compiled_row.get("question_date")),
    )
    return TemporalPlan(
        query_type=query_type,
        required_roles=roles,
        target_time=iso(target_time),
        requires_state_versions=query_type in {"attribute_at_time", "state_at_time"},
        requires_pairwise_operands=query_type in {"ordering", "elapsed", "duration"} or len(roles) > 1,
    )


def infer_question_target_time(question: str, question_time: datetime | None) -> datetime | None:
    if question_time is None:
        return None
    lowered = question.lower()
    match = _RELATIVE_RE.search(lowered)
    if match:
        # "the Wednesday two months ago" needs calendar/weekday binding,
        # not a blind month subtraction. Leave it symbolic until a planner
        # provides the exact anchor semantics.
        if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered):
            return None
        value, _, _ = normalize_relative(match.group(0), question_time)
        return value
    return None


def build_temporal_sidecar(
    *,
    question_id: str,
    pack: dict[str, Any],
    compiled_row: dict[str, Any],
) -> dict[str, Any]:
    """Build H1 sidecar without mutating the semantic graph or pack."""
    units = list(pack.get("evidence_units") or [])
    trusted = trusted_operand_index(compiled_row)
    nodes: dict[str, dict[str, Any]] = {}
    unit_to_event: dict[str, str] = {}
    for unit in units:
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id:
            continue
        event_id = f"temp:event:{unit_id}"
        unit_to_event[unit_id] = event_id
        trusted_operand = trusted.get(unit_id)
        time_payload = event_time_payload(unit, trusted_operand)
        nodes[event_id] = {
            "node_id": event_id,
            "node_type": "EVENT",
            "semantic_node_id": unit_id,
            "entity": unit.get("entity"),
            "aspect": unit.get("aspect"),
            "text": unit.get("normalized_claim"),
            "mentioned_at": time_payload["mentioned_at"],
            "occurred_start": time_payload["occurred_start"],
            "occurred_end": time_payload["occurred_end"],
            "time_expression": time_payload["time_expression"],
            "anchor_type": time_payload["anchor_type"],
            "anchor_id": time_payload["anchor_id"],
            "granularity": time_payload["granularity"],
            "normalization_status": time_payload["normalization_status"],
            "normalization_trace": time_payload["normalization_trace"],
            "confidence": time_payload["confidence"],
            "raw_message_ids": list(unit.get("raw_message_ids") or []),
            "session_id": unit.get("session_id"),
            "route_sources": list(unit.get("route_sources") or []),
            "ce_score": float(unit.get("ce_score") or 0.0),
        }
    edges: list[dict[str, Any]] = []
    add_query_operand_edges(nodes, edges, compiled_row, unit_to_event)
    add_temporal_pair_edges(nodes, edges, compiled_row, unit_to_event)
    state_nodes = add_state_view(nodes, edges, units, unit_to_event)
    anchor_nodes = add_anchor_view(nodes, edges, compiled_row, unit_to_event)
    return {
        "version": VERSION,
        "question_id": question_id,
        "view": "temporal_sidecar",
        "semantic_graph_mutated": False,
        "nodes": list(nodes.values()),
        "edges": edges,
        "diagnostics": {
            "event_nodes": sum(node["node_type"] == "EVENT" for node in nodes.values()),
            "state_nodes": state_nodes,
            "anchor_nodes": anchor_nodes,
            "relation_counts": relation_counts(edges),
            "resolved_event_times": sum(
                node["node_type"] == "EVENT" and bool(node.get("occurred_start")) for node in nodes.values()
            ),
        },
    }


def trusted_operand_index(compiled_row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for hypothesis in compiled_row.get("hypotheses") or []:
        validation = hypothesis.get("source_validation") or {}
        if not validation.get("eligible_for_normalization"):
            continue
        for operand in hypothesis.get("normalized_operands") or []:
            fact_id = str(operand.get("fact_id") or "")
            if fact_id and fact_id not in index:
                index[fact_id] = operand
    return index


def event_time_payload(unit: dict[str, Any], trusted: dict[str, Any] | None) -> dict[str, Any]:
    quote = first_quote(unit)
    mentioned = parse_datetime(unit.get("message_time") or quote.get("message_time"))
    claim = str(unit.get("normalized_claim") or "")
    expression = str((trusted or {}).get("time_expression") or "")
    if not expression:
        # Prefer the sentence-level claim. A raw quote may contain several
        # events and assigning one event's date to every sentence is unsafe.
        match = _RELATIVE_RE.search(claim)
        expression = match.group(0) if match else ""
    if trusted and trusted.get("occurred_at"):
        occurred = parse_datetime(trusted.get("occurred_at"))
        status = str(trusted.get("normalization_status") or "exact")
        trace = "trusted_q4_operand"
        confidence = 0.98
    elif expression:
        occurred, status, trace = normalize_relative(expression, mentioned)
        confidence = 0.84 if expression.lower().startswith("about ") and occurred else (0.90 if occurred else 0.45)
    elif _MONTH_DATE_RE.search(claim):
        occurred, status, trace = normalize_absolute_date(claim, mentioned)
        confidence = 0.94 if occurred else 0.45
        expression = _MONTH_DATE_RE.search(claim).group(0)
    elif _SAME_DAY_RETURN_RE.search(claim):
        occurred = mentioned
        status = "inferred_day"
        trace = "same_day_return_event@mentioned_at"
        confidence = 0.82 if occurred else 0.45
        expression = "same-day return"
    else:
        occurred = None
        status = "mentioned_only"
        trace = "message_time_not_assumed_as_event_time"
        confidence = 0.40
    return {
        "mentioned_at": iso(mentioned),
        "occurred_start": iso(occurred),
        "occurred_end": None,
        "time_expression": expression or None,
        "anchor_type": str((trusted or {}).get("anchor_type") or ("mentioned_at" if expression else "none")),
        "anchor_id": str((trusted or {}).get("anchor_id") or "") or None,
        "granularity": str((trusted or {}).get("precision") or ("day" if occurred else "unknown")),
        "normalization_status": status,
        "normalization_trace": trace,
        "confidence": confidence,
    }


def first_quote(unit: dict[str, Any]) -> dict[str, Any]:
    quotes = unit.get("raw_quotes") or []
    return quotes[0] if quotes else {}


def add_query_operand_edges(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    compiled_row: dict[str, Any],
    unit_to_event: dict[str, str],
) -> None:
    for hypothesis in compiled_row.get("hypotheses") or []:
        for binding in hypothesis.get("bindings") or []:
            event_id = unit_to_event.get(str(binding.get("fact_id") or ""))
            if not event_id:
                continue
            nodes[event_id].setdefault("query_roles", []).append(str(binding.get("role") or ""))


def add_temporal_pair_edges(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    compiled_row: dict[str, Any],
    unit_to_event: dict[str, str],
) -> None:
    seen: set[tuple[str, str, str]] = set()
    for hypothesis in compiled_row.get("hypotheses") or []:
        event_ids = [
            unit_to_event.get(str(binding.get("fact_id") or ""))
            for binding in hypothesis.get("bindings") or []
        ]
        event_ids = [event_id for event_id in event_ids if event_id in nodes]
        for left_index, left_id in enumerate(event_ids):
            for right_id in event_ids[left_index + 1 :]:
                left_time = parse_datetime(nodes[left_id].get("occurred_start"))
                right_time = parse_datetime(nodes[right_id].get("occurred_start"))
                if left_time is None or right_time is None:
                    continue
                if left_time == right_time:
                    add_edge(edges, seen, left_id, right_id, "SAME_TIME", 0.99, "query_operand_pair")
                elif left_time < right_time:
                    add_edge(edges, seen, left_id, right_id, "BEFORE", 0.99, "query_operand_pair")
                    add_edge(edges, seen, right_id, left_id, "AFTER", 0.99, "query_operand_pair")
                else:
                    add_edge(edges, seen, right_id, left_id, "BEFORE", 0.99, "query_operand_pair")
                    add_edge(edges, seen, left_id, right_id, "AFTER", 0.99, "query_operand_pair")


def add_state_view(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    units: Sequence[dict[str, Any]],
    unit_to_event: dict[str, str],
) -> int:
    states_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        metadata = unit.get("metadata") or {}
        if not (
            str(metadata.get("card_type") or "").lower() == "state"
            or str(metadata.get("nary_role") or "").lower() in {"old_state", "new_state"}
            or str(unit.get("claim_type") or "").lower() == "state"
        ):
            continue
        unit_id = str(unit.get("unit_id") or "")
        event_id = unit_to_event.get(unit_id)
        if not event_id:
            continue
        state_id = f"temp:state:{unit_id}"
        start = nodes[event_id].get("occurred_start") or nodes[event_id].get("mentioned_at")
        state = {
            "node_id": state_id,
            "node_type": "STATE",
            "semantic_node_id": unit_id,
            "entity": str(unit.get("entity") or "unknown"),
            "attribute": str(unit.get("aspect") or "unknown"),
            "value": unit.get("value") or unit.get("normalized_claim"),
            "valid_from": start,
            "valid_to": None,
            "status": unit.get("modality") or unit.get("state_status") or "asserted",
            "source_fact_id": unit_id,
            "source_raw_ids": list(unit.get("raw_message_ids") or []),
            "confidence": 0.85 if nodes[event_id].get("occurred_start") else 0.60,
        }
        nodes[state_id] = state
        states_by_key[(state["entity"].lower(), state["attribute"].lower())].append(state)
        edges.append(edge(state_id, event_id, "VALID_DURING", state["confidence"], "state_source_event"))
    for states in states_by_key.values():
        states.sort(key=lambda item: parse_datetime(item.get("valid_from")) or datetime.max)
        for old, new in zip(states, states[1:]):
            if new.get("valid_from"):
                old["valid_to"] = new["valid_from"]
            edges.append(edge(new["node_id"], old["node_id"], "SUPERSEDES", 0.82, "state_chronology"))
            edges.append(edge(old["node_id"], new["node_id"], "UPDATED_BY", 0.82, "state_chronology"))
            if normalized_value(old.get("value")) != normalized_value(new.get("value")):
                edges.append(edge(old["node_id"], new["node_id"], "CONTRADICTS", 0.65, "state_value_change"))
    return sum(node["node_type"] == "STATE" for node in nodes.values())


def add_anchor_view(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    compiled_row: dict[str, Any],
    unit_to_event: dict[str, str],
) -> int:
    for hypothesis in compiled_row.get("hypotheses") or []:
        for binding in hypothesis.get("bindings") or []:
            expression = str(binding.get("time_expression") or "").strip()
            anchor_id = str(binding.get("anchor_id") or "").strip()
            anchor_type = str(binding.get("anchor_type") or "").strip()
            if not expression or (anchor_type == "mentioned_at" and not anchor_id):
                continue
            event_id = unit_to_event.get(str(binding.get("fact_id") or ""))
            if not event_id:
                continue
            anchor_node_id = f"temp:anchor:{compiled_row.get('question_id')}:{len(nodes)}"
            nodes[anchor_node_id] = {
                "node_id": anchor_node_id,
                "node_type": "ANCHOR",
                "expression": expression,
                "anchor_type": anchor_type or "event",
                "anchor_semantic_id": anchor_id or None,
                "normalization_status": "resolved" if anchor_id else "symbolic",
                "confidence": 0.90 if anchor_id else 0.45,
            }
            edges.append(edge(event_id, anchor_node_id, "ANCHOR_OF", nodes[anchor_node_id]["confidence"], "relative_expression"))
    return sum(node["node_type"] == "ANCHOR" for node in nodes.values())


def add_edge(
    edges: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    src: str,
    dst: str,
    relation: str,
    confidence: float,
    reason: str,
) -> None:
    key = (src, dst, relation)
    if key not in seen:
        edges.append(edge(src, dst, relation, confidence, reason))
        seen.add(key)


def edge(src: str, dst: str, relation: str, confidence: float, reason: str) -> dict[str, Any]:
    if relation not in TEMPORAL_RELATIONS:
        raise ValueError(f"Unsupported temporal relation: {relation}")
    return {
        "edge_id": f"{src}->{relation}->{dst}",
        "src": src,
        "dst": dst,
        "relation": relation,
        "confidence": confidence,
        "reason": reason,
    }


def relation_counts(edges: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in edges:
        counts[str(item["relation"])] += 1
    return dict(sorted(counts.items()))


def close_operands(
    *,
    question_id: str,
    pack: dict[str, Any],
    compiled_row: dict[str, Any],
    sidecar: dict[str, Any],
    supplementary: SupplementaryRetriever | None = None,
    role_top_k: int = 3,
    minimum_role_score: float = 0.12,
) -> dict[str, Any]:
    """H2-H4 set-level role completion with soft temporal constraints."""
    plan = plan_query(compiled_row)
    supplementary = supplementary or NullSupplementaryRetriever()
    units = {str(unit.get("unit_id")): unit for unit in pack.get("evidence_units") or []}
    trusted_roles = trusted_role_bindings(compiled_row)
    candidates: dict[str, list[dict[str, Any]]] = {}
    supplementary_added: list[str] = []
    for role in plan.required_roles:
        scored = [
            role_candidate(role, unit, trusted_roles)
            for unit in units.values()
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        viable = [item for item in scored if item["score"] >= minimum_role_score][:role_top_k]
        if not viable:
            for unit in supplementary.retrieve(question_id, role, role_top_k):
                unit_id = str(unit.get("unit_id") or "")
                if not unit_id:
                    continue
                units[unit_id] = unit
                supplementary_added.append(unit_id)
                candidate = role_candidate(role, unit, trusted_roles)
                candidate["source"] = "supplementary"
                viable.append(candidate)
        candidates[role] = viable
    selected = select_role_set(plan.required_roles, candidates)
    selected = expand_single_role_duration_pair(plan, candidates, units, selected)
    selected_ids = [item["unit_id"] for item in selected.values()]
    node_by_semantic = {
        str(node.get("semantic_node_id")): node
        for node in sidecar.get("nodes") or []
        if node.get("semantic_node_id")
    }
    closure_ids = list(selected_ids)
    expansion_reasons: dict[str, list[str]] = defaultdict(list)
    for unit_id in selected_ids:
        node = node_by_semantic.get(unit_id)
        if not node:
            continue
        if node.get("anchor_id") and node["anchor_id"] in node_by_semantic:
            anchor_unit = str(node_by_semantic[node["anchor_id"]].get("semantic_node_id"))
            if anchor_unit not in closure_ids:
                closure_ids.append(anchor_unit)
                expansion_reasons[anchor_unit].append("anchor_expansion")
    if plan.requires_state_versions:
        selected_states = state_versions_for_selection(sidecar, selected_ids)
        for unit_id in selected_states:
            if unit_id not in closure_ids:
                closure_ids.append(unit_id)
                expansion_reasons[unit_id].append("state_version_expansion")
    selected_base_roles = {
        role.rsplit(":", 1)[0] if role.endswith((":start", ":end")) else role
        for role, item in selected.items()
        if item
    }
    covered_roles = [role for role in plan.required_roles if role in selected_base_roles]
    consistency = temporal_consistency(sidecar, closure_ids)
    return {
        "version": VERSION,
        "plan": plan.as_dict(),
        "role_candidates": candidates,
        "selected_by_role": selected,
        "selected_unit_ids": selected_ids,
        "closure_unit_ids": closure_ids,
        "expansion_reasons": dict(expansion_reasons),
        "covered_roles": covered_roles,
        "missing_roles": [role for role in plan.required_roles if role not in covered_roles],
        "operand_full_coverage": len(covered_roles) == len(plan.required_roles) and bool(plan.required_roles),
        "temporal_consistency": consistency,
        "supplementary_retrieval_attempted": bool(supplementary_added),
        "supplementary_added_unit_ids": supplementary_added,
        "set_score": set_score(selected, consistency, len(plan.required_roles)),
    }


def expand_single_role_duration_pair(
    plan: TemporalPlan,
    candidates: dict[str, list[dict[str, Any]]],
    units: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Turn one trip/activity role into explicit start/end operands when grounded."""
    if plan.query_type != "duration" or len(plan.required_roles) != 1:
        return selected
    role = plan.required_roles[0]
    starts: list[dict[str, Any]] = []
    ends: list[dict[str, Any]] = []
    for candidate in candidates.get(role) or []:
        unit = units.get(str(candidate.get("unit_id") or "")) or {}
        text = " ".join(
            str(value or "")
            for value in (unit.get("normalized_claim"), first_quote(unit).get("text"))
        )
        if _START_MARKER_RE.search(text):
            starts.append(candidate)
        if _END_MARKER_RE.search(text):
            ends.append(candidate)
    pair = next(
        (
            (start, end)
            for start in starts
            for end in ends
            if start.get("unit_id") != end.get("unit_id")
        ),
        None,
    )
    if pair is None:
        return selected
    return {f"{role}:start": pair[0], f"{role}:end": pair[1]}


def trusted_role_bindings(compiled_row: dict[str, Any]) -> dict[tuple[str, str], bool]:
    trusted: dict[tuple[str, str], bool] = {}
    for hypothesis in compiled_row.get("hypotheses") or []:
        validation = hypothesis.get("source_validation") or {}
        audits = {str(audit.get("role")): audit for audit in validation.get("binding_audits") or []}
        for binding in hypothesis.get("bindings") or []:
            role = str(binding.get("role") or "")
            fact_id = str(binding.get("fact_id") or "")
            audit = audits.get(role) or {}
            if fact_id and audit.get("deterministic_valid") and validation.get("identity_verified"):
                trusted[(role, fact_id)] = True
    return trusted


def role_candidate(
    role: str,
    unit: dict[str, Any],
    trusted_roles: dict[tuple[str, str], bool],
) -> dict[str, Any]:
    unit_id = str(unit.get("unit_id") or "")
    text = " ".join(
        str(value or "")
        for value in (
            unit.get("normalized_claim"),
            unit.get("entity"),
            unit.get("aspect"),
            unit.get("value"),
            first_quote(unit).get("text"),
        )
    )
    lexical = token_f1(role, text)
    ce = float(unit.get("ce_score") or 0.0)
    ce_soft = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, ce))))
    trusted = trusted_roles.get((role, unit_id), False)
    score = min(1.0, 0.72 * lexical + 0.08 * ce_soft + (0.35 if trusted else 0.0))
    return {
        "unit_id": unit_id,
        "score": score,
        "lexical_score": lexical,
        "ce_soft_score": ce_soft,
        "trusted_q4_binding": trusted,
        "source": "top50",
    }


def token_f1(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_RE.findall(left.lower()))
    right_tokens = set(_TOKEN_RE.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if not overlap:
        return 0.0
    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    return 2 * precision * recall / (precision + recall)


def select_role_set(
    roles: Sequence[str], candidates: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for role in sorted(roles, key=lambda item: len(candidates.get(item, []))):
        options = candidates.get(role) or []
        choice = next((item for item in options if item["unit_id"] not in used), None)
        if choice is None and len(roles) == 1 and options:
            choice = options[0]
        if choice:
            selected[role] = choice
            used.add(choice["unit_id"])
    return selected


def state_versions_for_selection(sidecar: dict[str, Any], selected_ids: Sequence[str]) -> list[str]:
    nodes = sidecar.get("nodes") or []
    selected_states = [
        node for node in nodes
        if node.get("node_type") == "STATE" and node.get("semantic_node_id") in selected_ids
    ]
    keys = {(str(node.get("entity")).lower(), str(node.get("attribute")).lower()) for node in selected_states}
    return [
        str(node.get("semantic_node_id"))
        for node in nodes
        if node.get("node_type") == "STATE"
        and (str(node.get("entity")).lower(), str(node.get("attribute")).lower()) in keys
    ]


def temporal_consistency(sidecar: dict[str, Any], selected_ids: Sequence[str]) -> dict[str, Any]:
    selected = set(selected_ids)
    node_to_semantic = {
        str(node.get("node_id")): str(node.get("semantic_node_id") or "")
        for node in sidecar.get("nodes") or []
    }
    conflicts = []
    for item in sidecar.get("edges") or []:
        if item.get("relation") != "CONTRADICTS":
            continue
        if node_to_semantic.get(str(item.get("src"))) in selected and node_to_semantic.get(str(item.get("dst"))) in selected:
            conflicts.append(item.get("edge_id"))
    return {"consistent": not conflicts, "conflict_edge_ids": conflicts}


def set_score(
    selected: dict[str, dict[str, Any]], consistency: dict[str, Any], required_role_count: int
) -> float:
    relevance = sum(item["score"] for item in selected.values())
    coverage = len(selected) / required_role_count if required_role_count else 0.0
    conflict_penalty = 1.0 if consistency.get("conflict_edge_ids") else 0.0
    return relevance + 1.5 * coverage - conflict_penalty


def compile_constraint_candidate(
    *,
    compiled_row: dict[str, Any],
    pack: dict[str, Any],
    sidecar: dict[str, Any],
    closure: dict[str, Any],
) -> dict[str, Any]:
    """Compile the closed evidence set into a minimal executable subgraph."""
    protected = protected_q4_candidate(compiled_row)
    if protected is not None:
        protected_ids = [str(item.get("fact_id") or "") for item in protected["operands"]]
        return {
            "version": VERSION,
            "question_id": compiled_row.get("question_id"),
            "query_type": compiled_row.get("query_type"),
            "required_roles": list(compiled_row.get("required_roles") or []),
            "operands": protected["operands"],
            "constraint_subgraph": minimal_subgraph(sidecar, protected_ids),
            "solution": protected["solution"],
            "candidate_source": "protected_source_q4",
            "eligible_for_q4_verifier": True,
        }
    unit_by_id = {str(unit.get("unit_id")): unit for unit in pack.get("evidence_units") or []}
    event_by_semantic = {
        str(node.get("semantic_node_id")): node
        for node in sidecar.get("nodes") or []
        if node.get("node_type") == "EVENT"
    }
    operands = []
    for role, selection in closure.get("selected_by_role", {}).items():
        unit_id = str(selection.get("unit_id") or "")
        unit = unit_by_id.get(unit_id) or {}
        event = event_by_semantic.get(unit_id) or {}
        operands.append(
            {
                "role": role,
                # The query role names the event being compared. Generic pack
                # entities such as "user" are provenance, not answer labels.
                "identity": role or unit.get("aspect") or unit.get("entity"),
                "fact_id": unit_id,
                "raw_id": (unit.get("raw_message_ids") or [None])[0],
                "evidence_span": first_quote(unit).get("text") or unit.get("normalized_claim"),
                "mentioned_at": event.get("mentioned_at"),
                "occurred_at": event.get("occurred_start"),
                "normalization_status": event.get("normalization_status"),
                "confidence": event.get("confidence"),
            }
        )
    solution = solve_constraint(
        str(compiled_row.get("question") or ""),
        str(compiled_row.get("query_type") or "other"),
        operands,
        parse_datetime(compiled_row.get("question_date")),
        sidecar,
        closure,
    )
    return {
        "version": VERSION,
        "question_id": compiled_row.get("question_id"),
        "query_type": compiled_row.get("query_type"),
        "required_roles": list(compiled_row.get("required_roles") or []),
        "operands": operands,
        "constraint_subgraph": minimal_subgraph(sidecar, closure.get("closure_unit_ids") or []),
        "solution": solution,
        "candidate_source": "8.4time_v_operand_closure",
        "eligible_for_q4_verifier": bool(
            closure.get("operand_full_coverage")
            and closure.get("temporal_consistency", {}).get("consistent")
            and solution.get("success")
            and compiled_row.get("pre_verifier", {}).get("restrictive_clause_grounded", True)
        ),
    }


def protected_q4_candidate(compiled_row: dict[str, Any]) -> dict[str, Any] | None:
    """Preserve an already executable Q4 candidate before adding retrieval risk."""
    if not compiled_row.get("pre_verifier", {}).get("eligible_to_call_solution_verifier"):
        return None
    eligible = [
        hypothesis
        for hypothesis in compiled_row.get("hypotheses") or []
        if hypothesis.get("source_validation", {}).get("eligible_for_normalization")
        and hypothesis.get("solution", {}).get("success")
    ]
    if not eligible:
        return None
    answers = {str(item.get("solution", {}).get("answer") or "").strip() for item in eligible}
    if len(answers) != 1 or not next(iter(answers), ""):
        return None
    first = eligible[0]
    return {
        "operands": list(first.get("normalized_operands") or []),
        "solution": dict(first.get("solution") or {}),
    }


def solve_constraint(
    question: str,
    query_type: str,
    operands: list[dict[str, Any]],
    question_time: datetime | None,
    sidecar: dict[str, Any],
    closure: dict[str, Any],
) -> dict[str, Any]:
    times = [(operand, parse_datetime(operand.get("occurred_at"))) for operand in operands]
    lowered = question.lower()
    if query_type == "ordering" and len(times) >= 2 and all(value for _, value in times):
        ordered = sorted(times, key=lambda item: item[1])
        return {"success": True, "answer": f"{ordered[0][0]['identity']} was first", "operation": "ORDERING"}
    if query_type == "duration" and len(times) > 2:
        return {
            "success": False,
            "answer": None,
            "operation": "DURATION_SUM",
            "failure_reason": "aggregate_duration_requires_per_item_durations",
        }
    if query_type in {"elapsed", "duration"} and len(times) == 2 and all(value for _, value in times):
        start, end = sorted((times[0][1], times[1][1]))
        if "day" in lowered:
            calendar_days = (end.date() - start.date()).days
            if query_type == "duration" and re.search(r"\b(?:spend|spent|stay|stayed)\b", lowered):
                calendar_days += 1
            return {
                "success": True,
                "answer": f"{calendar_days} days",
                "operation": "DURATION" if query_type == "duration" else "ELAPSED",
                "calendar_day_semantics": "inclusive" if query_type == "duration" else "between",
            }
        delta = abs((times[1][1] - times[0][1]).total_seconds())
        return duration_answer(delta, lowered)
    if query_type == "recency" and question_time and times and times[0][1]:
        delta = max(0.0, (question_time - times[0][1]).total_seconds())
        return duration_answer(delta, lowered, operation="RECENCY")
    if query_type in {"attribute_at_time", "state_at_time"}:
        state = solve_state_at_time(sidecar, closure, parse_datetime(closure.get("plan", {}).get("target_time")))
        if state:
            return {"success": True, "answer": state.get("value"), "operation": "STATE_AT_TIME", "state_id": state.get("node_id")}
    return {"success": False, "answer": None, "operation": query_type.upper(), "failure_reason": "non_executable_constraints"}


def duration_answer(seconds: float, question: str, operation: str = "ELAPSED") -> dict[str, Any]:
    days = seconds / 86400.0
    if "month" in question:
        value, unit = round(days / 30.0), "months"
    elif "week" in question:
        value, unit = round(days / 7.0), "weeks"
    elif "year" in question:
        value, unit = round(days / 365.0), "years"
    else:
        value, unit = round(days), "days"
    return {"success": True, "answer": f"{value} {unit}", "operation": operation, "delta_seconds": seconds}


def solve_state_at_time(
    sidecar: dict[str, Any], closure: dict[str, Any], target: datetime | None
) -> dict[str, Any] | None:
    if target is None:
        return None
    selected = set(closure.get("closure_unit_ids") or [])
    candidates = []
    for node in sidecar.get("nodes") or []:
        if node.get("node_type") != "STATE" or node.get("semantic_node_id") not in selected:
            continue
        if float(node.get("confidence") or 0.0) < 0.80:
            continue
        start = parse_datetime(node.get("valid_from"))
        end = parse_datetime(node.get("valid_to"))
        if start and start <= target and (end is None or target < end):
            candidates.append(node)
    return max(candidates, key=lambda node: parse_datetime(node.get("valid_from")) or datetime.min, default=None)


def minimal_subgraph(sidecar: dict[str, Any], semantic_ids: Sequence[str]) -> dict[str, Any]:
    selected = set(semantic_ids)
    nodes = [
        node for node in sidecar.get("nodes") or []
        if node.get("semantic_node_id") in selected or node.get("node_type") == "ANCHOR"
    ]
    node_ids = {node["node_id"] for node in nodes}
    edges = [
        item for item in sidecar.get("edges") or []
        if item.get("src") in node_ids and item.get("dst") in node_ids
    ]
    return {"nodes": nodes, "edges": edges}


def normalized_value(value: Any) -> str:
    return " ".join(_TOKEN_RE.findall(str(value or "").lower()))
