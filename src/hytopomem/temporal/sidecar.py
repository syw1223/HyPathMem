from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta
from typing import Any, Iterable

from hytopomem.temporal.schema import TemporalConstraint, TemporalEventRecord


NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
NUMBER = r"(?:\d+(?:\.\d+)?|" + "|".join(NUMBER_WORDS) + r")"
UNIT = r"hours?|days?|weeks?|months?|years?"


class TemporalSidecarBuilder:
    """Build query-local event-time records without mutating the memory graph."""

    def build(self, evidence_units: Iterable[dict[str, Any]]) -> tuple[list[TemporalEventRecord], list[TemporalConstraint]]:
        records: list[TemporalEventRecord] = []
        constraints: list[TemporalConstraint] = []
        for unit in evidence_units:
            quote = direct_quote(unit)
            # Normalize the sentence-level FACT. The full RAW quote remains
            # provenance but can contain unrelated temporal expressions.
            text = str(unit.get("normalized_claim") or "")
            mentioned = parse_datetime(str(quote.get("message_time") or unit.get("message_time") or ""))
            expressions = relative_expressions(text)
            if not expressions:
                records.append(base_record(unit, quote, mentioned, index=0))
                continue
            for index, expression in enumerate(expressions):
                record = base_record(unit, quote, mentioned, index=index)
                record.relative_expression = expression["text"]
                record.offset_value = expression["value"]
                record.offset_unit = expression["unit"]
                record.offset_direction = expression["direction"]
                record.anchor = expression["anchor"]
                record.granularity = expression["unit"]
                if mentioned and expression["anchor"] == "mentioned_at":
                    occurred = safe_shift_datetime(
                        mentioned,
                        expression["value"] * expression["direction"],
                        expression["unit"],
                    )
                    if occurred is not None:
                        record.occurred_start = occurred.isoformat()
                        record.normalization_status = "resolved"
                        record.time_confidence = 0.96
                    else:
                        record.normalization_status = "symbolic"
                        record.time_confidence = 0.35
                else:
                    record.normalization_status = "symbolic"
                    record.time_confidence = 0.78
                records.append(record)
                constraints.append(
                    TemporalConstraint(
                        constraint_id=f"constraint:{record.event_id}",
                        kind="OFFSET",
                        left_event_id=record.event_id,
                        right_anchor=expression["anchor"],
                        offset_value=expression["value"],
                        offset_unit=expression["unit"],
                        direction=expression["direction"],
                        expression=expression["constraint"],
                    )
                )
        return records, constraints


def base_record(unit: dict[str, Any], quote: dict[str, Any], mentioned: datetime | None, *, index: int) -> TemporalEventRecord:
    unit_id = str(unit.get("unit_id") or "")
    return TemporalEventRecord(
        event_id=f"temporal:{unit_id}:{index}",
        unit_id=unit_id,
        fact_ids=[unit_id.removeprefix("unit:")],
        raw_ids=[str(value) for value in unit.get("raw_message_ids") or []],
        session_id=str(unit.get("session_id") or quote.get("session_id") or ""),
        event=str(unit.get("normalized_claim") or ""),
        mentioned_at=mentioned.isoformat() if mentioned else None,
        source_quote=str(quote.get("text") or unit.get("normalized_claim") or ""),
        speaker=str(unit.get("speaker") or quote.get("speaker") or ""),
        provenance={
            "rank": (unit.get("metadata") or {}).get("rank"),
            "raw_message_id": quote.get("message_id"),
            "topology_score": unit.get("topology_score", 0.0),
            "ce_score": unit.get("ce_score", 0.0),
        },
    )


def direct_quote(unit: dict[str, Any]) -> dict[str, Any]:
    quotes = list(unit.get("raw_quotes") or [])
    return next((quote for quote in quotes if quote.get("support_kind") == "direct"), quotes[0] if quotes else {})


def relative_expressions(text: str) -> list[dict[str, Any]]:
    lowered = text.lower()
    matches: list[dict[str, Any]] = []
    patterns = [
        (rf"\b(?:exactly\s+|about\s+|around\s+)?({NUMBER})\s+({UNIT})\s+ago\b", "mentioned_at", -1),
        (rf"\b({NUMBER})\s+({UNIT})\s+(?:in advance|earlier|before)\b", "related_event", -1),
        (rf"\b({NUMBER})\s+({UNIT})\s+(?:later|after)\b", "related_event", 1),
    ]
    occupied: list[tuple[int, int]] = []
    for pattern, anchor, direction in patterns:
        for match in re.finditer(pattern, lowered):
            if any(start < match.end() and match.start() < end for start, end in occupied):
                continue
            occupied.append(match.span())
            value = parse_number(match.group(1))
            unit = match.group(2).rstrip("s")
            sign = "-" if direction < 0 else "+"
            matches.append(
                {
                    "text": match.group(0),
                    "value": value,
                    "unit": unit,
                    "direction": direction,
                    "anchor": anchor,
                    "constraint": f"event_time = {anchor} {sign} {format_number(value)} {unit}(s)",
                    "position": match.start(),
                }
            )
    shortcuts = [
        (r"\byesterday\b", 1, "day"),
        (r"\blast week\b", 1, "week"),
        (r"\blast month\b", 1, "month"),
        (r"\blast year\b", 1, "year"),
    ]
    for pattern, value, unit in shortcuts:
        for match in re.finditer(pattern, lowered):
            if any(start < match.end() and match.start() < end for start, end in occupied):
                continue
            matches.append(
                {
                    "text": match.group(0), "value": value, "unit": unit, "direction": -1,
                    "anchor": "mentioned_at", "constraint": f"event_time = mentioned_at - {value} {unit}(s)",
                    "position": match.start(),
                }
            )
    return sorted(matches, key=lambda item: item["position"])


def parse_datetime(value: str) -> datetime | None:
    cleaned = re.sub(r"\s*\([A-Za-z]{3}\)\s*", " ", value).strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def shift_datetime(value: datetime, amount: float, unit: str) -> datetime:
    if unit == "hour":
        return value + timedelta(hours=amount)
    if unit == "day":
        return value + timedelta(days=amount)
    if unit == "week":
        return value + timedelta(weeks=amount)
    months = int(amount * (12 if unit == "year" else 1))
    month_index = value.year * 12 + value.month - 1 + months
    year, month0 = divmod(month_index, 12)
    month = month0 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def safe_shift_datetime(value: datetime, amount: float, unit: str) -> datetime | None:
    try:
        shifted = shift_datetime(value, amount, unit)
    except (OverflowError, ValueError):
        return None
    return shifted if 1 <= shifted.year <= 9999 else None


def parse_number(value: str) -> float:
    return float(NUMBER_WORDS.get(value.lower(), value))


def format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)
