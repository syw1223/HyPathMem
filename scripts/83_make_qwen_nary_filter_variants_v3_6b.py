from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from common import read_json, resolve_path, write_json


ROLE_NAMES = (
    "old_state",
    "new_state",
    "preference_value",
    "polarity",
    "state_value",
    "plan_goal",
    "constraint",
    "temporal_scope",
    "reason_or_trigger",
    "exception",
    "context",
)

ALLOWED_BY_TYPE = {
    "change": {"old_state", "new_state", "temporal_scope", "reason_or_trigger", "context"},
    "preference": {"preference_value", "polarity", "temporal_scope", "reason_or_trigger", "exception", "context"},
    "state": {"state_value", "temporal_scope", "reason_or_trigger", "context"},
    "plan_constraint": {"plan_goal", "constraint", "temporal_scope", "reason_or_trigger", "exception", "context"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", default="outputs/nary_v3_6b/qwen30b_calibration_high_recall_maxtok800.json")
    parser.add_argument("--gpt", default="outputs/nary_v3_6b/high_recall_annotations.json")
    parser.add_argument("--output-dir", default="outputs/nary_v3_6b/filter_variants")
    args = parser.parse_args()

    qwen = read_json(resolve_path(args.qwen))
    gpt = read_json(resolve_path(args.gpt))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "qwen_all": qwen_records(qwen, lambda row: accepted(row)),
        "qwen_high_conf": qwen_records(qwen, lambda row: accepted(row) and confidence(row) >= 0.95),
        "qwen_schema_strict": qwen_records(qwen, schema_strict),
        "qwen_type_specific": qwen_records(qwen, type_specific),
        "gpt4o_clean": qwen_records(gpt, lambda row: accepted(row), source_payload=gpt),
    }
    for name, payload in variants.items():
        path = output_dir / f"{name}.json"
        write_json(payload, path)
        print(f"{name}: records={payload['count']} accepted={payload['accepted']} -> {path}")


def qwen_records(payload: dict, predicate, source_payload: dict | None = None) -> dict:
    source_payload = source_payload or payload
    records = []
    for row in payload.get("records", []):
        if predicate(row):
            records.append(row)
    usage = {}
    for row in records:
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    return {
        "candidates": source_payload.get("candidates", ""),
        "model": source_payload.get("model", ""),
        "prompt_version": source_payload.get("prompt_version", ""),
        "filter_variant": "derived",
        "count": len(records),
        "target_count": source_payload.get("target_count", source_payload.get("count", len(records))),
        "accepted": sum(bool((row.get("annotation") or {}).get("accept")) for row in records),
        "usage": usage,
        "records": records,
    }


def accepted(row: dict) -> bool:
    return bool((row.get("annotation") or {}).get("accept"))


def confidence(row: dict) -> float:
    try:
        return float((row.get("annotation") or {}).get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def relation_type(row: dict) -> str:
    return str((row.get("annotation") or {}).get("relation_type") or "").lower()


def source(row: dict) -> str:
    return str((row.get("candidate") or {}).get("candidate_source") or "")


def filled_roles(row: dict) -> set[str]:
    roles = (row.get("annotation") or {}).get("roles") or {}
    output = set()
    for role_name in ROLE_NAMES:
        role = roles.get(role_name) or {}
        if isinstance(role, dict) and str(role.get("value", "")).strip() and role.get("fact_ids"):
            output.add(role_name)
    return output


def cited_facts(row: dict) -> set[str]:
    roles = (row.get("annotation") or {}).get("roles") or {}
    output = set()
    for role in roles.values():
        if isinstance(role, dict):
            output.update(str(fact_id) for fact_id in role.get("fact_ids", []))
    return output


def schema_strict(row: dict) -> bool:
    if not accepted(row):
        return False
    typ = relation_type(row)
    allowed = ALLOWED_BY_TYPE.get(typ)
    if not allowed:
        return False
    roles = filled_roles(row)
    facts = cited_facts(row)
    if roles - allowed:
        return False
    if not fact_count_ok(typ, len(facts)):
        return False
    if len(roles) > max_role_count(typ):
        return False
    return required_roles(typ).issubset(roles)


def type_specific(row: dict) -> bool:
    if not accepted(row):
        return False
    typ = relation_type(row)
    src = source(row)
    roles = filled_roles(row)
    facts = cited_facts(row)
    if not required_roles(typ).issubset(roles):
        return False
    if len(facts) > 4 and typ in {"preference", "state"}:
        return False
    if typ == "preference":
        return src in {"event", "entity_aspect"} and len(roles) <= 4
    if typ == "state":
        return src in {"event", "entity_aspect", "update_context"} and len(roles) <= 3
    if typ == "plan_constraint":
        return src in {"event", "entity_aspect", "adjacent_event"} and len(roles) <= 5
    if typ == "change":
        return src in {"event", "adjacent_event", "update_context"} and 2 <= len(facts) <= 4 and len(roles) <= 5
    return False


def required_roles(typ: str) -> set[str]:
    if typ == "change":
        return {"old_state", "new_state"}
    if typ == "preference":
        return {"preference_value", "polarity"}
    if typ == "state":
        return {"state_value"}
    if typ == "plan_constraint":
        return {"plan_goal"}
    return set()


def fact_count_ok(typ: str, count: int) -> bool:
    if typ == "change":
        return 2 <= count <= 4
    if typ == "preference":
        return 1 <= count <= 4
    if typ == "state":
        return 1 <= count <= 4
    if typ == "plan_constraint":
        return 2 <= count <= 5
    return False


def max_role_count(typ: str) -> int:
    if typ == "change":
        return 5
    if typ == "preference":
        return 5
    if typ == "state":
        return 4
    if typ == "plan_constraint":
        return 5
    return 0


if __name__ == "__main__":
    main()
