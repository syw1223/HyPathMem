#!/usr/bin/env python3
"""Diagnose and convert 8.4 non-executable temporal constraints, fail closed."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient
from hytopomem.temporal_time_v8_5 import (
    VERSION,
    compile_enriched_candidate,
    diagnose_non_executable,
    validate_enrichment,
)


EXTRACT_SYSTEM = """You extract temporal constraints from exact conversation quotes.
Return JSON only. Never infer an event date from message time unless the quote explicitly
uses a same-message anchor such as today, yesterday, or N days/weeks/months ago. Never guess.
For each unresolved event, first verify that the quote describes the exact queried event.
Copy time_expression verbatim from the quote. If there is no explicit temporal expression,
return relation NONE and low confidence. Qwen extracts relations; deterministic code computes dates."""

EXTRACT_USER = """QUESTION: {question}
QUESTION_TIME: {question_time}
QUERY_TYPE: {query_type}
UNRESOLVED_OPERANDS_AND_EXACT_RAW:
{operands}
ALLOWED_ANCHOR_EVENTS:
{anchors}

Return:
{{
  "extractions": [
    {{
      "event_id": "exact fact_id",
      "identity_supported": true,
      "time_expression": "verbatim substring from evidence_span",
      "anchor_type": "mentioned_at or event",
      "anchor_event_id": null,
      "relation": "SAME_TIME|BEFORE_OFFSET|AFTER_OFFSET|ABSOLUTE|NONE",
      "offset_value": 0,
      "offset_unit": "day|week|month|year",
      "confidence": 0.0,
      "ambiguity": []
    }}
  ]
}}
"""


def load_verifier139() -> Any:
    path = ROOT / "scripts" / "139_verify_hypathmem_temporal_q4_v0_2.py"
    spec = importlib.util.spec_from_file_location("hytopomem_verifier139_time_v85", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_compiled(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_json(path)["rows"]:
            rows[str(row["question_id"])] = row
    return rows


def select_non_executable(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in payload["rows"]
        if row["audit"].get("time_v_operand_full_coverage")
        and not row["audit"].get("time_v_eligible_for_q4_verifier")
    ]


def audit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [row["conversion_audit"] for row in rows]
    return {
        "num_questions": len(rows),
        "structural_operand_coverage": sum(item["structural_roles_complete"] for item in audits),
        "validated_operand_coverage": sum(item["validated_operand_coverage"] for item in audits),
        "repairable_by_time_enrichment": sum(item["repairable_by_time_enrichment"] for item in audits),
        "failure_funnel": dict(sorted(Counter(item["failure_type"] for item in audits).items())),
        "new_raw_solver_success": sum(
            bool((row.get("enriched_candidate") or {}).get("solution", {}).get("success")) for row in rows
        ),
        "new_pre_verifier_eligible": sum(
            bool((row.get("enriched_candidate") or {}).get("eligible_for_q4_verifier")) for row in rows
        ),
        "new_verifier_safe": sum(bool((row.get("verifier") or {}).get("safe_to_override_d0")) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-v", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["audit", "enrich"], default="audit")
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--reuse-extractions", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("time_v", "output", "reuse_extractions"):
        path = getattr(args, name)
        if path is not None and not path.is_absolute():
            setattr(args, name, ROOT / path)
    args.compiled = [path if path.is_absolute() else ROOT / path for path in args.compiled]

    selected = select_non_executable(read_json(args.time_v))
    if len(selected) != 35:
        raise ValueError(f"expected frozen 35-row conversion set, got {len(selected)}")
    compiled = load_compiled(args.compiled)
    reused = {}
    if args.reuse_extractions is not None:
        reused = {
            str(row["question_id"]): row
            for row in read_json(args.reuse_extractions).get("rows", [])
            if row.get("extraction_called") and row.get("extraction")
        }
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        existing = {row["question_id"]: row for row in read_json(args.output).get("rows", [])}
    elif args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --resume")
    client = None
    verifier139 = None
    if args.mode == "enrich":
        client = OpenAICompatibleChatClient(api_key=args.api_key, base_url=args.base_url, timeout_seconds=180.0)
        verifier139 = load_verifier139()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(selected, start=1):
        qid = str(source["question_id"])
        if qid in existing:
            rows.append(existing[qid])
            continue
        audit = diagnose_non_executable(source)
        record: dict[str, Any] = {
            "question_id": qid,
            "question": source.get("question"),
            "question_date": source.get("question_date"),
            "conversion_audit": audit,
            "extraction_called": False,
            "validated_enrichments": [],
            "rejected_enrichments": [],
            "enriched_candidate": None,
            "verifier": {"called": False, "safe_to_override_d0": False},
        }
        if args.mode == "enrich" and audit["repairable_by_time_enrichment"]:
            assert client is not None and verifier139 is not None
            candidate = source["h5_constraint_candidate"]
            operands = list(candidate.get("operands") or [])
            unresolved = [item for item in operands if not item.get("occurred_at")]
            reused_row = reused.get(qid)
            if reused_row is not None:
                extraction = reused_row["extraction"]
                result = None
                attempts = 0
                record["extraction_reused_from"] = str(args.reuse_extractions)
            else:
                extraction, result, attempts = verifier139.call_json(
                    client,
                    model=args.model,
                    messages=[
                        ChatMessage(role="system", content=EXTRACT_SYSTEM),
                        ChatMessage(
                            role="user",
                            content=EXTRACT_USER.format(
                                question=source.get("question"),
                                question_time=source.get("question_date"),
                                query_type=candidate.get("query_type"),
                                operands=json.dumps(unresolved, ensure_ascii=False, separators=(",", ":")),
                                anchors=json.dumps(operands, ensure_ascii=False, separators=(",", ":")),
                            ),
                        ),
                    ],
                )
            record["extraction_called"] = True
            record["extraction"] = extraction
            record["extraction_usage"] = result.usage if result is not None else reused_row.get("extraction_usage", {})
            record["extraction_elapsed_seconds"] = result.elapsed_seconds if result is not None else 0.0
            record["extraction_attempts"] = attempts
            operand_by_id = {str(item.get("fact_id")): item for item in operands}
            allowed = set(operand_by_id)
            for item in extraction.get("extractions") or []:
                operand = operand_by_id.get(str(item.get("event_id") or ""))
                if operand is None or operand.get("occurred_at"):
                    record["rejected_enrichments"].append({"extraction": item, "reason": "not_unresolved_operand"})
                    continue
                validated, reason = validate_enrichment(item, operand, allowed)
                if validated:
                    record["validated_enrichments"].append(validated)
                else:
                    record["rejected_enrichments"].append({"extraction": item, "reason": reason})
            enriched = compile_enriched_candidate(source, record["validated_enrichments"])
            record["enriched_candidate"] = enriched
            if enriched["eligible_for_q4_verifier"]:
                verifier_system = verifier139.SYSTEM + """
Also enforce query-time alignment and reverse verification. Reject if the answer contains an
unresolved relative term (today, now, currently, yesterday), if an enriched date cannot be
traced to an exact raw time expression and allowed anchor, or if the supplied reverse check fails."""
                verification, verify_result, verify_attempts = verifier139.call_json(
                    client,
                    model=args.model,
                    messages=[
                        ChatMessage(role="system", content=verifier_system),
                        ChatMessage(
                            role="user",
                            content=verifier139.USER.format(
                                question=source.get("question"),
                                question_time=source.get("question_date"),
                                query_type=candidate.get("query_type"),
                                required_roles=json.dumps(candidate.get("required_roles") or [], ensure_ascii=False),
                                ambiguities=json.dumps(compiled[qid].get("ambiguities") or [], ensure_ascii=False),
                                operands=json.dumps(enriched["operands"], ensure_ascii=False, separators=(",", ":")),
                                solution=json.dumps(
                                    {**enriched["solution"], "reverse_verification": enriched["reverse_verification"]},
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            ),
                        ),
                    ],
                )
                checks = [
                    "all_question_operands_present",
                    "identities_match_question",
                    "anchor_bindings_supported",
                    "operation_matches_question",
                    "arithmetic_or_ordering_correct",
                    "answer_fully_entailed",
                    "safe_to_override_d0",
                ]
                safe = all(verification.get(key) is True for key in checks) and verification.get("unsupported_constraints") == []
                record["verifier"] = {
                    "called": True,
                    "safe_to_override_d0": safe,
                    "verification": verification,
                    "usage": verify_result.usage,
                    "elapsed_seconds": verify_result.elapsed_seconds,
                    "attempts": verify_attempts,
                }
        rows.append(record)
        payload = {
            "metadata": {
                "version": VERSION,
                "mode": args.mode,
                "source_8_4": str(args.time_v),
                "model": args.model if args.mode == "enrich" else None,
                "gold_provided_to_extractor_or_verifier": False,
                "packet_reader_enabled": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
            "summary": audit_summary(rows),
            "rows": rows,
        }
        write_json(args.output, payload)
        print(
            f"processed {index}/35 qid={qid} failure={audit['failure_type']} "
            f"repairable={audit['repairable_by_time_enrichment']} "
            f"safe={record['verifier']['safe_to_override_d0']}",
            flush=True,
        )
    print(json.dumps(audit_summary(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
