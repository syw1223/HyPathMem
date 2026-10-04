#!/usr/bin/env python3
"""Run E1/E2 evaluations for the independent HyPathMem 8.4time_v branch."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient


PACKET_SYSTEM = """You answer temporal questions from a frozen evidence packet.
Distinguish message time (when something was said) from occurrence time (when it happened).
Use only supported event times and exact evidence. Do not turn an unresolved message time into
an event time. Perform ordering or date arithmetic privately. If the packet is insufficient,
answer exactly: Insufficient evidence. Otherwise give only the concise answer."""

REFUSAL_RE = re.compile(
    r"\b(?:insufficient evidence|not enough (?:information|evidence)|cannot determine|"
    r"can't determine|unable to determine|unknown)\b",
    re.IGNORECASE,
)


def load_qa121() -> ModuleType:
    path = ROOT / "scripts" / "121_run_longmemeval_qa_eval.py"
    spec = importlib.util.spec_from_file_location("hytopomem_qa121_time_v84", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_verifier139() -> ModuleType:
    path = ROOT / "scripts" / "139_verify_hypathmem_temporal_q4_v0_2.py"
    spec = importlib.util.spec_from_file_location("hytopomem_verifier139_time_v84", path)
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


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def source_rows(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_json(path)["rows"]:
            question_id = str(row["question_id"])
            if question_id in rows:
                raise ValueError(f"duplicate compiled question_id: {question_id}")
            rows[question_id] = row
    return rows


def select_e1(current: dict[str, Any], previous: dict[str, Any]) -> list[dict[str, Any]]:
    previous_safe = {
        row["question_id"]
        for row in previous["rows"]
        if row["audit"].get("time_v_eligible_for_q4_verifier")
    }
    return [
        row
        for row in current["rows"]
        if row["audit"].get("time_v_eligible_for_q4_verifier")
        and row["question_id"] not in previous_safe
    ]


def select_e2(current: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in current["rows"]
        if row["audit"].get("time_v_operand_full_coverage")
        and not row["audit"].get("time_v_eligible_for_q4_verifier")
    ]


def failure_reason(row: dict[str, Any]) -> str:
    closure = row["h2_h4_operand_closure"]
    candidate = row["h5_constraint_candidate"]
    solution = candidate.get("solution") or {}
    if not closure.get("temporal_consistency", {}).get("consistent", True):
        return "temporal_conflict"
    if solution.get("failure_reason") == "aggregate_duration_requires_per_item_durations":
        return "aggregate_duration_missing_components"
    query_type = str(candidate.get("query_type") or "")
    operands = candidate.get("operands") or []
    if query_type in {"attribute_at_time", "state_at_time"} and not closure.get("plan", {}).get("target_time"):
        return "unresolved_state_target_time"
    if closure.get("plan", {}).get("requires_pairwise_operands") and len(operands) < 2:
        return "missing_pairwise_operands"
    unresolved = sum(not operand.get("occurred_at") for operand in operands)
    if unresolved:
        return "missing_occurred_at"
    if not solution.get("success"):
        return "unsupported_temporal_operation"
    if not row["audit"].get("time_v_eligible_for_q4_verifier"):
        return "safety_gate_rejected"
    return "none"


def packet_for_reader(row: dict[str, Any]) -> dict[str, Any]:
    closure = row["h2_h4_operand_closure"]
    candidate = row["h5_constraint_candidate"]
    events = {
        str(node.get("semantic_node_id")): node
        for node in row["h1_temporal_sidecar"].get("nodes") or []
        if node.get("node_type") == "EVENT" and node.get("semantic_node_id")
    }
    role_candidates = []
    for role, candidates in (closure.get("role_candidates") or {}).items():
        alternatives = []
        for item in candidates[:3]:
            event = events.get(str(item.get("unit_id") or "")) or {}
            alternatives.append(
                {
                    "fact_id": item.get("unit_id"),
                    "claim": event.get("text"),
                    "mentioned_at": event.get("mentioned_at"),
                    "occurred_at": event.get("occurred_start"),
                    "normalization_status": event.get("normalization_status"),
                    "confidence": event.get("confidence"),
                    "role_score": item.get("score"),
                    "trusted_q4_binding": item.get("trusted_q4_binding"),
                }
            )
        role_candidates.append({"role": role, "alternatives": alternatives})
    operands = []
    for operand in candidate.get("operands") or []:
        operands.append(
            {
                "role": operand.get("role"),
                "identity": operand.get("identity"),
                "fact_id": operand.get("fact_id"),
                "raw_id": operand.get("raw_id"),
                "evidence_quote": str(operand.get("evidence_span") or "")[:4000],
                "mentioned_at": operand.get("mentioned_at"),
                "occurred_at": operand.get("occurred_at"),
                "normalization_status": operand.get("normalization_status"),
                "confidence": operand.get("confidence"),
            }
        )
    return {
        "question_time": row.get("question_date"),
        "query_type": candidate.get("query_type"),
        "required_roles": candidate.get("required_roles"),
        "compiler_failure_reason": failure_reason(row),
        "selected_operands": operands,
        "role_candidates": role_candidates,
        "temporal_relations": candidate.get("constraint_subgraph", {}).get("edges") or [],
        "reader_instruction": "Resolve only what the packet supports; otherwise refuse.",
    }


def is_refusal(prediction: str) -> bool:
    return bool(REFUSAL_RE.search(prediction.strip()))


def classification(d0_correct: bool, packet_correct: bool) -> str:
    if not d0_correct and packet_correct:
        return "fix"
    if d0_correct and not packet_correct:
        return "break"
    if d0_correct and packet_correct:
        return "unchanged_correct"
    return "unchanged_wrong"


def grouped_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["failure_reason"])].append(row)
    result = {}
    for reason, items in sorted(grouped.items()):
        n = len(items)
        result[reason] = {
            "n": n,
            "packet_correct": sum(int(item["judge_correct"]) for item in items),
            "packet_accuracy": sum(int(item["judge_correct"]) for item in items) / n,
            "frozen_d0_correct": sum(int(item["frozen_d0_judge_correct"]) for item in items),
            "frozen_d0_accuracy": sum(int(item["frozen_d0_judge_correct"]) for item in items) / n,
            "fix": sum(item["comparison"] == "fix" for item in items),
            "break": sum(item["comparison"] == "break" for item in items),
            "reader_refusal": sum(item["reader_refusal"] for item in items),
        }
    return result


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    packet_correct = sum(int(row.get("judge_correct") or 0) for row in rows)
    d0_correct = sum(int(row.get("frozen_d0_judge_correct") or 0) for row in rows)
    counts = {name: sum(row.get("comparison") == name for row in rows) for name in (
        "fix", "break", "unchanged_correct", "unchanged_wrong"
    )}
    return {
        "num_questions": n,
        "packet_correct": packet_correct,
        "packet_accuracy": packet_correct / n if n else 0.0,
        "frozen_d0_correct": d0_correct,
        "frozen_d0_accuracy": d0_correct / n if n else 0.0,
        "delta_vs_frozen_d0": (packet_correct - d0_correct) / n if n else 0.0,
        **counts,
        "reader_refusal": sum(bool(row.get("reader_refusal")) for row in rows),
        "by_failure_reason": grouped_summary(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["e1", "e2"], required=True)
    parser.add_argument("--time-v", type=Path, required=True)
    parser.add_argument("--previous-time-v", type=Path)
    parser.add_argument("--compiled", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answer-model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--verifier-model", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--verifier-base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--verifier-api-key", default="EMPTY")
    parser.add_argument("--api-key-env", default="E_MEM_API_KEY")
    parser.add_argument("--base-url-env", default="E_MEM_BASE_URL")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("time_v", "previous_time_v", "output"):
        path = getattr(args, name)
        if path is not None and not path.is_absolute():
            setattr(args, name, ROOT / path)
    args.compiled = [path if path.is_absolute() else ROOT / path for path in args.compiled]

    current = read_json(args.time_v)
    if args.mode == "e1":
        if args.previous_time_v is None:
            raise ValueError("--previous-time-v is required for E1")
        selected = select_e1(current, read_json(args.previous_time_v))
    else:
        selected = select_e2(current)
    if args.expected_count is not None and len(selected) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} rows, selected {len(selected)}")

    compiled = source_rows(args.compiled)
    missing = [row["question_id"] for row in selected if row["question_id"] not in compiled]
    if missing:
        raise KeyError(f"missing compiled rows: {missing}")
    rows: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        rows = read_json(args.output).get("rows", [])
    elif args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --resume")
    done = {row["question_id"] for row in rows}
    log_path = args.output.with_suffix(".log")
    qa121 = load_qa121()
    verifier139 = load_verifier139() if args.mode == "e1" else None
    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=120.0,
    )
    verifier_client = (
        OpenAICompatibleChatClient(
            api_key=args.verifier_api_key,
            base_url=args.verifier_base_url,
            timeout_seconds=180.0,
        )
        if args.mode == "e1"
        else None
    )
    started = time.perf_counter()
    append_log(log_path, f"START mode={args.mode} selected={len(selected)} resumed={len(rows)}")
    for index, time_row in enumerate(selected, start=1):
        question_id = str(time_row["question_id"])
        if question_id in done:
            continue
        source = compiled[question_id]
        packet = packet_for_reader(time_row)
        generation_usage: dict[str, Any] = {}
        generation_elapsed = 0.0
        verifier_record: dict[str, Any] = {}
        if args.mode == "e1":
            candidate = time_row["h5_constraint_candidate"]
            candidate_prediction = str(candidate["solution"]["answer"])
            assert verifier139 is not None and verifier_client is not None
            verification, verifier_result, verifier_attempts = verifier139.call_json(
                verifier_client,
                model=args.verifier_model,
                messages=[
                    ChatMessage(role="system", content=verifier139.SYSTEM),
                    ChatMessage(
                        role="user",
                        content=verifier139.USER.format(
                            question=source["question"],
                            question_time=source["question_date"],
                            query_type=candidate.get("query_type"),
                            required_roles=json.dumps(candidate.get("required_roles") or [], ensure_ascii=False),
                            ambiguities=json.dumps(source.get("ambiguities") or [], ensure_ascii=False),
                            operands=json.dumps(candidate.get("operands") or [], ensure_ascii=False, separators=(",", ":")),
                            solution=json.dumps(candidate.get("solution") or {}, ensure_ascii=False, separators=(",", ":")),
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
            verifier_record = {
                "model": args.verifier_model,
                "verification": verification,
                "safe_to_override_d0": safe,
                "usage": verifier_result.usage,
                "elapsed_seconds": verifier_result.elapsed_seconds,
                "attempts": verifier_attempts,
            }
            prediction = candidate_prediction
        else:
            generation = client.chat_completion_with_metadata(
                model=args.answer_model,
                messages=[
                    ChatMessage(role="system", content=PACKET_SYSTEM),
                    ChatMessage(
                        role="user",
                        content=(
                            f"Question: {source['question']}\n"
                            f"Question time: {source['question_date']}\n"
                            "Frozen temporal packet:\n"
                            + json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=128,
            )
            prediction = generation.content.strip()
            generation_usage = qa121.normalize_usage(generation.usage)
            generation_elapsed = generation.elapsed_seconds
        judged = qa121.judge_answer(
            client=client,
            model=args.judge_model,
            question=source["question"],
            question_type="temporal-reasoning",
            gold_answer=str(source.get("gold_answer") or ""),
            prediction=prediction,
            is_abstention=bool(source.get("is_abstention")),
            max_tokens=160,
        )
        d0_correct = bool(source["frozen_d0"].get("judge_correct"))
        candidate_correct = bool(judged.get("judge_correct"))
        if args.mode == "e1" and not verifier_record["safe_to_override_d0"]:
            candidate_judged = judged
            prediction = str(source["frozen_d0"].get("prediction") or "")
            judged = {
                "judge_correct": int(d0_correct),
                "judge_label": "CORRECT" if d0_correct else "WRONG",
                "judge_reason": "Qwen verifier rejected candidate; reused frozen D0 result",
                "judge_usage": {},
                "judge_elapsed_seconds": 0.0,
            }
        else:
            candidate_judged = judged
        packet_correct = bool(judged.get("judge_correct"))
        row = {
            "question_id": question_id,
            "question": source["question"],
            "gold_answer": source.get("gold_answer"),
            "prediction": prediction,
            "packet": packet,
            "failure_reason": failure_reason(time_row),
            "reader_refusal": is_refusal(prediction),
            "frozen_d0_prediction": source["frozen_d0"].get("prediction"),
            "frozen_d0_judge_correct": int(d0_correct),
            "comparison": classification(d0_correct, packet_correct),
            "generation_usage": generation_usage,
            "generation_elapsed_seconds": generation_elapsed,
            "verifier": verifier_record,
            "candidate_prediction": candidate_prediction if args.mode == "e1" else prediction,
            "candidate_judge_correct": int(candidate_correct),
            "candidate_judge": candidate_judged,
            **judged,
        }
        rows.append(row)
        payload = {
            "metadata": {
                "version": "hypathmem_8.4time_v_e1_e2",
                "mode": args.mode,
                "answer_model": None if args.mode == "e1" else args.answer_model,
                "verifier_model": args.verifier_model if args.mode == "e1" else None,
                "judge_model": args.judge_model,
                "judge_repetitions": 1,
                "time_v_artifact": str(args.time_v),
                "uses_gold_for_packet": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
            "summary": summary(rows),
            "rows": rows,
        }
        write_json(args.output, payload)
        message = (
            f"processed {index}/{len(selected)} qid={question_id} judge={row['judge_label']} "
            f"comparison={row['comparison']} refusal={row['reader_refusal']}"
        )
        print(message, flush=True)
        append_log(log_path, message)
    final = {
        "metadata": {
            "version": "hypathmem_8.4time_v_e1_e2",
            "mode": args.mode,
            "answer_model": None if args.mode == "e1" else args.answer_model,
            "verifier_model": args.verifier_model if args.mode == "e1" else None,
            "judge_model": args.judge_model,
            "judge_repetitions": 1,
            "time_v_artifact": str(args.time_v),
            "uses_gold_for_packet": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "summary": summary(rows),
        "rows": rows,
    }
    write_json(args.output, final)
    append_log(log_path, "DONE " + json.dumps(final["summary"], ensure_ascii=False))
    print(json.dumps(final["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
