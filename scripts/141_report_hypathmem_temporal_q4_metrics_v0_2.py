#!/usr/bin/env python3
"""Build an auditable Q4 report from extraction through answer judging."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def usage_total(usage: dict[str, Any] | None, key: str) -> int:
    if not usage:
        return 0
    direct = usage.get(key)
    if isinstance(direct, (int, float)):
        return int(direct)
    return sum(usage_total(value, key) for value in usage.values() if isinstance(value, dict))


def has_executable_hypothesis(row: dict[str, Any]) -> bool:
    return any(
        hypothesis["source_validation"].get("eligible_for_normalization")
        and hypothesis["solution"].get("success")
        for hypothesis in row.get("hypotheses", [])
    )


def has_full_operand_coverage(row: dict[str, Any]) -> bool:
    return any(
        hypothesis["source_validation"].get("all_deterministic_checks_pass")
        and hypothesis["source_validation"].get("identity_verified")
        and not hypothesis["source_validation"].get("missing_required_roles")
        for hypothesis in row.get("hypotheses", [])
    )


def binding_anchor_counts(row: dict[str, Any]) -> tuple[int, int]:
    total = valid = 0
    for hypothesis in row.get("hypotheses", []):
        for audit in hypothesis.get("source_validation", {}).get("binding_audits", []):
            if audit.get("time_expression_grounded") is not None:
                total += 1
                valid += int(
                    audit.get("time_expression_grounded") is True
                    and audit.get("anchor_complete") is True
                )
    return valid, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("extraction", "compiled", "verified", "qa", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)

    extraction = read_json(args.extraction)
    compiled = read_json(args.compiled)
    verified = read_json(args.verified)
    qa = read_json(args.qa)
    extraction_by_id = {row["question_id"]: row for row in extraction["rows"]}
    compiled_by_id = {row["question_id"]: row for row in compiled["rows"]}
    verified_by_id = {row["question_id"]: row for row in verified["rows"]}
    qa_by_id = {row["question_id"]: row for row in qa["per_question"]}
    ids = [row["question_id"] for row in compiled["rows"]]
    missing = {
        "extraction": [qid for qid in ids if qid not in extraction_by_id],
        "verified": [qid for qid in ids if qid not in verified_by_id],
        "qa": [qid for qid in ids if qid not in qa_by_id],
    }
    if any(missing.values()):
        raise ValueError(f"Incomplete artifacts: {missing}")

    n = len(ids)
    correct = sum(int(qa_by_id[qid].get("judge_correct") or 0) for qid in ids)
    d0_correct = sum(int(qa_by_id[qid].get("frozen_d0_judge_correct") or 0) for qid in ids)
    takeover_ids = [qid for qid in ids if qa_by_id[qid].get("temporal_branch_activated")]
    fallback_ids = [qid for qid in ids if qa_by_id[qid].get("fallback_reused")]
    takeover_correct = sum(int(qa_by_id[qid].get("judge_correct") or 0) for qid in takeover_ids)
    fallback_correct = sum(int(qa_by_id[qid].get("judge_correct") or 0) for qid in fallback_ids)
    fixes = sum(
        not qa_by_id[qid].get("frozen_d0_judge_correct") and qa_by_id[qid].get("judge_correct")
        for qid in ids
    )
    breaks = sum(
        qa_by_id[qid].get("frozen_d0_judge_correct") and not qa_by_id[qid].get("judge_correct")
        for qid in ids
    )
    executable = sum(has_executable_hypothesis(compiled_by_id[qid]) for qid in ids)
    consensus_executable = sum(
        compiled_by_id[qid].get("pre_verifier", {}).get("eligible_to_call_solution_verifier") is True
        for qid in ids
    )
    full_coverage = sum(has_full_operand_coverage(compiled_by_id[qid]) for qid in ids)
    anchor_valid = anchor_total = 0
    for qid in ids:
        valid, total = binding_anchor_counts(compiled_by_id[qid])
        anchor_valid += valid
        anchor_total += total
    verifier_anchor_called = [
        verified_by_id[qid]
        for qid in ids
        if verified_by_id[qid].get("verifier_called")
    ]
    verifier_anchor_supported = sum(
        row.get("verification", {}).get("anchor_bindings_supported") is True
        for row in verifier_anchor_called
    )

    prompt_tokens = completion_tokens = total_tokens = 0
    qwen_seconds_by_id: dict[str, float] = defaultdict(float)
    qwen_successful_calls = 0
    qwen_api_attempts = 0
    known_unmetered_retries = 0
    for qid in ids:
        row = extraction_by_id[qid]
        prompt_tokens += usage_total(row.get("usage"), "prompt_tokens")
        completion_tokens += usage_total(row.get("usage"), "completion_tokens")
        total_tokens += usage_total(row.get("usage"), "total_tokens")
        qwen_seconds_by_id[qid] += sum(float(value) for value in row.get("elapsed_seconds", {}).values())
        qwen_successful_calls += 2
        qwen_api_attempts += 2
        verifier_row = verified_by_id[qid]
        if verifier_row.get("verifier_called"):
            prompt_tokens += usage_total(verifier_row.get("usage"), "prompt_tokens")
            completion_tokens += usage_total(verifier_row.get("usage"), "completion_tokens")
            total_tokens += usage_total(verifier_row.get("usage"), "total_tokens")
            qwen_seconds_by_id[qid] += float(verifier_row.get("elapsed_seconds") or 0.0)
            qwen_successful_calls += 1
            attempts = int(verifier_row.get("attempts") or 1)
            qwen_api_attempts += attempts
            known_unmetered_retries += max(0, attempts - 1)
    judge_prompt = sum(usage_total(qa_by_id[qid].get("judge_usage"), "prompt_tokens") for qid in takeover_ids)
    judge_completion = sum(
        usage_total(qa_by_id[qid].get("judge_usage"), "completion_tokens") for qid in takeover_ids
    )
    judge_total = sum(usage_total(qa_by_id[qid].get("judge_usage"), "total_tokens") for qid in takeover_ids)
    judge_seconds = sum(float(qa_by_id[qid].get("judge_elapsed_seconds") or 0.0) for qid in takeover_ids)
    qwen_latencies = list(qwen_seconds_by_id.values())

    by_type: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for qid in ids:
        grouped[str(compiled_by_id[qid].get("query_type") or "UNKNOWN")].append(qid)
    for query_type, group_ids in sorted(grouped.items()):
        group_correct = sum(int(qa_by_id[qid].get("judge_correct") or 0) for qid in group_ids)
        group_d0 = sum(int(qa_by_id[qid].get("frozen_d0_judge_correct") or 0) for qid in group_ids)
        group_takeover = [qid for qid in group_ids if qa_by_id[qid].get("temporal_branch_activated")]
        by_type[query_type] = {
            "n": len(group_ids),
            "accuracy": rate(group_correct, len(group_ids)),
            "frozen_d0_accuracy": rate(group_d0, len(group_ids)),
            "takeover_rate": rate(len(group_takeover), len(group_ids)),
            "takeover_precision": rate(
                sum(int(qa_by_id[qid].get("judge_correct") or 0) for qid in group_takeover),
                len(group_takeover),
            ),
        }

    report = {
        "metadata": {
            "version": "hypathmem_temporal_v0_2_q4_audit",
            "n": n,
            "artifact_paths": {name: str(getattr(args, name)) for name in ("extraction", "compiled", "verified", "qa")},
            "anchor_accuracy_note": (
                "LongMemEval has no gold anchor labels. The two anchor metrics below are "
                "automatic validation proxies, not true gold anchor-binding accuracy."
            ),
        },
        "effectiveness": {
            "accuracy": rate(correct, n),
            "num_correct": correct,
            "frozen_d0_accuracy": rate(d0_correct, n),
            "verified_takeover_rate": rate(len(takeover_ids), n),
            "verified_takeovers": len(takeover_ids),
            "takeover_precision": rate(takeover_correct, len(takeover_ids)),
            "takeover_correct": takeover_correct,
            "fixes": fixes,
            "breaks": breaks,
            "solver_executable_rate": rate(executable, n),
            "solver_executable": executable,
            "solver_consensus_executable_rate": rate(consensus_executable, n),
            "solver_consensus_executable": consensus_executable,
            "operand_full_coverage_rate": rate(full_coverage, n),
            "operand_full_coverage": full_coverage,
            "anchor_binding_validated_rate_proxy": rate(anchor_valid, anchor_total),
            "anchor_bindings_validated": anchor_valid,
            "anchor_bindings_total": anchor_total,
            "solution_verifier_anchor_support_rate_proxy": rate(
                verifier_anchor_supported, len(verifier_anchor_called)
            ),
            "fallback_accuracy": rate(fallback_correct, len(fallback_ids)),
            "fallback_correct": fallback_correct,
            "fallback_count": len(fallback_ids),
        },
        "cost": {
            "qwen_successful_calls": qwen_successful_calls,
            "qwen_api_attempts": qwen_api_attempts,
            "known_unmetered_application_retries": known_unmetered_retries,
            "qwen_prompt_tokens": prompt_tokens,
            "qwen_completion_tokens": completion_tokens,
            "qwen_total_tokens": total_tokens,
            "qwen_tokens_per_question": rate(total_tokens, n),
            "qwen_serial_latency_seconds": sum(qwen_latencies),
            "qwen_latency_mean_seconds": statistics.fmean(qwen_latencies) if qwen_latencies else 0.0,
            "qwen_latency_p50_seconds": percentile(qwen_latencies, 0.50),
            "qwen_latency_p95_seconds": percentile(qwen_latencies, 0.95),
            "judge_calls": len(takeover_ids),
            "judge_prompt_tokens": judge_prompt,
            "judge_completion_tokens": judge_completion,
            "judge_total_tokens": judge_total,
            "judge_latency_seconds": judge_seconds,
            "qwen_stage_wall_seconds": float(extraction.get("metadata", {}).get("elapsed_seconds") or 0.0)
            + float(verified.get("metadata", {}).get("elapsed_seconds") or 0.0),
            "judge_stage_wall_seconds": float(qa.get("metadata", {}).get("elapsed_seconds") or 0.0),
            "token_accounting_note": (
                "Tokens are exact for persisted successful responses. A discarded invalid-JSON "
                "retry response has no persisted usage and is excluded from token totals."
                if known_unmetered_retries
                else "All observed application attempts are represented in persisted token usage."
            ),
        },
        "by_query_type": by_type,
        "route_counts": dict(Counter(qa_by_id[qid].get("route", "UNKNOWN") for qid in ids)),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
