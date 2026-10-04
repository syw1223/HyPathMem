#!/usr/bin/env python3
"""Recompile cached LoCoMo Cat2 extraction with the independent v2 gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.locomo_time_v8_5_v2 import (
    VERSION,
    ambiguity_safe,
    normalize_locomo_binding_v2,
    solve_locomo_hypothesis_v2,
    typed_solution_safe,
)


def load_base() -> ModuleType:
    path = ROOT / "scripts/138_compile_hypathmem_temporal_q1_q4_v0_2.py"
    spec = importlib.util.spec_from_file_location("hypathmem_time_base138_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extractions",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_locomo_8.5time_v1/cat2_joint_extraction.json"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "outputs/qa/full_v3_9_expand120_light_quota_top20_hybrid_v11_"
            "chatanywhere_gpt41mini_judge_gpt4omini.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_locomo_8.5time_v2/cat2_compiled.json"),
    )
    args = parser.parse_args()
    for name in ("extractions", "baseline", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)

    base = load_base()
    extraction_payload = read(args.extractions)
    baseline = read(args.baseline)
    baseline_by_id = {str(row["question_id"]): row for row in baseline["per_question"]}
    rows = []
    for source in extraction_payload["rows"]:
        extraction = source["extraction"]
        audit_by_id = {str(item["hypothesis_id"]): item for item in source["validation"]["hypotheses"]}
        hypotheses = []
        for hypothesis in extraction.get("hypotheses") or []:
            audit = audit_by_id.get(str(hypothesis.get("hypothesis_id")), {})
            normalized = [
                normalize_locomo_binding_v2(binding, base.normalize_binding)
                for binding in hypothesis.get("bindings") or []
            ]
            solution = solve_locomo_hypothesis_v2(
                source["question"],
                str(extraction.get("query_type") or "other"),
                normalized,
                source.get("question_date") or "",
                base.solve_hypothesis,
            )
            hypotheses.append(
                {
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    "confidence": hypothesis.get("confidence", 0.0),
                    "source_validation": audit,
                    "bindings": hypothesis.get("bindings") or [],
                    "normalized_operands": normalized,
                    "solution": solution,
                }
            )

        eligible = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis["source_validation"].get("eligible_for_normalization")
            and float(hypothesis.get("confidence") or 0.0) >= 0.90
            and typed_solution_safe(source["question"], hypothesis["solution"])
            and all(operand.get("local_anchor_valid") for operand in hypothesis["normalized_operands"])
        ]
        clause_checks = [
            base.restrictive_clause_grounded(source["question"], hypothesis["normalized_operands"])[1]
            for hypothesis in eligible
        ]
        clause_grounded = bool(eligible) and all(
            base.restrictive_clause_grounded(source["question"], hypothesis["normalized_operands"])[0]
            for hypothesis in eligible
        )
        answers = {str(hypothesis["solution"]["answer"]).strip().lower() for hypothesis in eligible}
        ambiguity_gate = ambiguity_safe(extraction.get("ambiguities") or [])
        robust = bool(eligible) and len(answers) == 1 and clause_grounded and ambiguity_gate
        frozen = baseline_by_id[str(source["question_id"])]
        rows.append(
            {
                "question_id": source["question_id"],
                "conversation_id": frozen.get("conversation_id"),
                "question": source["question"],
                "question_date": source.get("question_date"),
                "category": 2,
                "gold_answer": frozen.get("gold_answer"),
                "frozen_d0": {
                    "prediction": frozen.get("prediction"),
                    "judge_correct": int(frozen.get("judge_correct") or 0),
                    "retrieval_hit": bool(frozen.get("retrieval_hit")),
                    "retrieval_full_cover": bool(frozen.get("retrieval_full_cover")),
                },
                "query_type": extraction.get("query_type"),
                "required_roles": extraction.get("required_roles") or [],
                "ambiguities": extraction.get("ambiguities") or [],
                "hypotheses": hypotheses,
                "pre_verifier": {
                    "eligible_solution_count": len(eligible),
                    "solution_answers": sorted(answers),
                    "multi_hypothesis_answer_consistent": bool(eligible) and len(answers) == 1,
                    "restrictive_clause_grounded": clause_grounded,
                    "ambiguity_gate_passed": ambiguity_gate,
                    "restrictive_clause_checks": clause_checks,
                    "candidate_answer": eligible[0]["solution"]["answer"] if robust else None,
                    "candidate_answer_type": eligible[0]["solution"].get("answer_type") if robust else None,
                    "eligible_to_call_solution_verifier": robust,
                },
            }
        )

    payload = {
        "metadata": {
            "version": VERSION,
            "dataset": "LoCoMo",
            "gold_evidence_or_answer_used_for_compilation": False,
            "fallback": "frozen_D0",
            "changes": [
                "evidence_local_anchor",
                "typed_when_answer",
                "compound_relative_day_fix",
                "granularity_preservation",
                "ambiguity_fail_closed",
            ],
        },
        "summary": {
            "num_questions": len(rows),
            "solver_success": sum(any(h["solution"].get("success") for h in row["hypotheses"]) for row in rows),
            "typed_candidate": sum(bool(row["pre_verifier"]["eligible_solution_count"]) for row in rows),
            "ambiguity_gate_passed": sum(row["pre_verifier"]["ambiguity_gate_passed"] for row in rows),
            "eligible_to_call_solution_verifier": sum(row["pre_verifier"]["eligible_to_call_solution_verifier"] for row in rows),
        },
        "rows": rows,
    }
    write(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
