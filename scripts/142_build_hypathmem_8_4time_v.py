#!/usr/bin/env python3
"""Build the non-destructive HyPathMem 8.4time_v H1-H5 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.temporal_time_v8_4 import (
    VERSION,
    build_temporal_sidecar,
    close_operands,
    compile_constraint_candidate,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_r_v0_1/top50_structured_raw_packs.json"),
    )
    parser.add_argument("--compiled", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_8.4time_v/longmemeval_temporal_h1_h5.json"),
    )
    parser.add_argument("--role-top-k", type=int, default=3)
    parser.add_argument("--minimum-role-score", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("packs", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    args.compiled = [path if path.is_absolute() else ROOT / path for path in args.compiled]
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force for this derived artifact only")

    packs_payload = read_json(args.packs)
    pack_by_id = {str(row["question_id"]): row for row in packs_payload["rows"]}
    compiled_rows: list[dict[str, Any]] = []
    for path in args.compiled:
        compiled_rows.extend(read_json(path)["rows"])
    seen: set[str] = set()
    compiled_rows = [
        row for row in compiled_rows
        if not (str(row["question_id"]) in seen or seen.add(str(row["question_id"])))
    ]
    if args.limit:
        compiled_rows = compiled_rows[: args.limit]

    rows = []
    for index, compiled in enumerate(compiled_rows, start=1):
        question_id = str(compiled["question_id"])
        pack_row = pack_by_id.get(question_id)
        if pack_row is None:
            raise KeyError(f"Missing frozen evidence pack for {question_id}")
        pack = pack_row["pack"]
        sidecar = build_temporal_sidecar(
            question_id=question_id,
            pack=pack,
            compiled_row=compiled,
        )
        closure = close_operands(
            question_id=question_id,
            pack=pack,
            compiled_row=compiled,
            sidecar=sidecar,
            role_top_k=args.role_top_k,
            minimum_role_score=args.minimum_role_score,
        )
        candidate = compile_constraint_candidate(
            compiled_row=compiled,
            pack=pack,
            sidecar=sidecar,
            closure=closure,
        )
        rows.append(
            {
                "question_id": question_id,
                "question": compiled.get("question"),
                "question_date": compiled.get("question_date"),
                "h1_temporal_sidecar": sidecar,
                "h2_h4_operand_closure": closure,
                "h5_constraint_candidate": candidate,
                "audit": {
                    "source_q4_operand_full_coverage": any(
                        hypothesis.get("source_validation", {}).get("all_deterministic_checks_pass")
                        and hypothesis.get("source_validation", {}).get("identity_verified")
                        for hypothesis in compiled.get("hypotheses") or []
                    ),
                    "source_q4_solver_executable": any(
                        hypothesis.get("source_validation", {}).get("eligible_for_normalization")
                        and hypothesis.get("solution", {}).get("success")
                        for hypothesis in compiled.get("hypotheses") or []
                    ),
                    "source_q4_consensus_executable": bool(
                        compiled.get("pre_verifier", {}).get("eligible_to_call_solution_verifier")
                    ),
                    "time_v_operand_full_coverage": closure["operand_full_coverage"],
                    "time_v_raw_solver_success": candidate["solution"]["success"],
                    "time_v_solver_executable": candidate["eligible_for_q4_verifier"],
                    "time_v_eligible_for_q4_verifier": candidate["eligible_for_q4_verifier"],
                },
            }
        )
        if index % 10 == 0 or index == len(compiled_rows):
            print(f"built 8.4time_v {index}/{len(compiled_rows)}", flush=True)

    payload = {
        "metadata": {
            "version": VERSION,
            "stages": [
                "H1_temporal_sidecar",
                "H2_top50_operand_closure",
                "H3_temporal_relation_expansion",
                "H4_supplementary_retrieval_interface",
                "H5_state_at_time_and_constraint_solver",
            ],
            "packs": str(args.packs),
            "packs_sha256": sha256(args.packs),
            "compiled": [str(path) for path in args.compiled],
            "compiled_sha256": [sha256(path) for path in args.compiled],
            "semantic_graph_mutated": False,
            "source_artifacts_overwritten": False,
            "uses_gold_answer": False,
            "uses_gold_evidence": False,
            "role_top_k": args.role_top_k,
            "minimum_role_score": args.minimum_role_score,
            "h4_note": "External retrieval is an injectable interface; this run only closes over frozen Top50.",
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    write_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    source_coverage = sum(row["audit"]["source_q4_operand_full_coverage"] for row in rows)
    source_executable = sum(row["audit"]["source_q4_solver_executable"] for row in rows)
    source_consensus = sum(row["audit"]["source_q4_consensus_executable"] for row in rows)
    time_coverage = sum(row["audit"]["time_v_operand_full_coverage"] for row in rows)
    time_executable = sum(row["audit"]["time_v_solver_executable"] for row in rows)
    raw_solutions = sum(row["audit"]["time_v_raw_solver_success"] for row in rows)
    eligible = sum(row["audit"]["time_v_eligible_for_q4_verifier"] for row in rows)
    return {
        "num_questions": n,
        "source_q4_operand_full_coverage": source_coverage,
        "source_q4_operand_full_coverage_rate": source_coverage / n if n else 0.0,
        "source_q4_solver_executable": source_executable,
        "source_q4_solver_executable_rate": source_executable / n if n else 0.0,
        "source_q4_consensus_executable": source_consensus,
        "source_q4_consensus_executable_rate": source_consensus / n if n else 0.0,
        "time_v_operand_full_coverage": time_coverage,
        "time_v_operand_full_coverage_rate": time_coverage / n if n else 0.0,
        "time_v_solver_executable": time_executable,
        "time_v_solver_executable_rate": time_executable / n if n else 0.0,
        "time_v_raw_solver_success": raw_solutions,
        "time_v_eligible_for_q4_verifier": eligible,
        "time_v_eligible_for_q4_verifier_rate": eligible / n if n else 0.0,
        "sidecar_event_nodes": sum(row["h1_temporal_sidecar"]["diagnostics"]["event_nodes"] for row in rows),
        "sidecar_state_nodes": sum(row["h1_temporal_sidecar"]["diagnostics"]["state_nodes"] for row in rows),
        "sidecar_anchor_nodes": sum(row["h1_temporal_sidecar"]["diagnostics"]["anchor_nodes"] for row in rows),
    }


if __name__ == "__main__":
    main()
