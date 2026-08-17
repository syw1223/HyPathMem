from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.temporal import TemporalPacketCompiler, TemporalQueryPlanner, TemporalSidecarBuilder, TemporalSolver


DEFAULT_PACKS = "outputs/reconstruction/hypathmem_r_v0_1/top50_structured_raw_packs.json"
DEFAULT_BASELINE = "outputs/qa/longmemeval_v3_10_cardquota_light_top50_c1_window_chunks_80k_gpt41mini_judge_gpt4omini.json"
DEFAULT_OUTPUT = "outputs/reconstruction/hypathmem_temporal_v0_1/frozen_d2_top50_t1_t4.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T1-T4 temporal packets from the frozen D2 Top50 pool.")
    parser.add_argument("--packs", default=DEFAULT_PACKS)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    packs_path = resolve(args.packs)
    baseline_path = resolve(args.baseline)
    output_path = resolve(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force for this derived artifact")

    packs = read_json(packs_path)
    baseline = read_json(baseline_path)
    baseline_rows = baseline.get("per_question") or baseline.get("rows") or []
    baseline_by_id = {str(row["question_id"]): row for row in baseline_rows}
    planner = TemporalQueryPlanner()
    sidecar = TemporalSidecarBuilder()
    solver = TemporalSolver()
    compiler = TemporalPacketCompiler()
    rows: list[dict[str, Any]] = []

    for index, pack_row in enumerate(packs["rows"], start=1):
        qid = str(pack_row["question_id"])
        base = baseline_by_id.get(qid, {})
        question = str(pack_row.get("question") or base.get("question") or "")
        plan = planner.plan(
            question,
            question_date=str(base.get("question_date") or "") or None,
            question_type=str(base.get("question_type") or ""),
        )
        if not plan.activated:
            continue
        units = list((pack_row.get("pack") or {}).get("evidence_units") or [])
        records, constraints = sidecar.build(units)
        solution = solver.solve(plan, records)
        packets = {
            stage.lower(): compiler.compile(question, plan, records, constraints, solution, stage=stage).model_dump(mode="json")
            for stage in ("T1", "T2", "T3", "T4")
        }
        rows.append(
            {
                "question_id": qid,
                "question": question,
                "question_type": base.get("question_type", ""),
                "question_date": base.get("question_date", ""),
                "gold_answer": base.get("gold_answer", pack_row.get("gold_answer", "")),
                "baseline_judge_correct": int(base.get("judge_correct") or 0),
                "baseline_retrieval_hit": bool(base.get("retrieval_hit")),
                "baseline_retrieval_full_cover": bool(base.get("retrieval_full_cover")),
                "plan": plan.model_dump(mode="json"),
                "event_records": [record.model_dump(mode="json") for record in records],
                "constraints": [constraint.model_dump(mode="json") for constraint in constraints],
                "solution": solution.model_dump(mode="json"),
                "packets": packets,
            }
        )
        if index % 50 == 0:
            print(f"scanned {index}/{len(packs['rows'])} temporal={len(rows)}", flush=True)

    payload = {
        "metadata": {
            "version": "hypathmem_temporal_v0_1",
            "candidate_pool": "frozen_D2_top50",
            "packs": str(packs_path),
            "packs_sha256": sha256(packs_path),
            "baseline": str(baseline_path),
            "baseline_sha256": sha256(baseline_path),
            "mutates_source_graph": False,
            "temporal_context_policy": "replace_D2_context_not_append",
            "non_temporal_policy": "unchanged_D2",
            "normalizer": "deterministic_relative_time_v0_1",
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    write_json(output_path, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solutions = [row["solution"] for row in rows]
    types = Counter(row["plan"]["query_type"] for row in rows)
    full_cover_wrong = [row for row in rows if row["baseline_retrieval_full_cover"] and not row["baseline_judge_correct"]]
    return {
        "num_temporal_questions": len(rows),
        "query_types": dict(sorted(types.items())),
        "solver_success": sum(bool(solution["success"]) for solution in solutions),
        "verified": sum(bool(solution["verified"]) for solution in solutions),
        "operand_full_coverage": sum(float(solution["operand_coverage"]) == 1.0 for solution in solutions),
        "full_cover_but_wrong": len(full_cover_wrong),
        "full_cover_but_wrong_verified": sum(bool(row["solution"]["verified"]) for row in full_cover_wrong),
        "failure_reasons": dict(
            sorted(Counter(solution.get("failure_reason") or "none" for solution in solutions if not solution["success"]).items())
        ),
    }


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
