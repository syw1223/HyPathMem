#!/usr/bin/env python3
"""Archive the frozen HyPathMem LongMemEval 500-question result."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/final/hypathmem_longmemeval_8.5time_v_500"

SOURCES = {
    "d2_non_temporal367": ROOT
    / "outputs/qa/hypathmem_r_v0_1_full_non_temporal367/"
    "d2_raw_grounded_gpt41mini_judge_gpt4omini.json",
    "q4_temporal_paired20": ROOT
    / "outputs/qa/hypathmem_temporal_v0_2_qwen_paired20/"
    "q4_verified_solver_formatfix_v2_gpt4omini.json",
    "q4_temporal_holdout113": ROOT
    / "outputs/qa/hypathmem_temporal_v0_2_qwen_holdout113/"
    "q4_verified_solver_gpt4omini.json",
    "time_v8_4_safe2": ROOT
    / "outputs/qa/hypathmem_8.4time_v/"
    "e1_new2_qwen_gpu2_verifier_solver_judge_gpt4omini.json",
    "time_v8_5_safe3": ROOT
    / "outputs/qa/hypathmem_8.5time_v/"
    "new_verified3_solver_judge_gpt4omini.json",
    "time_v8_5_conversion": ROOT
    / "outputs/reconstruction/hypathmem_8.5time_v/"
    "non_executable35_conversion_interval_v2_qwen_gpu2.json",
}

SNAPSHOTS = {
    "config": ROOT / "configs/hypathmem_8_5time_v.yaml",
    "method": ROOT / "src/hytopomem/temporal_time_v8_5.py",
    "runner": ROOT / "scripts/144_run_hypathmem_8_5time_v_conversion.py",
    "evaluator": ROOT / "scripts/145_eval_hypathmem_8_5time_v_candidates.py",
    "archiver": ROOT / "scripts/146_archive_hypathmem_longmemeval_final500.py",
    "documentation": ROOT / "docs/HYPATHMEM_8_5TIME_V.md",
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_base(row: dict, branch: str) -> dict:
    result = dict(row)
    result["final_branch"] = branch
    result["final_source"] = branch
    result["final_prediction"] = row["prediction"]
    result["final_judge_correct"] = int(row["judge_correct"])
    return result


def apply_override(base: dict, row: dict, source: str) -> dict:
    result = dict(base)
    result.update(
        {
            "prediction": row["prediction"],
            "judge_correct": int(row["judge_correct"]),
            "judge_label": row.get("judge_label"),
            "judge_reason": row.get("judge_reason"),
            "judge_raw_response": row.get("judge_raw_response"),
            "judge_usage": row.get("judge_usage", {}),
            "judge_elapsed_seconds": row.get("judge_elapsed_seconds", 0),
            "final_branch": "temporal_verified_solver",
            "final_source": source,
            "final_prediction": row["prediction"],
            "final_judge_correct": int(row["judge_correct"]),
            "override_comparison": row.get("comparison"),
            "pre_override_prediction": base.get("prediction"),
            "pre_override_judge_correct": int(base.get("judge_correct", 0)),
        }
    )
    return result


def summarize(rows: list[dict]) -> dict:
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in rows:
        correct = int(row["final_judge_correct"])
        question_type = row.get("question_type", "temporal")
        source = row["final_source"]
        by_type[question_type]["n"] += 1
        by_type[question_type]["correct"] += correct
        by_source[source]["n"] += 1
        by_source[source]["correct"] += correct

    def finish(groups: dict[str, dict[str, int]]) -> dict:
        return {
            key: {
                **counts,
                "accuracy": counts["correct"] / counts["n"],
            }
            for key, counts in sorted(groups.items())
        }

    correct = sum(int(row["final_judge_correct"]) for row in rows)
    return {
        "num_questions": len(rows),
        "num_correct": correct,
        "accuracy": correct / len(rows),
        "accuracy_percent": 100 * correct / len(rows),
        "by_question_type": finish(by_type),
        "by_final_source": finish(by_source),
    }


def main() -> None:
    missing = [str(path) for path in [*SOURCES.values(), *SNAPSHOTS.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing archive inputs:\n" + "\n".join(missing))

    d2 = load(SOURCES["d2_non_temporal367"])
    paired = load(SOURCES["q4_temporal_paired20"])
    holdout = load(SOURCES["q4_temporal_holdout113"])
    v84 = load(SOURCES["time_v8_4_safe2"])
    v85 = load(SOURCES["time_v8_5_safe3"])

    rows: dict[str, dict] = {}
    for row in d2["per_question"]:
        rows[row["question_id"]] = normalized_base(row, "d2_non_temporal")
    for payload in (paired, holdout):
        for row in payload["per_question"]:
            question_id = row["question_id"]
            if question_id in rows:
                raise ValueError(f"Duplicate base question: {question_id}")
            rows[question_id] = normalized_base(row, "q4_temporal")

    if len(rows) != 500:
        raise ValueError(f"Expected 500 unique base questions, found {len(rows)}")

    override_log = []
    for payload, source in ((v84, "8.4time_v_safe_override"), (v85, "8.5time_v_safe_override")):
        for row in payload["rows"]:
            question_id = row["question_id"]
            if question_id not in rows:
                raise KeyError(f"Override question missing from base: {question_id}")
            before = rows[question_id]
            rows[question_id] = apply_override(before, row, source)
            override_log.append(
                {
                    "question_id": question_id,
                    "source": source,
                    "before_correct": int(before["final_judge_correct"]),
                    "after_correct": int(row["judge_correct"]),
                    "comparison": row.get("comparison"),
                }
            )

    final_rows = sorted(rows.values(), key=lambda row: row["question_id"])
    summary = summarize(final_rows)
    summary.update(
        {
            "method": "HyPathMem",
            "dataset": "LongMemEval-S",
            "version": "8.5time_v",
            "non_temporal_correct": 305,
            "non_temporal_total": 367,
            "temporal_correct": 103,
            "temporal_total": 133,
            "frozen_d0_correct": 385,
            "frozen_d0_accuracy": 0.77,
            "delta_correct_vs_d0": summary["num_correct"] - 385,
            "delta_accuracy_points_vs_d0": round(100 * (summary["accuracy"] - 0.77), 2),
            "paired_overrides": override_log,
            "reporting_status": "post-hoc development result",
        }
    )
    if summary["num_correct"] != 408:
        raise ValueError(f"Expected 408 correct, found {summary['num_correct']}")

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    dump(
        ARCHIVE / "hypathmem_longmemeval_final500.json",
        {
            "metadata": {
                "method": "HyPathMem",
                "dataset": "LongMemEval-S",
                "version": "8.5time_v",
                "answer_model": "gpt-4.1-mini",
                "judge_model": "gpt-4o-mini",
                "routing": "D2 non-temporal; verified temporal solver; frozen D0 fallback",
                "reporting_status": "post-hoc development result",
            },
            "summary": summary,
            "per_question": final_rows,
        },
    )
    dump(ARCHIVE / "summary.json", summary)

    source_dir = ARCHIVE / "source_artifacts"
    snapshot_dir = ARCHIVE / "method_snapshot"
    source_dir.mkdir(exist_ok=True)
    snapshot_dir.mkdir(exist_ok=True)
    copied: dict[str, Path] = {}
    for name, source in SOURCES.items():
        target = source_dir / f"{name}{source.suffix}"
        shutil.copy2(source, target)
        copied[f"source_artifacts/{target.name}"] = target
    for name, source in SNAPSHOTS.items():
        target = snapshot_dir / f"{name}{source.suffix}"
        shutil.copy2(source, target)
        copied[f"method_snapshot/{target.name}"] = target

    final_path = ARCHIVE / "hypathmem_longmemeval_final500.json"
    summary_path = ARCHIVE / "summary.json"
    copied[final_path.name] = final_path
    copied[summary_path.name] = summary_path
    readme_path = ARCHIVE / "README.md"
    if readme_path.is_file():
        copied[readme_path.name] = readme_path
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": "HyPathMem LongMemEval-S final500 8.5time_v",
        "immutable_result_claim": "408/500 = 81.60%",
        "reporting_status": "post-hoc development result",
        "files": {
            name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in sorted(copied.items())
        },
        "original_sources": {name: str(path.relative_to(ROOT)) for name, path in SOURCES.items()},
    }
    dump(ARCHIVE / "MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
