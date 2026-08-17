#!/usr/bin/env python3
"""Archive the selected HyPathMem LoCoMo D0 + temporal-v2 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / (
    "outputs/qa/full_v3_9_expand120_light_quota_top20_hybrid_v11_"
    "chatanywhere_gpt41mini_judge_gpt4omini.json"
)
DEFAULT_TIME = ROOT / "outputs/qa/hypathmem_locomo_8.5time_v2/cat2_verified_solver_judge_gpt4omini.json"
DEFAULT_VERIFIED = ROOT / "outputs/reconstruction/hypathmem_locomo_8.5time_v2/cat2_q4_verified.json"
DEFAULT_COMPILED = ROOT / "outputs/reconstruction/hypathmem_locomo_8.5time_v2/cat2_compiled.json"
DEFAULT_D2 = ROOT / "outputs/qa/hypathmem_locomo_d2_v1/d2_top20_paired60_gpt41mini_judge_gpt4omini.json"
DEFAULT_ARCHIVE = ROOT / "outputs/final/hypathmem_locomo_d0_time_v2_final1540"
GRAPH = ROOT / "outputs/graphs/locomo_graph_v3_6b_qwen_all.json"
PATHS = ROOT / "outputs/paths/full_v3_9_card_guided_expand120_light_quota_top20.json"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--time", type=Path, default=DEFAULT_TIME)
    parser.add_argument("--verified", type=Path, default=DEFAULT_VERIFIED)
    parser.add_argument("--compiled", type=Path, default=DEFAULT_COMPILED)
    parser.add_argument("--d2", type=Path, default=DEFAULT_D2)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    for name in ("baseline", "time", "verified", "compiled", "d2", "archive"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    if args.archive.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {args.archive}")

    baseline = read(args.baseline)
    temporal = read(args.time)
    verified = read(args.verified)
    temporal_by_id = {str(row["question_id"]): row for row in temporal["per_question"]}
    verified_by_id = {str(row["question_id"]): row for row in verified["rows"]}
    merged = []
    for source in baseline["per_question"]:
        row = dict(source)
        question_id = str(row["question_id"])
        time_row = temporal_by_id.get(question_id)
        q4 = verified_by_id.get(question_id)
        row["frozen_d0_prediction"] = source.get("prediction")
        row["frozen_d0_judge_correct"] = int(source.get("judge_correct") or 0)
        row["temporal_branch_activated"] = bool(time_row and time_row.get("temporal_branch_activated"))
        row["temporal_q4_called"] = bool(q4 and q4.get("verifier_called"))
        row["temporal_q4_safe"] = bool(q4 and q4.get("safe_to_override_d0"))
        row["temporal_q4_usage"] = (q4 or {}).get("usage") or {}
        row["temporal_q4_elapsed_seconds"] = float((q4 or {}).get("elapsed_seconds") or 0.0)
        if row["temporal_branch_activated"]:
            row.update(
                {
                    "prediction": time_row["prediction"],
                    "draft_prediction": time_row["prediction"],
                    "answer_protocol": "8.5time_locomo_v2_verified_solver",
                    "final_source": "8.5time_locomo_v2_safe_override",
                    "temporal_answer_type": time_row.get("answer_type"),
                    "judge_correct": int(time_row.get("judge_correct") or 0),
                    "judge_label": time_row.get("judge_label"),
                    "judge_reason": time_row.get("judge_reason"),
                    "judge_raw_response": time_row.get("judge_raw_response"),
                    "judge_usage": time_row.get("judge_usage") or {},
                    "judge_elapsed_seconds": float(time_row.get("judge_elapsed_seconds") or 0.0),
                }
            )
        else:
            row["final_source"] = "frozen_D0"
            row["temporal_answer_type"] = None
        row["lexical_f1"] = token_f1(str(row.get("prediction") or ""), str(row.get("gold_answer") or ""))
        row["bleu1"] = bleu1(str(row.get("prediction") or ""), str(row.get("gold_answer") or ""))
        row["total_usage"] = sum_usage(
            row.get("generation_usage") or {},
            row.get("judge_usage") or {},
            row.get("temporal_q4_usage") or {},
        )
        row["total_api_elapsed_seconds"] = (
            float(row.get("generation_elapsed_seconds") or 0.0)
            + float(row.get("judge_elapsed_seconds") or 0.0)
            + float(row.get("temporal_q4_elapsed_seconds") or 0.0)
        )
        merged.append(row)

    assert len(merged) == 1540
    assert len(temporal_by_id) == 321
    assert sum(row["temporal_branch_activated"] for row in merged) == 36
    summary = summarize(merged)
    assert summary["num_correct"] == 1411
    assert math.isclose(summary["judge_accuracy"], 1411 / 1540)
    per_conversation = [summarize(group, conversation_id=conversation_id) for conversation_id, group in grouped(merged)]
    final_payload = {
        "metadata": {
            **baseline.get("metadata", {}),
            "method": "HyPathMem",
            "version": "D0+8.5time_locomo_v2",
            "dataset": "LoCoMo",
            "selection": "frozen D0 for all questions; verified temporal v2 override for 36 Cat2 questions",
            "reporting_status": "final selected post-hoc development result",
            "d2_selected": False,
            "d2_exclusion_reason": "paired60 was 49/60 versus frozen D0 51/60",
            "temporal_extraction_token_accounting_complete": False,
        },
        "summary": summary,
        "per_conversation": per_conversation,
        "per_question": merged,
    }

    archive = args.archive
    source_dir = archive / "source_artifacts"
    method_dir = archive / "method_snapshot"
    source_dir.mkdir(parents=True)
    method_dir.mkdir(parents=True)
    write(archive / "hypathmem_locomo_final1540.json", final_payload)
    write(archive / "summary.json", archive_summary(merged, temporal, read(args.d2), summary))
    copy_sources(args, source_dir, method_dir)
    (archive / "README.md").write_text(readme(summary), encoding="utf-8")

    manifest_files = {}
    for path in sorted(value for value in archive.rglob("*") if value.is_file() and value.name != "MANIFEST.json"):
        manifest_files[str(path.relative_to(archive))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": "HyPathMem LoCoMo final1540 D0+8.5time_locomo_v2",
        "immutable_result_claim": "1411/1540 = 91.6234%",
        "reporting_status": "final selected post-hoc development result",
        "files": manifest_files,
        "large_external_artifacts": {
            "graph": external_artifact(GRAPH),
            "paths": external_artifact(PATHS),
        },
        "original_sources": {
            "frozen_d0": str(args.baseline.relative_to(ROOT)),
            "temporal_v2_judge": str(args.time.relative_to(ROOT)),
            "temporal_v2_q4": str(args.verified.relative_to(ROOT)),
            "temporal_v2_compiled": str(args.compiled.relative_to(ROOT)),
            "d2_paired60_negative_control": str(args.d2.relative_to(ROOT)),
        },
    }
    write(archive / "MANIFEST.json", manifest)
    print(json.dumps(archive_summary(merged, temporal, read(args.d2), summary), indent=2))


def copy_sources(args: argparse.Namespace, source_dir: Path, method_dir: Path) -> None:
    sources = {
        args.baseline: "frozen_d0_full1540.json",
        args.time: "time_v2_cat2_judge.json",
        args.verified: "time_v2_cat2_q4_verified.json",
        args.compiled: "time_v2_cat2_compiled.json",
        args.d2: "d2_paired60_negative_control.json",
    }
    for source, name in sources.items():
        shutil.copy2(source, source_dir / name)
    methods = [
        ROOT / "src/hytopomem/locomo_time_v8_5_v2.py",
        ROOT / "scripts/152_compile_hypathmem_locomo_time_v8_5_v2.py",
        ROOT / "scripts/153_eval_hypathmem_locomo_time_v8_5_v2.py",
        ROOT / "scripts/154_archive_hypathmem_locomo_d0_time_v2_final.py",
        ROOT / "tests/test_locomo_time_v8_5_v2.py",
    ]
    for source in methods:
        shutil.copy2(source, method_dir / source.name)


def archive_summary(
    rows: list[dict[str, Any]], temporal: dict[str, Any], d2: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    by_category = {}
    for category, group in grouped_by(rows, "category"):
        correct = sum(int(row.get("judge_correct") or 0) for row in group)
        by_category[str(category)] = {"n": len(group), "correct": correct, "accuracy": correct / len(group)}
    return {
        "method": "HyPathMem",
        "dataset": "LoCoMo",
        "version": "D0+8.5time_locomo_v2",
        "num_questions": len(rows),
        "num_correct": summary["num_correct"],
        "accuracy": summary["judge_accuracy"],
        "accuracy_percent": summary["judge_accuracy"] * 100,
        "by_category": by_category,
        "by_final_source": {
            "frozen_D0": {"n": 1504, "correct": 1375},
            "8.5time_locomo_v2_safe_override": {"n": 36, "correct": 36},
        },
        "temporal_v2": temporal["summary"],
        "d2_paired60_negative_control": d2["summary"],
        "d2_selected": False,
        "delta_correct_vs_frozen_d0": 0,
        "reporting_status": "final selected post-hoc development result",
    }


def summarize(rows: list[dict[str, Any]], conversation_id: str | None = None) -> dict[str, Any]:
    n = len(rows)
    correct = sum(int(row.get("judge_correct") or 0) for row in rows)
    generation_tokens = sum(int((row.get("generation_usage") or {}).get("total_tokens") or 0) for row in rows)
    judge_tokens = sum(int((row.get("judge_usage") or {}).get("total_tokens") or 0) for row in rows)
    verifier_tokens = sum(int((row.get("temporal_q4_usage") or {}).get("total_tokens") or 0) for row in rows)
    value = {
        "num_questions": n,
        "judge_skipped": False,
        "macro_f1": sum(float(row.get("lexical_f1") or 0.0) for row in rows) / n,
        "macro_bleu1": sum(float(row.get("bleu1") or 0.0) for row in rows) / n,
        "generation_tokens": generation_tokens,
        "verifier_tokens": verifier_tokens,
        "judge_tokens": judge_tokens,
        "accounted_total_tokens": generation_tokens + verifier_tokens + judge_tokens,
        "temporal_extraction_tokens_recorded": False,
        "generation_api_seconds": sum(float(row.get("generation_elapsed_seconds") or 0.0) for row in rows),
        "verifier_api_seconds": sum(float(row.get("temporal_q4_elapsed_seconds") or 0.0) for row in rows),
        "judge_api_seconds": sum(float(row.get("judge_elapsed_seconds") or 0.0) for row in rows),
        "retrieval_hit": sum(bool(row.get("retrieval_hit")) for row in rows) / n,
        "retrieval_recall": sum(float(row.get("retrieval_recall") or 0.0) for row in rows) / n,
        "retrieval_full_cover": sum(bool(row.get("retrieval_full_cover")) for row in rows) / n,
        "retrieval_avg_tokens": sum(float(row.get("retrieval_tokens") or 0.0) for row in rows) / n,
        "retrieval_avg_path_len": sum(float(row.get("retrieval_path_len") or 0.0) for row in rows) / n,
        "num_correct": correct,
        "judge_accuracy": correct / n,
        "temporal_branch_activated": sum(bool(row.get("temporal_branch_activated")) for row in rows),
    }
    if conversation_id is not None:
        value = {"conversation_id": conversation_id, **value}
    return value


def grouped(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    return grouped_by(rows, "conversation_id")


def grouped_by(rows: list[dict[str, Any]], key: str) -> list[tuple[Any, list[dict[str, Any]]]]:
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row.get(key)].append(row)
    return sorted(buckets.items(), key=lambda item: str(item[0]))


def sum_usage(*values: dict[str, Any]) -> dict[str, int]:
    return {
        key: sum(int(value.get(key) or 0) for value in values)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def tokenize_metric_text(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\s]", "", str(text).lower())
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.split()


def multiset_overlap(left: list[str], right: list[str]) -> int:
    counts: dict[str, int] = {}
    for token in right:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in left:
        count = counts.get(token, 0)
        if count > 0:
            overlap += 1
            counts[token] = count - 1
    return overlap


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens, gold_tokens = tokenize_metric_text(prediction), tokenize_metric_text(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = multiset_overlap(pred_tokens, gold_tokens)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, gold: str) -> float:
    pred_tokens, gold_tokens = tokenize_metric_text(prediction), tokenize_metric_text(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    precision = multiset_overlap(pred_tokens, gold_tokens) / len(pred_tokens)
    penalty = 1.0 if len(pred_tokens) > len(gold_tokens) else math.exp(1 - len(gold_tokens) / len(pred_tokens))
    return precision * penalty


def external_artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def readme(summary: dict[str, Any]) -> str:
    return (
        "# HyPathMem LoCoMo Final Result\n\n"
        "Selected system: frozen D0 Top20 retrieval and GPT-4.1-mini answer generation, "
        "with fail-closed `8.5time_locomo_v2` replacement for 36 verified Cat2 answers.\n\n"
        f"- Accuracy: **{summary['num_correct']}/{summary['num_questions']} = "
        f"{summary['judge_accuracy'] * 100:.4f}%**\n"
        "- Time V2: 36/36 takeover answers correct; fix=0, break=0.\n"
        "- D2 was excluded: paired60 scored 49/60 versus frozen D0 51/60.\n"
        "- Generation model: GPT-4.1-mini.\n"
        "- Judge model: GPT-4o-mini, one judgment per answer.\n"
        "- Retrieval: frozen HyPathMem Top20 paths over one graph per LoCoMo conversation.\n\n"
        "## Reporting status\n\n"
        "This is the final selected post-hoc development result. Time V2 was developed after "
        "diagnosing LoCoMo V1 failures. It is not a preregistered or untouched test-set result.\n\n"
        "The graph and path files are referenced by path and SHA256 in `MANIFEST.json`; they are "
        "not duplicated in this archive. Temporal extraction token usage was not recorded by the "
        "source runner, so token accounting is explicitly marked incomplete.\n"
    )


if __name__ == "__main__":
    main()
