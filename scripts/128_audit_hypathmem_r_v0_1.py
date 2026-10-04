from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from common import read_json, resolve_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HyPathMem LongMemEval errors without changing retrieval.")
    parser.add_argument(
        "--qa-results",
        default="outputs/qa/longmemeval_v3_10_cardquota_light_top50_c1_window_chunks_80k_gpt41mini_judge_gpt4omini.json",
    )
    parser.add_argument("--output", default="outputs/reconstruction/hypathmem_r_v0_1/audit_top50.json")
    parser.add_argument("--force", action="store_true", help="Allow replacing this new audit artifact only.")
    args = parser.parse_args()

    source = resolve_path(args.qa_results)
    output = resolve_path(args.output)
    refuse_existing(output, args.force)
    payload = read_json(source)
    rows = list(payload.get("per_question") or [])
    audit = build_audit(rows)
    audit["source"] = {"path": str(source), "sha256": sha256(source)}
    write_json(audit, output)
    print(render_summary(audit))
    print(f"wrote {output}")


def build_audit(rows: list[dict]) -> dict:
    categories: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        categories[str(row.get("question_type") or "unknown")].append(row)
    return {
        "version": "hypathmem_r_audit_v0_1",
        "overall": summarize(rows),
        "by_question_type": {name: summarize(items) for name, items in sorted(categories.items())},
        "error_rows": [error_record(row) for row in rows if not correct(row)],
    }


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    num_correct = sum(correct(row) for row in rows)
    full = [row for row in rows if truthy(row.get("retrieval_full_cover"))]
    partial = [
        row for row in rows
        if truthy(row.get("retrieval_hit")) and not truthy(row.get("retrieval_full_cover"))
    ]
    no_hit = [row for row in rows if not truthy(row.get("retrieval_hit"))]
    wrong = [row for row in rows if not correct(row)]
    return {
        "num_questions": total,
        "num_correct": num_correct,
        "accuracy": ratio(num_correct, total),
        "retrieval_hit": ratio(sum(truthy(row.get("retrieval_hit")) for row in rows), total),
        "retrieval_full_cover": ratio(len(full), total),
        "full_cover": bucket(full),
        "hit_not_full_cover": bucket(partial),
        "no_hit": bucket(no_hit),
        "wrong": len(wrong),
        "wrong_despite_hit": sum(truthy(row.get("retrieval_hit")) for row in wrong),
        "wrong_despite_full_cover": sum(truthy(row.get("retrieval_full_cover")) for row in wrong),
        "wrong_abstention": sum(truthy(row.get("is_abstention")) for row in wrong),
    }


def bucket(rows: list[dict]) -> dict:
    num_correct = sum(correct(row) for row in rows)
    return {
        "n": len(rows),
        "correct": num_correct,
        "wrong": len(rows) - num_correct,
        "conditional_accuracy": ratio(num_correct, len(rows)),
    }


def error_record(row: dict) -> dict:
    if truthy(row.get("retrieval_full_cover")):
        evidence_state = "full_cover"
    elif truthy(row.get("retrieval_hit")):
        evidence_state = "partial"
    else:
        evidence_state = "no_hit"
    return {
        "question_id": row.get("question_id"),
        "question_type": row.get("question_type"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "prediction": row.get("prediction"),
        "evidence_state": evidence_state,
        "is_abstention": truthy(row.get("is_abstention")),
        "judge_reason": row.get("judge_reason"),
        "context_chars": row.get("context_chars"),
    }


def render_summary(audit: dict) -> str:
    overall = audit["overall"]
    return (
        f"questions={overall['num_questions']} correct={overall['num_correct']} "
        f"accuracy={overall['accuracy']:.4f} hit={overall['retrieval_hit']:.4f} "
        f"full_cover={overall['retrieval_full_cover']:.4f} "
        f"wrong_hit={overall['wrong_despite_hit']} wrong_full={overall['wrong_despite_full_cover']}"
    )


def correct(row: dict) -> bool:
    return int(row.get("judge_correct") or 0) == 1


def truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def refuse_existing(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {path}; pass --force explicitly")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
