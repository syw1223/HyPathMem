#!/usr/bin/env python3
"""Evaluate LoCoMo temporal v2 with verified takeover and frozen D0 fallback."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.eval.judge import OpenAICompatibleLLMJudge
from hytopomem.eval.openai_compatible import OpenAICompatibleChatClient
from hytopomem.locomo_time_v8_5_v2 import VERSION


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
        "--compiled",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_locomo_8.5time_v2/cat2_compiled.json"),
    )
    parser.add_argument(
        "--verified",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_locomo_8.5time_v2/cat2_q4_verified.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qa/hypathmem_locomo_8.5time_v2/cat2_verified_solver_judge_gpt4omini.json"),
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="E_MEM_API_KEY")
    parser.add_argument("--base-url-env", default="E_MEM_BASE_URL")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--max-judge-tokens", type=int, default=160)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("compiled", "verified", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    if not os.environ.get(args.api_key_env):
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    compiled = read(args.compiled)
    verified = read(args.verified)
    verifier_by_id = {str(row["question_id"]): row for row in verified["rows"]}
    rows = read(args.output).get("per_question", []) if args.resume and args.output.exists() else []
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --resume")
    done = {str(row["question_id"]) for row in rows}
    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=120.0,
    )
    judge = OpenAICompatibleLLMJudge(client=client, model=args.judge_model, max_tokens=args.max_judge_tokens)
    started = time.perf_counter()
    for index, source in enumerate(compiled["rows"], start=1):
        question_id = str(source["question_id"])
        if question_id in done:
            continue
        verifier = verifier_by_id[question_id]
        active = bool(verifier.get("safe_to_override_d0"))
        if active:
            prediction = str(source["pre_verifier"]["candidate_answer"])
            judged = judge.judge_with_metadata(
                str(source["question"]), str(source.get("gold_answer") or ""), prediction
            )
            route = "8.5time_locomo_v2_verified_solver"
        else:
            prediction = str(source["frozen_d0"]["prediction"])
            correct = int(source["frozen_d0"]["judge_correct"] or 0)
            judged = {
                "judge_correct": correct,
                "judge_label": "CORRECT" if correct else "WRONG",
                "judge_reason": "reused frozen D0 result",
                "judge_usage": {},
                "judge_elapsed_seconds": 0.0,
            }
            route = "frozen_D0_reused"
        before = int(source["frozen_d0"]["judge_correct"] or 0)
        after = int(judged["judge_correct"] or 0)
        rows.append(
            {
                "question_id": question_id,
                "conversation_id": source.get("conversation_id"),
                "question": source["question"],
                "gold_answer": source.get("gold_answer"),
                "category": 2,
                "prediction": prediction,
                "answer_type": source["pre_verifier"].get("candidate_answer_type") if active else None,
                "route": route,
                "temporal_branch_activated": active,
                "fallback_reused": not active,
                "frozen_d0_prediction": source["frozen_d0"]["prediction"],
                "frozen_d0_judge_correct": before,
                "comparison": compare(before, after),
                **judged,
            }
        )
        write(args.output, build_payload(rows, args, started))
        print(f"processed {index}/{len(compiled['rows'])} qid={question_id} route={route}", flush=True)
    write(args.output, build_payload(rows, args, started))


def compare(before: int, after: int) -> str:
    if not before and after:
        return "fix"
    if before and not after:
        return "break"
    return "unchanged_correct" if before else "unchanged_wrong"


def build_payload(rows: list[dict[str, Any]], args: argparse.Namespace, started: float) -> dict[str, Any]:
    correct = sum(int(row.get("judge_correct") or 0) for row in rows)
    baseline = sum(int(row.get("frozen_d0_judge_correct") or 0) for row in rows)
    return {
        "metadata": {
            "version": VERSION,
            "dataset": "LoCoMo",
            "selection": "official_category_2",
            "fallback": "frozen_D0_reused",
            "judge_model": args.judge_model,
            "judge_repetitions": 1,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "summary": {
            "num_questions": len(rows),
            "num_correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
            "frozen_d0_correct": baseline,
            "frozen_d0_accuracy": baseline / len(rows) if rows else 0.0,
            "temporal_branch_activated": sum(row["temporal_branch_activated"] for row in rows),
            "fallback_reused": sum(row["fallback_reused"] for row in rows),
            "fix": sum(row["comparison"] == "fix" for row in rows),
            "break": sum(row["comparison"] == "break" for row in rows),
            "unchanged_correct": sum(row["comparison"] == "unchanged_correct" for row in rows),
            "unchanged_wrong": sum(row["comparison"] == "unchanged_wrong" for row in rows),
        },
        "per_question": rows,
    }


if __name__ == "__main__":
    main()
