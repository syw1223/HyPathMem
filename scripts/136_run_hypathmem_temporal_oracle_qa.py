#!/usr/bin/env python3
"""Run O1/O2 generation and O1/O2/O3 judging on the fixed oracle15 set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient


def load_qa121() -> ModuleType:
    path = ROOT / "scripts" / "121_run_longmemeval_qa_eval.py"
    spec = importlib.util.spec_from_file_location("hytopomem_qa121_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
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


def build_payload(rows: list[dict[str, Any]], args: argparse.Namespace, started: float) -> dict[str, Any]:
    correct = sum(int(row.get("judge_correct") or 0) for row in rows)
    d0_correct = sum(int(row.get("frozen_d0_judge_correct") or 0) for row in rows)
    by_cohort: dict[str, dict[str, int | float]] = {}
    for cohort in sorted({str(row["cohort"]) for row in rows}):
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        cohort_correct = sum(int(row.get("judge_correct") or 0) for row in cohort_rows)
        by_cohort[cohort] = {
            "num_questions": len(cohort_rows),
            "num_correct": cohort_correct,
            "accuracy": cohort_correct / len(cohort_rows),
        }
    return {
        "metadata": {
            "version": "hypathmem_temporal_v0_2_oracle",
            "diagnostic_oracle_only": True,
            "variant": args.variant,
            "generation_model": args.model if args.variant != "o3" else None,
            "judge_model": args.judge_model,
            "judge_repetitions": 1,
            "fallback": "frozen_D0",
            "elapsed_seconds": time.perf_counter() - started,
        },
        "summary": {
            "num_questions": len(rows),
            "num_correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
            "frozen_d0_accuracy_on_same_rows": d0_correct / len(rows) if rows else 0.0,
            "delta_vs_frozen_d0": (correct - d0_correct) / len(rows) if rows else 0.0,
            "by_cohort": by_cohort,
        },
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["o1", "o2", "o3"], required=True)
    parser.add_argument(
        "--packets",
        type=Path,
        default=Path(
            "outputs/reconstruction/hypathmem_temporal_v0_2_oracle/oracle15_o1_o3_packets.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env", default="MRAGENT_ANSWER_BASE_URL")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--max-judge-tokens", type=int, default=160)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.packets.is_absolute():
        args.packets = ROOT / args.packets
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    if args.log is None:
        args.log = args.output.with_suffix(".log")
    elif not args.log.is_absolute():
        args.log = ROOT / args.log

    payload = read_json(args.packets)
    source_rows = payload["rows"]
    if args.dry_run:
        sample = source_rows[0]
        print(sample[args.variant].get("context", sample[args.variant].get("final_prediction")))
        return

    rows = []
    if args.resume and args.output.exists():
        rows = read_json(args.output).get("per_question", [])
    elif args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --resume")
    done = {row["question_id"] for row in rows}
    qa121 = load_qa121()
    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=90.0,
    )
    started = time.perf_counter()
    append_log(args.log, f"START variant={args.variant} n={len(source_rows)}")

    for index, source in enumerate(source_rows, start=1):
        if source["question_id"] in done:
            continue
        generation_usage: dict[str, int] = {}
        generation_elapsed = 0.0
        if args.variant == "o3":
            prediction = source["o3"]["final_prediction"]
            route = source["o3"]["route"]
        else:
            context = source[args.variant]["context"]
            answer_result = client.chat_completion_with_metadata(
                model=args.model,
                messages=[
                    ChatMessage(role="system", content=qa121.ANSWER_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=qa121.ANSWER_USER_PROMPT.format(
                            question=source["question"],
                            question_type="temporal-reasoning",
                            question_date=source.get("question_date") or "unknown",
                            task_instruction=qa121.task_instruction(
                                source["question"], "temporal-reasoning", source.get("question_date") or ""
                            ),
                            private_quant_instruction=qa121.private_quant_instruction("basic"),
                            context=context,
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=args.max_answer_tokens,
            )
            prediction = answer_result.content.strip()
            route = f"oracle_{args.variant}_reader"
            generation_usage = qa121.normalize_usage(answer_result.usage)
            generation_elapsed = answer_result.elapsed_seconds
        judged = qa121.judge_answer(
            client=client,
            model=args.judge_model,
            question=source["question"],
            question_type="temporal-reasoning",
            gold_answer=source["gold_answer"],
            prediction=prediction,
            is_abstention=bool(source.get("is_abstention", False)),
            max_tokens=args.max_judge_tokens,
        )
        row = {
            "question_id": source["question_id"],
            "cohort": source["cohort"],
            "question": source["question"],
            "gold_answer": source["gold_answer"],
            "prediction": prediction,
            "route": route,
            "oracle_top50_full_cover": source["oracle_top50_full_cover"],
            "solver_verified": bool(source["o3"]["solver_verified"]),
            "solver_trace": source["o3"]["solver_trace"],
            "frozen_d0_prediction": source["frozen_d0"]["prediction"],
            "frozen_d0_judge_correct": int(source["frozen_d0"]["judge_correct"] or 0),
            "generation_usage": generation_usage,
            "generation_elapsed_seconds": generation_elapsed,
            **judged,
        }
        rows.append(row)
        current = build_payload(rows, args, started)
        write_json(args.output, current)
        message = (
            f"processed {index}/{len(source_rows)} qid={source['question_id']} "
            f"judge={row['judge_label']} acc={current['summary']['accuracy']:.4f}"
        )
        print(message, flush=True)
        append_log(args.log, message)

    final = build_payload(rows, args, started)
    write_json(args.output, final)
    append_log(args.log, "DONE " + json.dumps(final["summary"], ensure_ascii=False))
    print(json.dumps(final["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
