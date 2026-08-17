#!/usr/bin/env python3
"""Evaluate Q1-Q4 with frozen-D0 reuse on every failed temporal branch."""

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
    spec = importlib.util.spec_from_file_location("hytopomem_qa121_qwen_temporal", path)
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


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def first_identity_eligible(source: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (
            hypothesis
            for hypothesis in source["hypotheses"]
            if hypothesis["source_validation"].get("eligible_for_normalization")
        ),
        None,
    )


def branch(source: dict[str, Any], variant: str, verifier: dict[str, Any]) -> dict[str, Any]:
    hypothesis = first_identity_eligible(source)
    if variant == "q1" and hypothesis:
        return {"active": True, "kind": "reader", "context": hypothesis["q1_context"], "route": "Q1_joint_binding_raw"}
    if variant == "q2" and hypothesis:
        return {"active": True, "kind": "reader", "context": hypothesis["q2_context"], "route": "Q2_normalized_packet"}
    if variant == "q3" and source["pre_verifier"]["eligible_to_call_solution_verifier"]:
        return {
            "active": True,
            "kind": "solver",
            "prediction": source["pre_verifier"]["candidate_answer"],
            "route": "Q3_deterministic_solver",
        }
    if variant == "q4" and verifier.get("safe_to_override_d0"):
        return {
            "active": True,
            "kind": "solver",
            # The verifier is a gate, not the answer source. Always use the
            # current compiled solver output so a later formatting fix cannot
            # be shadowed by a stale verifier artifact.
            "prediction": source["pre_verifier"]["candidate_answer"],
            "route": "Q4_verified_solver",
        }
    return {"active": False, "kind": "fallback", "route": "frozen_D0_reused"}


def build_payload(rows: list[dict[str, Any]], args: argparse.Namespace, started: float) -> dict[str, Any]:
    correct = sum(int(row.get("judge_correct") or 0) for row in rows)
    baseline = sum(int(row.get("frozen_d0_judge_correct") or 0) for row in rows)
    fixes = sum(not row.get("frozen_d0_judge_correct") and row.get("judge_correct") for row in rows)
    breaks = sum(row.get("frozen_d0_judge_correct") and not row.get("judge_correct") for row in rows)
    return {
        "metadata": {
            "version": "hypathmem_temporal_v0_2_qwen",
            "variant": args.variant,
            "fallback": "frozen_D0_reused",
            "generation_model": args.model if args.variant in {"q1", "q2"} else None,
            "judge_model": args.judge_model,
            "judge_repetitions": 1,
            "reuse_from": str(args.reuse_from) if args.reuse_from else None,
            "targeted_question_ids": list(args.question_id),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "summary": {
            "num_questions": len(rows),
            "num_correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
            "frozen_d0_accuracy": baseline / len(rows) if rows else 0.0,
            "delta_vs_frozen_d0": (correct - baseline) / len(rows) if rows else 0.0,
            "temporal_branch_activated": sum(row["temporal_branch_activated"] for row in rows),
            "fallback_reused": sum(row["fallback_reused"] for row in rows),
            "fixed_d0_wrong": fixes,
            "broke_d0_correct": breaks,
        },
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["q1", "q2", "q3", "q4"], required=True)
    parser.add_argument(
        "--compiled",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_temporal_v0_2_qwen/paired20_q1_q4_compiled.json"),
    )
    parser.add_argument(
        "--verified",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_temporal_v0_2_qwen/paired20_q4_verified_gpu4_v3.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env", default="MRAGENT_ANSWER_BASE_URL")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    for name in ("compiled", "verified", "output", "reuse_from"):
        path = getattr(args, name)
        if path is None:
            continue
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    compiled = read_json(args.compiled)
    verified = read_json(args.verified)
    verifier_by_id = {row["question_id"]: row for row in verified["rows"]}
    log_path = args.output.with_suffix(".log")
    selected_ids = set(args.question_id)
    source_rows = [
        row for row in compiled["rows"] if not selected_ids or row["question_id"] in selected_ids
    ]
    rows = []
    if args.reuse_from:
        rows = [
            row
            for row in read_json(args.reuse_from).get("per_question", [])
            if row["question_id"] not in {source["question_id"] for source in source_rows}
        ]
    elif args.resume and args.output.exists():
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
    append_log(log_path, f"START variant={args.variant} n={len(source_rows)} reused={len(rows)}")
    for index, source in enumerate(source_rows, start=1):
        if source["question_id"] in done:
            continue
        decision = branch(source, args.variant, verifier_by_id[source["question_id"]])
        if not decision["active"]:
            prediction = source["frozen_d0"]["prediction"]
            judged = {
                "judge_correct": int(source["frozen_d0"]["judge_correct"] or 0),
                "judge_label": "CORRECT" if source["frozen_d0"]["judge_correct"] else "WRONG",
                "judge_reason": "reused frozen D0 result",
                "judge_usage": {},
                "judge_elapsed_seconds": 0.0,
            }
            generation_usage = {}
        else:
            generation_usage = {}
            if decision["kind"] == "reader":
                result = client.chat_completion_with_metadata(
                    model=args.model,
                    messages=[
                        ChatMessage(role="system", content=qa121.ANSWER_SYSTEM_PROMPT),
                        ChatMessage(
                            role="user",
                            content=qa121.ANSWER_USER_PROMPT.format(
                                question=source["question"],
                                question_type="temporal-reasoning",
                                question_date=source["question_date"],
                                task_instruction=qa121.task_instruction(
                                    source["question"], "temporal-reasoning", source["question_date"]
                                ),
                                private_quant_instruction=qa121.private_quant_instruction("basic"),
                                context=decision["context"],
                            ),
                        ),
                    ],
                    temperature=0.0,
                    max_tokens=128,
                )
                prediction = result.content.strip()
                generation_usage = qa121.normalize_usage(result.usage)
            else:
                prediction = str(decision["prediction"])
            judged = qa121.judge_answer(
                client=client,
                model=args.judge_model,
                question=source["question"],
                question_type="temporal-reasoning",
                gold_answer=str(source["gold_answer"] or ""),
                prediction=prediction,
                is_abstention=bool(source["is_abstention"]),
                max_tokens=160,
            )
        row = {
            "question_id": source["question_id"],
            "question": source["question"],
            "gold_answer": source["gold_answer"],
            "prediction": prediction,
            "route": decision["route"],
            "temporal_branch_activated": bool(decision["active"]),
            "fallback_reused": not decision["active"],
            "frozen_d0_prediction": source["frozen_d0"]["prediction"],
            "frozen_d0_judge_correct": int(source["frozen_d0"]["judge_correct"] or 0),
            "generation_usage": generation_usage,
            **judged,
        }
        rows.append(row)
        current = build_payload(rows, args, started)
        write_json(args.output, current)
        message = (
            f"processed {index}/{len(source_rows)} qid={source['question_id']} "
            f"route={decision['route']} judge={row['judge_label']} acc={current['summary']['accuracy']:.4f}"
        )
        print(message, flush=True)
        append_log(log_path, message)
    final = build_payload(rows, args, started)
    write_json(args.output, final)
    append_log(log_path, "DONE " + json.dumps(final["summary"], ensure_ascii=False))
    print(json.dumps(final["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
