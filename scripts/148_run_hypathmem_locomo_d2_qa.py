#!/usr/bin/env python3
"""Run paired LoCoMo D2 QA over the frozen Top20 candidate set."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.eval.judge import OpenAICompatibleLLMJudge
from hytopomem.eval.openai_compatible import OpenAICompatibleChatClient
from hytopomem.eval.qa_runner import OpenAICompatibleQARunner


DEFAULT_PACKS = "outputs/reconstruction/hypathmem_locomo_d2_v1/top20_structured_raw_packs.json"
DEFAULT_BASELINE = (
    "outputs/qa/full_v3_9_expand120_light_quota_top20_hybrid_v11_"
    "chatanywhere_gpt41mini_judge_gpt4omini.json"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", default=DEFAULT_PACKS)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument(
        "--output",
        default="outputs/qa/hypathmem_locomo_d2_v1/d2_top20_gpt41mini_judge_gpt4omini.json",
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="E_MEM_API_KEY")
    parser.add_argument("--base-url-env", default="E_MEM_BASE_URL")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--max-context-chars", type=int, default=80_000)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--max-judge-tokens", type=int, default=160)
    parser.add_argument("--include-category2", action="store_true")
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    packs_path = resolve(args.packs)
    baseline_path = resolve(args.baseline)
    output_path = resolve(args.output)
    packs = read_json(packs_path)
    baseline = read_json(baseline_path)
    pack_by_id = {str(row["question_id"]): row for row in packs["rows"]}
    selected = [
        row for row in baseline["per_question"]
        if args.include_category2 or int(row.get("category") or 0) != 2
    ]
    if args.per_category:
        rng = random.Random(args.seed)
        sampled = []
        for category in sorted({int(row.get("category") or 0) for row in selected}):
            group = sorted(
                (row for row in selected if int(row.get("category") or 0) == category),
                key=lambda row: str(row["question_id"]),
            )
            if len(group) < args.per_category:
                raise ValueError(f"Category {category} has only {len(group)} questions")
            sampled.extend(rng.sample(group, args.per_category))
        selected = sorted(sampled, key=lambda row: (int(row.get("category") or 0), str(row["question_id"])))
    selected = selected[args.offset : args.offset + args.limit if args.limit else None]
    missing = [str(row["question_id"]) for row in selected if str(row["question_id"]) not in pack_by_id]
    if missing:
        raise KeyError(f"Missing D2 packs for {len(missing)} rows: {missing[:5]}")

    if args.dry_run:
        sample = selected[0]
        context = d2_context(pack_by_id[str(sample["question_id"])], args.max_context_chars)
        print(json.dumps({"n": len(selected), "first_question": sample["question"], "context_chars": len(context)}, indent=2))
        print(context[:3000])
        return

    if not os.environ.get(args.api_key_env):
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    existing = read_json(output_path).get("per_question", []) if args.resume and output_path.exists() else []
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_path}; use --resume")
    done = {str(row["question_id"]) for row in existing}
    rows = list(existing)
    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=120.0,
    )
    answerer = OpenAICompatibleQARunner(
        client=client,
        model=args.model,
        max_tokens=args.max_answer_tokens,
        answer_protocol="default",
    )
    judge = OpenAICompatibleLLMJudge(
        client=client,
        model=args.judge_model,
        max_tokens=args.max_judge_tokens,
    )
    log_path = output_path.with_suffix(".log")
    started = time.perf_counter()
    append_log(log_path, f"START n={len(selected)} resume={args.resume} topk=20")
    for index, base in enumerate(selected, start=1):
        question_id = str(base["question_id"])
        if question_id in done:
            continue
        context = d2_context(pack_by_id[question_id], args.max_context_chars)
        generated = answerer.answer_with_metadata(
            str(base["question"]),
            context,
            category=int(base.get("category") or 0),
            question_type=str(base.get("question_type") or ""),
        )
        judged = judge.judge_with_metadata(
            str(base["question"]),
            str(base.get("gold_answer") or ""),
            generated.content,
        )
        row = {
            "question_id": question_id,
            "conversation_id": base.get("conversation_id"),
            "question": base["question"],
            "gold_answer": base.get("gold_answer"),
            "category": base.get("category"),
            "question_type": base.get("question_type"),
            "prediction": generated.content,
            "context_chars": len(context),
            "pack_answerability": pack_by_id[question_id]["pack"].get("answerability"),
            "baseline_prediction": base.get("prediction"),
            "baseline_judge_correct": int(base.get("judge_correct") or 0),
            "baseline_retrieval_hit": bool(base.get("retrieval_hit")),
            "baseline_retrieval_full_cover": bool(base.get("retrieval_full_cover")),
            "generation_usage": generated.usage,
            "generation_elapsed_seconds": generated.elapsed_seconds,
            **judged,
        }
        row["comparison"] = comparison(row["baseline_judge_correct"], row["judge_correct"])
        rows.append(row)
        payload = build_payload(rows, args, packs_path, baseline_path, client.base_url, started)
        write_json(output_path, payload)
        message = (
            f"processed {index}/{len(selected)} qid={question_id} cat={row['category']} "
            f"judge={row['judge_label']} acc={payload['summary']['accuracy']:.4f} "
            f"fix={payload['summary']['fix']} break={payload['summary']['break']}"
        )
        append_log(log_path, message)
        print(message, flush=True)
    write_json(output_path, build_payload(rows, args, packs_path, baseline_path, client.base_url, started))


def d2_context(pack_row: dict[str, Any], max_chars: int) -> str:
    payload = json.loads(pack_row["answer_context"])
    payload["instructions"] = [
        "Use only claims supported by their attached exact raw quotes.",
        "Match named speakers, event identity, session, and time before answering.",
        "Use hyperpath provenance only to organize evidence; it is not itself a fact.",
        "Resolve relative dates from the raw quote's message time.",
        "Return a concise final answer.",
    ]
    payload["experiment_variant"] = "LoCoMo_D2_structured_claims_plus_exact_raw_quotes"
    while True:
        rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(rendered) <= max_chars or not payload.get("evidence"):
            return rendered
        removed = payload["evidence"].pop()
        removed_id = removed.get("unit_id")
        for group in payload.get("evidence_groups") or []:
            group["evidence_unit_ids"] = [
                value for value in group.get("evidence_unit_ids") or [] if value != removed_id
            ]


def comparison(before: int, after: int) -> str:
    if not before and after:
        return "fix"
    if before and not after:
        return "break"
    return "unchanged_correct" if before else "unchanged_wrong"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(int(row.get("judge_correct") or 0) for row in rows)
    baseline_correct = sum(int(row.get("baseline_judge_correct") or 0) for row in rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("category"))].append(row)
    return {
        "num_questions": len(rows),
        "num_correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "baseline_num_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / len(rows) if rows else 0.0,
        "fix": sum(row.get("comparison") == "fix" for row in rows),
        "break": sum(row.get("comparison") == "break" for row in rows),
        "unchanged_correct": sum(row.get("comparison") == "unchanged_correct" for row in rows),
        "unchanged_wrong": sum(row.get("comparison") == "unchanged_wrong" for row in rows),
        "by_category": {
            category: {
                "n": len(items),
                "correct": sum(int(item.get("judge_correct") or 0) for item in items),
                "baseline_correct": sum(int(item.get("baseline_judge_correct") or 0) for item in items),
            }
            for category, items in sorted(groups.items())
        },
    }


def build_payload(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    packs_path: Path,
    baseline_path: Path,
    base_url: str,
    started: float,
) -> dict[str, Any]:
    return {
        "metadata": {
            "version": "hypathmem_locomo_d2_v1",
            "dataset": "LoCoMo",
            "frozen_topk": 20,
            "packs": str(packs_path),
            "baseline": str(baseline_path),
            "generation_model": args.model,
            "judge_model": args.judge_model,
            "base_url": base_url,
            "categories": [1, 3, 4] if not args.include_category2 else [1, 2, 3, 4],
            "per_category": args.per_category,
            "sample_seed": args.seed if args.per_category else None,
            "retrieval_changed": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "summary": summarize(rows),
        "per_question": rows,
    }


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


if __name__ == "__main__":
    main()
