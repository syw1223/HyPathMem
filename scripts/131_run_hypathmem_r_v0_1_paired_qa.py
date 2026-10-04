from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient


DEFAULT_PACKS = "outputs/reconstruction/hypathmem_r_v0_1/top50_structured_raw_packs.json"
DEFAULT_BASELINE = "outputs/qa/longmemeval_v3_10_cardquota_light_top50_c1_window_chunks_80k_gpt41mini_judge_gpt4omini.json"
DEFAULT_MANIFEST = "outputs/reconstruction/hypathmem_r_v0_1/paired120_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired HyPathMem-R D1/D2 QA on a fixed selection.")
    parser.add_argument("--variant", choices=["d1", "d2"], required=True)
    parser.add_argument("--packs", default=DEFAULT_PACKS)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", default="")
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--selection",
        choices=["stratified", "all-non-temporal"],
        default="stratified",
        help="Use the original stratified sample or every non-temporal baseline row.",
    )
    parser.add_argument("--excluded-type", default="temporal-reasoning")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="E_MEM_API_KEY")
    parser.add_argument("--base-url-env", default="E_MEM_BASE_URL")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--max-judge-tokens", type=int, default=160)
    parser.add_argument("--max-context-chars", type=int, default=80_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qa121 = load_qa121()
    packs_payload = read_json(resolve(args.packs))
    baseline_payload = read_json(resolve(args.baseline))
    pack_by_id = {str(row["question_id"]): row for row in packs_payload["rows"]}
    baseline_rows = baseline_payload.get("per_question") or baseline_payload.get("rows") or []
    baseline_by_id = {str(row["question_id"]): row for row in baseline_rows}
    manifest_path = resolve(args.manifest)
    manifest = load_or_create_manifest(
        manifest_path,
        baseline_rows,
        per_type=args.per_type,
        seed=args.seed,
        selection=args.selection,
        excluded_type=args.excluded_type,
    )
    selected_ids = [str(row["question_id"]) for row in manifest["rows"]]
    missing = [qid for qid in selected_ids if qid not in pack_by_id or qid not in baseline_by_id]
    if missing:
        raise KeyError(f"sample contains {len(missing)} missing pack/baseline rows: {missing[:5]}")

    if args.dry_run:
        first = pack_by_id[selected_ids[0]]
        context = variant_context(first, args.variant, max_chars=args.max_context_chars)
        print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
        print(f"variant={args.variant} first_qid={selected_ids[0]} context_chars={len(context)}")
        print(context[:2000])
        return

    output_path = resolve(args.output)
    log_path = resolve(args.log) if args.log else output_path.with_suffix(".log")
    rows = load_existing_rows(output_path) if args.resume else []
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing result: {output_path}; use --resume")
    done = {str(row["question_id"]) for row in rows}
    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=90.0,
    )
    started = time.perf_counter()
    append_log(log_path, f"START variant={args.variant} questions={len(selected_ids)} resume={args.resume}")

    for index, qid in enumerate(selected_ids, start=1):
        if qid in done:
            continue
        baseline = baseline_by_id[qid]
        pack_row = pack_by_id[qid]
        context = variant_context(pack_row, args.variant, max_chars=args.max_context_chars)
        answer_result = client.chat_completion_with_metadata(
            model=args.model,
            messages=[
                ChatMessage(role="system", content=qa121.ANSWER_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=qa121.ANSWER_USER_PROMPT.format(
                        question=baseline["question"],
                        question_type=baseline.get("question_type", "unknown"),
                        question_date=baseline.get("question_date", "unknown"),
                        task_instruction=qa121.task_instruction(
                            question=str(baseline["question"]),
                            question_type=str(baseline.get("question_type", "")),
                            question_date=str(baseline.get("question_date", "")),
                            quant_mode="basic",
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
        judged = qa121.judge_answer(
            client=client,
            model=args.judge_model,
            question=str(baseline["question"]),
            question_type=str(baseline.get("question_type", "")),
            gold_answer=str(baseline.get("gold_answer") or ""),
            prediction=prediction,
            is_abstention=bool(baseline.get("is_abstention")),
            max_tokens=args.max_judge_tokens,
        )
        row = {
            "question_id": qid,
            "question": baseline["question"],
            "question_type": baseline.get("question_type", ""),
            "question_date": baseline.get("question_date", ""),
            "gold_answer": baseline.get("gold_answer", ""),
            "is_abstention": bool(baseline.get("is_abstention")),
            "prediction": prediction,
            "context_chars": len(context),
            "pack_answerability": pack_row["pack"].get("answerability"),
            "baseline_prediction": baseline.get("prediction", ""),
            "baseline_judge_correct": int(baseline.get("judge_correct") or 0),
            "baseline_retrieval_hit": bool(baseline.get("retrieval_hit")),
            "baseline_retrieval_full_cover": bool(baseline.get("retrieval_full_cover")),
            "generation_usage": qa121.normalize_usage(answer_result.usage),
            "generation_elapsed_seconds": answer_result.elapsed_seconds,
            **judged,
        }
        row["total_usage"] = qa121.add_usage(row["generation_usage"], row.get("judge_usage", {}))
        row["total_api_elapsed_seconds"] = row["generation_elapsed_seconds"] + float(
            row.get("judge_elapsed_seconds", 0.0)
        )
        rows.append(row)
        payload = build_payload(rows, args, manifest_path, client.base_url, started)
        write_json(output_path, payload)
        running = payload["summary_all"]
        message = (
            f"processed {index}/{len(selected_ids)} qid={qid} type={row['question_type']} "
            f"judge={row['judge_label']} acc={running['judge_accuracy']:.4f}"
        )
        append_log(log_path, message)
        print(message, flush=True)

    payload = build_payload(rows, args, manifest_path, client.base_url, started)
    write_json(output_path, payload)
    append_log(log_path, "DONE " + json.dumps(payload["summary_all"], ensure_ascii=False))
    print(json.dumps(payload["summary_all"], ensure_ascii=False, indent=2))


def variant_context(pack_row: dict[str, Any], variant: str, *, max_chars: int) -> str:
    payload = json.loads(str(pack_row["answer_context"]))
    if variant == "d1":
        for evidence in payload.get("evidence") or []:
            evidence.pop("raw_quotes", None)
        payload["instructions"] = [
            "Check every required slot before answering.",
            "Use only the structured claims in the evidence pack.",
            "Use speaker, session, time, and path provenance to resolve ambiguity.",
            "Do not treat retrieval scores or path metadata as facts.",
            "If a required slot is missing after retrieval, say that it cannot be determined.",
            "Return a concise final answer.",
        ]
    else:
        payload["instructions"] = [
            "Check every required slot before answering.",
            "Use only claims supported by the attached exact raw quotes.",
            "Use speaker, session, time, and path provenance to resolve ambiguity.",
            "Do not treat retrieval scores or path metadata as facts.",
            "If a required slot is missing after retrieval, say that it cannot be determined.",
            "Return a concise final answer.",
        ]
    payload["experiment_variant"] = (
        "D1_structured_claims_without_raw_quotes" if variant == "d1" else "D2_structured_claims_with_exact_raw_quotes"
    )
    payload["context_budget"] = {"max_chars": max_chars, "truncation": "drop_low_rank_whole_evidence_units"}
    while True:
        rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(rendered) <= max_chars or not payload.get("evidence"):
            return rendered
        removed = payload["evidence"].pop()
        removed_id = removed.get("unit_id")
        for group in payload.get("evidence_groups") or []:
            group["evidence_unit_ids"] = [
                unit_id for unit_id in group.get("evidence_unit_ids") or [] if unit_id != removed_id
            ]


def load_or_create_manifest(
    path: Path,
    baseline_rows: list[dict],
    *,
    per_type: int,
    seed: int,
    selection: str = "stratified",
    excluded_type: str = "temporal-reasoning",
) -> dict:
    if path.exists():
        payload = read_json(path)
        stored_selection = payload.get("selection", "stratified")
        if stored_selection != selection:
            raise ValueError("existing manifest uses a different selection mode")
        if selection == "stratified" and (
            payload.get("per_type") != per_type or payload.get("seed") != seed
        ):
            raise ValueError("existing paired manifest uses a different per-type count or seed")
        if selection == "all-non-temporal" and payload.get("excluded_type") != excluded_type:
            raise ValueError("existing manifest excludes a different question type")
        return payload
    if selection == "all-non-temporal":
        selected = [
            {
                "question_id": str(row["question_id"]),
                "question_type": str(row.get("question_type") or "unknown"),
                "baseline_judge_correct": int(row.get("judge_correct") or 0),
                "baseline_retrieval_hit": bool(row.get("retrieval_hit")),
                "baseline_retrieval_full_cover": bool(row.get("retrieval_full_cover")),
            }
            for row in baseline_rows
            if str(row.get("question_type") or "unknown") != excluded_type
        ]
        selected.sort(key=lambda row: (row["question_type"], row["question_id"]))
        payload = {
            "version": "hypathmem_r_all_non_temporal_v0_1",
            "selection": selection,
            "excluded_type": excluded_type,
            "summary": summarize_manifest(selected),
            "rows": selected,
        }
        write_json(path, payload)
        return payload
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in baseline_rows:
        grouped[str(row.get("question_type") or "unknown")].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    for question_type in sorted(grouped):
        candidates = sorted(grouped[question_type], key=lambda row: str(row["question_id"]))
        if len(candidates) < per_type:
            raise ValueError(f"question type {question_type!r} has only {len(candidates)} rows")
        for row in rng.sample(candidates, per_type):
            selected.append(
                {
                    "question_id": str(row["question_id"]),
                    "question_type": question_type,
                    "baseline_judge_correct": int(row.get("judge_correct") or 0),
                    "baseline_retrieval_hit": bool(row.get("retrieval_hit")),
                    "baseline_retrieval_full_cover": bool(row.get("retrieval_full_cover")),
                }
            )
    selected.sort(key=lambda row: (row["question_type"], row["question_id"]))
    payload = {
        "version": "hypathmem_r_paired120_v0_1",
        "selection": selection,
        "seed": seed,
        "per_type": per_type,
        "summary": summarize_manifest(selected),
        "rows": selected,
    }
    write_json(path, payload)
    return payload


def summarize_manifest(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["question_type"]].append(row)
    return {
        "num_questions": len(rows),
        "baseline_accuracy": ratio(sum(row["baseline_judge_correct"] for row in rows), len(rows)),
        "by_type": {
            key: {
                "num_questions": len(values),
                "baseline_accuracy": ratio(sum(row["baseline_judge_correct"] for row in values), len(values)),
                "retrieval_hit": ratio(sum(row["baseline_retrieval_hit"] for row in values), len(values)),
                "retrieval_full_cover": ratio(sum(row["baseline_retrieval_full_cover"] for row in values), len(values)),
            }
            for key, values in sorted(grouped.items())
        },
    }


def build_payload(rows: list[dict], args: argparse.Namespace, manifest: Path, base_url: str, started: float) -> dict:
    return {
        "metadata": {
            "version": "hypathmem_r_v0_1",
            "variant": args.variant,
            "selection": args.selection,
            "excluded_type": args.excluded_type if args.selection == "all-non-temporal" else None,
            "packs": str(resolve(args.packs)),
            "baseline": str(resolve(args.baseline)),
            "manifest": str(manifest),
            "generation_model": args.model,
            "judge_model": args.judge_model,
            "prompt_protocol": "same_as_121_longmemeval_locomo_style_hybrid_v3",
            "judge_protocol": "same_as_121_longmemeval_abstention_aware_count_tolerant_v2",
            "temperature": 0.0,
            "max_context_chars": args.max_context_chars,
            "base_url": base_url,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "summary_all": summarize(rows),
        "summary_by_type": summarize_by_type(rows),
        "per_question": rows,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    d0 = sum(int(row.get("baseline_judge_correct") or 0) for row in rows)
    current = sum(int(row.get("judge_correct") or 0) for row in rows)
    return {
        "num_questions": n,
        "num_correct": current,
        "judge_accuracy": ratio(current, n),
        "baseline_num_correct": d0,
        "baseline_accuracy": ratio(d0, n),
        "delta_accuracy": ratio(current - d0, n),
        "d0_wrong_variant_correct": sum(
            not bool(row.get("baseline_judge_correct")) and bool(row.get("judge_correct")) for row in rows
        ),
        "d0_correct_variant_wrong": sum(
            bool(row.get("baseline_judge_correct")) and not bool(row.get("judge_correct")) for row in rows
        ),
    }


def summarize_by_type(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("question_type"))].append(row)
    return {key: summarize(values) for key, values in sorted(grouped.items())}


def load_qa121() -> ModuleType:
    path = ROOT / "scripts" / "121_run_longmemeval_qa_eval.py"
    spec = importlib.util.spec_from_file_location("hytopomem_qa121", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load QA protocol module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = read_json(path)
    return list(payload.get("per_question") or [])


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
