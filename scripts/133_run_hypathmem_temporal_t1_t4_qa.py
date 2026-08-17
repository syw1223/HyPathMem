from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


DEFAULT_TEMPORAL = "outputs/reconstruction/hypathmem_temporal_v0_1/frozen_d2_top50_t1_t4.json"
DEFAULT_D2 = "outputs/reconstruction/hypathmem_r_v0_1/top50_structured_raw_packs.json"
DEFAULT_BASELINE = "outputs/qa/longmemeval_v3_10_cardquota_light_top50_c1_window_chunks_80k_gpt41mini_judge_gpt4omini.json"
DEFAULT_MANIFEST = "outputs/reconstruction/hypathmem_r_v0_1/paired120_manifest.json"
DEFAULT_D2_RESULTS = "outputs/qa/hypathmem_r_v0_1_paired120/d2_raw_grounded_gpt41mini_judge_gpt4omini.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled T1-T4 temporal QA with replacement contexts.")
    parser.add_argument("--variant", choices=["t1", "t2", "t3", "t4"], required=True)
    parser.add_argument("--temporal", default=DEFAULT_TEMPORAL)
    parser.add_argument("--d2-packs", default=DEFAULT_D2)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--d2-results", default=DEFAULT_D2_RESULTS)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", default="")
    parser.add_argument("--all-temporal", action="store_true", help="Run all temporal questions instead of paired-manifest temporal rows.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="E_MEM_API_KEY")
    parser.add_argument("--base-url-env", default="E_MEM_BASE_URL")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--max-answer-tokens", type=int, default=96)
    parser.add_argument("--max-judge-tokens", type=int, default=160)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qa121 = load_qa121()
    temporal = read_json(resolve(args.temporal))
    d2 = read_json(resolve(args.d2_packs))
    baseline = read_json(resolve(args.baseline))
    d2_results = read_json(resolve(args.d2_results))
    temporal_by_id = {str(row["question_id"]): row for row in temporal["rows"]}
    d2_by_id = {str(row["question_id"]): row for row in d2["rows"]}
    baseline_rows = baseline.get("per_question") or baseline.get("rows") or []
    baseline_by_id = {str(row["question_id"]): row for row in baseline_rows}
    d2_result_by_id = {
        str(row["question_id"]): row for row in d2_results.get("per_question") or d2_results.get("rows") or []
    }
    if args.all_temporal:
        selected_ids = sorted(temporal_by_id)
    else:
        manifest = read_json(resolve(args.manifest))
        selected_ids = [str(row["question_id"]) for row in manifest["rows"] if str(row["question_id"]) in temporal_by_id]

    if args.dry_run:
        for qid in selected_ids[:3]:
            context, route = context_for(qid, args.variant, temporal_by_id, d2_by_id)
            print(f"qid={qid} route={route} chars={len(context)}\n{context[:2400]}\n")
        return

    output_path = resolve(args.output)
    log_path = resolve(args.log) if args.log else output_path.with_suffix(".log")
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output_path}; use --resume")
    rows = list(read_json(output_path).get("per_question") or []) if args.resume and output_path.exists() else []
    done = {str(row["question_id"]) for row in rows}
    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=90.0,
    )
    started = time.perf_counter()
    append_log(log_path, f"START variant={args.variant} n={len(selected_ids)} replacement_context=true")

    for index, qid in enumerate(selected_ids, start=1):
        if qid in done:
            continue
        base = baseline_by_id[qid]
        solution = temporal_by_id[qid]["solution"]
        if args.variant == "t4" and not solution["verified"] and qid in d2_result_by_id:
            d2_row = d2_result_by_id[qid]
            row = {
                "question_id": qid,
                "question": base["question"],
                "question_type": base.get("question_type", ""),
                "question_date": base.get("question_date", ""),
                "gold_answer": base.get("gold_answer", ""),
                "prediction": d2_row.get("prediction", ""),
                "context_route": "D2_fallback_reused_frozen_result",
                "context_chars": d2_row.get("context_chars", 0),
                "solver_success": bool(solution["success"]),
                "solver_verified": False,
                "solver_answer": solution.get("answer"),
                "operand_coverage": solution.get("operand_coverage"),
                "baseline_judge_correct": int(base.get("judge_correct") or 0),
                "baseline_retrieval_full_cover": bool(base.get("retrieval_full_cover")),
                "generation_usage": d2_row.get("generation_usage", {}),
                "judge_label": d2_row.get("judge_label", ""),
                "judge_correct": int(d2_row.get("judge_correct") or 0),
                "judge_reason": d2_row.get("judge_reason", ""),
                "judge_usage": d2_row.get("judge_usage", {}),
                "fallback_reused": True,
            }
            rows.append(row)
            payload = build_payload(rows, args, started)
            write_json(output_path, payload)
            message = (
                f"processed {index}/{len(selected_ids)} qid={qid} route={row['context_route']} "
                f"judge={row['judge_label']} acc={payload['summary']['accuracy']:.4f}"
            )
            print(message, flush=True)
            append_log(log_path, message)
            continue
        context, route = context_for(qid, args.variant, temporal_by_id, d2_by_id)
        result = client.chat_completion_with_metadata(
            model=args.model,
            messages=[
                ChatMessage(role="system", content=qa121.ANSWER_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=qa121.ANSWER_USER_PROMPT.format(
                        question=base["question"],
                        question_type=base.get("question_type", "unknown"),
                        question_date=base.get("question_date", "unknown"),
                        task_instruction=qa121.task_instruction(
                            str(base["question"]), str(base.get("question_type", "")), str(base.get("question_date", ""))
                        ),
                        private_quant_instruction=qa121.private_quant_instruction("basic"),
                        context=context,
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=args.max_answer_tokens,
        )
        prediction = result.content.strip()
        judged = qa121.judge_answer(
            client=client,
            model=args.judge_model,
            question=str(base["question"]),
            question_type=str(base.get("question_type", "")),
            gold_answer=str(base.get("gold_answer") or ""),
            prediction=prediction,
            is_abstention=bool(base.get("is_abstention")),
            max_tokens=args.max_judge_tokens,
        )
        row = {
            "question_id": qid,
            "question": base["question"],
            "question_type": base.get("question_type", ""),
            "question_date": base.get("question_date", ""),
            "gold_answer": base.get("gold_answer", ""),
            "prediction": prediction,
            "context_route": route,
            "context_chars": len(context),
            "solver_success": bool(solution["success"]),
            "solver_verified": bool(solution["verified"]),
            "solver_answer": solution.get("answer"),
            "operand_coverage": solution.get("operand_coverage"),
            "baseline_judge_correct": int(base.get("judge_correct") or 0),
            "baseline_retrieval_full_cover": bool(base.get("retrieval_full_cover")),
            "generation_usage": qa121.normalize_usage(result.usage),
            **judged,
        }
        rows.append(row)
        payload = build_payload(rows, args, started)
        write_json(output_path, payload)
        message = f"processed {index}/{len(selected_ids)} qid={qid} route={route} judge={row['judge_label']} acc={payload['summary']['accuracy']:.4f}"
        print(message, flush=True)
        append_log(log_path, message)

    payload = build_payload(rows, args, started)
    write_json(output_path, payload)
    append_log(log_path, "DONE " + json.dumps(payload["summary"], ensure_ascii=False))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def context_for(
    qid: str,
    variant: str,
    temporal_by_id: dict[str, dict[str, Any]],
    d2_by_id: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    row = temporal_by_id[qid]
    packet = row["packets"][variant]
    if variant == "t4" and not packet["diagnostics"]["eligible_to_override"]:
        return d2_context(d2_by_id[qid]), "D2_fallback_unverified"
    return json.dumps(packet, ensure_ascii=False, separators=(",", ":")), f"TEMPORAL_{variant.upper()}_replacement"


def d2_context(pack_row: dict[str, Any]) -> str:
    payload = json.loads(str(pack_row["answer_context"]))
    payload["instructions"] = [
        "Check every required slot before answering.",
        "Use only claims supported by the attached exact raw quotes.",
        "Return a concise final answer.",
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_payload(rows: list[dict[str, Any]], args: argparse.Namespace, started: float) -> dict[str, Any]:
    correct = sum(int(row.get("judge_correct") or 0) for row in rows)
    baseline = sum(int(row.get("baseline_judge_correct") or 0) for row in rows)
    full_wrong = [row for row in rows if row["baseline_retrieval_full_cover"] and not row["baseline_judge_correct"]]
    return {
        "metadata": {
            "version": "hypathmem_temporal_v0_1",
            "variant": args.variant,
            "candidate_pool": "frozen_D2_top50",
            "temporal_context_policy": "replace_not_append",
            "generation_model": args.model,
            "judge_model": args.judge_model,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "summary": {
            "num_questions": len(rows),
            "num_correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
            "baseline_accuracy": baseline / len(rows) if rows else 0.0,
            "delta_accuracy": (correct - baseline) / len(rows) if rows else 0.0,
            "solver_verified": sum(row["solver_verified"] for row in rows),
            "full_cover_but_wrong_count": len(full_wrong),
            "full_cover_but_wrong_fixed": sum(bool(row.get("judge_correct")) for row in full_wrong),
        },
        "per_question": rows,
    }


def load_qa121() -> ModuleType:
    path = ROOT / "scripts" / "121_run_longmemeval_qa_eval.py"
    spec = importlib.util.spec_from_file_location("hytopomem_qa121_temporal", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


if __name__ == "__main__":
    main()
