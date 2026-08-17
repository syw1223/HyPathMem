#!/usr/bin/env python3
"""Judge only newly verified 8.5 solver candidates against frozen D0."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.eval.openai_compatible import OpenAICompatibleChatClient


def load_qa121() -> Any:
    path = ROOT / "scripts" / "121_run_longmemeval_qa_eval.py"
    spec = importlib.util.spec_from_file_location("hytopomem_qa121_time_v85", path)
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


def classify(d0: bool, current: bool) -> str:
    if not d0 and current:
        return "fix"
    if d0 and not current:
        return "break"
    return "unchanged_correct" if d0 else "unchanged_wrong"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversion", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="E_MEM_API_KEY")
    parser.add_argument("--base-url-env", default="E_MEM_BASE_URL")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    args = parser.parse_args()
    for name in ("conversion", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    args.compiled = [path if path.is_absolute() else ROOT / path for path in args.compiled]
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    sources = {}
    for path in args.compiled:
        for row in read_json(path)["rows"]:
            sources[str(row["question_id"])] = row
    candidates = [
        row for row in read_json(args.conversion)["rows"]
        if (row.get("verifier") or {}).get("safe_to_override_d0")
    ]
    qa121 = load_qa121()
    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=120.0,
    )
    started = time.perf_counter()
    rows = []
    for item in candidates:
        qid = str(item["question_id"])
        source = sources[qid]
        prediction = str(item["enriched_candidate"]["solution"]["answer"])
        judged = qa121.judge_answer(
            client=client,
            model=args.judge_model,
            question=str(source["question"]),
            question_type="temporal-reasoning",
            gold_answer=str(source.get("gold_answer") or ""),
            prediction=prediction,
            is_abstention=bool(source.get("is_abstention")),
            max_tokens=160,
        )
        d0 = bool(source["frozen_d0"].get("judge_correct"))
        current = bool(judged.get("judge_correct"))
        rows.append(
            {
                "question_id": qid,
                "question": source["question"],
                "gold_answer": source.get("gold_answer"),
                "prediction": prediction,
                "frozen_d0_prediction": source["frozen_d0"].get("prediction"),
                "frozen_d0_judge_correct": int(d0),
                "comparison": classify(d0, current),
                **judged,
            }
        )
        print(f"judged {qid} {judged['judge_label']} {rows[-1]['comparison']}", flush=True)
    n = len(rows)
    correct = sum(int(row["judge_correct"]) for row in rows)
    d0_correct = sum(row["frozen_d0_judge_correct"] for row in rows)
    summary = {
        "num_new_verified_candidates": n,
        "candidate_correct": correct,
        "candidate_accuracy": correct / n if n else 0.0,
        "frozen_d0_correct": d0_correct,
        "frozen_d0_accuracy": d0_correct / n if n else 0.0,
        "fix": sum(row["comparison"] == "fix" for row in rows),
        "break": sum(row["comparison"] == "break" for row in rows),
        "unchanged_correct": sum(row["comparison"] == "unchanged_correct" for row in rows),
        "unchanged_wrong": sum(row["comparison"] == "unchanged_wrong" for row in rows),
    }
    write_json(
        args.output,
        {
            "metadata": {
                "version": "8.5time_v_candidate_eval",
                "conversion": str(args.conversion),
                "judge_model": args.judge_model,
                "judge_repetitions": 1,
                "elapsed_seconds": time.perf_counter() - started,
            },
            "summary": summary,
            "rows": rows,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
