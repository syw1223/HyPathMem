#!/usr/bin/env python3
"""Prepare Cat2 temporal inputs for the LoCoMo 8.5 migration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.locomo_time_v8_5 import VERSION, latest_conversation_time


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/locomo/processed/locomo_mvp.json"))
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "outputs/qa/full_v3_9_expand120_light_quota_top20_hybrid_v11_"
            "chatanywhere_gpt41mini_judge_gpt4omini.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_locomo_8.5time_v1/cat2_temporal_inputs.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("processed", "baseline", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force")

    processed = read(args.processed)
    baseline = read(args.baseline)
    history_by_id = {str(row["conversation_id"]): row for row in processed}
    rows = []
    for base in baseline["per_question"]:
        if int(base.get("category") or 0) != 2:
            continue
        conversation_id = str(base["conversation_id"])
        latest = latest_conversation_time(history_by_id[conversation_id].get("turns") or [])
        rows.append(
            {
                "question_id": base["question_id"],
                "conversation_id": conversation_id,
                "question": base["question"],
                "question_date": latest.strftime("%Y-%m-%d %H:%M") if latest else "",
                "category": 2,
                "gold_answer": base.get("gold_answer"),
            }
        )
    if len(rows) != 321:
        raise ValueError(f"Expected 321 Cat2 questions, found {len(rows)}")
    write(
        args.output,
        {
            "metadata": {
                "version": VERSION,
                "dataset": "LoCoMo",
                "selection": "official_category_2",
                "uses_gold_answer_for_extraction": False,
                "question_time": "latest conversation timestamp, provenance only",
            },
            "summary": {"num_questions": len(rows), "question_time_resolved": sum(bool(row["question_date"]) for row in rows)},
            "rows": rows,
        },
    )
    manifest = args.output.with_name("cat2_manifest.json")
    write(manifest, {"selection": "all_cat2", "rows": [{"question_id": row["question_id"]} for row in rows]})
    print(json.dumps({"num_questions": len(rows), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
