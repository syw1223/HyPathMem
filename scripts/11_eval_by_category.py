from __future__ import annotations

import argparse
import json
from collections import defaultdict

from common import read_json, resolve_path


def load_question_categories(path: str) -> dict[str, str]:
    conversations = read_json(resolve_path(path))
    categories = {}
    for conversation in conversations:
        for qa in conversation.get("qa", []):
            categories[qa["question_id"]] = str(qa.get("category", "unknown"))
    return categories


def summarize(items: list[dict]) -> dict:
    if not items:
        return {
            "num_questions": 0,
            "hit": 0.0,
            "recall": 0.0,
            "full_cover": 0.0,
            "avg_tokens": 0.0,
            "avg_path_len": 0.0,
        }
    n = len(items)
    return {
        "num_questions": n,
        "hit": sum(float(item["hit"]) for item in items) / n,
        "recall": sum(float(item["recall"]) for item in items) / n,
        "full_cover": sum(float(item["full_cover"]) for item in items) / n,
        "avg_tokens": sum(float(item["tokens"]) for item in items) / n,
        "avg_path_len": sum(float(item["path_len"]) for item in items) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-files", nargs="+", required=True)
    parser.add_argument("--questions", default="data/locomo/processed/locomo_mvp.json")
    parser.add_argument("--output", default="outputs/eval/retrieval_summary_by_category_200_k5.md")
    args = parser.parse_args()

    categories = load_question_categories(args.questions)
    lines = [
        "# Retrieval Summary By Category",
        "",
        "| Method | Category | Questions | Hit@5 | Recall@5 | FullCover@5 | Avg Tokens | Avg Path Len |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for eval_file in args.eval_files:
        with resolve_path(eval_file).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        method = payload.get("method", eval_file)
        grouped = defaultdict(list)
        for item in payload.get("per_question", []):
            grouped[categories.get(item["question_id"], "unknown")].append(item)
        for category in sorted(grouped):
            summary = summarize(grouped[category])
            lines.append(
                "| {method} | {category} | {num_questions} | {hit:.4f} | {recall:.4f} | "
                "{full_cover:.4f} | {avg_tokens:.1f} | {avg_path_len:.2f} |".format(
                    method=method,
                    category=category,
                    **summary,
                )
            )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

