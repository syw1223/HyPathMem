from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def lightmem_tokens(value: object) -> list[str]:
    text = re.sub(r"\s+", " ", str(value or "").lower().strip())
    return re.findall(r"[a-z0-9]+", text)


def lightmem_f1(prediction: object, reference: object) -> float:
    pred_tokens = lightmem_tokens(prediction)
    ref_tokens = lightmem_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lightmem_bleu1(prediction: object, reference: object) -> float:
    pred_tokens = lightmem_tokens(prediction)
    ref_tokens = lightmem_tokens(reference)
    if not pred_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    precision = overlap / len(pred_tokens)
    brevity_penalty = (
        1.0
        if len(pred_tokens) > len(ref_tokens)
        else math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens)))
    )
    return brevity_penalty * precision


def emem_normalize(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def benchmark_tokens(value: object) -> list[str]:
    return emem_normalize(value).split()


def benchmark_f1(prediction: object, reference: object) -> float:
    pred_tokens = benchmark_tokens(prediction)
    ref_tokens = benchmark_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def benchmark_bleu1(prediction: object, reference: object) -> float:
    pred_tokens = benchmark_tokens(prediction)
    ref_tokens = benchmark_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    precision = overlap / len(pred_tokens)
    brevity_penalty = (
        1.0
        if len(pred_tokens) > len(ref_tokens)
        else math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens)))
    )
    return brevity_penalty * precision


def emem_f1(prediction: object, reference: object) -> float:
    pred_tokens = emem_normalize(prediction).split()
    ref_tokens = emem_normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def emem_tokens(value: object) -> list[str]:
    text = str(value or "").lower()
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def emem_bleu1(prediction: object, reference: object) -> float:
    pred_tokens = emem_tokens(prediction)
    ref_tokens = emem_tokens(reference)
    return sentence_bleu(
        [ref_tokens],
        pred_tokens,
        weights=(1, 0, 0, 0),
        smoothing_function=SmoothingFunction().method1,
    )


def emem_bleu4(prediction: object, reference: object) -> float:
    pred_tokens = emem_tokens(prediction)
    ref_tokens = emem_tokens(reference)
    return sentence_bleu(
        [ref_tokens],
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )


def summarize(rows: list[dict]) -> dict:
    scored = []
    for row in rows:
        prediction = row.get("prediction", "")
        reference = row.get("gold_answer", "")
        scored.append(
            {
                "benchmark_f1": benchmark_f1(prediction, reference),
                "benchmark_bleu1": benchmark_bleu1(prediction, reference),
                "lightmem_f1": lightmem_f1(prediction, reference),
                "lightmem_bleu1": lightmem_bleu1(prediction, reference),
                "emem_f1": emem_f1(prediction, reference),
                "emem_bleu1": emem_bleu1(prediction, reference),
                "emem_bleu4": emem_bleu4(prediction, reference),
            }
        )
    if not scored:
        return {"num_questions": 0}
    return {
        "num_questions": len(scored),
        **{name: mean(item[name] for item in scored) for name in scored[0]},
    }


def qa_summary(rows: list[dict]) -> dict:
    n = len(rows)
    generation_tokens = sum(int(row.get("generation_usage", {}).get("total_tokens", 0)) for row in rows)
    judge_tokens = sum(int(row.get("judge_usage", {}).get("total_tokens", 0)) for row in rows)
    generation_seconds = sum(float(row.get("generation_elapsed_seconds", 0.0)) for row in rows)
    judge_seconds = sum(float(row.get("judge_elapsed_seconds", 0.0)) for row in rows)
    calls = sum(bool(row.get("generation_usage")) for row in rows) + sum(
        bool(row.get("judge_usage")) for row in rows
    )
    return {
        "num_questions": n,
        "judge_skipped": False,
        "macro_f1": mean(float(row["lexical_f1"]) for row in rows) if rows else 0.0,
        "macro_bleu1": mean(float(row["bleu1"]) for row in rows) if rows else 0.0,
        "generation_tokens": generation_tokens,
        "judge_tokens": judge_tokens,
        "total_tokens": generation_tokens + judge_tokens,
        "avg_tokens_per_question": (generation_tokens + judge_tokens) / n if n else 0.0,
        "avg_tokens_per_call": (generation_tokens + judge_tokens) / calls if calls else 0.0,
        "generation_api_seconds": generation_seconds,
        "judge_api_seconds": judge_seconds,
        "total_api_seconds": generation_seconds + judge_seconds,
        "avg_api_seconds_per_question": (generation_seconds + judge_seconds) / n if n else 0.0,
        "retrieval_hit": mean(float(row.get("retrieval_hit", False)) for row in rows) if rows else 0.0,
        "retrieval_recall": mean(float(row.get("retrieval_recall", 0.0)) for row in rows) if rows else 0.0,
        "retrieval_full_cover": (
            mean(float(row.get("retrieval_full_cover", False)) for row in rows) if rows else 0.0
        ),
        "retrieval_avg_tokens": mean(float(row.get("retrieval_tokens", 0.0)) for row in rows) if rows else 0.0,
        "retrieval_avg_path_len": (
            mean(float(row.get("retrieval_path_len", 0.0)) for row in rows) if rows else 0.0
        ),
        "judge_accuracy": mean(float(row.get("judge_correct", 0)) for row in rows) if rows else 0.0,
        "num_correct": sum(int(row.get("judge_correct", 0)) for row in rows),
    }


def official_result(payload: dict, rows: list[dict]) -> dict:
    official_rows = []
    for source in rows:
        if int(source.get("category") or 0) not in {1, 2, 3, 4}:
            continue
        row = copy.deepcopy(source)
        row["lexical_f1"] = benchmark_f1(row.get("prediction", ""), row.get("gold_answer", ""))
        row["bleu1"] = benchmark_bleu1(row.get("prediction", ""), row.get("gold_answer", ""))
        row["metric_protocol"] = "locomo_squad_normalized_unigram"
        official_rows.append(row)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in official_rows:
        grouped[str(row.get("conversation_id", ""))].append(row)
    per_conversation = []
    for conversation_id, items in sorted(grouped.items()):
        summary = qa_summary(items)
        summary["conversation_id"] = conversation_id
        per_conversation.append(summary)

    metadata = copy.deepcopy(payload.get("metadata", {}))
    metadata.update(
        {
            "categories": [1, 2, 3, 4],
            "excluded_categories": [5],
            "metric_protocol": "locomo_squad_normalized_unigram",
            "metric_normalization": "lowercase, remove punctuation, remove a/an/the, collapse whitespace",
            "bleu1": "clipped unigram precision with brevity penalty",
            "derived_without_new_llm_calls": True,
        }
    )
    return {
        "metadata": metadata,
        "summary": qa_summary(official_rows),
        "per_conversation": per_conversation,
        "per_question": official_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--official-result-output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("per_question", payload)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category"))].append(row)
    official_rows = [row for row in rows if int(row.get("category") or 0) in {1, 2, 3, 4}]

    result = {
        "input": str(input_path),
        "notes": {
            "official_scope": "LoCoMo categories 1-4; LightMem skips category 5.",
            "benchmark_protocol": (
                "SQuAD-style lowercase/punctuation/article normalization; clipped unigram "
                "overlap; BLEU-1 includes brevity penalty."
            ),
            "category_5_warning": (
                "Category 5 adversarial_answer is not a standard gold answer and is excluded "
                "from the official aggregate."
            ),
            "emem_main_bleu": "E-mem main summary reports BLEU-4; both BLEU-1 and BLEU-4 are included.",
        },
        "official_categories_1_to_4": summarize(official_rows),
        "diagnostic_all_categories": summarize(rows),
        "by_category": {category: summarize(items) for category, items in sorted(by_category.items())},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")
    if args.official_result_output:
        official_path = Path(args.official_result_output).expanduser().resolve()
        official_path.parent.mkdir(parents=True, exist_ok=True)
        official_path.write_text(
            json.dumps(official_result(payload, rows), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {official_path}")


if __name__ == "__main__":
    main()
