from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_QA = "outputs/qa/longmemeval_v3_9_lgbm_top20_locomo_style_v3_gpt4omini_judge_gpt4omini.json"
DEFAULT_PATHS = "outputs/eval/longmemeval_v3_9_24_feature_card_selector/lgbm_top20_paths.json"
DEFAULT_NODES = "outputs/longmemeval_s/nodes.json"
DEFAULT_OUTPUT = "outputs/eval/longmemeval_answer_bearing_coverage_v3_9_top20.json"


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default=DEFAULT_QA)
    parser.add_argument("--paths", default=DEFAULT_PATHS)
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default="outputs/eval/cache/longmemeval_answer_bearing")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    qa_payload = read_json(Path(args.qa))
    qa_rows = qa_payload.get("per_question", []) if isinstance(qa_payload, dict) else []
    qa_by_qid = {str(row.get("question_id")): row for row in qa_rows}

    path_payload = read_json(Path(args.paths))
    path_items = path_payload if isinstance(path_payload, list) else path_payload.get("items", [])
    item_by_qid = {str(item.get("question_id")): item for item in path_items}

    target_qids = list(item_by_qid)
    target_node_ids = collect_top_node_ids(item_by_qid.values(), args.k)
    print(f"questions={len(target_qids)} qa_rows={len(qa_rows)} top_node_ids={len(target_node_ids)}", flush=True)

    cache_dir = Path(args.cache_dir) / f"k{args.k}"
    top_cache = cache_dir / "top_nodes.jsonl"
    raw_cache = cache_dir / "raw_nodes.jsonl"
    if args.refresh_cache or not top_cache.exists():
        extract_nodes_with_jq(Path(args.nodes), target_node_ids, top_cache)
    top_nodes = load_jsonl_nodes(top_cache)
    raw_ids = collect_raw_support_ids(top_nodes.values())
    missing_top = sorted(target_node_ids - set(top_nodes))
    print(
        f"loaded_top_nodes={len(top_nodes)} missing_top_nodes={len(missing_top)} raw_support_ids={len(raw_ids)}",
        flush=True,
    )

    if args.refresh_cache or not raw_cache.exists():
        extract_nodes_with_jq(Path(args.nodes), raw_ids, raw_cache)
    raw_nodes = load_jsonl_nodes(raw_cache)
    print(f"loaded_raw_nodes={len(raw_nodes)}", flush=True)

    per_question = []
    for qid in target_qids:
        item = item_by_qid[qid]
        qa_row = qa_by_qid.get(qid, {})
        row = diagnose_item(item, qa_row, top_nodes, raw_nodes, args.k)
        per_question.append(row)

    summary = summarize(per_question)
    payload = {
        "metadata": {
            "qa": args.qa,
            "paths": args.paths,
            "nodes": args.nodes,
            "k": args.k,
            "num_path_questions": len(path_items),
            "num_qa_rows": len(qa_rows),
            "num_missing_top_nodes": len(missing_top),
        },
        "summary": summary,
        "by_question_type": summarize_by(per_question, "question_type"),
        "fullcover_wrong_summary": summarize([r for r in per_question if r["qa_judged_wrong"] and r["gold_raw_fullcover"]]),
        "examples": build_examples(per_question, args.examples),
        "per_question": per_question,
    }
    out = Path(args.output)
    write_json(out, payload)
    write_markdown(out.with_suffix(".md"), payload)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.md')}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def collect_top_node_ids(items: Iterable[dict], k: int) -> set[str]:
    node_ids: set[str] = set()
    for item in items:
        for path in item.get("paths", [])[:k]:
            node_id = candidate_evidence_node_id(path)
            if node_id:
                node_ids.add(node_id)
    return node_ids


def candidate_evidence_node_id(path: dict) -> str | None:
    metadata = path.get("metadata") or {}
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    for node_id in reversed(path.get("node_ids", [])):
        if ":fact:" in node_id or ":raw:" in node_id:
            return str(node_id)
    return None


def scan_nodes_by_id(path: Path, target_ids: set[str]) -> dict[str, dict]:
    if not target_ids:
        return {}
    found: dict[str, dict] = {}
    for node in iter_json_array(path):
        node_id = str(node.get("node_id", ""))
        if node_id in target_ids:
            found[node_id] = node
            if len(found) >= len(target_ids):
                break
    return found


def extract_nodes_with_jq(nodes_path: Path, target_ids: set[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path = output_path.with_suffix(".ids.json")
    ids_path.write_text(json.dumps(sorted(target_ids), ensure_ascii=False), encoding="utf-8")
    if not target_ids:
        output_path.write_text("", encoding="utf-8")
        return
    print(f"extracting {len(target_ids)} nodes -> {output_path}", flush=True)
    jq_filter = '($ids | INDEX(.[])) as $idmap | .[] | select($idmap[.node_id])'
    with output_path.open("w", encoding="utf-8") as out:
        subprocess.run(
            ["jq", "-c", "--argfile", "ids", str(ids_path), jq_filter, str(nodes_path)],
            check=True,
            stdout=out,
        )


def load_jsonl_nodes(path: Path) -> dict[str, dict]:
    nodes: dict[str, dict] = {}
    if not path.exists():
        return nodes
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            node = json.loads(line)
            nodes[str(node.get("node_id"))] = node
    return nodes


def iter_json_array(path: Path, *, chunk_size: int = 1 << 20):
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk and not buffer:
                break
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    buffer = buffer[1:]
                    started = True
                    continue
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return
                try:
                    node, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                yield node
                buffer = buffer[end:]
            if not chunk:
                break


def collect_raw_support_ids(nodes: Iterable[dict]) -> set[str]:
    raw_ids: set[str] = set()
    for node in nodes:
        if node.get("type") == "RAW":
            raw_ids.add(str(node.get("node_id")))
            continue
        metadata = node.get("metadata") or {}
        for support_id in list(metadata.get("support_raw_ids") or []) + list(node.get("support_ids") or []):
            sid = str(support_id)
            if ":raw:" in sid:
                raw_ids.add(sid)
    return raw_ids


def diagnose_item(item: dict, qa_row: dict, top_nodes: dict[str, dict], raw_nodes: dict[str, dict], k: int) -> dict:
    qid = str(item.get("question_id"))
    question = str(item.get("question") or qa_row.get("question") or "")
    question_type = str(item.get("question_type") or qa_row.get("question_type") or "")
    gold_answer = str(item.get("answer") or qa_row.get("gold_answer") or "")
    gold_evidence = {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}
    if not gold_evidence and qa_row.get("matched_evidence_ids"):
        gold_evidence = set()

    gold_tokens = answer_tokens(gold_answer)
    gold_phrase = normalize_text(gold_answer)

    fact_texts = []
    raw_texts = []
    predicted_evidence: set[str] = set()
    rank_rows = []
    first_fact_rank = None
    first_raw_rank = None
    first_any_rank = None
    answer_sentence = ""

    for rank, path in enumerate(item.get("paths", [])[:k], start=1):
        node_id = candidate_evidence_node_id(path)
        node = top_nodes.get(node_id or "", {})
        if not node:
            continue
        claim = str(node.get("text") or "")
        raw_supports = raw_support_texts(node, raw_nodes)
        fact_texts.append(claim)
        raw_texts.extend(raw_supports)
        predicted_evidence.update(evidence_ids_for_node(node))
        for raw_id in raw_support_ids(node):
            raw = raw_nodes.get(raw_id, {})
            if raw:
                predicted_evidence.update(evidence_ids_for_node(raw))

        fact_bearing = contains_answer(claim, gold_tokens, gold_phrase)
        raw_bearing = contains_answer(" ".join(raw_supports), gold_tokens, gold_phrase)
        if fact_bearing and first_fact_rank is None:
            first_fact_rank = rank
        if raw_bearing and first_raw_rank is None:
            first_raw_rank = rank
            answer_sentence = find_answer_sentence(raw_supports, gold_tokens, gold_phrase)
        if (fact_bearing or raw_bearing) and first_any_rank is None:
            first_any_rank = rank
            if not answer_sentence:
                answer_sentence = find_answer_sentence([claim] + raw_supports, gold_tokens, gold_phrase)
        rank_rows.append(
            {
                "rank": rank,
                "node_id": node_id,
                "fact_answer_bearing": fact_bearing,
                "raw_answer_bearing": raw_bearing,
                "matched_gold_turn": bool(gold_evidence & evidence_ids_for_node(node)),
            }
        )

    fact_blob = " ".join(fact_texts)
    raw_blob = " ".join(raw_texts)
    matched = sorted(gold_evidence & predicted_evidence)
    token_recall_fact = token_recall(gold_tokens, fact_blob)
    token_recall_raw = token_recall(gold_tokens, raw_blob)
    qa_label = str(qa_row.get("judge_label") or "")
    qa_judged_wrong = qa_label.upper() == "WRONG"
    qa_judged_correct = qa_label.upper() == "CORRECT"
    fullcover = bool(gold_evidence) and gold_evidence.issubset(predicted_evidence)

    return {
        "question_id": qid,
        "question": question,
        "question_type": question_type,
        "gold_answer": gold_answer,
        "prediction": qa_row.get("prediction", ""),
        "judge_label": qa_label,
        "judge_reason": qa_row.get("judge_reason", ""),
        "qa_judged_correct": qa_judged_correct,
        "qa_judged_wrong": qa_judged_wrong,
        "gold_evidence_count": len(gold_evidence),
        "matched_gold_evidence_count": len(matched),
        "matched_gold_evidence_ids": matched,
        "gold_raw_hit": bool(matched),
        "gold_raw_fullcover": fullcover,
        "gold_answer_tokens": gold_tokens,
        "fact_answer_phrase": bool(gold_phrase and gold_phrase in normalize_text(fact_blob)),
        "fact_answer_any_token": any_token(gold_tokens, fact_blob),
        "fact_answer_all_tokens": all_tokens(gold_tokens, fact_blob),
        "fact_answer_token_recall": token_recall_fact,
        "raw_answer_phrase": bool(gold_phrase and gold_phrase in normalize_text(raw_blob)),
        "raw_answer_any_token": any_token(gold_tokens, raw_blob),
        "raw_answer_all_tokens": all_tokens(gold_tokens, raw_blob),
        "raw_answer_token_recall": token_recall_raw,
        "answer_bearing_fact_rank": first_fact_rank,
        "answer_bearing_raw_rank": first_raw_rank,
        "answer_bearing_rank": first_any_rank,
        "answer_bearing_sentence": one_line(answer_sentence, 360),
        "fullcover_wrong_fact_explicit": bool(qa_judged_wrong and fullcover and (first_fact_rank is not None)),
        "fullcover_wrong_raw_explicit": bool(qa_judged_wrong and fullcover and (first_raw_rank is not None)),
        "rank_diagnostics": rank_rows,
    }


def raw_support_ids(node: dict) -> list[str]:
    ids = []
    metadata = node.get("metadata") or {}
    for support_id in list(metadata.get("support_raw_ids") or []) + list(node.get("support_ids") or []):
        sid = str(support_id)
        if ":raw:" in sid and sid not in ids:
            ids.append(sid)
    return ids


def raw_support_texts(node: dict, raw_nodes: dict[str, dict]) -> list[str]:
    if node.get("type") == "RAW":
        return [str(node.get("text") or "")]
    texts = []
    metadata = node.get("metadata") or {}
    support_texts = [str(text) for text in metadata.get("support_texts") or []]
    for idx, raw_id in enumerate(raw_support_ids(node)):
        raw = raw_nodes.get(raw_id, {})
        text = str(raw.get("text") or "")
        if not text and idx < len(support_texts):
            text = support_texts[idx]
        if text and text not in texts:
            texts.append(text)
    if not texts:
        texts.extend(support_texts[:3])
    return texts


def evidence_ids_for_node(node: dict) -> set[str]:
    metadata = node.get("metadata") or {}
    evidence: set[str] = set()
    node_id = str(node.get("node_id") or "")
    if node.get("type") == "RAW":
        evidence.add(normalize_evidence_id(str(metadata.get("turn_id") or node_id)))
    elif node.get("type") == "FACT":
        if metadata.get("turn_id"):
            evidence.add(normalize_evidence_id(str(metadata["turn_id"])))
        for support_id in list(metadata.get("support_raw_ids") or []) + list(node.get("support_ids") or []):
            evidence.add(normalize_evidence_id(str(support_id)))
    return evidence


def normalize_evidence_id(value: Any) -> str:
    text = str(value)
    if ":raw:" in text:
        return text.split(":raw:", 1)[1]
    return text


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def answer_tokens(answer: str) -> list[str]:
    tokens = tokenize(answer)
    content = [tok for tok in tokens if tok not in STOPWORDS]
    return content or tokens


def any_token(tokens: list[str], text: str) -> bool:
    if not tokens:
        return False
    haystack = set(tokenize(text))
    return any(tok in haystack for tok in tokens)


def all_tokens(tokens: list[str], text: str) -> bool:
    if not tokens:
        return False
    haystack = set(tokenize(text))
    return all(tok in haystack for tok in tokens)


def token_recall(tokens: list[str], text: str) -> float:
    if not tokens:
        return 0.0
    haystack = set(tokenize(text))
    return sum(1 for tok in tokens if tok in haystack) / len(tokens)


def contains_answer(text: str, tokens: list[str], phrase: str) -> bool:
    norm = normalize_text(text)
    if phrase and phrase in norm:
        return True
    if len(tokens) <= 1:
        return any_token(tokens, text)
    return all_tokens(tokens, text)


def find_answer_sentence(texts: list[str], tokens: list[str], phrase: str) -> str:
    for text in texts:
        for sent in split_sentences(text):
            if contains_answer(sent, tokens, phrase):
                return sent
    return ""


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def one_line(text: Any, max_len: int = 200) -> str:
    line = re.sub(r"\s+", " ", str(text)).strip()
    if len(line) > max_len:
        return line[: max_len - 3] + "..."
    return line


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"num_questions": 0}
    return {
        "num_questions": len(rows),
        "num_with_qa_judge": sum(1 for r in rows if r["judge_label"]),
        "qa_accuracy": mean_bool(rows, "qa_judged_correct"),
        "gold_raw_hit": mean_bool(rows, "gold_raw_hit"),
        "gold_raw_fullcover": mean_bool(rows, "gold_raw_fullcover"),
        "fact_answer_any_token": mean_bool(rows, "fact_answer_any_token"),
        "fact_answer_all_tokens": mean_bool(rows, "fact_answer_all_tokens"),
        "fact_answer_phrase": mean_bool(rows, "fact_answer_phrase"),
        "fact_answer_token_recall": mean_float(rows, "fact_answer_token_recall"),
        "raw_answer_any_token": mean_bool(rows, "raw_answer_any_token"),
        "raw_answer_all_tokens": mean_bool(rows, "raw_answer_all_tokens"),
        "raw_answer_phrase": mean_bool(rows, "raw_answer_phrase"),
        "raw_answer_token_recall": mean_float(rows, "raw_answer_token_recall"),
        "answer_bearing_any": sum(1 for r in rows if r["answer_bearing_rank"] is not None) / len(rows),
        "answer_bearing_rank_mean": mean_optional_rank(rows, "answer_bearing_rank"),
        "answer_bearing_rank_median": median_optional_rank(rows, "answer_bearing_rank"),
        "fullcover_wrong": sum(1 for r in rows if r["qa_judged_wrong"] and r["gold_raw_fullcover"]),
        "fullcover_wrong_fact_explicit": mean_bool(
            [r for r in rows if r["qa_judged_wrong"] and r["gold_raw_fullcover"]],
            "fullcover_wrong_fact_explicit",
        ),
        "fullcover_wrong_raw_explicit": mean_bool(
            [r for r in rows if r["qa_judged_wrong"] and r["gold_raw_fullcover"]],
            "fullcover_wrong_raw_explicit",
        ),
    }


def summarize_by(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    return {name: summarize(group) for name, group in sorted(groups.items())}


def mean_bool(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key)) / len(rows)


def mean_float(rows: list[dict], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows]
    return sum(values) / len(values) if values else 0.0


def mean_optional_rank(rows: list[dict], key: str) -> float | None:
    values = [int(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def median_optional_rank(rows: list[dict], key: str) -> float | None:
    values = [int(row[key]) for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def build_examples(rows: list[dict], limit: int) -> dict[str, list[dict]]:
    buckets = {
        "fullcover_wrong_answer_in_fact": lambda r: r["qa_judged_wrong"] and r["gold_raw_fullcover"] and r["answer_bearing_fact_rank"] is not None,
        "fullcover_wrong_answer_only_in_raw": lambda r: (
            r["qa_judged_wrong"]
            and r["gold_raw_fullcover"]
            and r["answer_bearing_fact_rank"] is None
            and r["answer_bearing_raw_rank"] is not None
        ),
        "fullcover_wrong_no_answer_text": lambda r: r["qa_judged_wrong"] and r["gold_raw_fullcover"] and r["answer_bearing_rank"] is None,
        "raw_hit_but_answer_not_bearing": lambda r: r["gold_raw_hit"] and r["answer_bearing_rank"] is None,
        "no_raw_hit_but_answer_text_present": lambda r: (not r["gold_raw_hit"]) and r["answer_bearing_rank"] is not None,
    }
    examples: dict[str, list[dict]] = {}
    for name, predicate in buckets.items():
        selected = []
        for row in rows:
            if predicate(row):
                selected.append(example_row(row))
                if len(selected) >= limit:
                    break
        examples[name] = selected
    return examples


def example_row(row: dict) -> dict:
    return {
        "question_id": row["question_id"],
        "question_type": row["question_type"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "prediction": row["prediction"],
        "judge_label": row["judge_label"],
        "gold_raw_fullcover": row["gold_raw_fullcover"],
        "fact_answer_rank": row["answer_bearing_fact_rank"],
        "raw_answer_rank": row["answer_bearing_raw_rank"],
        "answer_bearing_sentence": row["answer_bearing_sentence"],
        "judge_reason": row["judge_reason"],
    }


def write_markdown(path: Path, payload: dict) -> None:
    lines = ["# LongMemEval Answer-Bearing Coverage Diagnosis", ""]
    lines.append("## Overall")
    lines.extend(summary_table(payload["summary"]))
    lines.append("")
    lines.append("## By Question Type")
    lines.append("| Type | N | QA Acc | Raw Hit | Raw FullCover | Fact AllTok | Raw AllTok | Bearing Any | Bearing Rank Med | FullCover Wrong |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, summary in payload["by_question_type"].items():
        lines.append(
            f"| {name} | {summary.get('num_questions', 0)} | {fmt(summary.get('qa_accuracy'))} | "
            f"{fmt(summary.get('gold_raw_hit'))} | {fmt(summary.get('gold_raw_fullcover'))} | "
            f"{fmt(summary.get('fact_answer_all_tokens'))} | {fmt(summary.get('raw_answer_all_tokens'))} | "
            f"{fmt(summary.get('answer_bearing_any'))} | {summary.get('answer_bearing_rank_median')} | "
            f"{summary.get('fullcover_wrong', 0)} |"
        )
    lines.append("")
    lines.append("## FullCover Wrong")
    lines.extend(summary_table(payload["fullcover_wrong_summary"]))
    lines.append("")
    lines.append("## Examples")
    for bucket, examples in payload["examples"].items():
        lines.append(f"### {bucket}")
        if not examples:
            lines.append("- none")
            continue
        for ex in examples:
            lines.append(
                f"- `{ex['question_id']}` type={ex['question_type']} fact_rank={ex['fact_answer_rank']} raw_rank={ex['raw_answer_rank']}\n"
                f"  Q: {one_line(ex['question'], 220)}\n"
                f"  Gold: {one_line(ex['gold_answer'], 160)}\n"
                f"  Pred: {one_line(ex['prediction'], 220)}\n"
                f"  Sentence: {one_line(ex['answer_bearing_sentence'], 260)}\n"
                f"  Judge: {one_line(ex['judge_reason'], 220)}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_table(summary: dict) -> list[str]:
    lines = ["| Metric | Value |", "|---|---:|"]
    for key, value in summary.items():
        lines.append(f"| {key} | {fmt(value)} |")
    return lines


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
