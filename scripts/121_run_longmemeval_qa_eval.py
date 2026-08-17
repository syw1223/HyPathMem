from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient


METRIC_PROTOCOL = "longmemeval_qa_llm_judge_with_abstention"

ANSWER_SYSTEM_PROMPT = """You answer memory QA questions using the provided evidence.

Important rules:
- The evidence comes from many sessions. Match the question's entity, event, object, and time.
- Be concise and answer directly.
- Use explicit evidence and reasonable inference from the context, including dates, comparisons, preferences, counts, totals, and likely outcomes.
- Do not answer "I don't know" if the evidence contains a direct or strongly supported answer, even if the evidence is scattered across sessions.
- Say "I don't know" only when the retrieved evidence contains no relevant information for the question.
- Use timestamps to resolve updates and relative time. A message timestamp is when the message was sent, not always the event date.
- If memories conflict, prefer evidence that matches the question date/event; prefer the latest state only when the question asks for the current/latest state.
- Relation-card text is an organization hint. Treat FACT claims and RAW supports as primary evidence.
"""

ANSWER_USER_PROMPT = """Question:
{question}

Question type: {question_type}
Question date: {question_date}

Task-specific instruction:
{task_instruction}

Retrieved evidence:
{context}

Use this private procedure before answering:
1. Identify all evidence items relevant to the question. Ignore unrelated retrieved items.
2. Identify the answer operation:
   - item-count: count unique matching items/events/people/projects/states.
   - duration-total: add durations such as days, weeks, months, or hours.
   - money-total: add monetary amounts.
   - date/detail/status: select the specific supported value.
{private_quant_instruction}
3. For item-count, do not double-count repeated mentions. For duration-total or money-total, add the supported numbers rather than counting entries.
4. If the question asks current/latest status, use the latest matching evidence by timestamp; otherwise use the evidence matching the event/time in the question.
5. If the evidence gives a specific name, place, date, object, count, amount, reason, or preference, return that specific answer.
6. If relevant evidence exists but is incomplete, give the best supported answer rather than refusing.
7. Verify the final number and unit before returning it. If you include a short item list, it must match the final number.

Return only the answer. If the evidence does not answer the question, return exactly: I don't know."""

JUDGE_PROMPT = """Your task is to judge a LongMemEval answer.

Question:
{question}

Question type: {question_type}
Gold answer:
{gold_answer}

Gold is abstention/no-answer: {is_abstention}

Generated answer:
{prediction}

Judging rules:
- If gold is abstention/no-answer, mark CORRECT only if the generated answer clearly says it does not know or lacks enough information.
- If gold is answerable, mark CORRECT if the generated answer expresses the same fact, entity, event, state, time, or answer as the gold answer, even with different wording.
- For count/total questions, mark CORRECT if the generated answer contains the same final count/total as the gold answer, even if it includes extra explanation.
- For recommendation/preference questions, mark CORRECT if the generated answer captures the same user preference, constraint, desired feature, or recommendation direction as the gold answer. It does not need to name the exact same external product/event unless the gold requires it.
- Be generous about paraphrases, abbreviations, and equivalent date formats.
- Do not mark WRONG merely because the generated answer includes additional non-contradictory supporting details.
- Mark WRONG if the generated answer is a refusal for an answerable question, contradicts the gold answer, or gives a different entity/time/event/count.

Return strict JSON:
{{
  "label": "CORRECT" or "WRONG",
  "reason": "short reason"
}}"""

RAW_INDEX_CACHE: dict[int, dict[str, Any]] = {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/longmemeval_s/graph_semantic_hierarchy_v3.json")
    parser.add_argument("--paths", default="outputs/eval/longmemeval_v3_9_24_feature_card_selector/lgbm_top20_paths.json")
    parser.add_argument("--processed", default="data/longmemeval/processed/longmemeval_s_mvp.json")
    parser.add_argument("--output", default="outputs/qa/longmemeval_v3_9_lgbm_top20_gpt4omini_judge_gpt4omini.json")
    parser.add_argument("--log", default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="E_MEM_API_KEY")
    parser.add_argument("--base-url-env", default="E_MEM_BASE_URL")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--max-context-chars", type=int, default=18000)
    parser.add_argument(
        "--max-message-chunks",
        type=int,
        default=0,
        help="Maximum support-centered MESSAGE_CHUNKS for c1_window_chunks. 0 keeps all chunks until context truncation.",
    )
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--max-judge-tokens", type=int, default=160)
    parser.add_argument("--compiler", choices=["none", "structured", "c1_window_chunks"], default="structured")
    parser.add_argument("--quant-mode", choices=["basic", "private_ie"], default="basic")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--include-abstention", action="store_true", default=True)
    parser.add_argument("--dry-run-context", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    graph_payload = read_json(Path(args.graph))
    nodes: dict[str, dict] = graph_payload.get("nodes", {})
    path_items: list[dict] = read_json(Path(args.paths))
    qa_map = load_longmemeval_qa(Path(args.processed))
    items = merge_path_items(path_items, qa_map, include_abstention=args.include_abstention)
    if args.offset:
        items = items[args.offset :]
    if args.limit:
        items = items[: args.limit]

    if args.dry_run_context:
        for item in items[: max(1, args.limit or 1)]:
            context = build_context(nodes, item, k=args.k, max_chars=args.max_context_chars, compiler=args.compiler)
            print(f"qid={item['question_id']} context_chars={len(context)}")
            print(context[:2000])
        return

    output_path = Path(args.output)
    log_path = Path(args.log) if args.log else output_path.with_suffix(".log")
    rows = load_existing_rows(output_path) if args.resume else []
    done = {str(row.get("question_id")) for row in rows}

    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=90.0,
    )
    append_log(log_path, f"START output={output_path} paths={args.paths} k={args.k} questions={len(items)} resume={args.resume}")

    for index, item in enumerate(items, start=1):
        qid = str(item["question_id"])
        if qid in done:
            print(f"skip {index}/{len(items)} qid={qid}", flush=True)
            continue
        context = build_context(
            nodes,
            item,
            k=args.k,
            max_chars=args.max_context_chars,
            compiler=args.compiler,
            max_message_chunks=args.max_message_chunks,
        )
        answer_result = client.chat_completion_with_metadata(
            model=args.model,
            messages=[
                ChatMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=ANSWER_USER_PROMPT.format(
                        question=item["question"],
                        question_type=item.get("question_type", "unknown"),
                        question_date=item.get("question_date", "unknown"),
                        task_instruction=task_instruction(
                            question=str(item["question"]),
                            question_type=str(item.get("question_type", "")),
                            question_date=str(item.get("question_date", "")),
                            quant_mode=args.quant_mode,
                        ),
                        private_quant_instruction=private_quant_instruction(args.quant_mode),
                        context=context,
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=args.max_answer_tokens,
        )
        prediction = answer_result.content.strip()
        retrieval = evaluate_retrieval(nodes, item, args.k)
        gold_answer = str(item.get("gold_answer") or "")
        is_abstention = bool(item.get("is_abstention")) or not gold_answer.strip()
        row = {
            "question_id": qid,
            "conversation_id": conversation_id(qid),
            "question": item["question"],
            "question_type": item.get("question_type", ""),
            "question_date": item.get("question_date", ""),
            "gold_answer": gold_answer,
            "is_abstention": is_abstention,
            "prediction": prediction,
            "context_chars": len(context),
            "context_node_ids": retrieval["context_node_ids"],
            "matched_evidence_ids": retrieval["matched_evidence_ids"],
            "retrieval_hit": retrieval["hit"],
            "retrieval_recall": retrieval["recall"],
            "retrieval_full_cover": retrieval["full_cover"],
            "retrieval_tokens": retrieval["tokens"],
            "lexical_f1": token_f1(prediction, gold_answer),
            "bleu1": bleu1(prediction, gold_answer),
            "generation_usage": normalize_usage(answer_result.usage),
            "generation_elapsed_seconds": answer_result.elapsed_seconds,
        }
        if args.skip_judge:
            row["judge_label"] = "SKIP"
            row["judge_correct"] = 0
        else:
            judge_result = judge_answer(
                client=client,
                model=args.judge_model,
                question=item["question"],
                question_type=str(item.get("question_type", "")),
                gold_answer=gold_answer,
                prediction=prediction,
                is_abstention=is_abstention,
                max_tokens=args.max_judge_tokens,
            )
            row.update(judge_result)
        row["total_usage"] = add_usage(row.get("generation_usage", {}), row.get("judge_usage", {}))
        row["total_api_elapsed_seconds"] = float(row.get("generation_elapsed_seconds", 0.0)) + float(row.get("judge_elapsed_seconds", 0.0))
        rows.append(row)
        if args.save_every and len(rows) % args.save_every == 0:
            write_payload(output_path, rows, args, started, client.base_url)
        append_log(log_path, format_question_log(index, len(items), row, time.perf_counter() - started))
        print(
            f"processed {index}/{len(items)} qid={qid} judge={row.get('judge_label')} "
            f"acc_so_far={accuracy(rows):.4f} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    payload = build_payload(rows, args, started, client.base_url)
    write_json(output_path, payload)
    append_log(log_path, format_summary_log(payload["summary_all"], prefix="OVERALL"))
    append_log(log_path, format_summary_log(payload["summary_answerable"], prefix="ANSWERABLE"))
    append_log(log_path, format_summary_log(payload["summary_abstention"], prefix="ABSTENTION"))
    print(f"wrote {output_path}")
    print(f"log {log_path}")
    print(json.dumps(payload["summary_all"], ensure_ascii=False, indent=2))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_longmemeval_qa(path: Path) -> dict[str, dict]:
    conversations = read_json(path)
    qa_map: dict[str, dict] = {}
    for conv in conversations:
        for qa in conv.get("qa", []):
            qid = str(qa["question_id"])
            qa_map[qid] = {
                "question_id": qid,
                "conversation_id": conv.get("conversation_id") or conversation_id(qid),
                "question": qa.get("question", ""),
                "question_type": qa.get("question_type") or conv.get("question_type") or "",
                "question_date": qa.get("question_date") or conv.get("question_date") or "",
                "gold_answer": str(qa.get("answer") or "").strip(),
                "gold_evidence": qa.get("gold_evidence") or qa.get("evidence") or [],
                "is_abstention": bool(qa.get("is_abstention")),
                "answer_session_ids": qa.get("answer_session_ids") or conv.get("answer_session_ids") or [],
            }
    return qa_map


def merge_path_items(path_items: list[dict], qa_map: dict[str, dict], *, include_abstention: bool) -> list[dict]:
    items: list[dict] = []
    path_by_qid = {str(item["question_id"]): item for item in path_items}
    for qid, qa in qa_map.items():
        if qa.get("is_abstention") and not include_abstention:
            continue
        item = dict(path_by_qid.get(qid, {"question_id": qid, "paths": []}))
        item.update({k: v for k, v in qa.items() if k not in {"paths"}})
        if "paths" not in item:
            item["paths"] = []
        items.append(item)
    return items


def build_context(
    nodes: dict[str, dict],
    item: dict,
    *,
    k: int,
    max_chars: int,
    compiler: str = "structured",
    max_message_chunks: int = 0,
) -> str:
    records = evidence_records(nodes, item, k)
    qtype = classify_question(str(item.get("question", "")), str(item.get("question_type", "")))
    operation = answer_operation(str(item.get("question", "")), str(item.get("question_type", "")))
    benchmark_type = str(item.get("question_type", ""))
    lines = [
        "LongMemEval evidence pack.",
        f"Benchmark question type: {benchmark_type or 'unknown'}. Answer operation: {operation}.",
        "Use FACT claims and RAW supports as evidence. Relation-card metadata is only a grouping hint.",
        "Use the relevant evidence to answer directly. If relevant evidence exists, do not refuse.",
        "For updates or conflicts, choose evidence matching the question date/event; use the latest state only for current/latest questions.",
    ]
    if "multi-session" in benchmark_type:
        lines.append("Multi-session note: combine relevant facts across sessions, then deduplicate repeated mentions before answering.")
    if "preference" in benchmark_type or is_recommendation_question(str(item.get("question", ""))):
        lines.append(
            "Preference note: infer the user's likely preferred recommendation/advice from remembered tastes, constraints, plans, and prior choices. Do not require the exact recommendation name to appear in evidence."
        )
    if qtype == "temporal":
        lines.append("Temporal note: resolve relative dates from the RAW message timestamp; do not confuse message time with event time.")
    elif qtype == "count":
        lines.append("Counting note: list the relevant unique items/events mentally, remove duplicates, verify the count equals that list, then return the final number.")
        lines.append("For count answers, start with the number, then optionally include a short parenthesized item list.")
    elif qtype == "inference":
        lines.append("Inference note: use explicit memories to infer a likely preference/reason/status; do not require the exact final wording to appear.")
    if compiler == "c1_window_chunks":
        return build_c1_window_context(
            nodes=nodes,
            item=item,
            records=records,
            base_lines=lines,
            max_chars=max_chars,
            qtype=qtype,
            max_message_chunks=max_message_chunks,
        )
    compiler_lines = build_compiler_section(str(item.get("question", "")), records, operation) if compiler == "structured" else []
    if compiler_lines:
        lines.append("")
        lines.extend(compiler_lines)
    lines.append("")
    lines.append("### Evidence")
    ordered = records
    if qtype == "temporal":
        ordered = sorted(records, key=lambda rec: (rec["timestamp"] or "", rec["rank"]))
    for record in ordered:
        lines.append(format_record(record, compact=False))
    context = "\n".join(lines)
    if len(context) > max_chars:
        return context[:max_chars] + "\n\n[context truncated]"
    return context


def build_c1_window_context(
    *,
    nodes: dict[str, dict],
    item: dict,
    records: list[dict],
    base_lines: list[str],
    max_chars: int,
    qtype: str,
    max_message_chunks: int,
) -> str:
    lines = list(base_lines)
    lines.extend(
        [
            "",
            "### Context Format",
            "The evidence is organized as selected FACTS plus nearby RAW message windows.",
            "Use FACTS for concise claims, then verify details, dates, speakers, and counts against MESSAGE_CHUNKS.",
            "Ignore unrelated messages inside a window; they are included only to preserve local context.",
            "",
            "### FACTS",
        ]
    )
    ordered = records
    if qtype == "temporal":
        ordered = sorted(records, key=lambda rec: (rec["timestamp"] or "", rec["rank"]))
    for record in ordered:
        lines.append(format_fact_record(record))
    window_lines = format_message_window_chunks(
        nodes,
        records,
        window=2,
        max_chunk_chars=1800,
        limit=max_message_chunks,
    )
    if window_lines:
        lines.extend(["", "### MESSAGE_CHUNKS"])
        lines.extend(window_lines)
    path_lines = format_evidence_paths(records)
    if path_lines:
        lines.extend(["", "### EVIDENCE_PATHS"])
        lines.extend(path_lines)
    context = "\n".join(lines)
    if len(context) > max_chars:
        return context[:max_chars] + "\n\n[context truncated]"
    return context


def answer_operation(question: str, question_type: str = "") -> str:
    q = question.lower()
    if re.search(r"\b(how much money|total money|amount|spend|spent|cost|worth|raise|raised|\$)\b", q):
        return "money-total"
    if re.search(r"\b(how many (hours|days|weeks|months|years)|total (hours|days|weeks|months|years)|how long)\b", q):
        return "duration-total"
    if re.search(r"\b(how many|number of)\b", q):
        return "item-count"
    if classify_question(question, question_type) == "temporal":
        return "temporal"
    if classify_question(question, question_type) == "inference":
        return "inference"
    return "detail"


def build_compiler_section(question: str, records: list[dict], operation: str) -> list[str]:
    if operation not in {"item-count", "duration-total", "money-total", "temporal"}:
        return []
    lines = [
        "### Structured Evidence View",
        f"Operation: {operation}",
    ]
    if operation == "money-total":
        lines.append("Use this as a candidate money table. Include only rows that match the question target; sum included monetary amounts.")
        rows = quantitative_rows(records, value_kind="money")
    elif operation == "duration-total":
        lines.append("Use this as a candidate duration table. Include only rows that match the question target; add durations with the requested unit.")
        rows = quantitative_rows(records, value_kind="duration")
    elif operation == "item-count":
        lines.append("Use this as a candidate item/event table. Count unique matching rows only; remove duplicates and unrelated rows.")
        rows = item_count_rows(question, records)
    else:
        lines.append("Use this as a timeline view. Prefer the evidence whose event/time matches the question.")
        rows = timeline_rows(records)
    if not rows:
        lines.append("No structured rows were extracted; fall back to the evidence list below.")
        return lines
    for row in rows[:14]:
        lines.append(row)
    return lines


def quantitative_rows(records: list[dict], *, value_kind: str) -> list[str]:
    rows = []
    for record in records:
        text = record_text(record)
        values = extract_money_values(text) if value_kind == "money" else extract_duration_values(text)
        if not values:
            continue
        rows.append(
            f"[E{record['rank']}] values={', '.join(values)} time={record['timestamp'] or 'unknown'} "
            f"candidate={one_line(record['claim'], max_len=220)}"
        )
    return rows


def item_count_rows(question: str, records: list[dict]) -> list[str]:
    rows = []
    target_terms = content_terms(question)
    for record in records:
        text = record_text(record)
        overlap = len(target_terms & content_terms(text))
        if overlap <= 0 and record["rank"] > 8:
            continue
        rows.append(
            f"[E{record['rank']}] overlap={overlap} time={record['timestamp'] or 'unknown'} "
            f"candidate_item_or_event={one_line(record['claim'], max_len=260)}"
        )
    return rows


def timeline_rows(records: list[dict]) -> list[str]:
    rows = []
    for record in sorted(records, key=lambda rec: (rec["timestamp"] or "", rec["rank"])):
        rows.append(
            f"[E{record['rank']}] time={record['timestamp'] or 'unknown'} "
            f"event={one_line(record['claim'], max_len=260)}"
        )
    return rows


def record_text(record: dict) -> str:
    return " ".join([str(record.get("claim", "")), " ".join(record.get("raw_supports", []))])


def extract_money_values(text: str) -> list[str]:
    matches = re.findall(r"\$\s?\d[\d,]*(?:\.\d+)?", text)
    return dedupe([match.replace(" ", "") for match in matches])


NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def extract_duration_values(text: str) -> list[str]:
    values = []
    pattern = r"\b\d+(?:\.\d+)?\s*(?:hours?|hrs?|days?|weeks?|months?|years?)\b"
    values.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    word_pattern = r"\b(" + "|".join(NUMBER_WORDS) + r")\s*(?:hours?|hrs?|days?|weeks?|months?|years?)\b"
    for match in re.finditer(word_pattern, text, flags=re.IGNORECASE):
        word = match.group(1).lower()
        unit = match.group(0)[len(match.group(1)) :].strip()
        values.append(f"{NUMBER_WORDS[word]} {unit}")
    hyphen_pattern = r"\b(" + "|".join(NUMBER_WORDS) + r"|\d+(?:\.\d+)?)[-\s](?:day|week|month|year|hour)[-\s]long\b"
    for match in re.finditer(hyphen_pattern, text, flags=re.IGNORECASE):
        value = match.group(1).lower()
        if value in NUMBER_WORDS:
            value = NUMBER_WORDS[value]
        phrase = match.group(0).lower()
        unit = "days" if "day" in phrase else "weeks" if "week" in phrase else "months" if "month" in phrase else "years" if "year" in phrase else "hours"
        values.append(f"{value} {unit}")
    return dedupe([one_line(value, max_len=40) for value in values])


STOP_TERMS = {
    "what", "when", "where", "which", "who", "why", "how", "many", "much", "total", "did",
    "do", "does", "am", "are", "is", "was", "were", "have", "has", "had", "the", "a", "an",
    "i", "me", "my", "mine", "of", "in", "on", "at", "to", "for", "from", "with", "and",
    "or", "this", "that", "these", "those", "there", "been", "being",
}


def content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]+", text.lower())
        if len(token) > 2 and token not in STOP_TERMS
    }


def evidence_records(nodes: dict[str, dict], item: dict, k: int) -> list[dict]:
    records = []
    seen: set[str] = set()
    for rank, path in enumerate(item.get("paths", [])[:k], start=1):
        node_id = candidate_evidence_node_id(nodes, path)
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        node = nodes.get(node_id, {})
        metadata = node.get("metadata") or {}
        path_metadata = path.get("metadata") or {}
        scores = path.get("scores") or {}
        raw_supports = raw_support_snippets(nodes, node, limit=3)
        raw_ids = raw_support_node_ids(nodes, node)
        records.append(
            {
                "rank": rank,
                "node_id": node_id,
                "node_type": node.get("type", ""),
                "claim": one_line(node.get("text", ""), max_len=500),
                "timestamp": first_timestamp(node),
                "evidence_turns": evidence_turn_ids(node),
                "raw_supports": raw_supports,
                "raw_ids": raw_ids,
                "route": str(path_metadata.get("route_source") or path_metadata.get("candidate_source") or ""),
                "event_id": str(path_metadata.get("event_node_id") or ""),
                "episode_id": str(path_metadata.get("episode_node_id") or ""),
                "topic_id": str(path_metadata.get("topic_node_id") or ""),
                "card": is_truthy(path_metadata.get("v3_9_query_card"))
                or is_truthy(path_metadata.get("is_nary_completion"))
                or is_truthy(path_metadata.get("card_quota_selected")),
                "card_type": str(path_metadata.get("nary_hyperedge_type") or ""),
                "cardce": float_or_zero(scores.get("v3_9_card_ce") or path_metadata.get("v39_card_ce_score")),
                "ce": float_or_zero(scores.get("cross_encoder")),
                "selector": float_or_zero(scores.get("topology_selector") or path.get("score")),
            }
        )
    return records


def format_fact_record(record: dict) -> str:
    card = " card=true" if record["card"] else ""
    if record["card_type"]:
        card += f" card_type={record['card_type']}"
    return (
        f"[E{record['rank']}] node={record['node_id']} type={record['node_type']} "
        f"time={record['timestamp'] or 'unknown'} evidence={','.join(record['evidence_turns']) or 'unknown'}"
        f"{card} ce={record['ce']:.4f} selector={record['selector']:.4f}\n"
        f"  fact={record['claim']}"
    )


def format_record(record: dict, *, compact: bool) -> str:
    card = " card=true" if record["card"] else ""
    if record["card_type"]:
        card += f" card_type={record['card_type']}"
    line = (
        f"[E{record['rank']}] node={record['node_id']} type={record['node_type']} "
        f"time={record['timestamp'] or 'unknown'} evidence={','.join(record['evidence_turns']) or 'unknown'}"
        f"{card} route={record['route']}\n"
        f"  fact={record['claim']}"
    )
    if record["raw_supports"]:
        raw_limit = 260 if compact else 520
        line += "\n  raw=" + " | ".join(record["raw_supports"])[:raw_limit]
    return line


def raw_support_node_ids(nodes: dict[str, dict], node: dict) -> list[str]:
    metadata = node.get("metadata") or {}
    ids = list(metadata.get("support_raw_ids") or []) + list(node.get("support_ids") or [])
    if node.get("type") == "RAW":
        own_id = str(node.get("id") or "")
        if own_id:
            ids.insert(0, own_id)
    out = []
    seen: set[str] = set()
    for raw_id in ids:
        raw_id = str(raw_id)
        raw = nodes.get(raw_id)
        if raw and raw.get("type") == "RAW" and raw_id not in seen:
            out.append(raw_id)
            seen.add(raw_id)
    return out


def raw_support_snippets(nodes: dict[str, dict], node: dict, *, limit: int) -> list[str]:
    metadata = node.get("metadata") or {}
    support_texts = list(metadata.get("support_texts") or [])
    support_ids = list(metadata.get("support_raw_ids") or []) + list(node.get("support_ids") or [])
    snippets = []
    seen: set[str] = set()
    for index, support_id in enumerate(support_ids):
        if len(snippets) >= limit:
            break
        raw = nodes.get(str(support_id), {})
        text = raw.get("text", "") if raw.get("type") == "RAW" else ""
        timestamp = raw.get("time") or ""
        if not text and index < len(support_texts):
            text = str(support_texts[index])
        text = one_line(text, max_len=420)
        if not text or text in seen:
            continue
        seen.add(text)
        prefix = f"time={timestamp} " if timestamp else ""
        snippets.append(prefix + text)
    return snippets


def format_message_window_chunks(
    nodes: dict[str, dict],
    records: list[dict],
    *,
    window: int,
    max_chunk_chars: int,
    limit: int,
) -> list[str]:
    raw_index = build_raw_index(nodes)
    if not raw_index:
        return []
    intervals_by_session: dict[str, list[tuple[int, int, set[int]]]] = {}
    for record in records:
        ranks = {int(record["rank"])}
        for raw_id in record.get("raw_ids") or []:
            raw_ref = raw_index["by_id"].get(str(raw_id))
            if not raw_ref:
                continue
            session_id, pos = raw_ref
            intervals_by_session.setdefault(session_id, []).append((max(0, pos - window), pos + window, ranks))
    chunks = []
    chunk_index = 1
    for session_id in sorted(intervals_by_session, key=session_sort_key):
        merged = merge_window_intervals(intervals_by_session[session_id])
        session_rows = raw_index["by_session"].get(session_id, [])
        for start, end, ranks in merged:
            rows = session_rows[start : min(end + 1, len(session_rows))]
            if not rows:
                continue
            evidence = ",".join(f"E{rank}" for rank in sorted(ranks))
            header = f"[C{chunk_index}] session={session_id} supports={evidence}"
            body_lines = []
            for raw_id, raw in rows:
                metadata = raw.get("metadata") or {}
                timestamp = raw.get("time") or metadata.get("timestamp") or metadata.get("date") or "unknown"
                speaker = metadata.get("speaker") or metadata.get("role") or raw.get("role") or ""
                speaker_prefix = f"{speaker}: " if speaker else ""
                body_lines.append(f"- time={timestamp} raw={raw_id} {speaker_prefix}{one_line(raw.get('text', ''), max_len=420)}")
            chunk = header + "\n" + "\n".join(body_lines)
            if len(chunk) > max_chunk_chars:
                chunk = chunk[: max_chunk_chars - 20] + "\n  [chunk truncated]"
            chunks.append(chunk)
            chunk_index += 1
            if limit and len(chunks) >= limit:
                return chunks
    return chunks


def build_raw_index(nodes: dict[str, dict]) -> dict[str, Any]:
    cache_key = id(nodes)
    cached = RAW_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    by_session: dict[str, list[tuple[str, dict]]] = {}
    by_id: dict[str, tuple[str, int]] = {}
    for node_id, node in nodes.items():
        if node.get("type") != "RAW":
            continue
        session_id = raw_session_id(str(node_id), node)
        by_session.setdefault(session_id, []).append((str(node_id), node))
    for session_id, rows in by_session.items():
        rows.sort(key=lambda item: raw_sort_key(item[0], item[1]))
        for pos, (node_id, node) in enumerate(rows):
            by_id[node_id] = (session_id, pos)
            turn_id = raw_turn_id(node_id, node)
            if turn_id:
                by_id.setdefault(turn_id, (session_id, pos))
    payload = {"by_session": by_session, "by_id": by_id}
    RAW_INDEX_CACHE[cache_key] = payload
    return payload


def raw_session_id(node_id: str, node: dict) -> str:
    metadata = node.get("metadata") or {}
    for key in ("session_id", "session", "date", "conversation_id"):
        if metadata.get(key):
            return str(metadata[key])
    match = re.search(r":raw:([^:]+)", node_id)
    if match:
        return match.group(1)
    parts = node_id.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else node_id


def raw_turn_id(node_id: str, node: dict) -> str:
    metadata = node.get("metadata") or {}
    if metadata.get("turn_id"):
        return str(metadata["turn_id"])
    return normalize_evidence_id(node_id)


def raw_sort_key(node_id: str, node: dict) -> tuple[str, int, str]:
    metadata = node.get("metadata") or {}
    timestamp = str(node.get("time") or metadata.get("timestamp") or metadata.get("date") or "")
    turn = str(metadata.get("turn_id") or normalize_evidence_id(node_id))
    nums = re.findall(r"\d+", turn)
    turn_num = int(nums[-1]) if nums else 0
    return (timestamp, turn_num, node_id)


def session_sort_key(session_id: str) -> tuple[str, int]:
    nums = re.findall(r"\d+", session_id)
    return (session_id, int(nums[-1]) if nums else 0)


def merge_window_intervals(intervals: list[tuple[int, int, set[int]]]) -> list[tuple[int, int, set[int]]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int, set[int]]] = []
    cur_start, cur_end, cur_ranks = intervals[0][0], intervals[0][1], set(intervals[0][2])
    for start, end, ranks in intervals[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
            cur_ranks.update(ranks)
            continue
        merged.append((cur_start, cur_end, cur_ranks))
        cur_start, cur_end, cur_ranks = start, end, set(ranks)
    merged.append((cur_start, cur_end, cur_ranks))
    return merged


def format_evidence_paths(records: list[dict]) -> list[str]:
    lines = []
    for record in records:
        path = " -> ".join(
            value
            for value in [
                record.get("topic_id") or "",
                record.get("episode_id") or "",
                record.get("event_id") or "",
                record.get("node_id") or "",
            ]
            if value
        )
        if not path:
            continue
        lines.append(f"[E{record['rank']}] {path}")
    return lines


def evaluate_retrieval(nodes: dict[str, dict], item: dict, k: int) -> dict:
    gold = {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}
    context_node_ids = candidate_evidence_node_ids(nodes, item.get("paths", []), k)
    predicted: set[str] = set()
    for node_id in context_node_ids:
        predicted.update(evidence_ids_for_node(nodes, node_id))
    matched = sorted(gold & predicted)
    return {
        "context_node_ids": context_node_ids,
        "matched_evidence_ids": matched,
        "hit": bool(matched),
        "recall": len(matched) / len(gold) if gold else 0.0,
        "full_cover": bool(gold) and gold.issubset(predicted),
        "tokens": sum(len(str(nodes.get(node_id, {}).get("text", "")).split()) for node_id in context_node_ids),
    }


def evidence_ids_for_node(nodes: dict[str, dict], node_id: str) -> set[str]:
    node = nodes.get(node_id, {})
    metadata = node.get("metadata") or {}
    evidence: set[str] = set()
    if node.get("type") == "RAW":
        evidence.add(normalize_evidence_id(str(metadata.get("turn_id") or node_id.rsplit(":raw:", 1)[-1])))
    elif node.get("type") == "FACT":
        if metadata.get("turn_id"):
            evidence.add(normalize_evidence_id(str(metadata["turn_id"])))
        for support_id in list(node.get("support_ids") or []) + list(metadata.get("support_raw_ids") or []):
            evidence.add(normalize_evidence_id(str(support_id)))
    return evidence


def candidate_evidence_node_id(nodes: dict[str, dict], path: dict) -> str | None:
    for node_id in reversed(path.get("node_ids", [])):
        node = nodes.get(node_id)
        if node and node.get("type") in {"FACT", "RAW"}:
            return node_id
    meta_node = (path.get("metadata") or {}).get("evidence_node_id")
    if meta_node in nodes and nodes[meta_node].get("type") in {"FACT", "RAW"}:
        return str(meta_node)
    return None


def candidate_evidence_node_ids(nodes: dict[str, dict], paths: list[dict], k: int) -> list[str]:
    selected = []
    seen: set[str] = set()
    for path in paths[:k]:
        node_id = candidate_evidence_node_id(nodes, path)
        if node_id and node_id not in seen:
            selected.append(node_id)
            seen.add(node_id)
    return selected


def judge_answer(
    *,
    client: OpenAICompatibleChatClient,
    model: str,
    question: str,
    question_type: str,
    gold_answer: str,
    prediction: str,
    is_abstention: bool,
    max_tokens: int,
) -> dict:
    result = client.chat_completion_with_metadata(
        model=model,
        messages=[
            ChatMessage(
                role="user",
                content=JUDGE_PROMPT.format(
                    question=question,
                    question_type=question_type,
                    gold_answer=gold_answer if gold_answer else "(no answer)",
                    is_abstention=str(bool(is_abstention)).lower(),
                    prediction=prediction,
                ),
            )
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    payload = parse_json_object(result.content)
    label = str(payload.get("label", "")).strip().upper()
    if label not in {"CORRECT", "WRONG"}:
        label = "WRONG"
    return {
        "judge_correct": 1 if label == "CORRECT" else 0,
        "judge_label": label,
        "judge_reason": str(payload.get("reason", "")).strip(),
        "judge_raw_response": result.content,
        "judge_usage": normalize_usage(result.usage),
        "judge_elapsed_seconds": result.elapsed_seconds,
    }


def parse_json_object(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", str(text).strip(), re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"label": "WRONG", "reason": f"judge returned non-JSON: {str(text)[:120]}"}


def build_payload(rows: list[dict], args: argparse.Namespace, started: float, base_url: str) -> dict:
    return {
        "metadata": {
            "dataset": "longmemeval_s",
            "paths": args.paths,
            "graph": args.graph,
            "processed": args.processed,
            "k": args.k,
            "offset": args.offset,
            "generation_model": args.model,
            "judge_model": None if args.skip_judge else args.judge_model,
            "base_url": base_url,
            "api_key_env": args.api_key_env,
            "elapsed_seconds": time.perf_counter() - started,
            "metric_protocol": METRIC_PROTOCOL,
            "prompt_version": "longmemeval_locomo_style_hybrid_v3",
            "judge_prompt_version": "longmemeval_abstention_aware_count_tolerant_v2",
            "compiler": args.compiler,
            "max_message_chunks": args.max_message_chunks,
            "quant_mode": args.quant_mode,
        },
        "summary_all": summarize(rows),
        "summary_answerable": summarize([row for row in rows if not row.get("is_abstention")]),
        "summary_abstention": summarize([row for row in rows if row.get("is_abstention")]),
        "summary_gold_bearing": summarize([row for row in rows if str(row.get("gold_answer") or "").strip()]),
        "per_question": rows,
    }


def write_payload(path: Path, rows: list[dict], args: argparse.Namespace, started: float, base_url: str) -> None:
    write_json(path, build_payload(rows, args, started, base_url))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not rows:
        return empty_summary()
    total_usage = [row.get("total_usage", {}) for row in rows]
    return {
        "num_questions": n,
        "num_correct": int(sum(int(row.get("judge_correct", 0)) for row in rows)),
        "judge_accuracy": sum(float(row.get("judge_correct", 0)) for row in rows) / n,
        "macro_f1": sum(float(row.get("lexical_f1", 0.0)) for row in rows) / n,
        "macro_bleu1": sum(float(row.get("bleu1", 0.0)) for row in rows) / n,
        "retrieval_hit": sum(float(bool(row.get("retrieval_hit"))) for row in rows) / n,
        "retrieval_recall": sum(float(row.get("retrieval_recall", 0.0)) for row in rows) / n,
        "retrieval_full_cover": sum(float(bool(row.get("retrieval_full_cover"))) for row in rows) / n,
        "generation_tokens": sum(int(row.get("generation_usage", {}).get("total_tokens", 0)) for row in rows),
        "judge_tokens": sum(int(row.get("judge_usage", {}).get("total_tokens", 0)) for row in rows),
        "total_tokens": sum(int(usage.get("total_tokens", 0)) for usage in total_usage),
        "avg_tokens_per_question": sum(int(usage.get("total_tokens", 0)) for usage in total_usage) / n,
        "total_api_seconds": sum(float(row.get("total_api_elapsed_seconds", 0.0)) for row in rows),
        "avg_api_seconds_per_question": sum(float(row.get("total_api_elapsed_seconds", 0.0)) for row in rows) / n,
    }


def empty_summary() -> dict:
    return {
        "num_questions": 0,
        "num_correct": 0,
        "judge_accuracy": 0.0,
        "macro_f1": 0.0,
        "macro_bleu1": 0.0,
        "retrieval_hit": 0.0,
        "retrieval_recall": 0.0,
        "retrieval_full_cover": 0.0,
        "generation_tokens": 0,
        "judge_tokens": 0,
        "total_tokens": 0,
        "avg_tokens_per_question": 0.0,
        "total_api_seconds": 0.0,
        "avg_api_seconds_per_question": 0.0,
    }


def load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("per_question"), list):
        return payload["per_question"]
    if isinstance(payload, list):
        return payload
    return []


def normalize_evidence_id(value: str) -> str:
    value = str(value).strip()
    if ":raw:" in value:
        return value.rsplit(":raw:", 1)[1]
    if ":fact:" in value:
        return value.rsplit(":fact:", 1)[1]
    return value


def evidence_turn_ids(node: dict) -> list[str]:
    metadata = node.get("metadata") or {}
    turns = []
    if metadata.get("turn_id"):
        turns.append(normalize_evidence_id(str(metadata["turn_id"])))
    for support_id in list(node.get("support_ids") or []) + list(metadata.get("support_raw_ids") or []):
        turns.append(normalize_evidence_id(str(support_id)))
    return dedupe(turns)


def first_timestamp(node: dict) -> str:
    metadata = node.get("metadata") or {}
    timestamps = list(metadata.get("support_timestamps") or [])
    if timestamps:
        return str(timestamps[0])
    return str(node.get("time") or "")


def classify_question(question: str, question_type: str = "") -> str:
    qt = question_type.lower()
    q = question.lower()
    if "temporal" in qt or re.search(r"\b(when|what date|which date|before|after|last week|next month|ago|year|month|day|weekend)\b", q):
        return "temporal"
    if re.search(r"\b(how many|number of|how much|times|months passed|years passed)\b", q):
        return "count"
    if re.search(r"\b(would|likely|might|could|if|considering|why|how does|how did)\b", q):
        return "inference"
    return "detail"


def task_instruction(question: str, question_type: str, question_date: str, quant_mode: str = "basic") -> str:
    op = classify_question(question, question_type)
    pieces: list[str] = []
    is_preference = "preference" in question_type.lower() or is_recommendation_question(question)
    if "multi-session" in question_type.lower():
        pieces.append(
            "This is a multi-session memory question. Search across all retrieved sessions, not only the top one. "
            "Collect every evidence item that matches the exact target in the question, exclude unrelated same-topic memories, "
            "deduplicate repeated mentions of the same event/item, and combine the remaining facts before answering."
        )
    elif is_preference:
        pieces.append(
            "This is a preference/recommendation memory question. The goal is not to retrieve a fixed factual answer, "
            "but to infer what kind of suggestion would fit the user's remembered preferences, constraints, plans, prior choices, "
            "and dislikes. If relevant preference evidence exists, give a personalized recommendation direction or advice; do not answer I don't know."
        )
    elif "single-session" in question_type.lower():
        pieces.append(
            "This is a single-session memory question. Prefer evidence from the session that directly matches the asked memory, "
            "and ignore unrelated retrieved sessions."
        )
    if op == "count":
        instruction = (
            "This is a quantitative question. First decide whether it asks for an item count, a duration total, or a money total. "
            "For item counts, count unique actual items/events/states only and remove duplicates. "
            "For duration or money totals, add the supported numbers with the same unit. "
            "Return the final number and unit first."
        )
        if quant_mode == "private_ie":
            instruction += (
                " Before answering, silently mark each relevant evidence item INCLUDE or EXCLUDE. "
                "Exclude duplicates, wrong object types, wrong time windows, examples, plans, and repeated mentions."
            )
        pieces.append(instruction)
    elif op == "temporal":
        pieces.append(
            f"This is a temporal question. The question date is {question_date or 'unknown'}. Use message timestamps and raw text to "
            "resolve relative expressions such as yesterday, last week, last month, recently, tomorrow, or next week. "
            "A message timestamp is the message time, not automatically the event date."
        )
    elif is_preference:
        pieces.append(
            "Return the user's likely preferred type/features/constraints in one concise answer. "
            "It is acceptable to say what they would prefer or not prefer, rather than naming a specific external item."
        )
    elif op == "inference":
        pieces.append(
            "This is an inference question. Infer the likely preference, reason, status, or outcome from explicit memories. "
            "Do not answer I don't know if the evidence strongly supports a likely answer."
        )
    else:
        pieces.append(
            "This is a detail question. Return the most specific supported name, object, place, date, amount, reason, or preference. "
            "Do not use a broad category when the evidence gives a specific answer."
        )
    return " ".join(pieces)


def is_recommendation_question(question: str) -> bool:
    q = question.lower()
    return bool(
        re.search(
            r"\b(recommend|suggest|suggestion|tips?|advice|what should i|any ideas|what to look for|helpful)\b",
            q,
        )
    )


def private_quant_instruction(quant_mode: str) -> str:
    if quant_mode != "private_ie":
        return ""
    return (
        "3. For quantitative questions, silently make an INCLUDE/EXCLUDE table over the evidence before answering:\n"
        "   - INCLUDE exactly one row for each unique counted item/event, or one row for each numeric value to sum.\n"
        "   - EXCLUDE duplicates, unrelated items, wrong object type, wrong time window, examples, plans, and repeated mentions.\n"
        "   - For twins, pairs, groups, or bundles, count the actual number of requested objects if the question asks for objects/people."
    )


def normalize_usage(usage: dict) -> dict:
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def add_usage(left: dict, right: dict) -> dict:
    left = normalize_usage(left)
    right = normalize_usage(right)
    return {
        "prompt_tokens": left["prompt_tokens"] + right["prompt_tokens"],
        "completion_tokens": left["completion_tokens"] + right["completion_tokens"],
        "total_tokens": left["total_tokens"] + right["total_tokens"],
    }


def tokenize_metric_text(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\s]", "", str(text).lower())
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.split()


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize_metric_text(prediction)
    gold_tokens = tokenize_metric_text(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = multiset_overlap(pred_tokens, gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize_metric_text(prediction)
    gold_tokens = tokenize_metric_text(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    precision = multiset_overlap(pred_tokens, gold_tokens) / len(pred_tokens)
    brevity_penalty = 1.0 if len(pred_tokens) > len(gold_tokens) else math.exp(1 - len(gold_tokens) / len(pred_tokens))
    return precision * brevity_penalty


def multiset_overlap(left: list[str], right: list[str]) -> int:
    counts: dict[str, int] = {}
    for token in right:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in left:
        count = counts.get(token, 0)
        if count <= 0:
            continue
        overlap += 1
        counts[token] = count - 1
    return overlap


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")


def format_question_log(index: int, total: int, row: dict, elapsed: float) -> str:
    return (
        f"QA {index}/{total} qid={row['question_id']} type={row.get('question_type','')} "
        f"abstention={int(bool(row.get('is_abstention')))} judge={row.get('judge_label')} "
        f"hit={int(bool(row.get('retrieval_hit')))} recall={float(row.get('retrieval_recall', 0.0)):.4f} "
        f"full={int(bool(row.get('retrieval_full_cover')))} f1={float(row.get('lexical_f1', 0.0)):.4f} "
        f"bleu1={float(row.get('bleu1', 0.0)):.4f} tokens={int(row.get('total_usage', {}).get('total_tokens', 0))} "
        f"api_time={float(row.get('total_api_elapsed_seconds', 0.0)):.2f}s elapsed={elapsed:.1f}s\n"
        f"  Q: {one_line(row.get('question', ''))}\n"
        f"  GOLD: {one_line(row.get('gold_answer', ''))}\n"
        f"  PRED: {one_line(row.get('prediction', ''))}"
    )


def format_summary_log(summary: dict, *, prefix: str) -> str:
    return (
        f"{prefix} n={summary['num_questions']} acc={summary['judge_accuracy']:.4f} "
        f"correct={summary['num_correct']}/{summary['num_questions']} "
        f"f1={summary['macro_f1']:.4f} bleu1={summary['macro_bleu1']:.4f} "
        f"hit={summary['retrieval_hit']:.4f} recall={summary['retrieval_recall']:.4f} "
        f"full={summary['retrieval_full_cover']:.4f} tokens={summary['total_tokens']} "
        f"api_time={summary['total_api_seconds']:.1f}s"
    )


def accuracy(rows: list[dict]) -> float:
    return sum(float(row.get("judge_correct", 0)) for row in rows) / len(rows) if rows else 0.0


def is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def float_or_zero(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def dedupe(values: list[str]) -> list[str]:
    out = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def conversation_id(question_id: str) -> str:
    return str(question_id).split(":", 1)[0]


def one_line(value: Any, max_len: int = 500) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


if __name__ == "__main__":
    main()
