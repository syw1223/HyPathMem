from __future__ import annotations

import argparse
from pathlib import Path
import re
import time

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.judge import OpenAICompatibleLLMJudge
from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient
from hytopomem.eval.qa_runner import OpenAICompatibleQARunner
from hytopomem.eval.retrieval_metrics import evaluate_item, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import NodeType


OFFICIAL_CATEGORIES = {1, 2, 3, 4}
METRIC_PROTOCOL = "locomo_squad_normalized_unigram"

VERIFIER_PROMPT = """Check the draft answer against the evidence.

Question:
{question}

Evidence:
{context}

Draft answer:
{draft_answer}

Verification checklist:
1. Does the answer directly answer the question?
2. Is the answer specific enough? If the evidence contains a more specific detail, revise the answer to include it.
3. For counting questions, does the answer count unique actual events/items and exclude plans, mentions, or duplicates?
4. For temporal questions, are relative dates resolved correctly using the message timestamp?
5. Does the answer include unsupported extra information or conflict with stronger evidence?
6. Is "not specified" or "I don't know" used even though the evidence contains a direct or strongly supported answer?

Return only the revised final answer. If the draft is already correct, return it unchanged."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url-env", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--answer-protocol", choices=["default", "v2", "v2_ops"], default="default")
    parser.add_argument(
        "--verify-answer",
        choices=["none", "all", "risk", "count_detail"],
        default="none",
        help=(
            "Optional second-pass answer verifier. "
            "risk verifies count/temporal/detail/inference; count_detail avoids temporal over-correction."
        ),
    )
    parser.add_argument("--max-verifier-tokens", type=int, default=128)
    parser.add_argument("--max-answer-tokens", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument(
        "--max-message-chunks",
        type=int,
        default=12,
        help="Maximum support-centered MESSAGE_CHUNKS for mnemis_c1_window/c4_window contexts.",
    )
    parser.add_argument(
        "--context-mode",
        choices=[
            "raw",
            "compiled",
            "hybrid",
            "mnemis_c1_chunks",
            "mnemis_c1_window_chunks",
            "mnemis_c2_entities",
            "mnemis_c3_cards",
            "mnemis_c4_full",
            "mnemis_c4_window_full",
        ],
        default="raw",
        help=(
            "raw keeps the original evidence list; compiled builds a type-aware evidence pack; "
            "hybrid compiles only temporal/count questions; mnemis_* builds structured memory-pack contexts."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log", default=None)
    parser.add_argument(
        "--categories",
        default="1,2,3,4",
        help="Comma-separated LoCoMo categories. Official evaluation uses 1,2,3,4.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config(args.config)
    qa_config = config.get("qa_eval", {})
    llm_config = config.get("llm", {})
    graph_path = resolve_path(args.graph or config["graph"]["graph_path"])
    paths_path = resolve_path(args.paths or qa_config.get("paths", "outputs/paths/full_graph_v2_event_topic_ce_top5.json"))
    output_path = resolve_path(args.output or qa_config.get("output", "outputs/qa/graph_v2_gpt4omini_smoke.json"))
    log_path = resolve_path(args.log) if args.log else output_path.with_suffix(".log")
    k = args.k if args.k is not None else int(qa_config.get("k", 5))
    max_context_chars = args.max_context_chars or int(qa_config.get("max_context_chars", 12000))
    answer_model = args.model or llm_config.get("generation_model", "gpt-4o-mini")
    judge_model = args.judge_model or llm_config.get("judge_model", answer_model)
    api_key_env = args.api_key_env or llm_config.get("api_key_env", "E_MEM_API_KEY")
    base_url_env = args.base_url_env or llm_config.get("base_url_env", "E_MEM_BASE_URL")
    base_url = args.base_url or llm_config.get("base_url")
    max_answer_tokens = args.max_answer_tokens or int(qa_config.get("max_answer_tokens", 128))
    categories = parse_categories(args.categories)

    graph = JsonGraphStore().load(graph_path)
    items = read_json(paths_path)
    items = [item for item in items if int(item.get("category") or 0) in categories]
    gold_answers = load_gold_answers(config)
    if args.limit:
        items = items[: args.limit]
    client = OpenAICompatibleChatClient.from_env(
        api_key_env=api_key_env,
        base_url_env=base_url_env,
        default_base_url=base_url,
    )
    runner = OpenAICompatibleQARunner(
        client=client,
        model=answer_model,
        max_tokens=max_answer_tokens,
        answer_protocol=args.answer_protocol,
    )
    judge = None if args.skip_judge else OpenAICompatibleLLMJudge(client=client, model=judge_model)

    per_question = load_existing_results(output_path) if args.resume else []
    completed = {row.get("question_id") for row in per_question}
    if completed:
        print(f"resuming {output_path}: loaded {len(completed)} completed questions", flush=True)
    append_log(
        log_path,
        f"START output={output_path} paths={paths_path} k={k} questions={len(items)} resume={args.resume}",
    )
    previous_completed_conv = latest_completed_conversation(per_question)
    for index, item in enumerate(items, start=1):
        if item["question_id"] in completed:
            print(f"skipped {index}/{len(items)} qid={item['question_id']} already_done", flush=True)
            continue
        effective_context_mode = effective_context_mode_for_item(args.context_mode, item)
        if effective_context_mode == "compiled":
            context = build_compiled_context(graph, item, k=k, max_chars=max_context_chars)
        elif effective_context_mode.startswith("mnemis_"):
            context = build_mnemis_style_context(
                graph,
                item,
                k=k,
                max_chars=max_context_chars,
                mode=effective_context_mode,
                max_message_chunks=args.max_message_chunks,
            )
        else:
            context = build_context(graph, item, k=k, max_chars=max_context_chars)
        context_node_ids = candidate_evidence_node_ids(graph, item.get("paths", []), k)
        retrieval = evaluate_item(graph, item, k)
        category = int(item.get("category") or 0)
        question_type = classify_question(item["question"])
        answer_result = runner.answer_with_metadata(
            item["question"],
            context,
            category=category,
            question_type=question_type,
        )
        draft_prediction = answer_result.content
        prediction = draft_prediction
        verifier_result = None
        if should_verify(args.verify_answer, question_type):
            verifier_result = verify_answer_with_metadata(
                client=client,
                model=answer_model,
                question=item["question"],
                context=context,
                draft_answer=draft_prediction,
                max_tokens=args.max_verifier_tokens,
            )
            prediction = verifier_result.content
        gold_answer, gold_answer_source = gold_answer_for_item(item, gold_answers)
        conv_id = conversation_id(item["question_id"])
        result = {
            "question_id": item["question_id"],
            "conversation_id": conv_id,
            "question": item["question"],
            "gold_answer": gold_answer,
            "gold_answer_source": gold_answer_source,
            "prediction": prediction,
            "draft_prediction": draft_prediction,
            "category": category,
            "question_type": question_type,
            "context_node_ids": context_node_ids,
            "context_mode": args.context_mode,
            "effective_context_mode": effective_context_mode,
            "answer_protocol": args.answer_protocol,
            "verify_answer": args.verify_answer,
            "context_chars": len(context),
            "context_truncated": context.endswith("[context truncated]"),
            "matched_evidence_ids": retrieval.matched_evidence_ids,
            "retrieval_hit": retrieval.hit,
            "retrieval_recall": retrieval.recall,
            "retrieval_full_cover": retrieval.full_cover,
            "retrieval_tokens": retrieval.tokens,
            "retrieval_path_len": retrieval.path_len,
            "generation_usage": normalize_usage(answer_result.usage),
            "generation_elapsed_seconds": answer_result.elapsed_seconds,
            "lexical_f1": token_f1(prediction, gold_answer),
            "bleu1": bleu1(prediction, gold_answer),
        }
        if verifier_result is not None:
            result["verifier_usage"] = normalize_usage(verifier_result.usage)
            result["verifier_elapsed_seconds"] = verifier_result.elapsed_seconds
        if judge is not None:
            result.update(judge.judge_with_metadata(item["question"], gold_answer, prediction))
        result["total_usage"] = add_usage(
            add_usage(result.get("generation_usage", {}), result.get("verifier_usage", {})),
            result.get("judge_usage", {}),
        )
        result["total_api_elapsed_seconds"] = float(result.get("generation_elapsed_seconds", 0.0)) + float(
            result.get("verifier_elapsed_seconds", 0.0)
        ) + float(
            result.get("judge_elapsed_seconds", 0.0)
        )
        per_question.append(result)
        if args.save_every and len(per_question) % args.save_every == 0:
            write_json(
                build_payload(
                    per_question,
                    paths_path=paths_path,
                    graph_path=graph_path,
                    k=k,
                    limit=args.limit or None,
                    answer_model=answer_model,
                    judge_model=None if args.skip_judge else judge_model,
                    base_url=client.base_url,
                    api_key_env=api_key_env,
                    started=started,
                    skipped_judge=args.skip_judge,
                    categories=categories,
                    context_mode=args.context_mode,
                    answer_protocol=args.answer_protocol,
                    verify_answer=args.verify_answer,
                    max_message_chunks=args.max_message_chunks,
                ),
                output_path,
            )
        append_log(log_path, format_question_log(index, len(items), result, time.perf_counter() - started))
        current_completed_conv = completed_conversation_if_boundary(items, index, per_question)
        if current_completed_conv and current_completed_conv != previous_completed_conv:
            conv_summary = summarize_by_conversation(per_question, skipped_judge=args.skip_judge)
            conv_payload = next((row for row in conv_summary if row["conversation_id"] == current_completed_conv), None)
            if conv_payload:
                append_log(log_path, format_conversation_log(conv_payload))
            previous_completed_conv = current_completed_conv
        print(
            f"processed {index}/{len(items)} qid={item['question_id']} "
            f"judge={result.get('judge_label', 'SKIP')} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    payload = build_payload(
        per_question,
        paths_path=paths_path,
        graph_path=graph_path,
        k=k,
        limit=args.limit or None,
        answer_model=answer_model,
        judge_model=None if args.skip_judge else judge_model,
        base_url=client.base_url,
        api_key_env=api_key_env,
        started=started,
        skipped_judge=args.skip_judge,
        categories=categories,
        context_mode=args.context_mode,
        answer_protocol=args.answer_protocol,
        verify_answer=args.verify_answer,
        max_message_chunks=args.max_message_chunks,
    )
    write_json(payload, output_path)
    append_log(log_path, format_overall_log(payload["summary"]))
    print(f"wrote {output_path}")
    print(f"log {log_path}")
    print(payload["summary"])


def load_existing_results(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    payload = read_json(output_path)
    if isinstance(payload, dict):
        rows = payload.get("per_question", [])
        if isinstance(rows, list):
            return rows
    if isinstance(payload, list):
        return payload
    return []


def should_verify(mode: str, question_type: str) -> bool:
    if mode == "all":
        return True
    if mode == "risk":
        return question_type in {"count", "temporal", "detail", "inference"}
    if mode == "count_detail":
        return question_type in {"count", "detail"}
    return False


def verify_answer_with_metadata(
    *,
    client: OpenAICompatibleChatClient,
    model: str,
    question: str,
    context: str,
    draft_answer: str,
    max_tokens: int,
):
    return client.chat_completion_with_metadata(
        model=model,
        messages=[
            ChatMessage(
                role="user",
                content=VERIFIER_PROMPT.format(
                    question=question,
                    context=context,
                    draft_answer=draft_answer,
                ),
            )
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )


def build_payload(
    per_question: list[dict],
    *,
    paths_path: Path,
    graph_path: Path,
    k: int,
    limit: int | None,
    answer_model: str,
    judge_model: str | None,
    base_url: str,
    api_key_env: str,
    started: float,
    skipped_judge: bool,
    categories: set[int],
    context_mode: str,
    answer_protocol: str,
    verify_answer: str,
    max_message_chunks: int = 12,
) -> dict:
    summary = summarize_qa(per_question, skipped_judge=skipped_judge)
    return {
        "metadata": {
            "paths": str(paths_path),
            "graph": str(graph_path),
            "k": k,
            "limit": limit,
            "generation_model": answer_model,
            "judge_model": judge_model,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "elapsed_seconds": time.perf_counter() - started,
            "categories": sorted(categories),
            "context_mode": context_mode,
            "max_message_chunks": max_message_chunks,
            "answer_protocol": answer_protocol,
            "verify_answer": verify_answer,
            "metric_protocol": METRIC_PROTOCOL,
            "metric_normalization": "lowercase, remove punctuation, remove a/an/the, collapse whitespace",
            "bleu1": "clipped unigram precision with brevity penalty",
        },
        "summary": summary,
        "per_conversation": summarize_by_conversation(per_question, skipped_judge=skipped_judge),
        "per_question": per_question,
    }


def build_context(graph, item: dict, *, k: int, max_chars: int) -> str:
    lines = []
    seen_supports: set[str] = set()
    for rank, node_id in enumerate(candidate_evidence_node_ids(graph, item.get("paths", []), k), start=1):
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        turn_id = node.metadata.get("turn_id") or node.node_id.rsplit(":", 1)[-1]
        evidence_turns = evidence_turn_ids(node)
        metadata_timestamps = [str(item) for item in node.metadata.get("support_timestamps", []) if item]
        timestamps = sorted(set(metadata_timestamps))
        if not timestamps and node.time:
            timestamps = [str(node.time)]
        time_part = f" time={'; '.join(timestamps)}" if timestamps else ""
        support_part = f" supports={','.join(evidence_turns)}" if evidence_turns else ""
        line = (
            f"[{rank}] node={node.node_id} evidence={turn_id} type={node.type.value}"
            f"{time_part}{support_part}\n{node.text.strip()}"
        )
        support_lines = support_context_lines(graph, node, seen_supports)
        if support_lines:
            line += "\n" + "\n".join(support_lines)
        lines.append(line)
    context = "\n\n".join(lines)
    if len(context) > max_chars:
        return context[:max_chars] + "\n\n[context truncated]"
    return context


def build_compiled_context(graph, item: dict, *, k: int, max_chars: int) -> str:
    records = evidence_records(graph, item, k)
    question = str(item.get("question", ""))
    question_type = classify_question(question)
    key_records = select_key_records(records)
    key_ids = {record["node_id"] for record in key_records}
    card_records = [record for record in records if record["is_card"] and record["node_id"] not in key_ids]
    additional = [record for record in records if record["node_id"] not in key_ids and record["node_id"] not in {r["node_id"] for r in card_records}]

    lines = [
        "Evidence Compiler v1.1.",
        f"Question type: {question_type}.",
        "Use only evidence that directly answers the question.",
        "Ignore evidence about the same person/topic if it refers to a different time, event, object, or constraint.",
        "Prefer the most specific supported answer. For dates, resolve relative times using evidence timestamps.",
        "Evidence timestamps are message times, not always event times. If the raw evidence says yesterday, last week, tomorrow, or next month, compute the event time relative to the message timestamp.",
        "Relation-card markers indicate grouping confidence only; do not treat relation-card summaries as evidence unless the raw/fact claim also supports them.",
    ]
    if question_type == "count":
        lines.append("For counting, count unique matching events/items only; do not count duplicate mentions.")
    elif question_type == "temporal":
        lines.append("For temporal questions, compare candidate dates and choose the one that matches the asked event.")
    elif question_type == "inference":
        lines.append("For inference questions, state the most likely answer supported by explicit evidence.")

    lines.extend(["", "### Key Evidence"])
    lines.extend(format_record(record, question_type=question_type) for record in key_records)

    if question_type == "temporal":
        lines.extend(["", "### Timeline Candidates"])
        for record in sorted(records, key=timestamp_sort_key):
            lines.append(format_record(record, compact=True, question_type=question_type))
    elif question_type == "count":
        lines.extend(["", "### Count Candidates"])
        for record in records:
            lines.append(format_record(record, compact=True, question_type=question_type))

    if card_records:
        lines.extend(["", "### Relation-card Evidence"])
        lines.extend(format_record(record, question_type=question_type) for record in card_records[:8])

    if additional:
        lines.extend(["", "### Additional Context"])
        lines.extend(format_record(record, compact=True, question_type=question_type) for record in additional[: max(0, k)])

    lines.extend(
        [
            "",
            "### Final Answer Requirements",
            "Return only the answer.",
            "If multiple evidence items conflict, choose the answer supported by the evidence that best matches the question's time/event/object.",
            "If the evidence supports a specific object, date, count, preference, or reason, include that specific detail.",
            "For temporal answers, do not simply repeat the message timestamp; use the event date implied by the claim and raw support.",
        ]
    )
    context = "\n".join(lines)
    if len(context) > max_chars:
        return context[:max_chars] + "\n\n[context truncated]"
    return context


def build_mnemis_style_context(
    graph,
    item: dict,
    *,
    k: int,
    max_chars: int,
    mode: str,
    max_message_chunks: int = 12,
) -> str:
    records = evidence_records(graph, item, k)
    question = str(item.get("question", ""))
    question_type = classify_question(question)
    include_chunks = mode in {"mnemis_c1_chunks", "mnemis_c4_full"}
    include_window_chunks = mode in {"mnemis_c1_window_chunks", "mnemis_c4_window_full"}
    include_entities = mode in {"mnemis_c2_entities", "mnemis_c4_full"}
    include_window_full = mode == "mnemis_c4_window_full"
    include_entities = include_entities or include_window_full
    include_cards = mode in {"mnemis_c3_cards", "mnemis_c4_full"} or include_window_full

    lines = [
        "# Structured Memory Pack",
        "Use this memory pack as historical conversation evidence. Prefer specific facts and raw message chunks over broad summaries.",
        "If summaries and raw messages disagree, trust the raw message chunks.",
        f"Question type: {question_type}.",
        "",
        "<QUERY>",
        question,
        "</QUERY>",
        "",
        "<ANSWER_RULES>",
        "- Return only the final answer, without explanation.",
        "- For temporal questions, resolve relative time using the message timestamp in MESSAGE_CHUNKS when available.",
        "- For count questions, count unique matching events/items only; exclude plans, repeated mentions, and hypotheticals.",
        "- For preference or likely/would questions, use relation cards and raw evidence to infer the most likely answer.",
        "- Say you do not know only if no relevant evidence is present.",
        "</ANSWER_RULES>",
    ]

    if include_entities:
        entity_lines = format_entity_summaries(graph, records, limit=14)
        lines.extend(["", "<ENTITIES>"])
        lines.extend(entity_lines or ["  - No high-confidence entity summary found for the selected evidence."])
        lines.append("</ENTITIES>")

    if include_cards:
        card_lines = format_relation_cards(records, limit=8)
        lines.extend(["", "<RELATION_CARDS>"])
        lines.extend(card_lines or ["  - No relation card selected for this question."])
        lines.append("</RELATION_CARDS>")

    lines.extend(["", "<FACTS>"])
    lines.extend(format_grouped_facts(graph, records))
    lines.append("</FACTS>")

    if include_chunks or include_window_chunks:
        if include_window_chunks:
            chunk_lines = format_message_window_chunks(graph, records, limit=max_message_chunks, window=2)
        else:
            chunk_lines = format_message_chunks(graph, records, limit=24)
        lines.extend(["", "<MESSAGE_CHUNKS>"])
        lines.extend(chunk_lines or ["  - No raw dialogue chunks found for the selected evidence."])
        lines.append("</MESSAGE_CHUNKS>")

    path_lines = format_evidence_paths(graph, item, k=k, limit=20)
    lines.extend(["", "<EVIDENCE_PATHS>"])
    lines.extend(path_lines or ["  - No graph paths available."])
    lines.append("</EVIDENCE_PATHS>")

    context = "\n".join(lines)
    if len(context) > max_chars:
        return context[:max_chars] + "\n\n[context truncated]"
    return context


def format_entity_summaries(graph, records: list[dict], *, limit: int) -> list[str]:
    selected_fact_ids = {record["node_id"] for record in records}
    selected_raw_ids = {raw_id for record in records for raw_id in record.get("raw_ids", [])}
    candidate_entities = {
        normalize_entity_name(record.get("card_entity", ""))
        for record in records
        if record.get("card_entity")
    }
    for record in records:
        node = graph.nodes.get(record["node_id"])
        if node is None:
            continue
        for related_id in [record.get("event_id"), record.get("topic_id")]:
            related = graph.nodes.get(str(related_id)) if related_id else None
            for entity in (related.metadata.get("entities", []) if related else []):
                normalized = normalize_entity_name(str(entity))
                if normalized:
                    candidate_entities.add(normalized)

    rows = []
    for node in graph.nodes.values():
        if node.type not in {NodeType.ENTITY, NodeType.ENTITY_STATE}:
            continue
        normalized = normalize_entity_name(str(node.metadata.get("normalized_entity") or node.metadata.get("entity") or ""))
        if candidate_entities and normalized and normalized not in candidate_entities:
            continue
        fact_ids = set(map(str, node.metadata.get("fact_ids", []))) | set(map(str, node.support_ids))
        raw_ids = set(map(str, node.metadata.get("raw_ids", [])))
        overlap = len(fact_ids & selected_fact_ids) + len(raw_ids & selected_raw_ids)
        if overlap <= 0 and candidate_entities:
            overlap = int(normalized in candidate_entities)
        if overlap <= 0:
            continue
        type_priority = 1 if node.type == NodeType.ENTITY_STATE else 0
        rows.append((type_priority, overlap, len(fact_ids), node))

    rows.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    output = []
    for _, overlap, fact_count, node in rows[:limit]:
        entity = node.metadata.get("entity") or node.metadata.get("normalized_entity") or node.node_id
        sessions = ",".join(map(str, node.metadata.get("session_ids", [])[:6]))
        output.append(
            f"  - entity={entity} node={node.node_id} related_facts={fact_count} overlap={overlap} "
            f"sessions={sessions}\n    summary={one_line(node.text, 420)}"
        )
    return output


def format_relation_cards(records: list[dict], *, limit: int) -> list[str]:
    cards: dict[str, dict] = {}
    for record in records:
        if not record.get("is_card"):
            continue
        key = record.get("card_key") or f"rank:{record.get('card_rank')}:summary:{record.get('card_summary')}"
        card = cards.setdefault(
            key,
            {
                "rank": record.get("card_rank") or "",
                "type": record.get("card_type") or "",
                "entity": record.get("card_entity") or "",
                "aspect": record.get("card_aspect") or "",
                "summary": record.get("card_summary") or "",
                "why": record.get("card_why") or "",
                "cardce": record.get("cardce", 0.0),
                "facts": [],
                "raws": [],
            },
        )
        card["cardce"] = max(float(card.get("cardce") or 0.0), float(record.get("cardce") or 0.0))
        card["facts"].append(record)
        card["raws"].extend(record.get("raw_ids", []))

    ranked = sorted(cards.values(), key=lambda card: (float(card.get("cardce") or 0.0), len(card["facts"])), reverse=True)
    output = []
    for index, card in enumerate(ranked[:limit], start=1):
        fact_refs = "; ".join(
            f"{fact['node_id']} role={fact.get('card_role') or ''} rank={fact['rank']}"
            for fact in card["facts"][:8]
        )
        raw_refs = ",".join(dedupe_keep_order(card["raws"])[:10])
        output.append(
            f"  - card={index} type={card['type']} entity={card['entity']} aspect={card['aspect']} "
            f"score={float(card.get('cardce') or 0.0):.3f}\n"
            f"    summary={one_line(card['summary'], 360)}\n"
            f"    why={one_line(card['why'], 260)}\n"
            f"    support_facts={fact_refs}\n"
            f"    support_raw_turns={raw_refs}"
        )
    return output


def format_grouped_facts(graph, records: list[dict]) -> list[str]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        key = (
            record.get("card_entity")
            or record.get("card_type")
            or record.get("event_id")
            or "ungrouped"
        )
        groups.setdefault(str(key), []).append(record)
    output = []
    for group, group_records in sorted(groups.items(), key=lambda row: min(record["rank"] for record in row[1])):
        output.append(f"  - group={group}")
        for record in sorted(group_records, key=lambda item: item["rank"]):
            time = "; ".join(record["timestamps"]) if record["timestamps"] else "unknown"
            raw_refs = ",".join(record.get("raw_ids", [])[:5])
            card = ""
            if record.get("is_card"):
                card = f" card=true type={record.get('card_type','')} role={record.get('card_role','')}"
            output.append(
                f"    [{record['rank']}] node={record['node_id']} time={time}{card} raw={raw_refs}\n"
                f"        fact={one_line(record['claim'], 420)}"
            )
    return output


def format_message_chunks(graph, records: list[dict], *, limit: int) -> list[str]:
    raw_ids = []
    rank_by_raw = {}
    for record in records:
        for raw_id in record.get("raw_ids", []):
            raw_ids.append(raw_id)
            rank_by_raw.setdefault(raw_id, record["rank"])
    raw_ids = dedupe_keep_order(raw_ids)
    raw_ids.sort(key=lambda raw_id: (raw_sort_key(raw_id), rank_by_raw.get(raw_id, 9999)))
    output = []
    for index, raw_id in enumerate(raw_ids[:limit], start=1):
        raw = graph.nodes.get(raw_id)
        if raw is None:
            continue
        session = raw.metadata.get("turn_id", raw_id).split(":", 1)[0]
        speaker = raw.metadata.get("speaker", "")
        timestamp = raw.time or ""
        output.append(
            f"Message Chunk {index} ({session}) source_fact_rank={rank_by_raw.get(raw_id, '')}:\n"
            f"[{timestamp}] {speaker}: {one_line(raw.text, 900)}"
        )
    return output


def format_message_window_chunks(graph, records: list[dict], *, limit: int, window: int) -> list[str]:
    raw_by_session = raw_session_index(graph)
    support_raw_ids = []
    rank_by_raw = {}
    for record in records:
        for raw_id in record.get("raw_ids", []):
            if raw_id not in graph.nodes:
                continue
            support_raw_ids.append(raw_id)
            rank_by_raw.setdefault(raw_id, record["rank"])

    intervals_by_session: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    for raw_id in dedupe_keep_order(support_raw_ids):
        raw = graph.nodes.get(raw_id)
        if raw is None:
            continue
        conv_id = raw_id.split(":", 1)[0]
        turn_id = str(raw.metadata.get("turn_id") or raw_id.split(":raw:", 1)[-1])
        session_id = turn_id.split(":", 1)[0]
        session_raws = raw_by_session.get((conv_id, session_id), [])
        if raw_id not in session_raws:
            continue
        position = session_raws.index(raw_id)
        start = max(0, position - window)
        end = min(len(session_raws), position + window + 1)
        intervals_by_session.setdefault((conv_id, session_id), []).append((start, end, rank_by_raw.get(raw_id, 9999)))

    windows = []
    for (conv_id, session_id), intervals in intervals_by_session.items():
        session_raws = raw_by_session.get((conv_id, session_id), [])
        merged = merge_intervals(intervals)
        for start, end, min_rank in merged:
            window_raws = session_raws[start:end]
            windows.append((min_rank, (start, end), conv_id, session_id, start, end, window_raws))
    windows.sort(key=lambda row: (row[0], row[1]))
    output = []
    for index, (_, _, conv_id, session_id, start, end, window_raws) in enumerate(windows[:limit], start=1):
        lines = []
        ranks = []
        for raw_id in window_raws:
            raw = graph.nodes.get(raw_id)
            if raw is None:
                continue
            rank = rank_by_raw.get(raw_id)
            if rank is not None:
                ranks.append(rank)
            speaker = raw.metadata.get("speaker", "")
            timestamp = raw.time or ""
            turn_id = raw.metadata.get("turn_id") or raw_id.split(":raw:", 1)[-1]
            prefix = f"[{timestamp}] " if timestamp else ""
            speaker_part = f"{speaker}: " if speaker else ""
            lines.append(f"{prefix}{speaker_part}{one_line(raw.text, 520)}")
        rank_part = f" source_fact_ranks={','.join(map(str, sorted(set(ranks))))}" if ranks else ""
        text = "\n".join(lines)
        output.append(
            f"Message Chunk {index} ({session_id}) turns={start}-{end - 1}{rank_part}:\n{text[:1800]}"
        )
    return output


def merge_intervals(intervals: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[tuple[int, int, int]] = []
    cur_start, cur_end, cur_rank = intervals[0]
    for start, end, rank in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            cur_rank = min(cur_rank, rank)
            continue
        merged.append((cur_start, cur_end, cur_rank))
        cur_start, cur_end, cur_rank = start, end, rank
    merged.append((cur_start, cur_end, cur_rank))
    return merged


def format_evidence_paths(graph, item: dict, *, k: int, limit: int) -> list[str]:
    output = []
    for rank, path in enumerate(item.get("paths", [])[: min(k, limit)], start=1):
        metadata = path.get("metadata", {})
        fact_id = candidate_evidence_node_id(graph, path) or metadata.get("evidence_node_id") or ""
        event_id = metadata.get("event_node_id") or ""
        topic_id = metadata.get("topic_node_id") or ""
        route = metadata.get("route_source") or metadata.get("candidate_source") or ""
        card = metadata.get("v3_9_card_summary") or ""
        output.append(
            f"  - rank={rank} fact={fact_id} event={event_id} topic={topic_id} route={route}"
            + (f"\n    relation_card={one_line(str(card), 240)}" if card else "")
        )
    return output


def effective_context_mode_for_item(context_mode: str, item: dict) -> str:
    if context_mode != "hybrid":
        return context_mode
    question_type = classify_question(str(item.get("question", "")))
    return "compiled" if question_type in {"temporal", "count"} else "raw"


def evidence_records(graph, item: dict, k: int) -> list[dict]:
    records = []
    seen: set[str] = set()
    for rank, path in enumerate(item.get("paths", [])[:k], start=1):
        node_id = candidate_evidence_node_id(graph, path)
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        metadata = path.get("metadata", {})
        scores = path.get("scores", {})
        timestamps = record_timestamps(node)
        support_raw_ids = raw_support_ids(node)
        raw_supports = raw_support_snippets(graph, node, limit=2)
        records.append(
            {
                "rank": rank,
                "node_id": node.node_id,
                "node_type": node.type.value,
                "turn_id": node.metadata.get("turn_id") or node.node_id.rsplit(":", 1)[-1],
                "evidence_turns": evidence_turn_ids(node),
                "timestamps": timestamps,
                "claim": " ".join(str(node.text).split()),
                "raw_ids": support_raw_ids,
                "raw_supports": raw_supports,
                "route": str(metadata.get("route_source", "") or metadata.get("candidate_source", "") or ""),
                "event_id": str(metadata.get("event_node_id", "") or ""),
                "topic_id": str(metadata.get("topic_node_id", "") or ""),
                "card_key": str(metadata.get("nary_hyperedge_id", "") or metadata.get("v3_9_card_rank", "") or ""),
                "card_summary": str(metadata.get("v3_9_card_summary", "") or ""),
                "card_type": str(metadata.get("nary_hyperedge_type", "") or ""),
                "card_role": str(metadata.get("nary_role", "") or ""),
                "card_rank": str(metadata.get("v3_9_card_rank", "") or ""),
                "card_entity": str(metadata.get("v3_9_card_entity", "") or ""),
                "card_aspect": str(metadata.get("v3_9_card_aspect", "") or ""),
                "card_why": str(metadata.get("v3_9_why_relevant", "") or ""),
                "is_card": is_truthy(metadata.get("v3_9_query_card"))
                or is_truthy(metadata.get("is_nary_completion"))
                or is_truthy(metadata.get("card_quota_selected")),
                "card_quota_selected": is_truthy(metadata.get("card_quota_selected")),
                "cardce": parse_float(scores.get("v3_9_card_ce") or metadata.get("card_quota_cardce_score")) or 0.0,
                "ce": parse_float(scores.get("cross_encoder")) or parse_float(path.get("score")) or 0.0,
                "selector": parse_float(scores.get("topology_selector")) or 0.0,
            }
        )
    return records


def select_key_records(records: list[dict]) -> list[dict]:
    ranked = sorted(records, key=key_record_score, reverse=True)
    selected = []
    seen = set()
    for record in ranked:
        if record["node_id"] in seen:
            continue
        selected.append(record)
        seen.add(record["node_id"])
        if len(selected) >= 8:
            break
    return sorted(selected, key=lambda record: record["rank"])


def key_record_score(record: dict) -> tuple[float, float, float]:
    quota_bonus = 3.0 if record["card_quota_selected"] else 0.0
    card_bonus = 1.0 if record["is_card"] else 0.0
    rank_bonus = 1.0 / max(float(record["rank"]), 1.0)
    return (quota_bonus + card_bonus + rank_bonus, record["cardce"], record["selector"] + record["ce"])


def format_record(record: dict, *, compact: bool = False, question_type: str = "detail") -> str:
    timestamps = "; ".join(record["timestamps"]) if record["timestamps"] else "unknown"
    evidence = ",".join(record["evidence_turns"]) if record["evidence_turns"] else str(record["turn_id"])
    card = ""
    if record["is_card"]:
        card = " card=true"
        if record["card_type"]:
            card += f" type={record['card_type']}"
    claim_limit = 220 if compact else 360
    raw_limit = 180 if compact else 320
    line = (
        f"[E{record['rank']}] time={timestamps} evidence={evidence}{card} "
        f"claim={record['claim'][:claim_limit]}"
    )
    if record["card_summary"] and not compact and question_type not in {"temporal", "count"}:
        line += f"\n  relation_card={record['card_summary'][:260]}"
    if record["raw_supports"]:
        raw = " | ".join(record["raw_supports"])[:raw_limit]
        line += f"\n  raw={raw}"
    return line


def classify_question(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(how many|number of|how much|times|months passed|years passed)\b", q):
        return "count"
    if re.search(r"\b(when|what date|which date|before|after|last week|next month|ago|year|month|day|weekend)\b", q):
        return "temporal"
    if re.search(r"\b(would|likely|might|could|if|considering|why|how does|how did)\b", q):
        return "inference"
    return "detail"


def record_timestamps(node) -> list[str]:
    timestamps = [str(item) for item in node.metadata.get("support_timestamps", []) if item]
    if not timestamps and node.time:
        timestamps = [str(node.time)]
    deduped = []
    seen = set()
    for timestamp in timestamps:
        if timestamp in seen:
            continue
        seen.add(timestamp)
        deduped.append(timestamp)
    return deduped


def raw_support_ids(node) -> list[str]:
    raw_ids = list(map(str, node.metadata.get("support_raw_ids", []))) or list(map(str, node.support_ids))
    output = []
    seen = set()
    for raw_id in raw_ids:
        if not raw_id or raw_id in seen:
            continue
        seen.add(raw_id)
        output.append(raw_id)
    return output


def timestamp_sort_key(record: dict) -> tuple[str, int]:
    timestamp = record["timestamps"][0] if record["timestamps"] else ""
    return (timestamp, int(record["rank"]))


def raw_support_snippets(graph, node, *, limit: int) -> list[str]:
    snippets = []
    support_texts = list(node.metadata.get("support_texts", []))
    support_raw_ids = raw_support_ids(node)
    for index, support_id in enumerate(support_raw_ids[:limit]):
        raw_node = graph.nodes.get(str(support_id))
        text = raw_node.text if raw_node is not None and raw_node.type == NodeType.RAW else ""
        if not text and index < len(support_texts):
            text = str(support_texts[index])
        text = " ".join(str(text).split())
        if text:
            snippets.append(text)
    return snippets


def normalize_entity_name(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    return re.sub(r"\s+", " ", text)


def dedupe_keep_order(items: list[str]) -> list[str]:
    output = []
    seen = set()
    for item in items:
        item = str(item)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def raw_sort_key(raw_id: str) -> tuple[int, int]:
    match = re.search(r":raw:D(\d+):(\d+)$", raw_id)
    if not match:
        return (10**9, 10**9)
    return (int(match.group(1)), int(match.group(2)))


def raw_session_index(graph) -> dict[tuple[str, str], list[str]]:
    cached = getattr(graph, "_hytopomem_raw_session_index", None)
    if cached is not None:
        return cached
    index: dict[tuple[str, str], list[str]] = {}
    for node in graph.iter_nodes(NodeType.RAW):
        conv_id = node.node_id.split(":", 1)[0]
        turn_id = str(node.metadata.get("turn_id") or node.node_id.split(":raw:", 1)[-1])
        session_id = turn_id.split(":", 1)[0]
        index.setdefault((conv_id, session_id), []).append(node.node_id)
    for raw_ids in index.values():
        raw_ids.sort(key=raw_sort_key)
    try:
        setattr(graph, "_hytopomem_raw_session_index", index)
    except Exception:
        pass
    return index


def one_line(text: str, limit: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


def candidate_evidence_node_id(graph, path: dict) -> str | None:
    return next(
        (
            node_id
            for node_id in reversed(path.get("node_ids", []))
            if node_id in graph.nodes and graph.nodes[node_id].type in {NodeType.FACT, NodeType.RAW}
        ),
        None,
    )


def candidate_evidence_node_ids(graph, paths: list[dict], k: int) -> list[str]:
    node_ids: list[str] = []
    seen: set[str] = set()
    for path in paths[:k]:
        candidate_id = next(
            (
                node_id
                for node_id in reversed(path.get("node_ids", []))
                if node_id in graph.nodes and graph.nodes[node_id].type in {NodeType.FACT, NodeType.RAW}
            ),
            None,
        )
        if candidate_id and candidate_id not in seen:
            seen.add(candidate_id)
            node_ids.append(candidate_id)
    return node_ids


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def support_context_lines(graph, node, seen_supports: set[str]) -> list[str]:
    lines = []
    support_timestamps = list(node.metadata.get("support_timestamps", []))
    support_texts = list(node.metadata.get("support_texts", []))
    support_raw_ids = list(node.metadata.get("support_raw_ids", [])) or list(node.support_ids)
    for index, support_id in enumerate(support_raw_ids[:2]):
        raw_node = graph.nodes.get(str(support_id))
        text = raw_node.text if raw_node is not None and raw_node.type == NodeType.RAW else ""
        timestamp = raw_node.time if raw_node is not None else None
        if not text and index < len(support_texts):
            text = str(support_texts[index])
        if not timestamp and index < len(support_timestamps):
            timestamp = support_timestamps[index]
        normalized_text = " ".join(str(text).split())
        support_key = str(support_id) if support_id else normalized_text
        if not normalized_text or support_key in seen_supports:
            continue
        seen_supports.add(support_key)
        time_part = f" time={timestamp}" if timestamp else ""
        lines.append(f"Raw support{time_part}: {normalized_text[:900]}")
    return lines


def load_gold_answers(config: dict) -> dict[str, str]:
    processed_path = resolve_path(config["data"]["processed_path"])
    conversations = read_json(processed_path)
    answers: dict[str, str] = {}
    for conversation in conversations:
        for item in conversation.get("qa", []):
            answer = str(item.get("answer", "")).strip()
            if answer:
                answers[str(item["question_id"])] = answer
    return answers


def gold_answer_for_item(item: dict, gold_answers: dict[str, str]) -> tuple[str, str]:
    path_answer = str(item.get("answer", "")).strip()
    if path_answer:
        return path_answer, "paths"
    mapped_answer = gold_answers.get(str(item["question_id"]), "")
    if mapped_answer:
        return mapped_answer, "processed_qa"
    return "", "missing"


def evidence_turn_ids(node) -> list[str]:
    turns: list[str] = []
    own_turn = node.metadata.get("turn_id")
    if own_turn:
        turns.append(normalize_evidence_id(str(own_turn)))
    for support_id in node.support_ids:
        turns.append(normalize_evidence_id(str(support_id)))
    deduped = []
    seen = set()
    for turn in turns:
        if turn in seen:
            continue
        seen.add(turn)
        deduped.append(turn)
    return deduped


def summarize_qa(rows: list[dict], *, skipped_judge: bool) -> dict:
    summary = {"num_questions": len(rows), "judge_skipped": skipped_judge}
    summary.update(metric_totals(rows))
    if skipped_judge:
        return summary
    if not rows:
        summary["judge_accuracy"] = 0.0
        return summary
    summary["judge_accuracy"] = sum(float(row.get("judge_correct", 0)) for row in rows) / len(rows)
    summary["num_correct"] = int(sum(int(row.get("judge_correct", 0)) for row in rows))
    return summary


def summarize_by_conversation(rows: list[dict], *, skipped_judge: bool) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("conversation_id") or conversation_id(row.get("question_id", ""))), []).append(row)

    summaries = []
    for conv_id, conv_rows in sorted(grouped.items()):
        summary = {
            "conversation_id": conv_id,
            "num_questions": len(conv_rows),
            "judge_skipped": skipped_judge,
            **metric_totals(conv_rows),
        }
        if not skipped_judge:
            summary["num_correct"] = int(sum(int(row.get("judge_correct", 0)) for row in conv_rows))
            summary["judge_accuracy"] = (
                sum(float(row.get("judge_correct", 0)) for row in conv_rows) / len(conv_rows) if conv_rows else 0.0
            )
        summaries.append(summary)
    return summaries


def metric_totals(rows: list[dict]) -> dict:
    if not rows:
        return {
            "macro_f1": 0.0,
            "macro_bleu1": 0.0,
            "generation_tokens": 0,
            "verifier_tokens": 0,
            "judge_tokens": 0,
            "total_tokens": 0,
            "generation_api_seconds": 0.0,
            "verifier_api_seconds": 0.0,
            "judge_api_seconds": 0.0,
            "total_api_seconds": 0.0,
            "retrieval_hit": 0.0,
            "retrieval_recall": 0.0,
            "retrieval_full_cover": 0.0,
            "retrieval_avg_tokens": 0.0,
            "retrieval_avg_path_len": 0.0,
        }
    generation_tokens = sum(int(row.get("generation_usage", {}).get("total_tokens", 0)) for row in rows)
    verifier_tokens = sum(int(row.get("verifier_usage", {}).get("total_tokens", 0)) for row in rows)
    judge_tokens = sum(int(row.get("judge_usage", {}).get("total_tokens", 0)) for row in rows)
    generation_seconds = sum(float(row.get("generation_elapsed_seconds", 0.0)) for row in rows)
    verifier_seconds = sum(float(row.get("verifier_elapsed_seconds", 0.0)) for row in rows)
    judge_seconds = sum(float(row.get("judge_elapsed_seconds", 0.0)) for row in rows)
    num_calls = (
        sum(1 for row in rows if row.get("generation_usage"))
        + sum(1 for row in rows if row.get("verifier_usage"))
        + sum(1 for row in rows if row.get("judge_usage"))
    )
    return {
        "macro_f1": sum(float(row.get("lexical_f1", 0.0)) for row in rows) / len(rows),
        "macro_bleu1": sum(float(row.get("bleu1", 0.0)) for row in rows) / len(rows),
        "generation_tokens": generation_tokens,
        "verifier_tokens": verifier_tokens,
        "judge_tokens": judge_tokens,
        "total_tokens": generation_tokens + verifier_tokens + judge_tokens,
        "avg_tokens_per_question": (generation_tokens + verifier_tokens + judge_tokens) / len(rows),
        "avg_tokens_per_call": (generation_tokens + verifier_tokens + judge_tokens) / num_calls if num_calls else 0.0,
        "generation_api_seconds": generation_seconds,
        "verifier_api_seconds": verifier_seconds,
        "judge_api_seconds": judge_seconds,
        "total_api_seconds": generation_seconds + verifier_seconds + judge_seconds,
        "avg_api_seconds_per_question": (generation_seconds + verifier_seconds + judge_seconds) / len(rows),
        "retrieval_hit": sum(float(row.get("retrieval_hit", False)) for row in rows) / len(rows),
        "retrieval_recall": sum(float(row.get("retrieval_recall", 0.0)) for row in rows) / len(rows),
        "retrieval_full_cover": sum(float(row.get("retrieval_full_cover", False)) for row in rows) / len(rows),
        "retrieval_avg_tokens": sum(float(row.get("retrieval_tokens", 0.0)) for row in rows) / len(rows),
        "retrieval_avg_path_len": sum(float(row.get("retrieval_path_len", 0.0)) for row in rows) / len(rows),
    }


def normalize_usage(usage: dict) -> dict:
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def add_usage(left: dict, right: dict) -> dict:
    left_norm = normalize_usage(left)
    right_norm = normalize_usage(right)
    return {
        "prompt_tokens": left_norm["prompt_tokens"] + right_norm["prompt_tokens"],
        "completion_tokens": left_norm["completion_tokens"] + right_norm["completion_tokens"],
        "total_tokens": left_norm["total_tokens"] + right_norm["total_tokens"],
    }


def conversation_id(question_id: str) -> str:
    return str(question_id).split(":", 1)[0]


def parse_categories(value: str) -> set[int]:
    categories = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not categories:
        raise ValueError("at least one category is required")
    return categories


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
    brevity_penalty = 1.0 if len(pred_tokens) > len(gold_tokens) else pow(2.718281828459045, 1 - len(gold_tokens) / len(pred_tokens))
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
        f"QA {index}/{total} qid={row['question_id']} conv={row['conversation_id']} "
        f"judge={row.get('judge_label', 'SKIP')} retrieval_hit={int(bool(row.get('retrieval_hit')))} "
        f"retrieval_recall={float(row.get('retrieval_recall', 0.0)):.4f} "
        f"f1={float(row.get('lexical_f1', 0.0)):.4f} bleu1={float(row.get('bleu1', 0.0)):.4f} "
        f"tokens={int(row.get('total_usage', {}).get('total_tokens', 0))} "
        f"api_time={float(row.get('total_api_elapsed_seconds', 0.0)):.2f}s elapsed={elapsed:.1f}s\n"
        f"  Q: {one_line(row.get('question', ''))}\n"
        f"  GOLD: {one_line(row.get('gold_answer', ''))}\n"
        f"  PRED: {one_line(row.get('prediction', ''))}"
    )


def format_conversation_log(summary: dict) -> str:
    return (
        f"CONV_SUMMARY conv={summary['conversation_id']} n={summary['num_questions']} "
        f"acc={float(summary.get('judge_accuracy', 0.0)):.4f} "
        f"f1={float(summary.get('macro_f1', 0.0)):.4f} bleu1={float(summary.get('macro_bleu1', 0.0)):.4f} "
        f"hit={float(summary.get('retrieval_hit', 0.0)):.4f} recall={float(summary.get('retrieval_recall', 0.0)):.4f} "
        f"full_cover={float(summary.get('retrieval_full_cover', 0.0)):.4f} "
        f"avg_tokens_call={float(summary.get('avg_tokens_per_call', 0.0)):.1f} "
        f"total_tokens={int(summary.get('total_tokens', 0))} api_time={float(summary.get('total_api_seconds', 0.0)):.1f}s"
    )


def format_overall_log(summary: dict) -> str:
    return (
        f"OVERALL n={summary['num_questions']} acc={float(summary.get('judge_accuracy', 0.0)):.4f} "
        f"correct={int(summary.get('num_correct', 0))}/{summary['num_questions']} "
        f"f1={float(summary.get('macro_f1', 0.0)):.4f} bleu1={float(summary.get('macro_bleu1', 0.0)):.4f} "
        f"hit={float(summary.get('retrieval_hit', 0.0)):.4f} recall={float(summary.get('retrieval_recall', 0.0)):.4f} "
        f"full_cover={float(summary.get('retrieval_full_cover', 0.0)):.4f} "
        f"avg_tokens_call={float(summary.get('avg_tokens_per_call', 0.0)):.1f} "
        f"total_tokens={int(summary.get('total_tokens', 0))} api_time={float(summary.get('total_api_seconds', 0.0)):.1f}s"
    )


def one_line(value: object, max_len: int = 500) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def latest_completed_conversation(rows: list[dict]) -> str | None:
    if not rows:
        return None
    return str(rows[-1].get("conversation_id") or conversation_id(rows[-1].get("question_id", "")))


def completed_conversation_if_boundary(items: list[dict], current_index: int, rows: list[dict]) -> str | None:
    if not rows:
        return None
    current_conv = str(rows[-1].get("conversation_id") or conversation_id(rows[-1].get("question_id", "")))
    if current_index >= len(items):
        return current_conv
    next_conv = conversation_id(items[current_index]["question_id"])
    if next_conv != current_conv:
        return current_conv
    return None


if __name__ == "__main__":
    main()
