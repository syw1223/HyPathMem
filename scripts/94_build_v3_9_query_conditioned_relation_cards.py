from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from common import read_json, resolve_path, write_json
from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient
from hytopomem.eval.retrieval_metrics import evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore


SYSTEM_PROMPT = """You construct query-conditioned relation cards for long-term memory QA.

You are given one question and the top retrieved FACTS for that question.
Your job is to create only relation cards that are directly useful for answering the question.

Return STRICT JSON only:
{
  "cards": [
    {
      "type": "change|preference|state|plan_constraint|temporal",
      "entity": "",
      "aspect": "",
      "summary": "",
      "why_relevant_to_query": "",
      "needed_roles": ["role_name"],
      "roles": {
        "role_name": {"value": "", "fact_ids": []}
      },
      "support_facts": [],
      "confidence": 0.0
    }
  ]
}

Rules:
- Use ONLY provided fact_ids. Do not invent facts, ids, entities, or dates.
- Use ONLY these role names: old_state, new_state, preference_value, polarity, state_value, plan_goal, constraint, temporal_scope, reason_or_trigger, exception, context, location, decision, progress, evidence.
- Each accepted card must contain at least 2 distinct support facts.
- Create at most 3 cards.
- Only create cards relevant to the current question.
- needed_roles must be the minimal roles needed for this query, not all possible roles.
- If no useful relation card exists, return {"cards": []}.
- Keep summaries concise.
"""

ROLE_WHITELIST = {
    "old_state",
    "new_state",
    "preference_value",
    "polarity",
    "state_value",
    "plan_goal",
    "constraint",
    "temporal_scope",
    "reason_or_trigger",
    "exception",
    "context",
    "location",
    "decision",
    "progress",
    "evidence",
}
TYPE_WHITELIST = {"change", "preference", "state", "plan_constraint", "temporal"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument(
        "--base-paths",
        default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_notemporal_top150_D_true_bu_euhyp_td_euhyp_paths.json",
    )
    parser.add_argument("--base-topn", type=int, default=100)
    parser.add_argument("--context-topn", type=int, default=50)
    parser.add_argument("--max-cards", type=int, default=3)
    parser.add_argument("--model", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--base-url-env", default="VLLM_BASE_URL")
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache", default="outputs/v3_9_query_cards/qwen3_cards.jsonl")
    parser.add_argument("--output", default="outputs/v3_9_query_cards/qwen3_card_annotated_base100_paths.json")
    parser.add_argument("--summary-json", default="outputs/eval/V3_9_QUERY_CARD_SUMMARY.json")
    parser.add_argument("--summary-md", default="outputs/eval/V3_9_QUERY_CARD_SUMMARY.md")
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    items = read_json(resolve_path(args.base_paths))
    if args.limit:
        items = items[: args.limit]
    if args.dry_run:
        for item in items[:2]:
            print(user_prompt(graph, item, args))
        return

    cache_path = resolve_path(args.cache)
    cache = load_cache(cache_path) if args.resume or cache_path.exists() else {}
    disable_proxy_for_local_endpoint(args.base_url)
    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )

    records_by_qid = {}
    pending = []
    for item in items:
        qid = str(item["question_id"])
        key = cache_key(qid, args.model, args.context_topn, args.max_cards)
        if key in cache:
            records_by_qid[qid] = cache[key]
        else:
            pending.append((key, item))

    completed = len(records_by_qid)
    failures = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(call_cards, client, graph, item, args): (key, item) for key, item in pending}
        for future in as_completed(futures):
            key, item = futures[future]
            qid = str(item["question_id"])
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append({"question_id": qid, "error": f"{type(exc).__name__}: {exc}"})
                print(f"failed {qid}: {exc}", flush=True)
                continue
            append_cache(cache_path, key, record)
            records_by_qid[qid] = record
            completed += 1
            if completed % 20 == 0 or completed == len(items):
                cards = sum(len(row.get("cards", [])) for row in records_by_qid.values())
                print(f"cards {completed}/{len(items)} total_cards={cards}", flush=True)

    output_items = []
    summary_rows = []
    for item in items:
        qid = str(item["question_id"])
        record = records_by_qid.get(qid, {"cards": []})
        annotated, row = annotate_item(graph, item, record, args)
        output_items.append(annotated)
        summary_rows.append(row)

    output = resolve_path(args.output)
    write_json(output_items, output)
    summary = build_summary(args, output, summary_rows, failures)
    write_json(summary, resolve_path(args.summary_json))
    resolve_path(args.summary_md).write_text(render_summary(summary), encoding="utf-8")
    print(render_summary(summary))
    print(f"wrote {output}")


def call_cards(client: OpenAICompatibleChatClient, graph, item: dict, args) -> dict:
    started = time.perf_counter()
    result = client.chat_completion_with_metadata(
        model=args.model,
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt(graph, item, args)),
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        response_format={"type": "json_object"},
    )
    cards = validate_cards(parse_json_object(result.content), set(context_fact_ids(item, args)), args.max_cards)
    return {
        "question_id": item["question_id"],
        "model": args.model,
        "cards": cards,
        "raw_content": result.content,
        "usage": result.usage,
        "elapsed_seconds": result.elapsed_seconds or (time.perf_counter() - started),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def user_prompt(graph, item: dict, args) -> str:
    facts = []
    for rank, path in enumerate(non_completion_paths(item)[: args.context_topn], start=1):
        fact_id = evidence_node_id(path)
        node = graph.nodes.get(fact_id)
        if node is None:
            continue
        metadata = path.get("metadata", {})
        facts.append(
            {
                "rank": rank,
                "fact_id": fact_id,
                "text": truncate(node.text, 70),
                "time": node.time or metadata.get("time") or "",
                "event_id": metadata.get("event_node_id", ""),
                "episode_id": metadata.get("episode_node_id", ""),
                "topic_id": metadata.get("topic_node_id", ""),
                "route_source": metadata.get("route_source", metadata.get("candidate_source", "")),
                "ce_score": path.get("scores", {}).get("cross_encoder", path.get("score", 0.0)),
            }
        )
    payload = {
        "question_id": item.get("question_id"),
        "question": item.get("question", ""),
        "task": "Build query-conditioned relation cards using only the facts below.",
        "facts": facts,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def disable_proxy_for_local_endpoint(base_url: str) -> None:
    if "127.0.0.1" not in base_url and "localhost" not in base_url:
        return
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(name, None)
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    values = [item.strip() for item in existing.split(",") if item.strip()]
    for item in ("127.0.0.1", "localhost"):
        if item not in values:
            values.append(item)
    os.environ["NO_PROXY"] = ",".join(values)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


def validate_cards(payload: dict, valid_fact_ids: set[str], max_cards: int) -> list[dict]:
    cards = payload.get("cards") if isinstance(payload, dict) else []
    if not isinstance(cards, list):
        return []
    output = []
    for index, card in enumerate(cards[:max_cards]):
        if not isinstance(card, dict):
            continue
        relation_type = normalize_token(card.get("type"))
        if relation_type not in TYPE_WHITELIST:
            continue
        roles_in = card.get("roles") or {}
        roles = {}
        support_facts = set()
        if isinstance(roles_in, dict):
            for role_name, role_payload in roles_in.items():
                role = normalize_token(role_name)
                if not isinstance(role_payload, dict):
                    continue
                if role not in ROLE_WHITELIST:
                    role = "evidence"
                fact_ids = [str(fid) for fid in role_payload.get("fact_ids", []) if str(fid) in valid_fact_ids]
                value = str(role_payload.get("value", "")).strip()
                if fact_ids:
                    roles[role] = {"value": value, "fact_ids": dedupe(fact_ids), "confidence": safe_float(role_payload.get("confidence"), 1.0)}
                    support_facts.update(fact_ids)
        for fact_id in card.get("support_facts", []) or []:
            if str(fact_id) in valid_fact_ids:
                support_facts.add(str(fact_id))
        if len(support_facts) < 2:
            continue
        if not roles and support_facts:
            roles["evidence"] = {"value": "", "fact_ids": sorted(support_facts), "confidence": safe_float(card.get("confidence"), 0.8)}
        needed_roles = [normalize_token(role) for role in card.get("needed_roles", []) if normalize_token(role) in roles]
        if not needed_roles:
            needed_roles = list(roles)[:3]
        if not needed_roles:
            continue
        output.append(
            {
                "card_index": index,
                "type": relation_type,
                "entity": str(card.get("entity", "")).strip(),
                "aspect": str(card.get("aspect", "")).strip(),
                "summary": str(card.get("summary", "")).strip(),
                "why_relevant_to_query": str(card.get("why_relevant_to_query", "")).strip(),
                "needed_roles": needed_roles,
                "roles": roles,
                "support_facts": sorted(support_facts),
                "confidence": max(0.0, min(1.0, safe_float(card.get("confidence"), 0.8))),
            }
        )
    return output


def annotate_item(graph, item: dict, record: dict, args) -> tuple[dict, dict]:
    base_paths = non_completion_paths(item)[: args.base_topn]
    path_by_fact = {evidence_node_id(path): dict(path) for path in base_paths if evidence_node_id(path)}
    card_fact_payloads = build_card_fact_payloads(record.get("cards", []))
    for fact_id, payloads in card_fact_payloads.items():
        if fact_id not in path_by_fact:
            continue
        path = dict(path_by_fact[fact_id])
        metadata = dict(path.get("metadata", {}))
        best = max(payloads, key=lambda row: (row["card_confidence"], -row["card_index"]))
        metadata.update(best["metadata"])
        path["metadata"] = metadata
        scores = dict(path.get("scores", {}))
        scores["nary_card_confidence"] = best["card_confidence"]
        scores["nary_needed_role_score"] = best["needed_role_score"]
        path["scores"] = scores
        path_by_fact[fact_id] = path
    paths = list(path_by_fact.values())
    paths.sort(key=path_score, reverse=True)
    copied = dict(item)
    copied["paths"] = paths
    metadata = dict(copied.get("metadata", {}))
    metadata.update(
        {
            "method": "v3_9_query_conditioned_relation_cards",
            "base_topn": args.base_topn,
            "context_topn": args.context_topn,
            "query_card_count": len(record.get("cards", [])),
            "query_card_fact_count": len(card_fact_payloads),
        }
    )
    copied["metadata"] = metadata
    gold = gold_set(item)
    card_evidence = set()
    for fact_id in card_fact_payloads:
        card_evidence.update(fact_evidence(graph, fact_id))
    return copied, {
        "question_id": item["question_id"],
        "cards": len(record.get("cards", [])),
        "card_facts": len(card_fact_payloads),
        "card_gold_facts": len(card_evidence & gold),
        "card_hit": bool(card_evidence & gold),
        "card_full_cover": bool(gold) and gold.issubset(card_evidence),
        "usage": record.get("usage", {}),
        "elapsed_seconds": record.get("elapsed_seconds", 0.0),
    }


def build_card_fact_payloads(cards: list[dict]) -> dict[str, list[dict]]:
    output = defaultdict(list)
    for card_rank, card in enumerate(cards, start=1):
        card_id = f"query_card:{card_rank:02d}"
        needed_roles = set(card.get("needed_roles") or [])
        support_facts = card.get("support_facts") or []
        role_to_facts = []
        for role, payload in (card.get("roles") or {}).items():
            if needed_roles and role not in needed_roles:
                continue
            for fact_id in payload.get("fact_ids", []) or []:
                role_to_facts.append((str(fact_id), role, payload))
        if not role_to_facts:
            role_to_facts = [(str(fid), "evidence", {"confidence": card.get("confidence", 0.8)}) for fid in support_facts]
        hyperedge_size = len(set(support_facts))
        for completion_rank, (fact_id, role, role_payload) in enumerate(role_to_facts, start=1):
            metadata = {
                "v3_9_query_card": "true",
                "is_nary_completion": "true",
                "nary_hyperedge_id": card_id,
                "nary_hyperedge_type": card["type"],
                "nary_role": role,
                "nary_seed_fact_id": "",
                "nary_seed_fact_rank": "0",
                "nary_seed_fact_score": "0.000000",
                "nary_seed_route_origin": "query_conditioned_card",
                "nary_hyperedge_size": str(hyperedge_size),
                "nary_hyperedge_confidence": f"{card.get('confidence', 0.8):.6f}",
                "nary_role_confidence": f"{safe_float(role_payload.get('confidence'), card.get('confidence', 0.8)):.6f}",
                "nary_extractor_type": "qwen3_v3_9_query_card",
                "nary_completion_rank": str(completion_rank),
                "nary_same_hyperedge_count_in_candidate_pool": str(hyperedge_size),
                "nary_role_coverage_potential": str(len(card.get("roles") or {})),
                "nary_pool_covered_roles_count": str(len(card.get("roles") or {})),
                "nary_pool_required_roles_covered": "1",
                "nary_pool_has_preference_and_constraint": str(int("preference_value" in needed_roles and "constraint" in needed_roles)),
                "nary_pool_has_old_and_new_state": str(int("old_state" in needed_roles and "new_state" in needed_roles)),
                "nary_pool_has_reason": str(int("reason_or_trigger" in needed_roles)),
                "nary_pool_has_time_scope": str(int("temporal_scope" in needed_roles)),
                "v3_9_card_rank": str(card_rank),
                "v3_9_card_summary": card.get("summary", ""),
                "v3_9_card_entity": card.get("entity", ""),
                "v3_9_card_aspect": card.get("aspect", ""),
                "v3_9_why_relevant": card.get("why_relevant_to_query", ""),
            }
            output[fact_id].append(
                {
                    "metadata": metadata,
                    "card_confidence": float(card.get("confidence", 0.8)),
                    "needed_role_score": 1.0 if role in needed_roles else 0.5,
                    "card_index": card_rank,
                }
            )
    return dict(output)


def build_summary(args, output: Path, rows: list[dict], failures: list[dict]) -> dict:
    usage = defaultdict(float)
    for row in rows:
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] += value
    return {
        "output": str(output),
        "base_paths": str(resolve_path(args.base_paths)),
        "graph": str(resolve_path(args.graph)),
        "model": args.model,
        "questions": len(rows),
        "failures": failures,
        "card_questions": sum(row["cards"] > 0 for row in rows),
        "avg_cards": mean([row["cards"] for row in rows]) if rows else 0.0,
        "avg_card_facts": mean([row["card_facts"] for row in rows]) if rows else 0.0,
        "card_hit": sum(row["card_hit"] for row in rows) / max(len(rows), 1),
        "card_full_cover": sum(row["card_full_cover"] for row in rows) / max(len(rows), 1),
        "total_card_gold_facts": sum(row["card_gold_facts"] for row in rows),
        "usage": dict(usage),
        "avg_elapsed_seconds": mean([float(row.get("elapsed_seconds") or 0.0) for row in rows]) if rows else 0.0,
    }


def render_summary(summary: dict) -> str:
    return "\n".join(
        [
            "# V3.9 Query-conditioned Relation Cards",
            "",
            f"- Questions: {summary['questions']}",
            f"- Card questions: {summary['card_questions']}",
            f"- Avg cards/question: {summary['avg_cards']:.3f}",
            f"- Avg card facts/question: {summary['avg_card_facts']:.3f}",
            f"- Card Hit: {summary['card_hit']:.4f}",
            f"- Card FullCover: {summary['card_full_cover']:.4f}",
            f"- Total card gold facts: {summary['total_card_gold_facts']}",
            f"- Avg LLM seconds/question: {summary['avg_elapsed_seconds']:.2f}",
            "",
            "```json",
            json.dumps({k: summary[k] for k in ['output', 'model', 'usage']}, indent=2),
            "```",
            "",
        ]
    )


def context_fact_ids(item: dict, args) -> list[str]:
    return [evidence_node_id(path) for path in non_completion_paths(item)[: args.context_topn] if evidence_node_id(path)]


def non_completion_paths(item: dict) -> list[dict]:
    return [path for path in item.get("paths", []) if not is_completion(path)]


def is_completion(path: dict) -> bool:
    metadata = path.get("metadata", {})
    return str(metadata.get("is_nary_completion", "")).lower() == "true" or metadata.get("candidate_source") == "nary_completion"


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def path_score(path: dict) -> float:
    scores = path.get("scores", {})
    return safe_float(scores.get("cross_encoder"), safe_float(path.get("score"), 0.0))


def fact_evidence(graph, fact_id: str) -> set[str]:
    return {normalize_evidence_id(eid) for eid in evidence_ids_for_node(graph, fact_id)}


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}


def parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}


def load_cache(path: Path) -> dict:
    cache = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["cache_key"]] = row["record"]
    return cache


def append_cache(path: Path, key: str, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"cache_key": key, "record": record}, ensure_ascii=False) + "\n")


def cache_key(qid: str, model: str, context_topn: int, max_cards: int) -> str:
    return f"{model}|ctx{context_topn}|cards{max_cards}|{qid}"


def normalize_token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def truncate(text: str, words: int) -> str:
    parts = str(text).split()
    return " ".join(parts[:words])


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
