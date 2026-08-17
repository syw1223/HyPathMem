from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import read_json, resolve_path, write_json
from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient


PROMPT_VERSION = "typed_nary_memory_hyperedge_v2_high_recall"
SYSTEM_PROMPT = """You extract typed n-ary memory relations from conversation FACTS.

Judge whether the FACTS jointly express ONE coherent Change, Preference, State, or PlanConstraint relation.
Do not invent information. Every non-empty role must cite one or more provided fact_ids.

Return STRICT JSON only:
{
  "accept": true,
  "relation_type": "change|preference|state|plan_constraint",
  "entity": "",
  "aspect": "",
  "roles": {
    "old_state": {"value": "", "fact_ids": []},
    "new_state": {"value": "", "fact_ids": []},
    "preference_value": {"value": "", "fact_ids": []},
    "polarity": {"value": "", "fact_ids": []},
    "state_value": {"value": "", "fact_ids": []},
    "plan_goal": {"value": "", "fact_ids": []},
    "constraint": {"value": "", "fact_ids": []},
    "temporal_scope": {"value": "", "fact_ids": []},
    "reason_or_trigger": {"value": "", "fact_ids": []},
    "exception": {"value": "", "fact_ids": []},
    "context": {"value": "", "fact_ids": []}
  },
  "confidence": 0.0,
  "reason": ""
}

Rules:
- Change requires a supported old/new distinction or explicit update.
- Preference requires a supported like/dislike/prefer/want signal and polarity.
- State requires a concrete entity state.
- PlanConstraint requires a supported plan/goal plus a constraint, schedule, temporal scope, or condition.
- Facts must form one semantically closed relation, not merely share an Event, Episode, or Entity.
- Set accept=false for loose groups, duplicates, mixed subjects, conversational noise, or unsupported roles.
- confidence is 0 to 1.
- Keep role values concise.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/nary_v3_6b/high_recall_candidates.json")
    parser.add_argument("--output", default="outputs/nary_v3_6b/high_recall_annotations.json")
    parser.add_argument("--cache", default="outputs/nary_v3_6b/high_recall_annotations.jsonl")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api-key-env", default="E_MEM_API_KEY")
    parser.add_argument("--base-url-env", default="E_MEM_BASE_URL")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=420)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidate_payload = read_json(resolve_path(args.candidates))
    candidates = [
        row
        for row in candidate_payload["candidates"]
        if float(row.get("candidate_score", 0.0)) >= args.min_candidate_score
    ]
    if args.limit:
        candidates = candidates[: args.limit]
    if args.dry_run:
        for row in candidates[:3]:
            print(user_prompt(row))
        return

    cache_path = resolve_path(args.cache)
    cache = load_cache(cache_path) if args.resume or cache_path.exists() else {}
    client = OpenAICompatibleChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        default_base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    records_by_id = {}
    pending = []
    for candidate in candidates:
        key = cache_key(candidate["candidate_id"], args.model)
        if key in cache:
            records_by_id[candidate["candidate_id"]] = {"candidate": candidate, **cache[key]}
        else:
            pending.append((key, candidate))

    completed = len(records_by_id)
    attempted = completed
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(call_annotation, client, candidate, args): (key, candidate)
            for key, candidate in pending
        }
        for future in as_completed(futures):
            key, candidate = futures[future]
            attempted += 1
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"failed attempted={attempted}/{len(candidates)} "
                    f"cached={completed} candidate={candidate['candidate_id']} error={exc}",
                    flush=True,
                )
                continue
            append_cache(cache_path, key, record)
            records_by_id[candidate["candidate_id"]] = {"candidate": candidate, **record}
            completed += 1
            if completed % 5 == 0 or completed == len(candidates):
                accepted = sum(bool(row["annotation"].get("accept")) for row in records_by_id.values())
                total_tokens = sum(
                    int((row.get("usage") or {}).get("total_tokens", 0))
                    for row in records_by_id.values()
                )
                print(
                    f"annotated {completed}/{len(candidates)} accepted={accepted} "
                    f"last={record.get('elapsed_seconds', 0.0):.1f}s tokens={total_tokens}",
                    flush=True,
                )

    records = [
        records_by_id[candidate["candidate_id"]]
        for candidate in candidates
        if candidate["candidate_id"] in records_by_id
    ]
    usage = {}
    for record in records:
        for key, value in (record.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    payload = {
        "candidates": str(resolve_path(args.candidates)),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "count": len(records),
        "target_count": len(candidates),
        "failed_count": len(failures),
        "failures": failures,
        "accepted": sum(bool(row["annotation"].get("accept")) for row in records),
        "usage": usage,
        "records": records,
    }
    output = resolve_path(args.output)
    write_json(payload, output)
    print(json.dumps({"count": payload["count"], "accepted": payload["accepted"], "usage": usage}, indent=2))
    print(f"wrote {output}")


def call_annotation(client: OpenAICompatibleChatClient, candidate: dict, args) -> dict:
    result = client.chat_completion_with_metadata(
        model=args.model,
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt(candidate)),
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        response_format={"type": "json_object"},
    )
    annotation = validate_annotation(parse_json_object(result.content), candidate)
    return {
        "candidate_id": candidate["candidate_id"],
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "annotation": annotation,
        "usage": result.usage,
        "elapsed_seconds": result.elapsed_seconds,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def user_prompt(candidate: dict) -> str:
    facts = [
        {
            "fact_id": fact["fact_id"],
            "text": fact["text"],
            "time": fact.get("time", ""),
            "status": fact.get("status", ""),
        }
        for fact in candidate["facts"]
    ]
    payload = {
        "candidate_source": candidate["candidate_source"],
        "relation_hints": candidate.get("relation_hints", []),
        "entity_hints": candidate.get("entity_hints", []),
        "aspect_keywords": candidate.get("aspect_keywords", []),
        "facts": facts,
    }
    return "CANDIDATE FACT GROUP:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def validate_annotation(annotation: dict, candidate: dict) -> dict:
    valid_ids = set(candidate["fact_ids"])
    relation_type = str(annotation.get("relation_type", "")).lower()
    if relation_type not in {"change", "preference", "state", "plan_constraint"}:
        annotation["accept"] = False
    roles = annotation.get("roles")
    if not isinstance(roles, dict):
        roles = {}
    for role_name, role in list(roles.items()):
        if not isinstance(role, dict):
            roles[role_name] = {"value": "", "fact_ids": []}
            continue
        role["fact_ids"] = [str(fact_id) for fact_id in role.get("fact_ids", []) if str(fact_id) in valid_ids]
        role["value"] = str(role.get("value", "")).strip()
    annotation["roles"] = roles
    annotation["entity"] = str(annotation.get("entity", "")).strip()
    annotation["aspect"] = str(annotation.get("aspect", "")).strip()
    annotation["confidence"] = max(0.0, min(1.0, safe_float(annotation.get("confidence"))))
    annotation["accept"] = bool(annotation.get("accept")) and role_complete(relation_type, roles)
    return annotation


def role_complete(relation_type: str, roles: dict) -> bool:
    def filled(name: str) -> bool:
        role = roles.get(name) or {}
        return bool(role.get("value")) and bool(role.get("fact_ids"))

    if relation_type == "change":
        return filled("old_state") and filled("new_state")
    if relation_type == "preference":
        return filled("preference_value") and filled("polarity")
    if relation_type == "state":
        return filled("state_value")
    if relation_type == "plan_constraint":
        return filled("plan_goal") and (filled("constraint") or filled("temporal_scope"))
    return False


def parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise TypeError("expected JSON object")
    return payload


def cache_key(candidate_id: str, model: str) -> str:
    return f"{candidate_id}|{model}|{PROMPT_VERSION}"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    output = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output[str(row["cache_key"])] = dict(row["record"])
    return output


def append_cache(path: Path, key: str, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"cache_key": key, "record": record}, ensure_ascii=False) + "\n")


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
