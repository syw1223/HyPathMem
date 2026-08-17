from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import read_json, resolve_path, write_json
from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient


PROMPT_VERSION = "typed_nary_memory_hyperedge_v1"
SYSTEM_PROMPT = """You extract typed n-ary memory relations from conversation FACTS.

Judge whether the FACTS jointly express ONE coherent Change, Preference, or State relation.
Do not invent information. Every non-empty role must cite one or more provided fact_ids.

Return STRICT JSON only:
{
  "accept": true,
  "relation_type": "change|preference|state",
  "entity": "",
  "aspect": "",
  "roles": {
    "old_state": {"value": "", "fact_ids": []},
    "new_state": {"value": "", "fact_ids": []},
    "preference_value": {"value": "", "fact_ids": []},
    "polarity": {"value": "", "fact_ids": []},
    "state_value": {"value": "", "fact_ids": []},
    "temporal_scope": {"value": "", "fact_ids": []},
    "reason_or_constraint": {"value": "", "fact_ids": []},
    "exception": {"value": "", "fact_ids": []}
  },
  "confidence": 0.0,
  "reason": ""
}

Rules:
- relation_type must match the dominant relation actually supported by FACTS.
- Change requires an old/new distinction or an explicit update.
- Preference requires a like/dislike/prefer/want/constraint signal.
- State requires a concrete entity state in a time/context scope.
- Set accept=false when facts are only loosely related, conversational noise, redundant, or roles are unsupported.
- confidence is 0 to 1.
- Keep values concise.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/nary_v3_6/nary_hyperedge_candidates.json")
    parser.add_argument("--output", default="outputs/nary_v3_6/nary_hyperedge_annotations.json")
    parser.add_argument("--cache", default="outputs/nary_v3_6/nary_hyperedge_annotations.jsonl")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api-key-env", default="E_MEM_API_KEY")
    parser.add_argument("--base-url-env", default="E_MEM_BASE_URL")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
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
    records = []
    for index, candidate in enumerate(candidates, start=1):
        key = cache_key(candidate["candidate_id"], args.model)
        record = cache.get(key)
        if record is None:
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
            record = {
                "candidate_id": candidate["candidate_id"],
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "annotation": annotation,
                "usage": result.usage,
                "elapsed_seconds": result.elapsed_seconds,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            append_cache(cache_path, key, record)
            cache[key] = record
        records.append({"candidate": candidate, **record})
        if index % 25 == 0 or index == len(candidates):
            accepted = sum(bool(row["annotation"].get("accept")) for row in records)
            print(f"annotated {index}/{len(candidates)} accepted={accepted}", flush=True)

    payload = {
        "candidates": str(resolve_path(args.candidates)),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "count": len(records),
        "accepted": sum(bool(row["annotation"].get("accept")) for row in records),
        "records": records,
    }
    output = resolve_path(args.output)
    write_json(payload, output)
    print(f"wrote {output}")


def user_prompt(candidate: dict) -> str:
    payload = {
        "candidate_type": candidate["candidate_type"],
        "entity_hint": candidate.get("entity_hint", ""),
        "aspect_keywords": candidate.get("aspect_keywords", []),
        "facts": candidate["facts"],
    }
    return "CANDIDATE FACT GROUP:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def validate_annotation(annotation: dict, candidate: dict) -> dict:
    valid_ids = set(candidate["fact_ids"])
    relation_type = str(annotation.get("relation_type", "")).lower()
    if relation_type not in {"change", "preference", "state"}:
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
