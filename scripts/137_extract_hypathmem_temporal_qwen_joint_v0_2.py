#!/usr/bin/env python3
"""Query-conditioned joint event/time/anchor extraction with Qwen3-30B.

The extractor never receives LongMemEval gold evidence or gold answers. It sees
only the frozen D2 Top50 candidates and their local raw provenance. Results are
saved as a new v0.2 artifact and cannot override D0 until validation succeeds.
"""

from __future__ import annotations

import argparse
import json
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


SYSTEM_PROMPT = """You bind temporal question operands to retrieved evidence.
Return one JSON object only. Do not answer the question and do not invent dates.
Jointly resolve event identity, time expression, and its anchor. Preserve 2-3
plausible hypotheses when ambiguity is real. Every binding must cite an exact
fact_id, raw_id, and verbatim evidence_span from the supplied candidates.
Relative time expressions must cite a verbatim time_expression and identify an
anchor_type: mentioned_at, question_time, or another_event. If uncertain, put
the issue in ambiguities and lower confidence instead of guessing.

Required schema rules:
1. required_roles enumerates every distinct event/entity operand, never an
   operation name such as event_order.
2. Every binding.role must exactly equal one entry in required_roles.
3. time_expression must be copied verbatim from the cited raw text. Use an
   empty string when there is no explicit expression; never synthesize words
   such as "today" from "just" or from a timestamp.
4. A hypothesis is complete only when every required role has one binding."""

# Appended separately to keep the core instructions readable to the model.
SYSTEM_PROMPT += """
5. Comparison operands must be separate roles. For example, "bus or train"
   requires roles "bus ride" and "train ride"; "Tom or Alex" requires roles
   "Tom became a parent" and "Alex became a parent".
6. If a required operand is absent, keep it in required_roles and list it in
   hypothesis.missing_roles. Never bind the absent role to another entity.
7. For an open-ended ordered list, create one unique role per qualifying event
   found in the supplied candidates. Do not reuse one generic role for several
   bindings."""


USER_TEMPLATE = """QUESTION: {question}
QUESTION_TIME: {question_time}

Return this schema:
{{
  "query_type": "ordering|recency|elapsed|duration|date|attribute_at_time|state|other",
  "required_roles": ["short semantic role required by the question"],
  "hypotheses": [
    {{
      "hypothesis_id": "H1",
      "bindings": [
        {{
          "role": "one required role",
          "fact_id": "exact supplied fact_id",
          "raw_id": "exact supplied raw_id",
          "identity": "event/entity identity in this quote",
          "evidence_span": "exact verbatim span",
          "mentioned_at": "supplied message_time",
          "time_expression": "exact verbatim expression or empty",
          "anchor_type": "mentioned_at|question_time|another_event|none",
          "anchor_id": "raw_id, role, or empty"
        }}
      ],
      "missing_roles": [],
      "confidence": 0.0
    }}
  ],
  "ambiguities": []
}}

CANDIDATES:
{candidates}
"""


VERIFY_SYSTEM = """You are a strict semantic binding verifier. Return JSON only.
Check whether each proposed binding really denotes the event/entity role asked
by the question. Do not reward keyword overlap. Reject a binding that refers to
a similar but different event, state, person, object, or occurrence. This step
does not answer the question and receives no gold answer."""


VERIFY_TEMPLATE = """QUESTION: {question}
PROPOSED_EXTRACTION:
{extraction}

For every hypothesis return:
{{
  "hypotheses": [
    {{
      "hypothesis_id": "H1",
      "role_checks": [
        {{"role": "...", "identity_verified": true, "reason": "..."}}
      ],
      "all_required_identities_verified": true
    }}
  ]
}}
"""


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise ValueError("Response JSON is not an object")
    return value


def render_candidates(pack: dict[str, Any], max_chars: int) -> tuple[str, dict[str, dict[str, Any]]]:
    rendered: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    used = 2
    for rank, unit in enumerate(pack.get("evidence_units", []), start=1):
        raw_quotes = []
        for quote in unit.get("raw_quotes", []):
            raw = {
                "raw_id": quote.get("message_id"),
                "message_time": quote.get("message_time"),
                "speaker": quote.get("speaker"),
                "text": quote.get("text"),
            }
            raw_quotes.append(raw)
        item = {
            "rank": rank,
            "fact_id": unit.get("unit_id"),
            "claim": unit.get("normalized_claim"),
            "entity": unit.get("entity"),
            "aspect": unit.get("aspect"),
            "raw_quotes": raw_quotes,
        }
        chunk = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if rendered and used + len(chunk) + 1 > max_chars:
            break
        rendered.append(item)
        used += len(chunk) + 1
        lookup[str(item["fact_id"])] = item
    return json.dumps(rendered, ensure_ascii=False, separators=(",", ":")), lookup


def deterministic_validate(
    extraction: dict[str, Any], candidate_lookup: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    required_roles = {str(role) for role in extraction.get("required_roles", [])}
    audits = []
    for hypothesis in extraction.get("hypotheses", []):
        bindings = hypothesis.get("bindings", [])
        binding_audits = []
        bound_roles = set()
        for binding in bindings:
            role = str(binding.get("role", ""))
            fact_id = str(binding.get("fact_id", ""))
            raw_id = str(binding.get("raw_id", ""))
            span = str(binding.get("evidence_span", ""))
            time_expression = str(binding.get("time_expression", ""))
            anchor_type = str(binding.get("anchor_type", "none"))
            anchor_id = str(binding.get("anchor_id", ""))
            fact = candidate_lookup.get(fact_id)
            raw = None
            if fact:
                raw = next(
                    (quote for quote in fact.get("raw_quotes", []) if str(quote.get("raw_id")) == raw_id),
                    None,
                )
            raw_text = str(raw.get("text", "")) if raw else ""
            fact_valid = fact is not None
            raw_valid = raw is not None
            span_valid = bool(span) and span in raw_text
            expression_valid = not time_expression or time_expression in raw_text
            relative = bool(
                re.search(
                    r"\b(?:ago|yesterday|today|just|last\s+\w+|before|after|earlier|later|past\s+\w+)\b",
                    time_expression,
                    flags=re.I,
                )
            )
            anchor_complete = not relative or (
                anchor_type in {"mentioned_at", "question_time", "another_event"} and bool(anchor_id)
            )
            role_declared = role in required_roles
            if role:
                bound_roles.add(role)
            binding_audits.append(
                {
                    "role": role,
                    "fact_valid": fact_valid,
                    "raw_valid": raw_valid,
                    "span_grounded": span_valid,
                    "time_expression_grounded": expression_valid,
                    "anchor_complete": anchor_complete,
                    "role_declared": role_declared,
                    "deterministic_valid": all(
                        [fact_valid, raw_valid, span_valid, expression_valid, anchor_complete, role_declared]
                    ),
                }
            )
        model_missing = {str(role) for role in hypothesis.get("missing_roles", [])}
        missing = sorted((required_roles - bound_roles) | model_missing)
        binding_roles = [str(binding.get("role", "")) for binding in bindings]
        duplicate_roles = sorted({role for role in binding_roles if binding_roles.count(role) > 1})
        audits.append(
            {
                "hypothesis_id": hypothesis.get("hypothesis_id"),
                "binding_audits": binding_audits,
                "missing_required_roles": missing,
                "duplicate_binding_roles": duplicate_roles,
                "all_deterministic_checks_pass": bool(binding_audits)
                and not missing
                and not duplicate_roles
                and all(audit["deterministic_valid"] for audit in binding_audits),
            }
        )
    return {"required_roles": sorted(required_roles), "hypotheses": audits}


def merge_verification(
    deterministic: dict[str, Any], semantic: dict[str, Any]
) -> dict[str, Any]:
    semantic_by_id = {
        str(item.get("hypothesis_id")): item for item in semantic.get("hypotheses", [])
    }
    merged = []
    for audit in deterministic["hypotheses"]:
        semantic_audit = semantic_by_id.get(str(audit.get("hypothesis_id")), {})
        identity_verified = bool(semantic_audit.get("all_required_identities_verified", False))
        item = dict(audit)
        item["semantic_identity"] = semantic_audit
        item["identity_verified"] = identity_verified
        item["eligible_for_normalization"] = bool(audit["all_deterministic_checks_pass"] and identity_verified)
        merged.append(item)
    return {"required_roles": deterministic["required_roles"], "hypotheses": merged}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def call_json(
    client: OpenAICompatibleChatClient,
    *,
    model: str,
    messages: list[ChatMessage],
    max_tokens: int,
    attempts: int = 3,
) -> tuple[dict[str, Any], Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = client.chat_completion_with_metadata(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return parse_json_object(result.content), result
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"JSON application retries exhausted: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_r_v0_1/top50_structured_raw_packs.json"),
    )
    parser.add_argument(
        "--temporal",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_temporal_v0_1/frozen_d2_top50_t1_t4.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_r_v0_1/paired120_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_temporal_v0_2_qwen/paired20_joint_extraction.json"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8008/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--max-candidate-chars", type=int, default=60000)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=["paired", "remaining-temporal", "all-temporal"],
        default="paired",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("packs", "temporal", "manifest", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)

    packs = json.loads(args.packs.read_text(encoding="utf-8"))
    temporal = json.loads(args.temporal.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pack_by_id = {row["question_id"]: row for row in packs["rows"]}
    paired_ids = {str(row["question_id"]) for row in manifest["rows"]}
    if args.selection == "paired":
        selected = [row for row in temporal["rows"] if str(row["question_id"]) in paired_ids]
    elif args.selection == "remaining-temporal":
        selected = [row for row in temporal["rows"] if str(row["question_id"]) not in paired_ids]
    else:
        selected = list(temporal["rows"])
    if args.limit:
        selected = selected[: args.limit]
    if args.dry_run:
        row = selected[0]
        candidates, _ = render_candidates(pack_by_id[row["question_id"]]["pack"], args.max_candidate_chars)
        print(USER_TEMPLATE.format(question=row["question"], question_time=row["question_date"], candidates=candidates))
        return

    rows = []
    if args.resume and args.output.exists():
        rows = json.loads(args.output.read_text(encoding="utf-8")).get("rows", [])
    elif args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --resume")
    done = {row["question_id"] for row in rows}
    client = OpenAICompatibleChatClient(api_key=args.api_key, base_url=args.base_url, timeout_seconds=180.0)
    started = time.perf_counter()
    for index, row in enumerate(selected, start=1):
        question_id = row["question_id"]
        if question_id in done:
            continue
        candidates, lookup = render_candidates(pack_by_id[question_id]["pack"], args.max_candidate_chars)
        extraction, result = call_json(
            client,
            model=args.model,
            messages=[
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=USER_TEMPLATE.format(
                        question=row["question"], question_time=row["question_date"], candidates=candidates
                    ),
                ),
            ],
            max_tokens=args.max_output_tokens,
        )
        deterministic = deterministic_validate(extraction, lookup)
        semantic, semantic_result = call_json(
            client,
            model=args.model,
            messages=[
                ChatMessage(role="system", content=VERIFY_SYSTEM),
                ChatMessage(
                    role="user",
                    content=VERIFY_TEMPLATE.format(
                        question=row["question"],
                        extraction=json.dumps(extraction, ensure_ascii=False, separators=(",", ":")),
                    ),
                ),
            ],
            max_tokens=1536,
        )
        validation = merge_verification(deterministic, semantic)
        rows.append(
            {
                "question_id": question_id,
                "question": row["question"],
                "question_date": row["question_date"],
                "candidate_chars": len(candidates),
                "candidate_fact_count": len(lookup),
                "extraction": extraction,
                "validation": validation,
                "raw_extraction_response": result.content,
                "raw_semantic_verifier_response": semantic_result.content,
                "usage": {"extraction": result.usage, "semantic_verifier": semantic_result.usage},
                "elapsed_seconds": {
                    "extraction": result.elapsed_seconds,
                    "semantic_verifier": semantic_result.elapsed_seconds,
                },
            }
        )
        eligible = sum(item["eligible_for_normalization"] for item in validation["hypotheses"])
        output = {
            "metadata": {
                "version": "hypathmem_temporal_v0_2_qwen_joint",
                "model": args.model,
                "base_url": args.base_url,
                "gold_evidence_or_answer_provided": False,
                "max_candidate_chars": args.max_candidate_chars,
                "selection": args.selection,
                "elapsed_seconds": time.perf_counter() - started,
            },
            "summary": {
                "num_questions": len(rows),
                "questions_with_eligible_hypothesis": sum(
                    any(h["eligible_for_normalization"] for h in saved["validation"]["hypotheses"])
                    for saved in rows
                ),
            },
            "rows": rows,
        }
        write_json(args.output, output)
        print(
            f"processed {index}/{len(selected)} qid={question_id} "
            f"hypotheses={len(extraction.get('hypotheses', []))} eligible={eligible}",
            flush=True,
        )


if __name__ == "__main__":
    main()
