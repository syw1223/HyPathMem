#!/usr/bin/env python3
"""Use Qwen3-30B to verify solver inputs/operation without seeing gold answers."""

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


SYSTEM = """You verify whether a deterministic temporal solution may safely
override an existing answer. Return JSON only. You receive no gold answer.
Reject if any semantic condition in the question is omitted, if compared/listed
events are incomplete, if a relative expression uses the wrong anchor, if the
operator does not match the question, or if arithmetic/ordering is unsupported.
Before checking the supplied required_roles, independently decompose every
semantic condition in the question, including subordinate clauses. For example,
"the class when I made my friend's birthday cake" requires evidence connecting
both the class and that cake-making condition. A role list containing only
"class" is incomplete. Be conservative: uncertainty means
safe_to_override_d0=false."""


USER = """QUESTION: {question}
QUESTION_TIME: {question_time}
QUERY_TYPE: {query_type}
REQUIRED_ROLES: {required_roles}
AMBIGUITIES: {ambiguities}
NORMALIZED_OPERANDS_AND_SOURCE:
{operands}
SOLVER_OUTPUT:
{solution}

Return:
{{
  "question_constraints": ["each independently identified semantic condition"],
  "unsupported_constraints": [],
  "all_question_operands_present": true,
  "identities_match_question": true,
  "anchor_bindings_supported": true,
  "operation_matches_question": true,
  "arithmetic_or_ordering_correct": true,
  "answer_fully_entailed": true,
  "safe_to_override_d0": true,
  "reason": "short explanation"
}}
"""


def parse_object(text: str) -> dict[str, Any]:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        if start < 0:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(value[start:])
    if not isinstance(parsed, dict):
        raise ValueError("Verifier response is not a JSON object")
    return parsed


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
    max_attempts: int = 3,
) -> tuple[dict[str, Any], Any, int]:
    """Retry application-level JSON failures without weakening the verifier."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        result = client.chat_completion_with_metadata(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        try:
            return parse_object(result.content), result, attempt
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            if attempt < max_attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"Verifier returned invalid JSON after {max_attempts} attempts") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiled",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_temporal_v0_2_qwen/paired20_q1_q4_compiled.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_temporal_v0_2_qwen/paired20_q4_verified_gpu4.json"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8008/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("compiled", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    compiled = json.loads(args.compiled.read_text(encoding="utf-8"))
    rows = []
    if args.resume and args.output.exists():
        rows = json.loads(args.output.read_text(encoding="utf-8")).get("rows", [])
    elif args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --resume")
    done = {row["question_id"] for row in rows}
    client = OpenAICompatibleChatClient(api_key=args.api_key, base_url=args.base_url, timeout_seconds=180.0)
    started = time.perf_counter()
    for index, source in enumerate(compiled["rows"], start=1):
        if source["question_id"] in done:
            continue
        if not source["pre_verifier"]["eligible_to_call_solution_verifier"]:
            rows.append(
                {
                    "question_id": source["question_id"],
                    "verifier_called": False,
                    "safe_to_override_d0": False,
                    "reason": "pre_verifier_failed",
                }
            )
        else:
            eligible = [
                hypothesis
                for hypothesis in source["hypotheses"]
                if hypothesis["source_validation"].get("eligible_for_normalization")
                and hypothesis["solution"]["success"]
            ]
            verification, result, attempts = call_json(
                client,
                model=args.model,
                messages=[
                    ChatMessage(role="system", content=SYSTEM),
                    ChatMessage(
                        role="user",
                        content=USER.format(
                            question=source["question"],
                            question_time=source["question_date"],
                            query_type=source["query_type"],
                            required_roles=json.dumps(source["required_roles"], ensure_ascii=False),
                            ambiguities=json.dumps(source["ambiguities"], ensure_ascii=False),
                            operands=json.dumps(
                                [hypothesis["normalized_operands"] for hypothesis in eligible],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            solution=json.dumps(
                                [hypothesis["solution"] for hypothesis in eligible],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        ),
                    ),
                ],
            )
            required_checks = [
                "all_question_operands_present",
                "identities_match_question",
                "anchor_bindings_supported",
                "operation_matches_question",
                "arithmetic_or_ordering_correct",
                "answer_fully_entailed",
                "safe_to_override_d0",
            ]
            safe = (
                all(verification.get(key) is True for key in required_checks)
                and verification.get("unsupported_constraints") == []
            )
            rows.append(
                {
                    "question_id": source["question_id"],
                    "verifier_called": True,
                    "verification": verification,
                    "safe_to_override_d0": safe,
                    "candidate_answer": source["pre_verifier"]["candidate_answer"],
                    "usage": result.usage,
                    "elapsed_seconds": result.elapsed_seconds,
                    "attempts": attempts,
                }
            )
        output = {
            "metadata": {
                "version": "hypathmem_temporal_v0_2_q4_verifier",
                "model": args.model,
                "gold_evidence_or_answer_provided": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
            "summary": {
                "num_questions": len(rows),
                "verifier_called": sum(row["verifier_called"] for row in rows),
                "safe_to_override_d0": sum(row["safe_to_override_d0"] for row in rows),
            },
            "rows": rows,
        }
        write_json(args.output, output)
        print(
            f"processed {index}/{len(compiled['rows'])} qid={source['question_id']} "
            f"called={rows[-1]['verifier_called']} safe={rows[-1]['safe_to_override_d0']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
