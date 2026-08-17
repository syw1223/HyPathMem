from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "temporal_qwen_joint", ROOT / "scripts" / "137_extract_hypathmem_temporal_qwen_joint_v0_2.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_fenced_json() -> None:
    assert MODULE.parse_json_object('```json\n{"query_type":"ordering"}\n```')["query_type"] == "ordering"


def test_deterministic_validation_requires_exact_span_and_anchor() -> None:
    extraction = {
        "required_roles": ["purchase"],
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "bindings": [
                    {
                        "role": "purchase",
                        "fact_id": "f1",
                        "raw_id": "r1",
                        "evidence_span": "bought it three weeks ago",
                        "time_expression": "three weeks ago",
                        "anchor_type": "mentioned_at",
                        "anchor_id": "r1",
                    }
                ],
            }
        ],
    }
    candidates = {
        "f1": {
            "raw_quotes": [
                {"raw_id": "r1", "text": "I bought it three weeks ago.", "message_time": "2024-05-21"}
            ]
        }
    }
    audit = MODULE.deterministic_validate(extraction, candidates)
    assert audit["hypotheses"][0]["all_deterministic_checks_pass"]


def test_relative_expression_without_anchor_fails() -> None:
    extraction = {
        "required_roles": ["event"],
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "bindings": [
                    {
                        "role": "event",
                        "fact_id": "f1",
                        "raw_id": "r1",
                        "evidence_span": "three weeks ago",
                        "time_expression": "three weeks ago",
                        "anchor_type": "none",
                        "anchor_id": "",
                    }
                ],
            }
        ],
    }
    candidates = {"f1": {"raw_quotes": [{"raw_id": "r1", "text": "three weeks ago"}]}}
    audit = MODULE.deterministic_validate(extraction, candidates)
    assert not audit["hypotheses"][0]["all_deterministic_checks_pass"]
