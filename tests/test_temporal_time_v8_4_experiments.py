from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "143_run_hypathmem_8_4time_v_e1_e2.py"
SPEC = importlib.util.spec_from_file_location("time_v84_experiments", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(*, full: bool, safe: bool, success: bool = False, occurred: bool = False) -> dict:
    return {
        "question_id": "q1",
        "question": "How long between A and B?",
        "question_date": "2025-01-10T00:00:00",
        "audit": {
            "time_v_operand_full_coverage": full,
            "time_v_eligible_for_q4_verifier": safe,
        },
        "h1_temporal_sidecar": {"nodes": []},
        "h2_h4_operand_closure": {
            "plan": {"query_type": "elapsed", "required_roles": ["A", "B"], "requires_pairwise_operands": True},
            "role_candidates": {},
            "temporal_consistency": {"consistent": True},
        },
        "h5_constraint_candidate": {
            "query_type": "elapsed",
            "required_roles": ["A", "B"],
            "operands": [
                {"role": "A", "occurred_at": "2025-01-01" if occurred else None},
                {"role": "B", "occurred_at": "2025-01-02" if occurred else None},
            ],
            "solution": {"success": success, "answer": "1 day" if success else None},
            "constraint_subgraph": {"edges": []},
        },
    }


def test_e1_selects_only_new_safe_candidates() -> None:
    old = row(full=True, safe=True, success=True, occurred=True)
    old["question_id"] = "old"
    new = row(full=True, safe=True, success=True, occurred=True)
    new["question_id"] = "new"
    assert [item["question_id"] for item in MODULE.select_e1({"rows": [old, new]}, {"rows": [old]})] == ["new"]


def test_e2_is_exactly_full_cover_non_executable_slice() -> None:
    full_unsafe = row(full=True, safe=False)
    missing = row(full=False, safe=False)
    safe = row(full=True, safe=True, success=True, occurred=True)
    assert MODULE.select_e2({"rows": [full_unsafe, missing, safe]}) == [full_unsafe]


def test_failure_reason_and_comparison_are_deterministic() -> None:
    item = row(full=True, safe=False, occurred=False)
    assert MODULE.failure_reason(item) == "missing_occurred_at"
    assert MODULE.classification(False, True) == "fix"
    assert MODULE.classification(True, False) == "break"
    assert MODULE.classification(True, True) == "unchanged_correct"
    assert MODULE.classification(False, False) == "unchanged_wrong"


def test_refusal_detection_is_explicit() -> None:
    assert MODULE.is_refusal("Insufficient evidence.")
    assert not MODULE.is_refusal("7 days")
