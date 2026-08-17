from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "temporal_q4_runner", ROOT / "scripts" / "140_run_hypathmem_temporal_q1_q4_qa_v0_2.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_q4_uses_current_solver_answer_not_stale_verifier_answer() -> None:
    source = {
        "hypotheses": [],
        "pre_verifier": {
            "eligible_to_call_solution_verifier": True,
            "candidate_answer": "Page Turners was first",
        },
    }
    decision = MODULE.branch(
        source,
        "q4",
        {"safe_to_override_d0": True, "candidate_answer": "Page Turners"},
    )
    assert decision["prediction"] == "Page Turners was first"
