from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "temporal_oracle_compiler", ROOT / "scripts" / "135_compile_hypathmem_temporal_oracle_o1_o3.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ordering_solver() -> None:
    answer, verified, _ = MODULE.solve(
        {
            "status": "verified",
            "operation": "ORDERING",
            "operands": [
                {"label": "later", "time": "2023-05-02"},
                {"label": "earlier", "time": "2023-05-01"},
            ],
        }
    )
    assert verified
    assert answer == "earlier, then later"


def test_unresolved_fails_closed() -> None:
    answer, verified, trace = MODULE.solve(
        {
            "status": "oracle_unresolvable",
            "operation": "ELAPSED",
            "notes": "inconsistent evidence",
        }
    )
    assert answer is None
    assert not verified
    assert trace == "inconsistent evidence"


def test_duration_at_event_solver() -> None:
    answer, verified, _ = MODULE.solve(
        {
            "status": "verified",
            "operation": "DURATION_AT_EVENT",
            "unit": "weeks",
            "operands": [{"duration": 6}, {"offset": 2}],
        }
    )
    assert verified
    assert answer == "4 weeks"
