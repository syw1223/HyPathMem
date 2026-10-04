from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "temporal_qwen_compiler", ROOT / "scripts" / "138_compile_hypathmem_temporal_q1_q4_v0_2.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def normalized(expression: str) -> dict:
    return MODULE.normalize_binding(
        {
            "role": "event",
            "mentioned_at": "2023/05/20 (Sat) 20:08",
            "time_expression": expression,
        }
    )


def test_three_weeks_ago() -> None:
    assert normalized("exactly three weeks ago")["occurred_at"].startswith("2023-04-29")


def test_calendar_month_shift() -> None:
    assert normalized("about a month ago")["occurred_at"].startswith("2023-04-20")


def test_absolute_month_day() -> None:
    assert normalized("on the 15th of March")["occurred_at"].startswith("2023-03-15")


def test_duration_solver() -> None:
    operands = [normalized("six weeks now"), normalized("two weeks ago")]
    solution = MODULE.solve_hypothesis(
        "How long had I been taking lessons when I bought an amp?",
        "duration",
        operands,
        "2023/05/25 (Thu) 19:07",
    )
    assert solution["answer"] == "4 weeks"


def test_restrictive_when_clause_must_be_grounded() -> None:
    grounded, audit = MODULE.restrictive_clause_grounded(
        "How many days ago was the class when I made my friend's birthday cake?",
        [{"identity": "baking class", "evidence_span": "I took a baking class yesterday."}],
    )
    assert not grounded
    assert audit["coverage"] == 0.0


def test_restrictive_when_clause_accepts_supported_amp_event() -> None:
    grounded, _ = MODULE.restrictive_clause_grounded(
        "How long had I taken lessons when I bought the new guitar amp?",
        [{"identity": "new guitar amp purchase", "evidence_span": "I just got a new amp two weeks ago."}],
    )
    assert grounded


def test_first_answer_states_relation_explicitly() -> None:
    operands = [
        {"identity": "Page Turners", "occurred_at": "2023-05-18T00:00:00"},
        {"identity": "Marketing Professionals", "occurred_at": "2023-05-24T00:00:00"},
    ]
    solution = MODULE.solve_hypothesis(
        "Which group did I join first, Page Turners or Marketing Professionals?",
        "ordering",
        operands,
        "2023/05/25 (Thu) 08:24",
    )
    assert solution["answer"] == "Page Turners was first"
