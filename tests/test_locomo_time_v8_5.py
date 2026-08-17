from datetime import datetime

from hytopomem.locomo_time_v8_5 import (
    normalize_locomo_binding,
    parse_locomo_datetime,
    solve_locomo_hypothesis,
)


def fake_base_normalize(binding: dict) -> dict:
    return {
        **binding,
        "occurred_at": None,
        "precision": "unknown",
        "normalization_status": "unresolved",
        "normalization_trace": [],
    }


def test_parse_locomo_message_time() -> None:
    assert parse_locomo_datetime("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56)


def test_last_named_weekday_is_strictly_previous() -> None:
    result = normalize_locomo_binding(
        {"mentioned_at": "1:14 pm on 25 May, 2023", "time_expression": "last Saturday"},
        fake_base_normalize,
    )
    assert result["occurred_at"].startswith("2023-05-20")
    assert result["precision"] == "day"


def test_direct_date_solver_formats_day() -> None:
    solution = solve_locomo_hypothesis(
        "When did Melanie run the race?",
        "date",
        [{"occurred_at": "2023-05-20T13:14:00", "precision": "day", "normalization_status": "resolved"}],
        "2023-09-01 00:00",
        lambda *_: {},
    )
    assert solution["success"]
    assert solution["answer"] == "20 May 2023"


def test_direct_date_solver_rejects_approximate_recent() -> None:
    solution = solve_locomo_hypothesis(
        "When did it happen?",
        "date",
        [{"occurred_at": "2023-05-20T13:14:00", "precision": "approximate_recent", "normalization_status": "resolved"}],
        "2023-09-01 00:00",
        lambda *_: {},
    )
    assert not solution["success"]


def test_explicit_duration_precedes_conversation_end_arithmetic() -> None:
    solution = solve_locomo_hypothesis(
        "How long has Caroline had her current group of friends for?",
        "elapsed",
        [
            {
                "occurred_at": "2019-06-09T19:55:00",
                "duration_value": 4,
                "duration_unit": "year",
                "precision": "year",
                "normalization_status": "resolved",
            }
        ],
        "2023-10-01 00:00",
        lambda *_: {"success": True, "answer": "4.33 years"},
    )
    assert solution["answer"] == "4 years"
