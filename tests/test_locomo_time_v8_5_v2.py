from hytopomem.locomo_time_v8_5_v2 import (
    ambiguity_safe,
    infer_answer_type,
    normalize_locomo_binding_v2,
    solve_locomo_hypothesis_v2,
    typed_solution_safe,
)


def fake_base_normalize(binding: dict) -> dict:
    expression = str(binding.get("time_expression") or "").lower()
    if "around 3 years ago" in expression:
        occurred_at, precision, status = "2019-01-21T19:31:00", "approximate", "resolved"
    elif "last week" in expression:
        occurred_at, precision, status = "2023-04-26T17:41:00", "week", "resolved"
    elif "last summer" in expression:
        occurred_at, precision, status = "2022-07-01T00:00:00", "season", "resolved"
    else:
        occurred_at, precision, status = None, "unknown", "unresolved"
    return {
        **binding,
        "occurred_at": occurred_at,
        "precision": precision,
        "normalization_status": status,
        "normalization_trace": [],
    }


def binding(expression: str, mentioned: str = "5:13 pm on 9 July, 2022") -> dict:
    return {
        "raw_id": "raw:1",
        "anchor_id": "raw:1",
        "anchor_type": "mentioned_at",
        "mentioned_at": mentioned,
        "time_expression": expression,
    }


def test_compound_relative_days_are_not_reduced_to_substrings() -> None:
    after = normalize_locomo_binding_v2(binding("the day after tomorrow evening"), fake_base_normalize)
    before = normalize_locomo_binding_v2(
        binding("the day before yesterday", "9:17 am on 26 June, 2023"), fake_base_normalize
    )
    assert after["occurred_at"].startswith("2022-07-11")
    assert before["occurred_at"].startswith("2023-06-24")


def test_named_weekday_fails_closed() -> None:
    result = normalize_locomo_binding_v2(
        binding("last Sunday", "1:59 pm on 31 July, 2023"), fake_base_normalize
    )
    assert result["normalization_status"] == "unresolved"
    assert result["precision"] == "weekday_relative_ambiguous"


def test_when_overrides_recency_event_identity_with_temporal_answer() -> None:
    operand = normalize_locomo_binding_v2(
        binding("around 3 years ago", "7:31 pm on 21 January, 2022"), fake_base_normalize
    )
    solution = solve_locomo_hypothesis_v2(
        "When did Joanna first watch the movie?", "recency", [operand], "", lambda *_: {}
    )
    assert solution["answer"] == "around 2019"
    assert solution["answer_type"] == "YEAR"


def test_week_and_season_preserve_granularity() -> None:
    week = normalize_locomo_binding_v2(
        binding("last week", "5:41 pm on 3 May, 2023"), fake_base_normalize
    )
    summer = normalize_locomo_binding_v2(
        binding("last summer", "4:12 pm on 22 February, 2023"), fake_base_normalize
    )
    week_solution = solve_locomo_hypothesis_v2("When did it happen?", "date", [week], "", lambda *_: {})
    summer_solution = solve_locomo_hypothesis_v2("When was it taken?", "date", [summer], "", lambda *_: {})
    assert week_solution["answer"] == "the week before 3 May 2023"
    assert summer_solution["answer"] == "summer 2022"


def test_invalid_anchor_and_non_temporal_when_answer_are_rejected() -> None:
    bad = binding("tomorrow")
    bad["anchor_id"] = "raw:other"
    operand = normalize_locomo_binding_v2(bad, fake_base_normalize)
    solution = solve_locomo_hypothesis_v2("When did it happen?", "recency", [operand], "", lambda *_: {})
    assert not solution["success"]
    assert not typed_solution_safe(
        "When did it happen?", {"success": True, "answer": "the event", "answer_type": "DATE_POINT"}
    )


def test_ambiguity_gate_and_answer_type_parser() -> None:
    assert ambiguity_safe([])
    assert not ambiguity_safe(["last week is ambiguous"])
    assert infer_answer_type("24 June 2023") == "DATE_POINT"
    assert infer_answer_type("the week before 14 August 2022") == "DATE_WINDOW"
    assert infer_answer_type("someone") is None
