from __future__ import annotations

from hytopomem.temporal_time_v8_4 import (
    build_temporal_sidecar,
    close_operands,
    compile_constraint_candidate,
    normalize_absolute_date,
    normalize_relative,
    parse_datetime,
)


def unit(unit_id: str, claim: str, when: str, *, role: str = "fact") -> dict:
    return {
        "unit_id": unit_id,
        "normalized_claim": claim,
        "claim_type": "state" if role == "state" else "fact",
        "entity": "user",
        "aspect": "activity",
        "value": claim,
        "raw_quotes": [{"message_id": f"raw:{unit_id}", "text": claim, "message_time": when}],
        "raw_message_ids": [f"raw:{unit_id}"],
        "message_time": when,
        "metadata": {"card_type": role},
        "ce_score": 5.0,
    }


def compiled() -> dict:
    return {
        "question_id": "q1",
        "question": "Which happened first, buying a guitar or attending the concert?",
        "question_date": "2025/05/30 (Fri) 10:00",
        "query_type": "ordering",
        "required_roles": ["buying a guitar", "attending the concert"],
        "hypotheses": [],
    }


def test_relative_month_is_calendar_aware() -> None:
    value, status, _ = normalize_relative("one month ago", parse_datetime("2024/03/31"))
    assert value is not None
    assert value.date().isoformat() == "2024-02-29"
    assert status == "exact"


def test_implausible_relative_year_fails_closed() -> None:
    value, status, trace = normalize_relative("4000 years ago", parse_datetime("2024/03/31"))
    assert value is None
    assert status == "ambiguous"
    assert trace == "relative_amount_out_of_range"


def test_explicit_month_day_borrows_message_year() -> None:
    value, status, trace = normalize_absolute_date("on October 15th", parse_datetime("2023/11/01"))
    assert value is not None
    assert value.date().isoformat() == "2023-10-15"
    assert status == "exact"
    assert trace == "explicit_month_day"


def test_raw_quote_date_is_not_assigned_to_unrelated_sentence() -> None:
    item = unit("u1", "I need gift ideas.", "2025/05/20 (Tue) 09:00")
    item["raw_quotes"][0]["text"] = "I need gift ideas. Rachel got engaged on May 15th."
    sidecar = build_temporal_sidecar(question_id="q1", pack={"evidence_units": [item]}, compiled_row=compiled())
    event = next(node for node in sidecar["nodes"] if node["node_type"] == "EVENT")
    assert event["occurred_start"] is None


def test_sidecar_does_not_assume_message_time_is_event_time() -> None:
    pack = {"evidence_units": [unit("u1", "I bought a guitar.", "2025/05/20 (Tue) 09:00")]}
    sidecar = build_temporal_sidecar(question_id="q1", pack=pack, compiled_row=compiled())
    event = next(node for node in sidecar["nodes"] if node["node_type"] == "EVENT")
    assert event["mentioned_at"].startswith("2025-05-20")
    assert event["occurred_start"] is None
    assert event["normalization_status"] == "mentioned_only"


def test_operand_closure_and_solver_use_distinct_roles() -> None:
    pack = {
        "evidence_units": [
            unit("u1", "I bought a guitar today.", "2025/05/01 (Thu) 09:00"),
            unit("u2", "I attended the concert today.", "2025/05/20 (Tue) 21:00"),
        ]
    }
    row = compiled()
    sidecar = build_temporal_sidecar(question_id="q1", pack=pack, compiled_row=row)
    closure = close_operands(question_id="q1", pack=pack, compiled_row=row, sidecar=sidecar)
    assert closure["operand_full_coverage"] is True
    assert len(set(closure["selected_unit_ids"])) == 2
    candidate = compile_constraint_candidate(compiled_row=row, pack=pack, sidecar=sidecar, closure=closure)
    assert candidate["solution"]["success"] is True
    assert "guitar" in candidate["solution"]["answer"]


def test_state_view_has_validity_and_update_edges() -> None:
    row = {
        **compiled(),
        "query_type": "attribute_at_time",
        "required_roles": ["activity"],
        "question": "What was my activity one month ago?",
    }
    pack = {
        "evidence_units": [
            unit("s1", "I worked at Company A today.", "2025/03/01 (Sat) 09:00", role="state"),
            unit("s2", "I work at Company B today.", "2025/05/01 (Thu) 09:00", role="state"),
        ]
    }
    sidecar = build_temporal_sidecar(question_id="q1", pack=pack, compiled_row=row)
    states = [node for node in sidecar["nodes"] if node["node_type"] == "STATE"]
    assert len(states) == 2
    assert any(item["valid_to"] for item in states)
    relations = {item["relation"] for item in sidecar["edges"]}
    assert {"VALID_DURING", "UPDATED_BY", "SUPERSEDES"} <= relations


def test_existing_executable_q4_candidate_is_protected() -> None:
    row = compiled()
    row["pre_verifier"] = {"eligible_to_call_solution_verifier": True}
    row["hypotheses"] = [
        {
            "source_validation": {"eligible_for_normalization": True},
            "normalized_operands": [
                {"role": "buying a guitar", "fact_id": "u1", "occurred_at": "2025-05-01T09:00:00"},
                {"role": "attending the concert", "fact_id": "u2", "occurred_at": "2025-05-20T21:00:00"},
            ],
            "solution": {"success": True, "answer": "buying a guitar was first", "operation": "ORDERING"},
        }
    ]
    pack = {
        "evidence_units": [
            unit("u1", "I bought a guitar today.", "2025/05/01 (Thu) 09:00"),
            unit("u2", "I attended the concert today.", "2025/05/20 (Tue) 21:00"),
        ]
    }
    sidecar = build_temporal_sidecar(question_id="q1", pack=pack, compiled_row=row)
    closure = close_operands(question_id="q1", pack=pack, compiled_row=row, sidecar=sidecar)
    candidate = compile_constraint_candidate(compiled_row=row, pack=pack, sidecar=sidecar, closure=closure)
    assert candidate["candidate_source"] == "protected_source_q4"
    assert candidate["solution"]["answer"] == "buying a guitar was first"


def test_locally_solved_but_pre_verifier_rejected_candidate_is_not_protected() -> None:
    row = compiled()
    row["pre_verifier"] = {"eligible_to_call_solution_verifier": False, "restrictive_clause_grounded": False}
    row["hypotheses"] = [
        {
            "source_validation": {"eligible_for_normalization": True},
            "normalized_operands": [],
            "solution": {"success": True, "answer": "unsafe local answer"},
        }
    ]
    pack = {"evidence_units": []}
    sidecar = build_temporal_sidecar(question_id="q1", pack=pack, compiled_row=row)
    closure = close_operands(question_id="q1", pack=pack, compiled_row=row, sidecar=sidecar)
    candidate = compile_constraint_candidate(compiled_row=row, pack=pack, sidecar=sidecar, closure=closure)
    assert candidate["candidate_source"] == "8.4time_v_operand_closure"
    assert candidate["eligible_for_q4_verifier"] is False


def test_aggregate_duration_without_per_item_durations_fails_closed() -> None:
    row = {
        **compiled(),
        "query_type": "duration",
        "required_roles": ["reading A", "reading B", "reading C"],
        "question": "How many weeks in total did I spend reading A, B and C?",
    }
    pack = {
        "evidence_units": [
            unit("a", "I finished reading A today.", "2025/05/01 (Thu) 09:00"),
            unit("b", "I finished reading B today.", "2025/05/08 (Thu) 09:00"),
            unit("c", "I finished reading C today.", "2025/05/15 (Thu) 09:00"),
        ]
    }
    sidecar = build_temporal_sidecar(question_id="q1", pack=pack, compiled_row=row)
    closure = close_operands(question_id="q1", pack=pack, compiled_row=row, sidecar=sidecar)
    candidate = compile_constraint_candidate(compiled_row=row, pack=pack, sidecar=sidecar, closure=closure)
    assert candidate["solution"]["success"] is False
    assert candidate["eligible_for_q4_verifier"] is False


def test_single_role_trip_duration_expands_to_start_and_end() -> None:
    row = {
        **compiled(),
        "query_type": "duration",
        "required_roles": ["solo camping trip"],
        "question": "How many days did I spend on my solo camping trip?",
    }
    pack = {
        "evidence_units": [
            unit("start", "I started my solo camping trip today.", "2025/05/15 (Thu) 09:00"),
            unit("end", "I got back from my solo camping trip today.", "2025/05/17 (Sat) 09:00"),
        ]
    }
    sidecar = build_temporal_sidecar(question_id="q1", pack=pack, compiled_row=row)
    closure = close_operands(question_id="q1", pack=pack, compiled_row=row, sidecar=sidecar)
    assert set(closure["selected_by_role"]) == {"solo camping trip:start", "solo camping trip:end"}
    candidate = compile_constraint_candidate(compiled_row=row, pack=pack, sidecar=sidecar, closure=closure)
    assert candidate["solution"]["answer"] == "3 days"
    assert candidate["eligible_for_q4_verifier"] is True


def test_elapsed_days_uses_calendar_dates_not_timestamp_rounding() -> None:
    row = {
        **compiled(),
        "query_type": "elapsed",
        "required_roles": ["museum A", "museum B"],
        "question": "How many days passed between museum A and museum B?",
    }
    pack = {
        "evidence_units": [
            unit("a", "I returned from a museum visit today.", "2025/01/08 (Wed) 23:50"),
            unit("b", "I attended museum B today.", "2025/01/15 (Wed) 00:10"),
        ]
    }
    sidecar = build_temporal_sidecar(question_id="q1", pack=pack, compiled_row=row)
    closure = close_operands(question_id="q1", pack=pack, compiled_row=row, sidecar=sidecar)
    candidate = compile_constraint_candidate(compiled_row=row, pack=pack, sidecar=sidecar, closure=closure)
    assert candidate["solution"]["answer"] == "7 days"
    assert candidate["solution"]["calendar_day_semantics"] == "between"


def test_ambiguous_weekday_month_anchor_fails_closed() -> None:
    row = {
        **compiled(),
        "query_type": "attribute_at_time",
        "required_roles": ["activity"],
        "question": "What did I do on the Wednesday two months ago?",
    }
    pack = {"evidence_units": [unit("s1", "I worked today.", "2025/03/01 (Sat) 09:00", role="state")]}
    sidecar = build_temporal_sidecar(question_id="q1", pack=pack, compiled_row=row)
    closure = close_operands(question_id="q1", pack=pack, compiled_row=row, sidecar=sidecar)
    candidate = compile_constraint_candidate(compiled_row=row, pack=pack, sidecar=sidecar, closure=closure)
    assert closure["plan"]["target_time"] is None
    assert candidate["eligible_for_q4_verifier"] is False
