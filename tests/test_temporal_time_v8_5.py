from hytopomem.temporal_time_v8_5 import (
    diagnose_non_executable,
    propagate_constraints,
    resolve_grounded_interval,
    validate_enrichment,
)


def operand(fact_id="event-a", occurred_at=None, evidence="It happened two weeks ago."):
    return {
        "fact_id": fact_id,
        "raw_id": f"raw-{fact_id}",
        "evidence_span": evidence,
        "mentioned_at": "2025-05-20T10:00:00",
        "occurred_at": occurred_at,
    }


def test_diagnosis_selects_only_trusted_raw_grounded_missing_time():
    row = {
        "h2_h4_operand_closure": {
            "operand_full_coverage": True,
            "selected_by_role": {"event": {"trusted_q4_binding": True}},
            "temporal_consistency": {"consistent": True},
            "plan": {"requires_pairwise_operands": False},
        },
        "h5_constraint_candidate": {
            "query_type": "recency",
            "required_roles": ["event"],
            "operands": [operand()],
        },
    }
    audit = diagnose_non_executable(row)
    assert audit["failure_type"] == "occurred_time_missing"
    assert audit["repairable_by_time_enrichment"] is True
    assert audit["validated_operand_coverage"] is False


def test_identity_failure_blocks_enrichment():
    row = {
        "h2_h4_operand_closure": {
            "operand_full_coverage": True,
            "selected_by_role": {"event": {"trusted_q4_binding": False}},
            "temporal_consistency": {"consistent": True},
            "plan": {"requires_pairwise_operands": False},
        },
        "h5_constraint_candidate": {
            "query_type": "recency",
            "required_roles": ["event"],
            "operands": [operand()],
        },
    }
    audit = diagnose_non_executable(row)
    assert audit["failure_type"] == "event_identity_unreliable"
    assert audit["repairable_by_time_enrichment"] is False


def test_validate_enrichment_resolves_relative_time_from_message_anchor():
    extraction = {
        "event_id": "event-a",
        "identity_supported": True,
        "time_expression": "two weeks ago",
        "anchor_type": "mentioned_at",
        "anchor_event_id": None,
        "relation": "BEFORE_OFFSET",
        "offset_value": 2,
        "offset_unit": "week",
        "confidence": 0.95,
        "ambiguity": [],
    }
    validated, reason = validate_enrichment(extraction, operand(), {"event-a"})
    assert reason == "accepted"
    assert validated["resolved_at"].startswith("2025-05-06")


def test_validate_enrichment_rejects_unquoted_expression():
    extraction = {
        "event_id": "event-a",
        "identity_supported": True,
        "time_expression": "three weeks ago",
        "anchor_type": "mentioned_at",
        "anchor_event_id": None,
        "relation": "BEFORE_OFFSET",
        "offset_value": 3,
        "offset_unit": "week",
        "confidence": 0.99,
        "ambiguity": [],
    }
    validated, reason = validate_enrichment(extraction, operand(), {"event-a"})
    assert validated is None
    assert reason == "time_expression_not_in_raw"


def test_chain_propagation_composes_offsets():
    operands = [
        operand("festival", "2025-05-01T10:00:00"),
        operand("trip"),
        operand("booking"),
    ]
    constraints = [
        {
            "event_id": "trip",
            "anchor_event_id": "festival",
            "relation": "AFTER_OFFSET",
            "offset_value": 2,
            "offset_unit": "week",
        },
        {
            "event_id": "booking",
            "anchor_event_id": "trip",
            "relation": "BEFORE_OFFSET",
            "offset_value": 1,
            "offset_unit": "week",
        },
    ]
    result = propagate_constraints(operands, constraints)
    assert result["consistent"] is True
    assert result["resolved_times"]["trip"].startswith("2025-05-15")
    assert result["resolved_times"]["booking"].startswith("2025-05-08")


def test_propagation_rejects_conflicting_known_path():
    operands = [
        operand("festival", "2025-05-01T10:00:00"),
        operand("trip", "2025-05-20T10:00:00"),
    ]
    constraints = [
        {
            "event_id": "trip",
            "anchor_event_id": "festival",
            "relation": "AFTER_OFFSET",
            "offset_value": 2,
            "offset_unit": "week",
        }
    ]
    result = propagate_constraints(operands, constraints)
    assert result["consistent"] is False
    assert result["conflicts"]


def test_last_weekend_is_a_grounded_interval_not_a_guessed_point():
    interval = resolve_grounded_interval("last weekend", __import__("datetime").datetime(2023, 5, 30, 7, 8))
    assert interval[0].date().isoformat() == "2023-05-27"
    assert interval[1].date().isoformat() == "2023-05-28"


def test_low_model_confidence_can_be_replaced_by_deterministic_interval():
    value = operand(evidence="I finished it last weekend.")
    extraction = {
        "event_id": "event-a",
        "identity_supported": True,
        "time_expression": "last weekend",
        "anchor_type": "mentioned_at",
        "anchor_event_id": "raw-id-that-must-be-ignored",
        "relation": "NONE",
        "offset_value": 0,
        "offset_unit": "day",
        "confidence": 0.45,
        "ambiguity": [],
    }
    validated, reason = validate_enrichment(extraction, value, {"event-a"})
    assert reason == "accepted"
    assert validated["relation"] == "INTERVAL"
    assert validated["resolved_start"].startswith("2025-05-17")
