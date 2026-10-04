from hytopomem.temporal import TemporalPacketCompiler, TemporalQueryPlanner, TemporalSidecarBuilder, TemporalSolver
from hytopomem.temporal.schema import TemporalQueryType


def unit(unit_id: str, claim: str, timestamp: str, *, rank: int) -> dict:
    return {
        "unit_id": unit_id,
        "normalized_claim": claim,
        "raw_message_ids": [f"raw:{unit_id}"],
        "session_id": "session",
        "speaker": "user",
        "raw_quotes": [
            {
                "message_id": f"raw:{unit_id}",
                "text": f"user: {claim}",
                "speaker": "user",
                "session_id": "session",
                "message_time": timestamp,
                "support_kind": "direct",
            }
        ],
        "metadata": {"rank": rank},
    }


def test_planner_treats_how_many_months_ago_as_elapsed_not_count() -> None:
    plan = TemporalQueryPlanner().plan(
        "How many months ago did I book the Airbnb in San Francisco?",
        question_date="2023-05-21",
        question_type="temporal-reasoning",
    )

    assert plan.query_type == TemporalQueryType.DURATION
    assert plan.operator == "ELAPSED_TO_QUESTION"
    assert plan.answer_unit == "month"
    assert plan.required_roles[0].description == "book the airbnb in san francisco"


def test_sidecar_distinguishes_message_time_from_occurrence_time() -> None:
    records, constraints = TemporalSidecarBuilder().build(
        [unit("binoculars", "I got my binoculars exactly three weeks ago.", "2023/05/20 (Sat) 20:08", rank=1)]
    )

    assert records[0].mentioned_at == "2023-05-20T20:08:00"
    assert records[0].occurred_start == "2023-04-29T20:08:00"
    assert records[0].normalization_status == "resolved"
    assert constraints[0].expression == "event_time = mentioned_at - 3 week(s)"


def test_solver_composes_elapsed_event_and_in_advance_offset() -> None:
    question = "How many months ago did I book the Airbnb in San Francisco?"
    plan = TemporalQueryPlanner().plan(question, question_date="2023-05-21", question_type="temporal-reasoning")
    records, _ = TemporalSidecarBuilder().build(
        [
            unit(
                "booking",
                "I stayed in San Francisco for my friend's wedding and had to book the Airbnb three months in advance.",
                "2023/05/21 (Sun) 18:59",
                rank=1,
            ),
            unit(
                "trip",
                "I went to San Francisco exactly two months ago for my friend's wedding.",
                "2023/05/21 (Sun) 17:13",
                rank=2,
            ),
        ]
    )
    solution = TemporalSolver().solve(plan, records)

    assert solution.success
    assert solution.verified
    assert solution.answer == "5 months"


def test_elapsed_uses_question_time_not_relative_expression_alone() -> None:
    question = "How many days ago did I attend a baking class?"
    plan = TemporalQueryPlanner().plan(question, question_date="2022/04/15 (Fri) 18:46", question_type="temporal-reasoning")
    records, _ = TemporalSidecarBuilder().build(
        [unit("class", "I attended the baking class yesterday.", "2022/03/21 (Mon) 18:46", rank=1)]
    )
    solution = TemporalSolver().solve(plan, records)

    assert solution.success
    assert solution.answer == "26 days"


def test_planner_extracts_three_ordering_operands() -> None:
    plan = TemporalQueryPlanner().plan(
        "Which three events happened in the order from first to last: the day I prepared a nursery, the day I picked baby-shower gifts, and the day I ordered a phone case?",
        question_type="temporal-reasoning",
    )

    assert plan.query_type == TemporalQueryType.ORDERING
    assert plan.operator == "SORT_EVENTS"
    assert [role.role for role in plan.required_roles] == ["event_1", "event_2", "event_3"]


def test_temporal_lookup_is_not_verified_as_a_date_answer() -> None:
    question = "What did I buy four weeks ago?"
    plan = TemporalQueryPlanner().plan(question, question_date="2023-05-20", question_type="temporal-reasoning")
    records, _ = TemporalSidecarBuilder().build(
        [unit("tools", "I bought sculpting tools four weeks ago.", "2023/05/20 (Sat) 20:08", rank=1)]
    )
    solution = TemporalSolver().solve(plan, records)

    assert plan.operator == "SELECT_EVENT_AT_TIME"
    assert not solution.success
    assert not solution.verified


def test_planner_extracts_two_choice_ordering_operands() -> None:
    plan = TemporalQueryPlanner().plan(
        "Which gift did I buy first, the necklace for my sister or the photo album for my mom?",
        question_type="temporal-reasoning",
    )

    assert plan.query_type == TemporalQueryType.ORDERING
    assert [role.description for role in plan.required_roles] == [
        "the necklace for my sister",
        "the photo album for my mom",
    ]
    assert plan.subtype == "first_of_candidates"


def test_solver_computes_difference_between_two_relative_events() -> None:
    question = "How long did I have binoculars before I saw the goldfinches return?"
    plan = TemporalQueryPlanner().plan(question, question_date="2023-05-20", question_type="temporal-reasoning")
    records, _ = TemporalSidecarBuilder().build(
        [
            unit("binoculars", "I got my binoculars exactly three weeks ago.", "2023/05/20 (Sat) 20:08", rank=1),
            unit("birds", "I saw the goldfinches return exactly one week ago.", "2023/05/20 (Sat) 20:08", rank=2),
        ]
    )
    solution = TemporalSolver().solve(plan, records)

    assert solution.success
    assert solution.answer == "2 weeks"


def test_temporal_packet_replaces_d2_and_keeps_one_quote_per_operand() -> None:
    question = "When did I get my binoculars?"
    plan = TemporalQueryPlanner().plan(question, question_date="2023-05-20", question_type="temporal-reasoning")
    records, constraints = TemporalSidecarBuilder().build(
        [unit("binoculars", "I got my binoculars exactly three weeks ago.", "2023/05/20 (Sat) 20:08", rank=1)]
    )
    solution = TemporalSolver().solve(plan, records)
    packet = TemporalPacketCompiler().compile(question, plan, records, constraints, solution, stage="T4")

    assert packet.diagnostics["context_replaces_d2_for_temporal"] is True
    assert packet.diagnostics["eligible_to_override"] is True
    assert len(packet.operands) == 1
    assert "quote" in packet.operands[0]
    assert packet.computed_result and packet.computed_result.answer == "2023-04-29"
