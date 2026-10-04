from hytopomem.reconstruction import AnswerPackCompiler, EvidenceUnitBuilder, HeuristicQueryRequirementCompiler
from hytopomem.reconstruction.schema import QueryOperation, SpeakerRole


def synthetic_nodes() -> dict:
    return {
        "conv:raw:session:t000": {
            "node_id": "conv:raw:session:t000",
            "type": "RAW",
            "text": "user: Which flash should I buy for my camera?",
            "time": "2025-01-01",
            "metadata": {"speaker": "user", "session_id": "session"},
        },
        "conv:raw:session:t001": {
            "node_id": "conv:raw:session:t001",
            "type": "RAW",
            "text": "assistant: I recommend the Godox V1-S flash.",
            "time": "2025-01-01",
            "metadata": {"speaker": "assistant", "session_id": "session"},
        },
        "conv:fact:001": {
            "node_id": "conv:fact:001",
            "type": "FACT",
            "text": "The assistant recommended the Godox V1-S flash.",
            "time": "2025-01-01",
            "support_ids": ["conv:raw:session:t001"],
            "metadata": {
                "speaker": "assistant",
                "session_id": "session",
                "support_raw_ids": ["conv:raw:session:t001"],
            },
        },
    }


def synthetic_path() -> dict:
    return {
        "node_ids": ["conv:topic:1", "conv:event:1", "conv:fact:001"],
        "score": 2.0,
        "scores": {"topology_selector": 2.0, "cross_encoder": 5.0},
        "metadata": {
            "evidence_node_id": "conv:fact:001",
            "route_source": "bottom_up+eu_event+hyp_event",
            "episode_node_id": "conv:episode:1",
        },
    }


def test_query_compiler_uses_question_text_for_assistant_recall() -> None:
    contract = HeuristicQueryRequirementCompiler().compile("What flash did you recommend for my camera?")

    assert contract.operation == QueryOperation.ASSISTANT_RECALL
    assert contract.target_speaker == SpeakerRole.ASSISTANT
    assert contract.required_slots[0].slot_id == "assistant_response"


def test_support_closure_preserves_assistant_quote_and_user_request() -> None:
    contract = HeuristicQueryRequirementCompiler().compile("What flash did you recommend for my camera?")
    units = EvidenceUnitBuilder(synthetic_nodes()).build([synthetic_path()], contract, k=1)

    assert len(units) == 1
    assert [quote.support_kind for quote in units[0].raw_quotes] == ["request_pair", "direct"]
    assert "Godox V1-S" in units[0].raw_quotes[1].text
    assert units[0].covered_slot_ids == ["assistant_response"]


def test_answer_pack_is_supported_and_raw_grounded() -> None:
    contract = HeuristicQueryRequirementCompiler().compile("What flash did you recommend for my camera?")
    units = EvidenceUnitBuilder(synthetic_nodes()).build([synthetic_path()], contract, k=1)
    compiler = AnswerPackCompiler(token_budget=1000)
    pack = compiler.compile("What flash did you recommend for my camera?", contract, units)
    rendered = compiler.render_json(pack)

    assert pack.answerability == "SUPPORTED"
    assert pack.target_speaker_covered
    assert pack.diagnostics["raw_grounded_units"] == 1
    assert pack.diagnostics["answerability_is_qa_prediction"] is False
    assert "Godox V1-S" in rendered
    assert '"evidence_groups"' in rendered
    assert "question_type" not in rendered
