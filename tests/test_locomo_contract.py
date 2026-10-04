from hytopomem.reconstruction.locomo_contract import LoCoMoQueryRequirementCompiler
from hytopomem.reconstruction.schema import QueryOperation, SpeakerRole


def test_named_speaker_does_not_trigger_user_assistant_gate() -> None:
    contract = LoCoMoQueryRequirementCompiler().compile(
        "When did Caroline go to the LGBTQ support group?"
    )
    assert contract.operation == QueryOperation.TEMPORAL
    assert contract.target_speaker == SpeakerRole.ANY
    assert contract.target_entities == ["Caroline"]
    assert contract.compiler == "locomo_named_speaker_adapter_v1"


def test_count_contract_is_preserved() -> None:
    contract = LoCoMoQueryRequirementCompiler().compile(
        "How many concerts did Melanie attend?"
    )
    assert contract.operation == QueryOperation.COUNT
    assert contract.target_speaker == SpeakerRole.ANY
