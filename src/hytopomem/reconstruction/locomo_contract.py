"""LoCoMo-specific query contracts for HyPathMem reconstruction.

LoCoMo conversations use named speakers rather than the fixed user/assistant
roles in LongMemEval. This adapter keeps the operation and slot inference from
the frozen D2 compiler while disabling the incompatible speaker gate.
"""

from __future__ import annotations

import re

from hytopomem.reconstruction.query_requirement_compiler import (
    HeuristicQueryRequirementCompiler,
)
from hytopomem.reconstruction.schema import QueryContract, SpeakerRole


_CAPITALIZED_NAME = re.compile(r"\b[A-Z][a-z]+\b")
_QUESTION_WORDS = {
    "After",
    "Before",
    "Did",
    "Does",
    "How",
    "If",
    "In",
    "On",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
    "Would",
}


class LoCoMoQueryRequirementCompiler:
    """Adapt the D2 query contract without inspecting gold annotations."""

    def __init__(self) -> None:
        self.base = HeuristicQueryRequirementCompiler()

    def compile(self, question: str, *, question_date: str | None = None) -> QueryContract:
        contract = self.base.compile(question, question_date=question_date)
        names = [
            token
            for token in _CAPITALIZED_NAME.findall(question)
            if token not in _QUESTION_WORDS
        ]
        return contract.model_copy(
            update={
                "target_speaker": SpeakerRole.ANY,
                "target_entities": list(dict.fromkeys(names)),
                "compiler": "locomo_named_speaker_adapter_v1",
            }
        )
