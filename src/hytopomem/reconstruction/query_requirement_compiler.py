from __future__ import annotations

import re

from hytopomem.reconstruction.schema import (
    QueryContract,
    QueryOperation,
    RequirementSlot,
    SpeakerRole,
    TemporalScope,
)


_COUNT_RE = re.compile(r"\bhow many\b", re.IGNORECASE)
_TEMPORAL_RE = re.compile(
    r"\b(when|what date|how long|how much time|before|after|earlier|later|ago|first|last)\b",
    re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    r"\b(current|currently|now|latest|previous|previously|used to|changed|change|update|moved|no longer|still)\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(prefer|preference|favorite|favourite|like best|recommend for me|would i prefer)\b",
    re.IGNORECASE,
)
_ASSISTANT_RE = re.compile(
    r"\b(did you|have you|you (?:recommend|suggest|tell|advise|explain|say|give)|your (?:answer|advice|suggestion|recommendation))\b",
    re.IGNORECASE,
)
_MULTI_RE = re.compile(
    r"\b(all|each|every|across|over time|throughout|different times|multiple|in total|altogether)\b",
    re.IGNORECASE,
)
_LIST_RE = re.compile(r"\b(what (?:are|were)|which (?:things|items|activities|places)|list|name all)\b", re.IGNORECASE)
_COMPARE_RE = re.compile(r"\b(compare|difference|both|between)\b", re.IGNORECASE)
_ANSWERABILITY_RE = re.compile(r"\b(enough information|can (?:you|we) determine|do we know)\b", re.IGNORECASE)


class HeuristicQueryRequirementCompiler:
    """Compile a query contract from question text only.

    The compiler intentionally ignores benchmark category metadata. It is a
    deterministic baseline for isolating reconstruction gains before adding an
    LLM planner.
    """

    def compile(self, question: str, *, question_date: str | None = None) -> QueryContract:
        operation = self._operation(question)
        target_speaker = SpeakerRole.ASSISTANT if operation == QueryOperation.ASSISTANT_RECALL else SpeakerRole.USER
        temporal_mode = self._temporal_mode(question, operation)
        slots = self._slots(operation)
        needs_multiple = bool(_MULTI_RE.search(question)) or operation in {
            QueryOperation.AGGREGATE,
            QueryOperation.COMPARE,
        }
        confidence = 0.92 if operation != QueryOperation.LOOKUP else 0.72
        return QueryContract(
            operation=operation,
            target_entities=[target_speaker.value],
            required_slots=slots,
            target_speaker=target_speaker,
            temporal_scope=TemporalScope(mode=temporal_mode, anchor=question_date),
            needs_multiple_sessions=needs_multiple,
            needs_raw_quote=operation in {
                QueryOperation.ASSISTANT_RECALL,
                QueryOperation.COUNT,
                QueryOperation.LIST,
                QueryOperation.UPDATE,
                QueryOperation.TEMPORAL,
            },
            answerability_requires=[slot.slot_id for slot in slots if slot.required],
            confidence=confidence,
        )

    @staticmethod
    def _operation(question: str) -> QueryOperation:
        if _ANSWERABILITY_RE.search(question):
            return QueryOperation.ANSWERABILITY
        if _ASSISTANT_RE.search(question):
            return QueryOperation.ASSISTANT_RECALL
        if _COUNT_RE.search(question):
            return QueryOperation.COUNT
        if _UPDATE_RE.search(question):
            return QueryOperation.UPDATE
        if _TEMPORAL_RE.search(question):
            return QueryOperation.TEMPORAL
        if _PREFERENCE_RE.search(question):
            return QueryOperation.PREFERENCE
        if _COMPARE_RE.search(question):
            return QueryOperation.COMPARE
        if _LIST_RE.search(question):
            return QueryOperation.LIST
        if _MULTI_RE.search(question):
            return QueryOperation.AGGREGATE
        return QueryOperation.LOOKUP

    @staticmethod
    def _temporal_mode(question: str, operation: QueryOperation) -> str:
        lowered = question.lower()
        if any(token in lowered for token in ("current", "currently", "now", "latest")):
            return "latest"
        if any(token in lowered for token in ("previous", "previously", "used to")):
            return "historical"
        if operation == QueryOperation.TEMPORAL:
            return "relative_or_event_time"
        return "unspecified"

    @staticmethod
    def _slots(operation: QueryOperation) -> list[RequirementSlot]:
        if operation == QueryOperation.UPDATE:
            return [
                RequirementSlot(slot_id="previous_state", aspect="state", required=False),
                RequirementSlot(slot_id="current_state", aspect="state"),
                RequirementSlot(slot_id="change_time", aspect="time", required=False),
            ]
        if operation == QueryOperation.TEMPORAL:
            return [
                RequirementSlot(slot_id="event", aspect="event"),
                RequirementSlot(slot_id="time_anchor", aspect="time"),
            ]
        if operation == QueryOperation.COUNT:
            return [RequirementSlot(slot_id="counted_items", aspect="items", cardinality="multiple")]
        if operation in {QueryOperation.LIST, QueryOperation.AGGREGATE}:
            return [RequirementSlot(slot_id="items", aspect="items", cardinality="multiple")]
        if operation == QueryOperation.COMPARE:
            return [
                RequirementSlot(slot_id="left_item", aspect="comparison"),
                RequirementSlot(slot_id="right_item", aspect="comparison"),
            ]
        if operation == QueryOperation.PREFERENCE:
            return [
                RequirementSlot(slot_id="preference", aspect="preference"),
                RequirementSlot(slot_id="constraints", aspect="constraint", required=False),
            ]
        if operation == QueryOperation.ASSISTANT_RECALL:
            return [RequirementSlot(slot_id="assistant_response", aspect="assistant_content")]
        if operation == QueryOperation.ANSWERABILITY:
            return [RequirementSlot(slot_id="direct_support", aspect="answerability")]
        return [RequirementSlot(slot_id="answer", aspect="fact")]
