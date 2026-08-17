from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class QueryOperation(StrEnum):
    LOOKUP = "lookup"
    LIST = "list"
    COUNT = "count"
    AGGREGATE = "aggregate"
    COMPARE = "compare"
    UPDATE = "update"
    TEMPORAL = "temporal"
    PREFERENCE = "preference"
    ASSISTANT_RECALL = "assistant_recall"
    ANSWERABILITY = "answerability"


class SpeakerRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    ANY = "any"


class RequirementSlot(BaseModel):
    slot_id: str
    aspect: str
    cardinality: str = "one"
    required: bool = True
    description: str = ""


class TemporalScope(BaseModel):
    mode: str = "unspecified"
    anchor: str | None = None


class QueryContract(BaseModel):
    operation: QueryOperation = QueryOperation.LOOKUP
    target_entities: list[str] = Field(default_factory=list)
    required_slots: list[RequirementSlot] = Field(default_factory=list)
    target_speaker: SpeakerRole = SpeakerRole.ANY
    temporal_scope: TemporalScope = Field(default_factory=TemporalScope)
    needs_multiple_sessions: bool = False
    needs_raw_quote: bool = True
    answerability_requires: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    compiler: str = "heuristic_v1"


class RawQuote(BaseModel):
    message_id: str
    text: str
    speaker: str = ""
    session_id: str = ""
    message_time: str | None = None
    support_kind: str = "direct"


class EvidenceUnit(BaseModel):
    unit_id: str
    normalized_claim: str
    claim_type: str = "fact"
    entity: str | None = None
    aspect: str | None = None
    value: str | None = None
    polarity: str = "positive"
    raw_quotes: list[RawQuote] = Field(default_factory=list)
    raw_message_ids: list[str] = Field(default_factory=list)
    speaker: str = ""
    session_id: str = ""
    episode_id: str | None = None
    event_time_start: str | None = None
    event_time_end: str | None = None
    message_time: str | None = None
    state_status: str = "unknown"
    modality: str = "asserted"
    permanence: str = "unknown"
    path_node_ids: list[str] = Field(default_factory=list)
    path_relation_types: list[str] = Field(default_factory=list)
    route_sources: list[str] = Field(default_factory=list)
    topology_score: float = 0.0
    ce_score: float = 0.0
    route_agreement: float = 0.0
    token_cost: int = 0
    covered_slot_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceGroup(BaseModel):
    group_id: str
    requirement: str
    evidence_unit_ids: list[str] = Field(default_factory=list)


class AnswerPack(BaseModel):
    version: str = "hypathmem_r_v0_1"
    question: str
    contract: QueryContract
    evidence_units: list[EvidenceUnit] = Field(default_factory=list)
    evidence_groups: list[EvidenceGroup] = Field(default_factory=list)
    covered_slots: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    distinct_sessions: list[str] = Field(default_factory=list)
    target_speaker_covered: bool = False
    answerability: str = "PARTIALLY_SUPPORTED"
    token_cost: int = 0
    diagnostics: dict[str, Any] = Field(default_factory=dict)
