from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TemporalQueryType(StrEnum):
    DATE = "DATE"
    DURATION = "DURATION"
    ORDERING = "ORDERING"
    STATE_AT_TIME = "STATE_AT_TIME"


class TemporalRole(BaseModel):
    role: str
    description: str


class TemporalQueryPlan(BaseModel):
    activated: bool = False
    query_type: TemporalQueryType | None = None
    subtype: str = ""
    question_time: str | None = None
    required_roles: list[TemporalRole] = Field(default_factory=list)
    operator: str = ""
    answer_unit: str | None = None
    complexity: str = "L0"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class TemporalEventRecord(BaseModel):
    event_id: str
    unit_id: str
    fact_ids: list[str] = Field(default_factory=list)
    raw_ids: list[str] = Field(default_factory=list)
    session_id: str = ""
    event: str
    mentioned_at: str | None = None
    occurred_start: str | None = None
    occurred_end: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    relative_expression: str | None = None
    anchor: str | None = None
    offset_value: float | None = None
    offset_unit: str | None = None
    offset_direction: int | None = None
    duration_value: float | None = None
    duration_unit: str | None = None
    granularity: str = "unknown"
    time_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    normalization_status: str = "unresolved"
    source_quote: str = ""
    speaker: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)


class TemporalConstraint(BaseModel):
    constraint_id: str
    kind: str
    left_event_id: str
    right_anchor: str
    offset_value: float | None = None
    offset_unit: str | None = None
    direction: int | None = None
    expression: str = ""


class TemporalSolution(BaseModel):
    success: bool = False
    operator: str = ""
    answer: str | None = None
    value: float | str | list[str] | None = None
    unit: str | None = None
    selected_event_ids: list[str] = Field(default_factory=list)
    constraint_satisfaction: float = 0.0
    operand_coverage: float = 0.0
    verified: bool = False
    failure_reason: str | None = None
    trace: list[str] = Field(default_factory=list)


class TemporalEvidencePacket(BaseModel):
    version: str = "hypathmem_temporal_v0_1"
    stage: str
    question: str
    plan: TemporalQueryPlan
    operands: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[TemporalConstraint] = Field(default_factory=list)
    computed_result: TemporalSolution | None = None
    instructions: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
