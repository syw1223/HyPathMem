from __future__ import annotations

from typing import Any

from hytopomem.temporal.schema import (
    TemporalConstraint,
    TemporalEventRecord,
    TemporalEvidencePacket,
    TemporalQueryPlan,
    TemporalSolution,
)
from hytopomem.temporal.solver import select_operands


class TemporalPacketCompiler:
    """Compile replacement temporal contexts; this never appends to a full D2 context."""

    def compile(
        self,
        question: str,
        plan: TemporalQueryPlan,
        records: list[TemporalEventRecord],
        constraints: list[TemporalConstraint],
        solution: TemporalSolution,
        *,
        stage: str,
    ) -> TemporalEvidencePacket:
        selected = select_operands(plan, records)
        operands = [self._operand(role, record, stage) for role, record in selected.items()]
        selected_ids = {record.event_id for record in selected.values()}
        packet_constraints = [constraint for constraint in constraints if constraint.left_event_id in selected_ids]
        instructions = [
            "Use only the temporal operands in this packet; ignore other same-topic memories.",
            "A mention timestamp is not automatically an event occurrence time.",
            "Return one concise answer with the requested unit.",
        ]
        computed = None
        if stage in {"T3", "T4"}:
            computed = solution
            instructions.insert(0, "Use the deterministic computed result as-is; do not recalculate it.")
        if stage == "T4":
            instructions.insert(0, "Use the computed result only when verified=true; otherwise fall back to the standard reader.")
        return TemporalEvidencePacket(
            stage=stage,
            question=question,
            plan=plan,
            operands=operands,
            constraints=packet_constraints if stage in {"T2", "T3", "T4"} else [],
            computed_result=computed,
            instructions=instructions,
            diagnostics={
                "candidate_pool": "frozen_D2_top50",
                "context_replaces_d2_for_temporal": True,
                "selected_operands": len(operands),
                "required_operands": len(plan.required_roles),
                "eligible_to_override": stage == "T4" and solution.verified,
                "fallback": "D2_reader",
            },
        )

    @staticmethod
    def _operand(role: str, record: TemporalEventRecord, stage: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": role,
            "event": record.event,
            "mentioned_at": record.mentioned_at,
            "quote": record.source_quote,
            "provenance": {
                "unit_id": record.unit_id,
                "raw_ids": record.raw_ids,
                "session_id": record.session_id,
            },
        }
        if stage in {"T2", "T3", "T4"}:
            payload.update(
                {
                    "occurred_start": record.occurred_start,
                    "occurred_end": record.occurred_end,
                    "relative_expression": record.relative_expression,
                    "anchor": record.anchor,
                    "offset_value": record.offset_value,
                    "offset_unit": record.offset_unit,
                    "normalization_status": record.normalization_status,
                    "time_confidence": record.time_confidence,
                }
            )
        return payload
