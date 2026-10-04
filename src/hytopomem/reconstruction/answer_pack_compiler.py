from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence

from hytopomem.reconstruction.schema import AnswerPack, EvidenceGroup, EvidenceUnit, QueryContract, SpeakerRole


class AnswerPackCompiler:
    """Compile ranked units into a bounded, provenance-first answer pack."""

    def __init__(self, *, token_budget: int = 10_000) -> None:
        self.token_budget = token_budget

    def compile(self, question: str, contract: QueryContract, units: Sequence[EvidenceUnit]) -> AnswerPack:
        selected = self._bounded_units(contract, units)
        covered = sorted({slot_id for unit in selected for slot_id in unit.covered_slot_ids})
        required = [slot.slot_id for slot in contract.required_slots if slot.required]
        missing = [slot_id for slot_id in required if slot_id not in covered]
        sessions = sorted({unit.session_id for unit in selected if unit.session_id})
        target_speaker_covered = contract.target_speaker == SpeakerRole.ANY or any(
            unit.speaker.lower() == contract.target_speaker.value for unit in selected
        )
        if contract.needs_multiple_sessions and len(sessions) < 2:
            missing.append("distinct_sessions")
        if not target_speaker_covered:
            missing.append(f"speaker:{contract.target_speaker.value}")
        answerability = "SUPPORTED" if not missing else ("PARTIALLY_SUPPORTED" if covered else "UNSUPPORTED")
        groups_by_slot: dict[str, list[str]] = defaultdict(list)
        for unit in selected:
            for slot_id in unit.covered_slot_ids:
                groups_by_slot[slot_id].append(unit.unit_id)
        groups = [
            EvidenceGroup(group_id=f"requirement:{slot.slot_id}", requirement=slot.description or slot.aspect,
                          evidence_unit_ids=groups_by_slot.get(slot.slot_id, []))
            for slot in contract.required_slots
        ]
        return AnswerPack(
            question=question,
            contract=contract,
            evidence_units=selected,
            evidence_groups=groups,
            covered_slots=covered,
            missing_slots=list(dict.fromkeys(missing)),
            distinct_sessions=sessions,
            target_speaker_covered=target_speaker_covered,
            answerability=answerability,
            token_cost=sum(unit.token_cost for unit in selected),
            diagnostics={
                "candidate_units": len(units),
                "selected_units": len(selected),
                "raw_grounded_units": sum(bool(unit.raw_quotes) for unit in selected),
                "token_budget": self.token_budget,
                "selection": "rank_preserving_budget_v0_1",
                "answerability_gate": "deterministic_required_slot_coverage_v0_1",
                "answerability_is_qa_prediction": False,
            },
        )

    def render_json(self, pack: AnswerPack) -> str:
        payload = {
            "instructions": [
                "Check every required slot before answering.",
                "Use only claims supported by exact raw quotes.",
                "Do not treat retrieval scores or path metadata as facts.",
                "If a required slot is missing after retrieval, say that it cannot be determined.",
                "Return a concise final answer.",
            ],
            "question": pack.question,
            "query_contract": pack.contract.model_dump(mode="json"),
            "answerability": pack.answerability,
            "covered_slots": pack.covered_slots,
            "missing_slots": pack.missing_slots,
            "evidence_groups": [group.model_dump(mode="json") for group in pack.evidence_groups],
            "evidence": [self._render_unit(unit) for unit in pack.evidence_units],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _bounded_units(self, contract: QueryContract, units: Sequence[EvidenceUnit]) -> list[EvidenceUnit]:
        ordered = list(units)
        if contract.target_speaker != SpeakerRole.ANY:
            ordered.sort(
                key=lambda unit: (
                    unit.speaker.lower() != contract.target_speaker.value,
                    int(unit.metadata.get("rank") or 0),
                )
            )
        selected: list[EvidenceUnit] = []
        used = 0
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for unit in ordered:
            fingerprint = (unit.normalized_claim.lower(), tuple(unit.raw_message_ids))
            if fingerprint in seen:
                continue
            if selected and used + unit.token_cost > self.token_budget:
                continue
            selected.append(unit)
            seen.add(fingerprint)
            used += unit.token_cost
        return selected

    @staticmethod
    def _render_unit(unit: EvidenceUnit) -> dict:
        return {
            "unit_id": unit.unit_id,
            "claim": unit.normalized_claim,
            "speaker": unit.speaker,
            "session_id": unit.session_id,
            "event_time": unit.event_time_start,
            "covered_slots": unit.covered_slot_ids,
            "raw_quotes": [quote.model_dump(mode="json") for quote in unit.raw_quotes],
            "provenance": {
                "path_node_ids": unit.path_node_ids,
                "route_sources": unit.route_sources,
            },
        }
