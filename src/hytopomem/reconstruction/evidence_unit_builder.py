from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from hytopomem.reconstruction.schema import EvidenceUnit, QueryContract, QueryOperation, SpeakerRole
from hytopomem.reconstruction.support_closure import SupportClosureBuilder


class EvidenceUnitBuilder:
    def __init__(self, nodes: Mapping[str, Mapping[str, Any]]) -> None:
        self.nodes = nodes
        self.support_closure = SupportClosureBuilder(nodes)

    def build(self, paths: Sequence[Mapping[str, Any]], contract: QueryContract, *, k: int) -> list[EvidenceUnit]:
        units: list[EvidenceUnit] = []
        seen_nodes: set[str] = set()
        for rank, path in enumerate(paths[:k], start=1):
            node_id = evidence_node_id(path)
            if not node_id or node_id in seen_nodes:
                continue
            node = self.nodes.get(node_id)
            if not node:
                continue
            seen_nodes.add(node_id)
            units.append(self._unit(node_id, node, path, contract, rank))
        return units

    def _unit(
        self,
        node_id: str,
        node: Mapping[str, Any],
        path: Mapping[str, Any],
        contract: QueryContract,
        rank: int,
    ) -> EvidenceUnit:
        node_metadata = node.get("metadata") or {}
        path_metadata = path.get("metadata") or {}
        scores = path.get("scores") or {}
        quotes = self.support_closure.build(node)
        speaker = str(node_metadata.get("speaker") or next((quote.speaker for quote in quotes if quote.speaker), ""))
        session_id = str(
            node_metadata.get("session_id")
            or next((quote.session_id for quote in quotes if quote.session_id), "")
        )
        claim = normalize_claim(str(node.get("text") or ""))
        route_sources = route_tokens(str(path_metadata.get("route_source") or path_metadata.get("candidate_source") or ""))
        topology_score = number(scores.get("topology_selector"), number(path.get("score")))
        ce_score = number(scores.get("cross_encoder"))
        route_agreement = min(1.0, len(set(route_sources)) / 6.0)
        aspect = str(path_metadata.get("v3_9_card_aspect") or "") or None
        entity = str(path_metadata.get("v3_9_card_entity") or "") or None
        covered_slots = covered_slot_ids(contract, speaker=speaker, node=node, path_metadata=path_metadata, quotes=quotes)
        text_chars = len(claim) + sum(len(quote.text) for quote in quotes)
        return EvidenceUnit(
            unit_id=f"unit:{node_id}",
            normalized_claim=claim,
            claim_type=str(node.get("type") or "fact").lower(),
            entity=entity,
            aspect=aspect,
            value=claim or None,
            polarity="negative" if has_negation(claim) else "positive",
            raw_quotes=quotes,
            raw_message_ids=[quote.message_id for quote in quotes],
            speaker=speaker,
            session_id=session_id,
            episode_id=str(path_metadata.get("episode_node_id") or "") or None,
            event_time_start=str(node.get("time") or "") or None,
            message_time=next((quote.message_time for quote in quotes if quote.message_time), None),
            state_status=str(node.get("status") or "unknown"),
            modality=infer_modality(claim),
            permanence=infer_permanence(claim),
            path_node_ids=[str(value) for value in path.get("node_ids") or []],
            path_relation_types=[str(value) for value in path.get("edge_ids") or []],
            route_sources=route_sources,
            topology_score=topology_score,
            ce_score=ce_score,
            route_agreement=route_agreement,
            token_cost=max(1, math.ceil(text_chars / 4)),
            covered_slot_ids=covered_slots,
            metadata={
                "rank": rank,
                "nary_role": path_metadata.get("nary_role"),
                "card_type": path_metadata.get("nary_hyperedge_type"),
                "raw_grounded": bool(quotes),
            },
        )


def evidence_node_id(path: Mapping[str, Any]) -> str:
    metadata = path.get("metadata") or {}
    explicit = metadata.get("evidence_node_id")
    if explicit:
        return str(explicit)
    node_ids = list(path.get("node_ids") or [])
    return str(node_ids[-1]) if node_ids else ""


def covered_slot_ids(
    contract: QueryContract,
    *,
    speaker: str,
    node: Mapping[str, Any],
    path_metadata: Mapping[str, Any],
    quotes: Sequence[Any],
) -> list[str]:
    covered: list[str] = []
    role = str(path_metadata.get("nary_role") or "").lower()
    has_time = bool(node.get("time") or any(quote.message_time for quote in quotes))
    for slot in contract.required_slots:
        if slot.slot_id == "assistant_response":
            match = speaker.lower() == SpeakerRole.ASSISTANT.value and bool(quotes)
        elif slot.slot_id == "previous_state":
            match = role in {"old_state", "previous_state", "historical_state"}
        elif slot.slot_id == "current_state":
            match = role not in {"old_state", "previous_state"}
        elif slot.slot_id in {"change_time", "time_anchor"}:
            match = has_time
        else:
            match = bool(node.get("text")) and bool(quotes)
        if match:
            covered.append(slot.slot_id)
    return covered


def normalize_claim(text: str) -> str:
    return " ".join(text.split())


def route_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[+,]", value) if token]


def has_negation(text: str) -> bool:
    return bool(re.search(r"\b(no|not|never|no longer|didn't|doesn't|isn't|wasn't)\b", text, re.IGNORECASE))


def infer_modality(text: str) -> str:
    if re.search(r"\b(plan|planning|intend|hope|want|might|may|could)\b", text, re.IGNORECASE):
        return "planned_or_hypothetical"
    return "asserted"


def infer_permanence(text: str) -> str:
    if re.search(r"\b(today|tonight|this time|for now|temporarily)\b", text, re.IGNORECASE):
        return "temporary"
    if re.search(r"\b(always|usually|generally|permanently|moved|graduated)\b", text, re.IGNORECASE):
        return "persistent"
    return "unknown"


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
