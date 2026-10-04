from __future__ import annotations

import math
from dataclasses import dataclass


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(frozen=True)
class AnchorConfidenceSignals:
    support_score: float = 0.0
    repeat_score: float = 0.0
    source_score: float = 0.0
    recency_score: float = 0.0
    conflict_score: float = 0.0
    uncertainty_score: float = 0.0


def anchor_confidence(signals: AnchorConfidenceSignals) -> float:
    raw = (
        1.2 * signals.support_score
        + 0.8 * signals.repeat_score
        + 1.0 * signals.source_score
        + 0.5 * signals.recency_score
        - 1.2 * signals.conflict_score
        - 0.8 * signals.uncertainty_score
    )
    return sigmoid(raw)


def source_score(source: str) -> float:
    source = source.lower()
    if "raw" in source:
        return 1.0
    if "summary" in source:
        return 0.7
    if "inferred" in source or "llm" in source:
        return 0.55
    return 0.4

