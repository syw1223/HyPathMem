from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PathScoreWeights:
    relevance: float = 1.0
    geometry: float = 0.7
    temporal: float = 0.3
    support: float = 0.5
    conflict: float = 0.8
    redundancy: float = 0.25
    token_cost: float = 0.002
    length_cost: float = 0.05

