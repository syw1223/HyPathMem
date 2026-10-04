from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class CalibrationWeights:
    ce: float = 1.0
    bm25: float = 0.15
    entity: float = 0.08
    seed_prior: float = 0.18
    sibling_prior: float = 0.10
    edge: float = 0.04
    degree_penalty: float = 0.06
    hop_penalty: float = 0.08
    redundancy_penalty: float = 0.0


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def calibrate_path(path: dict, weights: CalibrationWeights) -> dict:
    calibrated = dict(path)
    scores = dict(path.get("scores", {}))
    metadata = dict(path.get("metadata", {}))

    ce_score = as_float(scores.get("cross_encoder", path.get("score", 0.0)))
    bm25_norm = as_float(metadata.get("bm25_norm", 0.0))
    entity_overlap = as_float(metadata.get("entity_overlap", 0.0))
    anchor_degree = as_int(metadata.get("anchor_degree", 0))
    hop = as_int(metadata.get("hop", 0))
    is_seed = str(metadata.get("is_seed", "")).lower() == "true"
    source = str(metadata.get("candidate_source", ""))
    edge_score = as_float(scores.get("edge", 0.0))

    source_prior = 0.0
    if is_seed or source == "bm25_fact":
        source_prior += weights.seed_prior
    if source in {"filtered_sibling", "seed_sibling"} and not is_seed:
        source_prior += weights.sibling_prior

    calibrated_score = (
        weights.ce * ce_score
        + weights.bm25 * bm25_norm
        + weights.entity * entity_overlap
        + source_prior
        + weights.edge * edge_score
        - weights.degree_penalty * math.log1p(anchor_degree)
        - weights.hop_penalty * hop
    )
    scores["calibrated"] = calibrated_score
    scores["source_prior"] = source_prior
    scores["degree_penalty"] = math.log1p(anchor_degree)
    scores["hop_penalty"] = float(hop)
    calibrated["score"] = calibrated_score
    calibrated["scores"] = scores
    metadata["calibrated"] = "true"
    calibrated["metadata"] = metadata
    return calibrated


def calibrate_item(item: dict, top_k: int, weights: CalibrationWeights) -> dict:
    calibrated = dict(item)
    paths = [calibrate_path(path, weights) for path in item.get("paths", [])]
    paths = sorted(paths, key=lambda path: path.get("score", 0.0), reverse=True)
    calibrated["paths"] = paths[:top_k]
    metadata = dict(item.get("metadata", {}))
    metadata["calibrated_top_k"] = top_k
    calibrated["metadata"] = metadata
    return calibrated


def calibrate_items(items: List[dict], top_k: int, weights: CalibrationWeights) -> List[dict]:
    return [calibrate_item(item, top_k, weights) for item in items]

