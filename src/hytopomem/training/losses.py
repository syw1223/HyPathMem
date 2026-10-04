from __future__ import annotations

from typing import Iterable

import numpy as np

from hytopomem.geometry.lorentz import LorentzManifold, cone_violation


def relevance_contrastive_loss(
    query: np.ndarray,
    positive: np.ndarray,
    negatives: Iterable[np.ndarray],
    temperature: float = 0.1,
    curvature: float = 1.0,
) -> float:
    manifold = LorentzManifold(curvature=curvature)
    pos_logit = -float(manifold.distance(query, positive)) / temperature
    neg_logits = [-float(manifold.distance(query, neg)) / temperature for neg in negatives]
    logits = np.array([pos_logit] + neg_logits, dtype=np.float64)
    logits = logits - np.max(logits)
    return float(-(logits[0] - np.log(np.exp(logits).sum())))


def radial_order_loss(
    child: np.ndarray,
    parent: np.ndarray,
    margin: float = 0.1,
    curvature: float = 1.0,
) -> float:
    manifold = LorentzManifold(curvature=curvature)
    return max(0.0, margin + float(manifold.radius(parent)) - float(manifold.radius(child)))


def cone_order_loss(parent: np.ndarray, child: np.ndarray, confidence: float) -> float:
    return cone_violation(parent, child, confidence)


def update_margin_loss(new_score: float, old_score: float, margin: float = 0.1) -> float:
    return max(0.0, margin - new_score + old_score)


def stability_loss(previous_distance: float, current_distance: float) -> float:
    return abs(current_distance - previous_distance)

