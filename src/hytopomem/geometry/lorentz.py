from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPS = 1e-7


def arcosh(value: np.ndarray | float) -> np.ndarray | float:
    return np.arccosh(np.maximum(value, 1.0 + EPS))


@dataclass(frozen=True)
class LorentzManifold:
    """Lorentz model with positive curvature parameter c = -K."""

    curvature: float = 1.0

    def __post_init__(self) -> None:
        if self.curvature <= 0:
            raise ValueError("curvature must be positive")

    @property
    def origin(self) -> np.ndarray:
        return np.array([1.0 / math.sqrt(self.curvature), 0.0], dtype=np.float64)

    def origin_like(self, dim: int) -> np.ndarray:
        if dim < 2:
            raise ValueError("Lorentz point dimension must be at least 2")
        point = np.zeros(dim, dtype=np.float64)
        point[0] = 1.0 / math.sqrt(self.curvature)
        return point

    def inner(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return -x[..., 0] * y[..., 0] + np.sum(x[..., 1:] * y[..., 1:], axis=-1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        spatial_sq = np.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)
        time = np.sqrt((1.0 / self.curvature) + spatial_sq)
        return np.concatenate([time, x[..., 1:]], axis=-1)

    def expmap0(self, tangent: np.ndarray) -> np.ndarray:
        tangent = np.asarray(tangent, dtype=np.float64)
        if tangent.ndim == 1:
            tangent = tangent[None, :]
            squeeze = True
        else:
            squeeze = False
        spatial_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
        sqrt_c = math.sqrt(self.curvature)
        coef_time = np.cosh(sqrt_c * spatial_norm) / sqrt_c
        coef_spatial = np.sinh(sqrt_c * spatial_norm) / np.maximum(sqrt_c * spatial_norm, EPS)
        point = np.concatenate([coef_time, coef_spatial * tangent], axis=-1)
        point = self.project(point)
        return point[0] if squeeze else point

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        prod = -self.curvature * self.inner(x, y)
        return arcosh(prod) / math.sqrt(self.curvature)

    def radius(self, x: np.ndarray) -> np.ndarray:
        return self.distance(self.origin_like(np.asarray(x).shape[-1]), x)

    def to_poincare(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        denom = x[..., :1] + (1.0 / math.sqrt(self.curvature))
        return x[..., 1:] / np.maximum(denom, EPS)


def cone_aperture(confidence: float, psi_min: float = 0.1, psi_max: float = 1.0) -> float:
    confidence = min(1.0, max(0.0, confidence))
    return psi_min + (1.0 - confidence) * (psi_max - psi_min)


def cone_violation(parent: np.ndarray, child: np.ndarray, confidence: float) -> float:
    p = np.asarray(parent, dtype=np.float64)[1:]
    c = np.asarray(child, dtype=np.float64)[1:]
    denom = max(float(np.linalg.norm(p) * np.linalg.norm(c)), EPS)
    angle = math.acos(max(-1.0, min(1.0, float(np.dot(p, c) / denom))))
    return max(0.0, angle - cone_aperture(confidence))

