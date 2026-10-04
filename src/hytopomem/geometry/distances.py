from __future__ import annotations

import numpy as np

from hytopomem.geometry.lorentz import LorentzManifold


def lorentz_distance(x: np.ndarray, y: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    return LorentzManifold(curvature=curvature).distance(x, y)

