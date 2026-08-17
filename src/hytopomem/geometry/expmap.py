from __future__ import annotations

import numpy as np

from hytopomem.geometry.lorentz import LorentzManifold


def expmap0(tangent: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    return LorentzManifold(curvature=curvature).expmap0(tangent)

