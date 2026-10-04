from __future__ import annotations

import numpy as np

from hytopomem.geometry.lorentz import LorentzManifold


def lorentz_to_poincare(x: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    return LorentzManifold(curvature=curvature).to_poincare(x)

