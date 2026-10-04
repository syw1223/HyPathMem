import numpy as np

from hytopomem.geometry.lorentz import LorentzManifold


def test_expmap_projects_to_lorentz_manifold():
    manifold = LorentzManifold(curvature=1.0)
    point = manifold.expmap0(np.array([0.2, -0.1, 0.3]))
    assert point[0] > 0
    assert abs(float(manifold.inner(point, point)) + 1.0) < 1e-6


def test_distance_is_non_negative():
    manifold = LorentzManifold(curvature=1.0)
    x = manifold.expmap0(np.array([0.1, 0.0]))
    y = manifold.expmap0(np.array([0.0, 0.2]))
    assert float(manifold.distance(x, y)) >= 0.0

