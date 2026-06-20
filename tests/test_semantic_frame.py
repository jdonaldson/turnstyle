"""BipolarAxis / SemanticFrame — contrastive fit, projection, serialization.

No model needed: fit from synthetic vectors, project, round-trip."""

import numpy as np

from turnstyle.semantic_frame import (
    BipolarAxis, SemanticFrame, fit_axis_from_vectors,
)


def _temp_axis():
    # high pole at +x, low at -x; off-axis dims symmetric so the fit direction is
    # pure-x (real activations have comparable per-feature variance — this mimics it)
    high = np.array([[2.0, 0.3, -0.3], [2.0, -0.3, 0.3], [2.0, 0.3, 0.3], [2.0, -0.3, -0.3]])
    low = np.array([[-2.0, 0.3, -0.3], [-2.0, -0.3, 0.3], [-2.0, 0.3, 0.3], [-2.0, -0.3, -0.3]])
    return fit_axis_from_vectors("temperature", "cold", "hot", 14, high, low)


def test_axis_signs():
    ax = _temp_axis()
    assert ax.project(np.array([3.0, 0, 0])) > 0      # toward high pole
    assert ax.project(np.array([-3.0, 0, 0])) < 0     # toward low pole
    assert ax.pole(np.array([3.0, 0, 0])) == "hot"
    assert ax.pole(np.array([-3.0, 0, 0])) == "cold"


def test_axis_centered_on_boundary():
    ax = _temp_axis()
    # the midpoint between the pole means projects to ~0
    assert abs(ax.project(np.array([0.0, 0.05, 0.05]))) < 0.3


def test_axis_roundtrip():
    ax = _temp_axis()
    bx = BipolarAxis.from_dict(ax.to_dict())
    for v in (np.array([2.0, 0, 0]), np.array([-1.0, 0.5, 0.2])):
        assert np.isclose(bx.project(v), ax.project(v))
    assert bx.high_label == "hot" and bx.low_label == "cold"


def test_frame_coordinates_stack():
    ax1 = _temp_axis()
    # a second, orthogonal axis on +y (symmetric off-axis dims → pure-y direction)
    hi = np.array([[0.3, 2.0, -0.3], [-0.3, 2.0, 0.3], [0.3, 2.0, 0.3], [-0.3, 2.0, -0.3]])
    lo = np.array([[0.3, -2.0, -0.3], [-0.3, -2.0, 0.3], [0.3, -2.0, 0.3], [-0.3, -2.0, -0.3]])
    ax2 = fit_axis_from_vectors("size", "small", "big", 14, hi, lo)
    frame = SemanticFrame(axes=[ax1, ax2], layer=14)
    assert frame.names == ["temperature", "size"]
    coords = frame.coordinates(np.array([3.0, 3.0, 0.0]))
    assert coords.shape == (2,)
    assert coords[0] > 0 and coords[1] > 0            # hot AND big
    mat = frame.project_matrix(np.array([[3.0, 3.0, 0.0], [-3.0, -3.0, 0.0]]))
    assert mat.shape == (2, 2)
    assert mat[1, 0] < 0 and mat[1, 1] < 0            # cold AND small


def test_frame_roundtrip():
    frame = SemanticFrame(axes=[_temp_axis()], layer=14)
    g = SemanticFrame.from_dict(frame.to_dict())
    assert g.layer == 14 and g.names == ["temperature"]
    assert np.isclose(g.coordinates(np.array([2.0, 0, 0]))[0],
                      frame.coordinates(np.array([2.0, 0, 0]))[0])
