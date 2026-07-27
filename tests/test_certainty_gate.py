"""No-model tests for the certainty abstention gate.

The gate is the frame family's first CONTROL-surface consumer, so the tests focus on
the properties that make it safe to wire into `solve()`: the coordinate matches the
frame's own projection, the threshold is a calibrated quantile rather than a magic
number, and it round-trips so a shipped gate carries its provenance.
"""
from __future__ import annotations

import numpy as np

from turnstyle.certainty_gate import CertaintyGate, accuracy_at_coverage
from turnstyle.frame_library import Frame
from turnstyle.semantic_frame import BipolarAxis


def _frame(H=6, layer=16):
    d = np.zeros(H)
    d[0] = 1.0
    ax = BipolarAxis("certainty", "impossible", "certain", layer,
                     np.zeros(H), np.ones(H), d, 0.0)
    return Frame(ax, data={"impossible": -3, "certain": 3}, coord_scale=1.0)


def test_coord_matches_the_frames_own_projection():
    """The gate must read the SAME axis the library measures — if these diverge, a
    gate calibrated on frame coordinates would threshold something else."""
    f = _frame()
    g = CertaintyGate.from_frame(f, threshold=0.0)
    for v in (np.array([2.0, 0, 0, 0, 0, 0]), np.array([-1.5, 3, 1, 0, 0, 0])):
        assert abs(g.coord(v) - f.project_norm(v)) < 1e-9


def test_should_abstain_is_below_threshold():
    g = CertaintyGate.from_frame(_frame(), threshold=0.5)
    assert g.should_abstain(np.array([0.1, 0, 0, 0, 0, 0])) is True      # unsure
    assert g.should_abstain(np.array([2.0, 0, 0, 0, 0, 0])) is False     # confident


def test_threshold_is_a_coverage_quantile():
    coords = np.arange(100, dtype=float)
    t80 = CertaintyGate.pick_threshold(coords, coverage=0.8)
    kept = (coords >= t80).mean()
    assert 0.75 <= kept <= 0.85          # keeps ~80%
    # full coverage must abstain on nothing
    t100 = CertaintyGate.pick_threshold(coords, coverage=1.0)
    assert (coords >= t100).all()


def test_threshold_rejects_bad_coverage():
    for bad in (0.0, -0.1, 1.5):
        try:
            CertaintyGate.pick_threshold([1.0, 2.0], coverage=bad)
        except ValueError:
            continue
        raise AssertionError(f"coverage={bad} should have raised")


def test_accuracy_at_coverage_prefers_confident_examples():
    # certainty perfectly ordered with correctness -> tightening coverage must help
    coords = np.array([0.0, 1.0, 2.0, 3.0])
    correct = np.array([0, 0, 1, 1])
    assert accuracy_at_coverage(coords, correct, 1.0) == 0.5
    assert accuracy_at_coverage(coords, correct, 0.5) == 1.0


def test_accuracy_at_coverage_flat_for_useless_signal():
    rng = np.random.default_rng(0)
    correct = rng.integers(0, 2, 400)
    coords = rng.normal(size=400)                    # independent of correctness
    base = correct.mean()
    at50 = accuracy_at_coverage(coords, correct, 0.5)
    assert abs(at50 - base) < 0.12                   # no free lunch from noise


def test_round_trip_preserves_gate_and_provenance():
    g = CertaintyGate.from_frame(_frame(), threshold=0.25,
                                 fingerprint="abc123", auc=0.653, n=200)
    back = CertaintyGate.from_dict(g.to_dict())
    assert back.threshold == g.threshold and back.layer == g.layer
    assert back.model_fingerprint == "abc123"
    # provenance must survive: a shipped gate has to be auditable without re-deriving
    assert back.auc == 0.653 and back.n_calibration == 200
    v = np.array([1.0, 2, 3, 0, 0, 0])
    assert abs(back.coord(v) - g.coord(v)) < 1e-9
