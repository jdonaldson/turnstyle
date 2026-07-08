"""No-model tests for turnstyle.operators (NegationDetector + OperatorTable)."""
from __future__ import annotations

import numpy as np
import pytest

from turnstyle.operators import NegationDetector, OperatorTable


def _detector(h=8, z_negated=3.5, z_clear=2.0):
    d = np.zeros(h)
    d[0] = 1.0
    return NegationDetector(layer=14, direction=d, mu0=np.ones(h), s0=2.0,
                            z_negated=z_negated, z_clear=z_clear)


def test_score_is_projection_in_sd_units():
    det = _detector()
    state = np.ones(8)
    state[0] += 5.0                      # 5 raw units along the hedge dir
    assert det.score(state) == pytest.approx(2.5)   # / s0
    assert det.score(np.ones(8)) == pytest.approx(0.0)


def test_detect_bands():
    det = _detector()
    hi = np.ones(8); hi[0] += 2.0 * 4.0        # z = 4.0 -> negated
    lo = np.ones(8); lo[0] += 2.0 * 1.0        # z = 1.0 -> clear
    mid = np.ones(8); mid[0] += 2.0 * 2.6      # z = 2.6 -> abstain
    assert det.detect(hi) is True
    assert det.detect(lo) is False
    assert det.detect(mid) is None


def test_detector_round_trip():
    det = _detector()
    det.meta = {"fit_frames": ["warmth"], "n_pairs": 15}
    back = NegationDetector.from_dict(det.to_dict())
    assert back.layer == det.layer
    assert back.s0 == pytest.approx(det.s0)
    assert np.allclose(back.direction, det.direction)
    assert back.meta == det.meta
    state = np.ones(8); state[0] += 3.0
    assert back.score(state) == pytest.approx(det.score(state))


def test_operator_table_apply():
    t = OperatorTable(table={"warmth": {"not": {"slope": 0.4,
                                                "intercept": -3.4, "r2": 0.69}}})
    assert t.apply("warmth", "not", 10.0) == pytest.approx(0.6)


def test_operator_chain_surface_order():
    # "not very X": very innermost, not outermost -> [very, not]
    t = OperatorTable(table={"f": {
        "very": {"slope": 0.9, "intercept": 1.0, "r2": 0.99},
        "not": {"slope": 0.5, "intercept": -2.0, "r2": 0.9}}})
    c = t.apply_chain("f", ["very", "not"], 4.0)
    assert c == pytest.approx(0.5 * (0.9 * 4.0 + 1.0) - 2.0)


def test_double_negation_composes_in_code():
    # the model collapses "not un-X"; symbolic application does not: two
    # applications of an attenuating-negation affine walk back toward the
    # hedge fixed point, they do NOT stick at the single-negation landing.
    t = OperatorTable(table={"f": {"not": {"slope": 0.4, "intercept": -3.0,
                                           "r2": 0.7}}})
    once = t.apply("f", "not", 10.0)
    twice = t.apply("f", "not", once)
    assert once == pytest.approx(1.0)
    assert twice == pytest.approx(-2.6)
    assert twice != once


def test_store_round_trip(tmp_path, monkeypatch):
    from turnstyle import operators

    monkeypatch.setattr(operators, "_store_path",
                        lambda fp: tmp_path / f"{fp}.negation.json")
    det = _detector()
    p = operators.save_detector(det, "fp123")
    assert p.exists()
    back = operators.load_detector("fp123")
    assert back is not None and back.layer == det.layer
    state = np.ones(8); state[0] += 4.0
    assert back.score(state) == pytest.approx(det.score(state))
    assert operators.load_detector("missing-fp") is None


def test_from_results_json(tmp_path):
    payload = {"frames": {"size": {"ops": {"not": {"slope": 0.75,
                                                   "intercept": -0.07,
                                                   "r2": 0.99}}}}}
    p = tmp_path / "res.json"
    p.write_text(__import__("json").dumps(payload))
    t = OperatorTable.from_results_json(p)
    assert t.apply("size", "not", 2.0) == pytest.approx(1.43)
