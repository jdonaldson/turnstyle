"""PolarityProbe inference + serialization. No model/sklearn needed."""

import numpy as np

from turnstyle.polarity import (
    HIGH, LOW, POLARITY_LEXICON, PolarityCapability, PolarityProbe,
)


def _toy_probe():
    # 3-d space; polarity lives entirely on axis 0 (positive = HIGH)
    return PolarityProbe(
        layer=11,
        mean=np.zeros(3),
        scale=np.ones(3),
        coef=np.array([2.0, 0.0, 0.0]),
        intercept=0.0,
        capability=PolarityCapability(
            layer=11, loo_axis=0.92,
            per_axis={"size": 1.0, "value": 1.0, "age": 0.35}, ship=True),
        lexicon_axes=["size", "value", "age"],
    )


def test_pole_sign():
    p = _toy_probe()
    assert p.pole(np.array([1.0, 0, 0])) == HIGH
    assert p.pole(np.array([-1.0, 0, 0])) == LOW
    # orthogonal direction → near the boundary, leans on intercept (0 → LOW)
    assert p.pole(np.array([0.0, 5.0, 0])) == LOW


def test_confidence_monotone():
    p = _toy_probe()
    assert p.confidence(np.array([3.0, 0, 0])) > p.confidence(np.array([0.5, 0, 0]))


def test_standardization_applied():
    p = PolarityProbe(layer=0, mean=np.array([10.0]), scale=np.array([2.0]),
                      coef=np.array([1.0]), intercept=0.0)
    # input 10 → standardized 0 → score 0 → LOW; input 14 → z=2 → HIGH
    assert p.pole(np.array([10.0])) == LOW
    assert p.pole(np.array([14.0])) == HIGH


def test_roundtrip():
    p = _toy_probe()
    q = PolarityProbe.from_dict(p.to_dict())
    assert q.layer == p.layer
    np.testing.assert_allclose(q.coef, p.coef)
    np.testing.assert_allclose(q.mean, p.mean)
    assert q.capability.ship is True
    assert q.capability.per_axis["age"] == 0.35
    # predictions identical after round-trip
    for v in (np.array([1.0, 0, 0]), np.array([-3.0, 1, 2])):
        assert q.pole(v) == p.pole(v)


def test_capability_axes_shipping():
    cap = PolarityCapability(layer=11, loo_axis=0.92,
                             per_axis={"size": 1.0, "value": 0.98, "age": 0.35},
                             ship=True)
    shipping = cap.axes_shipping(gate=0.75)
    assert "size" in shipping and "value" in shipping
    assert "age" not in shipping


def test_lexicon_wellformed():
    # every entry is (positive, comparative, superlative, pole∈{+1,-1})
    for axis, words in POLARITY_LEXICON.items():
        for entry in words:
            assert len(entry) == 4
            *_forms, pole = entry
            assert pole in (HIGH, LOW)
            assert all(isinstance(f, str) and f for f in _forms)
