"""No-model tests for the route-classification probe (linear params + softmax)."""
import numpy as np

from turnstyle.route import RouteProbe


def _probe():
    return RouteProbe(layer=5, classes=["a", "b", "c"], mean=[0.0, 0.0], std=[1.0, 1.0],
                      coef=[[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]],
                      intercept=[0.0, 0.0, 0.0], threshold=0.8)


def test_roundtrip():
    p = _probe()
    q = RouteProbe.from_dict(p.to_dict())
    assert q.layer == 5 and q.classes == ["a", "b", "c"] and q.threshold == 0.8
    assert np.allclose(q.coef, p.coef) and np.allclose(q.std, p.std)


def test_scores_are_softmax():
    p = _probe()
    s = p._scores(np.array([1.0, 0.0], dtype=np.float32))
    assert abs(float(s.sum()) - 1.0) < 1e-6
    assert s.argmax() == 0          # input aligned with class "a"'s coef row


def test_profile_router_roundtrip():
    from turnstyle.profile import ModelProfile
    prof = ModelProfile(fingerprint="x", model_id="m")
    assert prof.get_router() is None
    prof.set_router(_probe())
    d = prof.to_dict()
    assert "router" in d and d["router"]["classes"] == ["a", "b", "c"]
    r = prof.get_router()
    assert r is not None and r.classes == ["a", "b", "c"]
