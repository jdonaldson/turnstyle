"""ModelProfile — fingerprint + serialization. No LLM needed."""

import json

import pytest
import torch

from turnstyle.profile import (
    CALIBRATION_VERSION,
    ModelProfile,
    _probe_from_dict,
    _probe_to_dict,
    model_fingerprint,
)


class _Cfg:
    model_type = "fake"
    hidden_size = 4
    num_hidden_layers = 2
    vocab_size = 10
    _name_or_path = "fake/model"


class _FakeModel:
    """Minimal stand-in: a config + a state_dict + a dtype is all the fingerprint reads."""

    def __init__(self, seed: int = 0, dtype=torch.float32):
        g = torch.Generator().manual_seed(seed)
        self._sd = {
            "a.weight": torch.randn(4, 4, generator=g),
            "b.weight": torch.randn(10, 4, generator=g),
        }
        self.config = _Cfg()
        self.dtype = dtype

    def state_dict(self):
        return self._sd


# ── fingerprint ──────────────────────────────────────────────────────────────

def test_fingerprint_deterministic():
    m = _FakeModel(seed=1)
    assert model_fingerprint(m) == model_fingerprint(m)


def test_fingerprint_weight_sensitive():
    # different weights (a fine-tune) → different profile key
    assert model_fingerprint(_FakeModel(seed=1)) != model_fingerprint(_FakeModel(seed=2))


def test_fingerprint_dtype_sensitive():
    # fp16 vs fp32 give different activations → different key
    a = model_fingerprint(_FakeModel(seed=1, dtype=torch.float32))
    b = model_fingerprint(_FakeModel(seed=1, dtype=torch.float16))
    assert a != b


# ── profile (de)serialization ────────────────────────────────────────────────

def test_profile_roundtrip(tmp_path):
    prof = ModelProfile(fingerprint="abc123", model_id="fake/model")
    prof.support["arith"] = {"method": "structural", "shipped": True}
    path = prof.save(tmp_path / "abc123.json")

    back = ModelProfile.load(path)
    assert back.fingerprint == "abc123"
    assert back.model_id == "fake/model"
    assert back.calibration_version == CALIBRATION_VERSION
    assert back.support["arith"]["shipped"] is True


def test_get_probe_missing_returns_none():
    assert ModelProfile(fingerprint="x", model_id="y").get_probe("nope") is None


def test_probe_dict_roundtrip():
    """A fitted probe survives _probe_to_dict → JSON → _probe_from_dict intact."""
    pytest.importorskip("sklearn")
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from turnstyle.autoprobe import DEFAULT_FINDERS, ProbeArtifact

    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y = [0, 1, 0, 1]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression().fit(sc.transform(X), y)

    finder_name = next(iter(DEFAULT_FINDERS))
    art = ProbeArtifact(
        finder=DEFAULT_FINDERS[finder_name], layer=3, mode="per_option",
        classes=["positive"], scaler=sc, classifier=clf, answer_format="letter_paren",
    )

    d = json.loads(json.dumps(_probe_to_dict(art, finder_name)))  # through JSON
    art2 = _probe_from_dict(d)

    assert art2.layer == 3
    assert art2.mode == "per_option"
    assert art2.answer_format == "letter_paren"
    np.testing.assert_allclose(art2.scaler.mean_, sc.mean_)
    np.testing.assert_allclose(art2.classifier.coef_, clf.coef_)
    np.testing.assert_allclose(art2.classifier.intercept_, clf.intercept_)
