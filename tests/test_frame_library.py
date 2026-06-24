"""No-model tests for FrameLibrary: pure-numpy ridge/CV math, BipolarAxis fitting from
graded data, projection, orthogonality on synthetic directions, and JSON round-trip."""
from __future__ import annotations

import numpy as np

import turnstyle.frame_library as fl
from turnstyle.frame_library import (Frame, FrameLibrary, CANONICAL_FRAMES,
                                     _ridge_dir, _cv_r, _axis_from_scalar,
                                     save_library, load_library)
from turnstyle.semantic_frame import BipolarAxis


def _mk_frame(name, layer=0, H=6):
    ax = BipolarAxis(name, "-", "+", layer, np.zeros(H), np.ones(H), np.eye(H)[0], 0.0)
    return Frame(ax, data={"a": 0, "b": 1})


def test_ridge_dir_recovers_planted_direction():
    rng = np.random.default_rng(0)
    H, N = 50, 60
    true = rng.standard_normal(H); true /= np.linalg.norm(true)
    Z = rng.standard_normal((N, H))
    y = Z @ true + 0.01 * rng.standard_normal(N)
    d = _ridge_dir(Z, y, alpha=1.0)
    assert abs(float(d @ true)) > 0.9          # recovered (up to sign)


def test_cv_r_high_for_signal_low_for_noise():
    rng = np.random.default_rng(1)
    H, N = 40, 80
    w = rng.standard_normal(H)
    X = rng.standard_normal((N, H))
    y = X @ w
    assert _cv_r(X, y) > 0.8
    assert abs(_cv_r(X, rng.standard_normal(N))) < 0.5    # noise target


def test_axis_from_scalar_orders_by_value():
    rng = np.random.default_rng(2)
    H = 30
    axisdir = rng.standard_normal(H)
    y = np.linspace(-3, 3, 12)
    X = np.outer(y, axisdir) + 0.05 * rng.standard_normal((12, H))
    ax = _axis_from_scalar("v", X, y, layer=5, low_label="lo", high_label="hi")
    projs = np.array([ax.project(X[i]) for i in range(12)])
    # projection must increase with the planted value
    assert np.corrcoef(projs, y)[0, 1] > 0.95
    assert ax.layer == 5 and ax.high_label == "hi"


def test_frame_and_library_json_roundtrip():
    H = 8
    ax = BipolarAxis("size", "tiny", "huge", 4, np.zeros(H), np.ones(H),
                     np.eye(H)[0], 0.0)
    lib = FrameLibrary().add(Frame(ax, template="It is a {w}.", cv_r=0.9,
                                   data={"tiny": -1, "huge": 1}))
    lib2 = FrameLibrary.from_dict(lib.to_dict())
    assert lib2.names == ["size"]
    f = lib2.frames["size"]
    assert f.cv_r == 0.9 and f.layer == 4 and f.template == "It is a {w}."
    v = np.zeros(H); v[0] = 2.0
    assert f.project(v) == lib.frames["size"].project(v)


def test_library_save_load(tmp_path):
    H = 6
    ax = BipolarAxis("age", "new", "old", 2, np.zeros(H), np.ones(H), np.eye(H)[1], 0.0)
    lib = FrameLibrary().add(Frame(ax, data={"new": 0, "old": 1}))
    p = lib.save(tmp_path / "frames.json")
    lib2 = FrameLibrary.load(p)
    assert "age" in lib2 and len(lib2) == 1


def test_npz_roundtrip_preserves_projection(tmp_path):
    H = 12
    rng = np.random.default_rng(3)
    d = rng.standard_normal(H); d /= np.linalg.norm(d)
    ax = BipolarAxis("size", "tiny", "huge", 7, rng.standard_normal(H),
                     np.abs(rng.standard_normal(H)) + 0.1, d, 0.5)
    lib = FrameLibrary(fingerprint="fp9", model_id="toy").add(
        Frame(ax, template="It is a {w}.", pool="last", cv_r=0.9, data={"tiny": -1}))
    p = lib.save_npz(tmp_path / "frames.npz")
    lib2 = FrameLibrary.load_npz(p)
    assert lib2.fingerprint == "fp9" and lib2.frames["size"].layer == 7
    assert lib2.frames["size"].template == "It is a {w}." and lib2.frames["size"].cv_r == 0.9
    v = rng.standard_normal(H)
    # float32 storage → projections agree to float precision
    assert abs(lib.frames["size"].project(v) - lib2.frames["size"].project(v)) < 1e-3


def test_orthogonality_math_on_synthetic_axes():
    # two BipolarAxis with known orthogonal vs aligned directions, project-only check
    H = 10
    d0, d1 = np.eye(H)[0], np.eye(H)[1]
    a = BipolarAxis("a", "-", "+", 0, np.zeros(H), np.ones(H), d0, 0.0)
    b = BipolarAxis("b", "-", "+", 0, np.zeros(H), np.ones(H), d1, 0.0)
    assert abs(float(a.direction @ b.direction)) < 1e-9      # orthogonal
    c = BipolarAxis("c", "-", "+", 0, np.zeros(H), np.ones(H), d0.copy(), 0.0)
    assert abs(float(a.direction @ c.direction)) > 0.99      # aligned


def test_fingerprint_store_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "_USER_FRAMES", tmp_path / "user")
    monkeypatch.setattr(fl, "_BUNDLED_FRAMES", tmp_path / "bundled")
    lib = FrameLibrary().add(_mk_frame("size"))
    save_library(lib, "fp123", model_id="toy")
    got = load_library("fp123")
    assert got is not None and got.names == ["size"] and got.fingerprint == "fp123"
    assert load_library("nope") is None


def test_fingerprint_store_user_overlays_bundled(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "_USER_FRAMES", tmp_path / "user")
    monkeypatch.setattr(fl, "_BUNDLED_FRAMES", tmp_path / "bundled")
    # bundled ships size+age; user re-fits size (layer 9) and adds color
    bundled = FrameLibrary().add(_mk_frame("size", layer=1)).add(_mk_frame("age"))
    bundled.save(tmp_path / "bundled" / "fp.json")
    user = FrameLibrary().add(_mk_frame("size", layer=9)).add(_mk_frame("color"))
    user.save(tmp_path / "user" / "fp.json")
    merged = load_library("fp")
    assert set(merged.names) == {"size", "age", "color"}     # union
    assert merged.frames["size"].layer == 9                  # user wins per-frame
    assert "age" in merged                                   # bundled fills the gap


def test_canonical_frames_wellformed():
    assert {"opinion", "size", "age", "shape", "space", "material",
            "number", "time"} <= set(CANONICAL_FRAMES)
    for name, spec in CANONICAL_FRAMES.items():
        assert len(spec["data"]) >= 8                         # enough words to CV
        assert len(set(spec["data"].values())) >= 2           # a real gradient
