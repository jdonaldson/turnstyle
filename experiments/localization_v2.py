"""A null-calibrated, shape-aware localization statistic — replacing `spread`.

`spread` (max-min over layers) was retracted: shuffled labels produce MORE spread
than real ones (mean null .586, every frame inside the null 95th pct), because a
meaningless target gives r near zero bouncing layer to layer while a real target
gives r high and stable. Spread measured fit instability, not localization.

What it missed is SHAPE. A null curve has large range but is JAGGED — adjacent
layers uncorrelated. A genuinely localized frame varies SYSTEMATICALLY with depth:
it rises, peaks, and falls (or rises to a plateau), so neighbouring layers agree.

Three statistics, each z-scored against a label-permutation null so the scale is
calibrated rather than eyeballed:

  peak_r    max CV r over layers      — is there signal anywhere at all?
  smooth    lag-1 autocorrelation     — systematic vs jagged
  shape_r2  R^2 of a quadratic in normalized depth — how much of the curve is
            explained by depth itself (captures both a bump and a monotone rise)

A frame LOCALIZES iff its depth structure is above null (shape_z >= 2) AND there is
signal to localize (peak_z >= 2). Both gates, because a jagged curve can trend and
a flat-high curve has signal without structure.

Includes a SYNTHETIC GATE that runs first: a planted bump must score high and pure
noise must score low. If the instrument cannot separate those, nothing it says about
real frames is worth reading — the check skipped last time.

    uv run python experiments/localization_v2.py

Runs off cached Gemma 3n activations; no model load.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turnstyle.frame_library import CANONICAL_FRAMES, _cv_r  # noqa: E402

CACHE = Path(__file__).parent / "results" / "_gemma3n_acts"
RESULTS = Path(__file__).parent / "results"
N_PERM = 40
Z_GATE = 2.0


# ── the statistics ────────────────────────────────────────────────────────────

def smoothness(c):
    """Lag-1 autocorrelation. Systematic depth structure => neighbours agree."""
    a, b = c[:-1], c[1:]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def shape_r2(c):
    """Fraction of the curve's variance that SURVIVES smoothing — shape-agnostic.

    A quadratic fit was tried first and the synthetic gate rejected it: it scores a
    monotone rise at .986 but a narrow bump at only .414, so it would have called
    sharply-localized frames unstructured — exactly backwards. Smoothing is
    agnostic to WHICH shape: structured curves survive a moving average, noise does
    not.
    """
    if len(c) < 5:
        return 0.0
    k = np.ones(3) / 3.0
    sm = np.convolve(c, k, mode="valid")
    v_raw = float(np.var(c[1:-1]))
    return 0.0 if v_raw < 1e-12 else min(1.0, float(np.var(sm)) / v_raw)


def stats_of(c):
    return {"peak_r": float(np.nanmax(c)), "smooth": smoothness(c),
            "shape_r2": shape_r2(c)}


# ── synthetic gate: does the instrument work at all? ──────────────────────────

def synthetic_gate(rng, n_layers=31):
    print("=" * 80)
    print("SYNTHETIC GATE — can the statistic tell a planted bump from noise?")
    print("=" * 80)
    x = np.linspace(0, 1, n_layers)
    cases = {
        "localized bump":   0.3 + 0.6 * np.exp(-((x - 0.6) ** 2) / 0.02),
        "monotone rise":    0.2 + 0.7 * x,
        "flat high":        0.9 + np.zeros_like(x),
        "pure noise":       np.zeros_like(x),
    }
    print(f"  {'case':18s} {'peak_r':>7s} {'smooth':>7s} {'shape_r2':>9s}   expected")
    ok = True
    for name, base in cases.items():
        c = base + rng.normal(0, 0.03, n_layers)
        s = stats_of(c)
        if name == "pure noise":
            c = rng.normal(0, 0.25, n_layers)
            s = stats_of(c)
        exp = {"localized bump": "high shape+smooth", "monotone rise": "high shape+smooth",
               "flat high": "high peak, LOW shape", "pure noise": "low everything"}[name]
        print(f"  {name:18s} {s['peak_r']:>7.3f} {s['smooth']:>7.3f} "
              f"{s['shape_r2']:>9.3f}   {exp}")
        if name in ("localized bump", "monotone rise") and not (
                s["shape_r2"] >= 0.7 and s["smooth"] >= 0.7):
            ok = False
        if name == "pure noise" and (s["shape_r2"] > 0.5 or s["smooth"] > 0.4):
            ok = False
        if name == "flat high" and s["smooth"] > 0.4:
            ok = False       # flat-high has SIGNAL but must show no depth structure
    print(f"\n  gate: {'PASS — instrument separates structure from noise' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("synthetic gate failed; do not read the real results")
    print()
    return ok


# ── real frames ───────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(0)
    synthetic_gate(rng)

    files = sorted(glob.glob(str(CACHE / "*.npz")))
    if not files:
        raise SystemExit("no cached activations — run gemma3n_readout_sweep.py first")

    print("=" * 80)
    print(f"REAL FRAMES — z against {N_PERM} label permutations (Gemma 3n, stream0)")
    print("=" * 80)
    print(f"  {'frame':14s} {'peak_r':>7s} {'peak_z':>7s} {'shape':>7s} {'shape_z':>8s} "
          f"{'smooth':>7s}   verdict")
    out = {}
    for f in files:
        name = Path(f).stem
        if name not in CANONICAL_FRAMES:
            continue
        z = np.load(f)
        words = list(z["words"])
        y = np.array([CANONICAL_FRAMES[name]["data"][w] for w in words], float)
        n_layers = z[f"raw::{words[0]}"].shape[0]
        X = [np.array([z[f"raw::{w}"][L, 0, :] for w in words]) for L in range(n_layers)]

        def curve(yy):
            c = np.array([_cv_r(Xi, yy) for Xi in X], float)
            return np.nan_to_num(c, nan=0.0)

        real = stats_of(curve(y))
        null = [stats_of(curve(rng.permutation(y))) for _ in range(N_PERM)]
        zs = {}
        for k in ("peak_r", "shape_r2", "smooth"):
            v = np.array([n[k] for n in null])
            zs[k] = float((real[k] - v.mean()) / (v.std() + 1e-9))

        localizes = zs["shape_r2"] >= Z_GATE and zs["peak_r"] >= Z_GATE
        verdict = ("LOCALIZES" if localizes else
                   "signal, no depth structure" if zs["peak_r"] >= Z_GATE else
                   "no signal above null")
        out[name] = {**real, "peak_z": zs["peak_r"], "shape_z": zs["shape_r2"],
                     "smooth_z": zs["smooth"], "localizes": bool(localizes)}
        print(f"  {name:14s} {real['peak_r']:>7.3f} {zs['peak_r']:>7.1f} "
              f"{real['shape_r2']:>7.3f} {zs['shape_r2']:>8.1f} {real['smooth']:>7.3f}"
              f"   {verdict}", flush=True)

    loc = sorted(k for k, v in out.items() if v["localizes"])
    sig = sorted(k for k, v in out.items() if v["peak_z"] >= Z_GATE and not v["localizes"])
    print(f"\n  LOCALIZES (depth structure above null): {loc or 'none'}")
    print(f"  signal but NO depth structure:          {sig or 'none'}")
    print(f"  no signal above null:                   "
          f"{sorted(k for k, v in out.items() if v['peak_z'] < Z_GATE) or 'none'}")

    p = RESULTS / "localization_v2.json"
    p.write_text(json.dumps({"n_perm": N_PERM, "z_gate": Z_GATE, "frames": out}, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
