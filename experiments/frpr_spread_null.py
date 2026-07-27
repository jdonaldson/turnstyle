"""frpr: is `spread` calibrated? And is the L0/peak law definitional?

Two re-derivations of claims this thread has been building on, with prior reasoning
suspended.

CHECK 1 -- the null distribution of `spread`.
Every FLAT/real verdict here rests on curve spread against a cutoff picked by eye
(0.20), and spread has NEVER been compared to what a MEANINGLESS target produces.
A CV-r curve fitted to shuffled labels still wanders layer to layer from estimation
variance alone, and at n=12 words that wander could be large. If the null spread is
~0.15 then "concreteness never localizes" (max .133 across six models) is just
"concreteness sits at the noise floor" -- true but differently meaningful -- and any
frame near the cutoff is unclassifiable.

CHECK 2 -- is corr(L0/peak, spread) = -0.945 definitional?
spread = max(c[1:]) - min(c[1:]), which excludes L0, so it is not circular. But a
curve starting low has more room to rise. If `min(c[1:])/max` predicts spread just
as well as `c[0]/max`, then the "law" is only "curves with a low point vary more"
and L0 -- the embedding, the part with the mechanistic story -- adds nothing.
The test is whether L0 survives partialling out the curve's own minimum.

Runs off the cached Gemma 3n activations; no model load.

    uv run python experiments/frpr_spread_null.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turnstyle.frame_library import CANONICAL_FRAMES, _cv_r  # noqa: E402

CACHE = Path(__file__).parent / "results" / "_gemma3n_acts"
RESULTS = Path(__file__).parent / "results"
N_PERM = 40


def curve_spread(X_by_layer, y):
    c = np.array([_cv_r(X, y) for X in X_by_layer], float)
    c[np.isnan(c)] = -9.0
    c = np.where(c < -1, np.nan, c)
    return float(np.nanmax(c[1:]) - np.nanmin(c[1:]))


def main():
    files = sorted(glob.glob(str(CACHE / "*.npz")))
    if not files:
        raise SystemExit("no cached activations — run gemma3n_readout_sweep.py first")
    rng = np.random.default_rng(0)

    print("=" * 84)
    print("CHECK 1 — NULL DISTRIBUTION OF `spread` (labels shuffled, same activations)")
    print("=" * 84)
    print(f"  {'frame':14s} {'n':>4s} {'real':>7s} {'null mean':>10s} {'null p95':>9s} "
          f"{'z':>6s}   verdict")
    rows = []
    for f in files:
        name = Path(f).stem
        if name not in CANONICAL_FRAMES:
            continue
        z = np.load(f)
        words = list(z["words"])
        y = np.array([CANONICAL_FRAMES[name]["data"][w] for w in words], float)
        # stream0 readout, matching the sweep
        X = [np.array([z[f"raw::{w}"][L, 0, :] for w in words])
             for L in range(z[f"raw::{words[0]}"].shape[0])]
        real = curve_spread(X, y)
        null = np.array([curve_spread(X, rng.permutation(y)) for _ in range(N_PERM)])
        p95 = float(np.percentile(null, 95))
        zs = (real - null.mean()) / (null.std() + 1e-9)
        verdict = "above null" if real > p95 else "INDISTINGUISHABLE from null"
        rows.append((name, len(words), real, float(null.mean()), p95, float(zs), verdict))
        print(f"  {name:14s} {len(words):>4d} {real:>7.3f} {null.mean():>10.3f} "
              f"{p95:>9.3f} {zs:>6.1f}   {verdict}", flush=True)

    nulls = np.array([r[3] for r in rows])
    print(f"\n  mean null spread across frames = {nulls.mean():.3f}  "
          f"(the floor `spread` can never go below)")
    print(f"  the 0.20 FLAT cutoff sits at "
          f"{'ABOVE' if 0.20 > nulls.mean() else 'BELOW'} the mean null")
    bad = [r[0] for r in rows if r[2] <= r[4]]
    print(f"  frames whose REAL spread is inside the null 95th pct: {bad or 'none'}")

    print("\n" + "=" * 84)
    print("CHECK 2 — is the L0/peak law definitional?")
    print("=" * 84)
    l0r, minr, spr = [], [], []
    for f in files:
        name = Path(f).stem
        if name not in CANONICAL_FRAMES:
            continue
        z = np.load(f)
        words = list(z["words"])
        y = np.array([CANONICAL_FRAMES[name]["data"][w] for w in words], float)
        X = [np.array([z[f"raw::{w}"][L, 0, :] for w in words])
             for L in range(z[f"raw::{words[0]}"].shape[0])]
        c = np.array([_cv_r(Xi, y) for Xi in X], float)
        c[np.isnan(c)] = -9.0
        c = np.where(c < -1, np.nan, c)
        mx = np.nanmax(c)
        l0r.append(c[0] / mx)
        minr.append(np.nanmin(c[1:]) / mx)
        spr.append(np.nanmax(c[1:]) - np.nanmin(c[1:]))
    l0r, minr, spr = np.array(l0r), np.array(minr), np.array(spr)

    def pcorr(a, b, ctrl):
        """corr(a,b) with ctrl partialled out of both."""
        ra = a - np.polyval(np.polyfit(ctrl, a, 1), ctrl)
        rb = b - np.polyval(np.polyfit(ctrl, b, 1), ctrl)
        return float(np.corrcoef(ra, rb)[0, 1])

    print(f"  corr(L0/peak,  spread) = {np.corrcoef(l0r, spr)[0, 1]:+.3f}   the claim")
    print(f"  corr(min/peak, spread) = {np.corrcoef(minr, spr)[0, 1]:+.3f}   "
          f"the definitional part (min IS in spread)")
    print(f"  partial corr(L0/peak, spread | min/peak) = {pcorr(l0r, spr, minr):+.3f}")
    print(f"  ^ if this collapses toward 0, the 'law' is just 'curves with a low point")
    print(f"    vary more', and L0 -- the embedding, the mechanistic story -- adds nothing.")

    out = RESULTS / "frpr_spread_null.json"
    out.write_text(json.dumps({
        "n_perm": N_PERM,
        "null": [{"frame": r[0], "n": r[1], "real": r[2], "null_mean": r[3],
                  "null_p95": r[4], "z": r[5], "verdict": r[6]} for r in rows],
        "l0_check": {"corr_l0": float(np.corrcoef(l0r, spr)[0, 1]),
                     "corr_min": float(np.corrcoef(minr, spr)[0, 1]),
                     "partial_l0_given_min": pcorr(l0r, spr, minr)}}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
