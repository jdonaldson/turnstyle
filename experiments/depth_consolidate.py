"""Consolidation: which depth claims survive, and are they threshold artifacts?

Every citability claim in this thread rests on one arbitrary constant -- a curve is
"FLAT" if its spread is below 0.20 ([[peak_layer_not_load_bearing]]).  That number
was picked by eye off the Gemma 3n sweep.  If the headline results move with it,
they are properties of the threshold, not of the models.

Sweeps the threshold and reports, at each value:

  citable%      fraction of (frame, model) pairs with a locatable peak
  flat-always   frames flat on EVERY model -- the strongest frame-level claim
  early/late    P(late deeper than early) over citable pairs, the surviving half
                of the lexical-early / relational-late gradient

A claim is only worth keeping if it holds across the plausible range.

    uv run python experiments/depth_consolidate.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
EARLY = {"opinion", "size", "shape", "number"}
LATE = {"concreteness", "time", "animacy", "certainty", "potency", "competence"}
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)


def load():
    models = {}
    for p in sorted(glob.glob(str(RESULTS / "depth_stability_*.json"))):
        d = json.loads(Path(p).read_text())
        models[d["model"].split("/")[-1]] = {k: v["curve"] for k, v in d["frames"].items()}
    g = RESULTS / "gemma3n_readout_sweep_alphatuned.json"
    if g.exists():
        d = json.loads(g.read_text())
        models["gemma-3n-E2B"] = {k: v["stream0"]["curve"] for k, v in d["results"].items()}
    return models


def curve_stats(curve):
    c = np.array([-9.0 if x is None else float(x) for x in curve])
    c[c < -1] = np.nan
    mx = float(np.nanmax(c))
    pk = int(np.nanargmax(c))
    return {"peak_depth": pk / (len(c) - 1),
            "within_02": int(np.nansum(c >= mx - 0.02)),
            "spread": float(np.nanmax(c[1:]) - np.nanmin(c[1:]))}


def main():
    models = load()
    tags = list(models)
    frames = sorted({f for m in models.values() for f in m})
    S = {t: {f: curve_stats(c) for f, c in models[t].items()} for t in tags}
    pairs = [(t, f) for t in tags for f in frames if f in S[t]]
    print(f"models={len(tags)}  frames={len(frames)}  pairs={len(pairs)}\n")

    print("=" * 88)
    print("THRESHOLD SENSITIVITY — do the headline claims move with the FLAT cutoff?")
    print("=" * 88)
    print(f"  {'cutoff':>7s} {'citable':>9s} {'flat on ALL models':>44s} {'P(late>early)':>14s}")
    for th in THRESHOLDS:
        real = [(t, f) for t, f in pairs
                if S[t][f]["spread"] >= th and S[t][f]["within_02"] <= 6]
        flat_all = sorted(f for f in frames
                          if all(S[t][f]["spread"] < th for t in tags if f in S[t]))
        e = [S[t][f]["peak_depth"] for t, f in real if f in EARLY]
        l = [S[t][f]["peak_depth"] for t, f in real if f in LATE]
        if e and l:
            p = sum(1 for a in e for b in l if b > a) / (len(e) * len(l))
            ptxt = f"{p:.2f} (n={len(e)}x{len(l)})"
        else:
            ptxt = "untestable"
        print(f"  {th:>7.2f} {len(real):>4d}/{len(pairs)} {len(real) / len(pairs):>4.0%} "
              f"{', '.join(flat_all) or '(none)':>44s} {ptxt:>14s}")

    print("\n" + "=" * 88)
    print("WHAT SURVIVES REGARDLESS OF THRESHOLD")
    print("=" * 88)
    # frames flat on every model at the LOOSEST cutoff = flat beyond argument
    strict = sorted(f for f in frames
                    if all(S[t][f]["spread"] < min(THRESHOLDS) for t in tags if f in S[t]))
    loose = sorted(f for f in frames
                   if all(S[t][f]["spread"] < max(THRESHOLDS) for t in tags if f in S[t]))
    print(f"  flat on all models even at cutoff {min(THRESHOLDS):.2f} (strongest claim): "
          f"{strict or '(none)'}")
    print(f"  flat on all models at cutoff {max(THRESHOLDS):.2f} (weakest claim):        {loose}")

    # spread is the raw quantity -- report it threshold-free per frame
    print(f"\n  RAW spread per frame, min across models (threshold-free):")
    rows = sorted(frames, key=lambda f: max(S[t][f]["spread"] for t in tags if f in S[t]))
    print(f"  {'frame':14s} {'min':>7s} {'median':>7s} {'MAX':>7s}   reading")
    for f in rows:
        v = np.array([S[t][f]["spread"] for t in tags if f in S[t]])
        rd = ("never localizes anywhere" if v.max() < 0.20 else
              "localizes on some models" if v.min() < 0.20 else
              "localizes on ALL models")
        print(f"  {f:14s} {v.min():>7.3f} {np.median(v):>7.3f} {v.max():>7.3f}   {rd}")

    print(f"\n  => The threshold-free statement: a frame's MAX spread across models is")
    print(f"     what decides whether 'where does it live' is ever a real question.")


if __name__ == "__main__":
    main()
