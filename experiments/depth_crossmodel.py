"""Is there ANY model-invariant depth structure in the frame family?

Aggregates every `depth_stability_<model>.json` (plus the Gemma 3n readout sweep,
which stores its stream0 curve in a different shape) and asks the question the
lexical-early / relational-late gradient in [[frame_sweep_patterns]] presupposes:
does a frame sit at a consistent RELATIVE depth across backbones?

Two gates, both required before a frame may enter the comparison:

  1. the peak must be real on that model -- not a flat curve
     ([[peak_layer_not_load_bearing]])
  2. depth is compared as a FRACTION of stack depth, never as a layer index
     (360M/1.5B/1.7B/3.8B/7B/Gemma-3n do not share a layer axis)

Reports per-frame citability across models, depth spread among the models where the
peak is real, and a direct test of the early/late split.

    uv run python experiments/depth_crossmodel.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"

# The gradient as recorded in [[frame_sweep_patterns]], stated so it can be tested.
EARLY = {"opinion", "size", "shape", "number"}
LATE = {"concreteness", "time", "animacy", "certainty", "potency", "competence"}


def stab(curve):
    c = np.array([-9.0 if x is None else float(x) for x in curve])
    c[c < -1] = np.nan
    mx = float(np.nanmax(c))
    pk = int(np.nanargmax(c))
    return {
        "r": mx, "layer": pk, "depth": pk / (len(c) - 1),
        "w2": int(np.nansum(c >= mx - 0.02)),
        "spread": float(np.nanmax(c[1:]) - np.nanmin(c[1:])),
    }


def verdict(s):
    if s["spread"] < 0.20:
        return "FLAT"
    return "soft" if s["w2"] > 6 else "real"


def load_all():
    models = {}
    for p in sorted(glob.glob(str(RESULTS / "depth_stability_*.json"))):
        d = json.loads(Path(p).read_text())
        tag = d["model"].split("/")[-1]
        models[tag] = {k: stab(v["curve"]) for k, v in d["frames"].items()}
    g = RESULTS / "gemma3n_readout_sweep_alphatuned.json"
    if g.exists():
        d = json.loads(g.read_text())
        models["gemma-3n-E2B"] = {k: stab(v["stream0"]["curve"])
                                  for k, v in d["results"].items()}
    return models


def main():
    models = load_all()
    tags = list(models)
    frames = sorted({f for m in models.values() for f in m})
    print(f"models ({len(tags)}): {', '.join(tags)}\n")

    print("=" * 96)
    print("CITABILITY — is the peak real on each model?")
    print("=" * 96)
    print(f"  {'frame':14s} " + "".join(f"{t[:11]:>13s}" for t in tags) + f"{'real/n':>9s}")
    citable = {}
    for f in frames:
        cells, n_real = [], 0
        for t in tags:
            s = models[t].get(f)
            if s is None:
                cells.append("-")
                continue
            v = verdict(s)
            n_real += v == "real"
            cells.append(f"{v}" if v != "real" else f"real {s['depth']:.0%}")
        citable[f] = n_real
        print(f"  {f:14s} " + "".join(f"{c:>13s}" for c in cells) + f"{n_real:>6d}/{len(tags)}")

    print("\n" + "=" * 96)
    print("DEPTH AGREEMENT — only across models where the peak is REAL")
    print("=" * 96)
    print(f"  {'frame':14s} {'n':>3s} {'mean':>7s} {'sd':>7s} {'min':>6s} {'max':>6s} "
          f"{'range':>7s}   interpretation")
    agree = {}
    for f in frames:
        ds = [models[t][f]["depth"] for t in tags
              if f in models[t] and verdict(models[t][f]) == "real"]
        if len(ds) < 2:
            continue
        rng = max(ds) - min(ds)
        agree[f] = (len(ds), float(np.mean(ds)), float(np.std(ds)), rng)
        tag = ("CONSISTENT" if rng <= 0.20 else
               "loose" if rng <= 0.35 else "NOT consistent")
        print(f"  {f:14s} {len(ds):>3d} {np.mean(ds):>6.0%} {np.std(ds):>6.0%} "
              f"{min(ds):>5.0%} {max(ds):>5.0%} {rng:>6.0%}   {tag}")

    print("\n" + "=" * 96)
    print("THE EARLY/LATE CLAIM — tested only on citable (frame, model) pairs")
    print("=" * 96)
    e = [models[t][f]["depth"] for f in EARLY for t in tags
         if f in models[t] and verdict(models[t][f]) == "real"]
    l = [models[t][f]["depth"] for f in LATE for t in tags
         if f in models[t] and verdict(models[t][f]) == "real"]
    print(f"  claimed EARLY {sorted(EARLY)}")
    print(f"    citable pairs: {len(e):3d}   mean depth {np.mean(e):.0%}"
          if e else "    citable pairs: 0 — nothing to test")
    print(f"  claimed LATE  {sorted(LATE)}")
    print(f"    citable pairs: {len(l):3d}   mean depth {np.mean(l):.0%}"
          if l else "    citable pairs: 0 — nothing to test")
    if e and l:
        print(f"\n  separation: LATE - EARLY = {np.mean(l) - np.mean(e):+.0%}")
        # Mann-Whitney U without scipy: count dominating pairs.
        wins = sum(1 for a in e for b in l if b > a)
        tot = len(e) * len(l)
        print(f"  P(late deeper than early) over all pairs = {wins / tot:.2f}  "
              f"(0.50 = no ordering; n={tot} pairs)")

    total_pairs = sum(1 for f in frames for t in tags if f in models[t])
    real_pairs = sum(1 for f in frames for t in tags
                     if f in models[t] and verdict(models[t][f]) == "real")
    print(f"\n  overall: {real_pairs}/{total_pairs} (frame, model) pairs have a real peak "
          f"({real_pairs / total_pairs:.0%})")
    never = [f for f in frames if citable[f] == 0]
    if never:
        print(f"  never citable on ANY model: {never}")


if __name__ == "__main__":
    main()
