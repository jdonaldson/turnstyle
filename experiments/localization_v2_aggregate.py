"""Across six backbones: is the handful of "localizes" hits more than chance?

`localization_v2` gates a frame as localizing on shape_z >= 2 AND peak_z >= 2, both
against a label-permutation null. Per model the hit count is 0-1 out of 14. The
question this answers is whether that is a small real effect or exactly the
false-positive rate of the gate.

Two things decide it:

  1. TOTAL hits vs expected. One-sided z >= 2 is p ~ .0228 under the null, and the
     gate needs shape AND peak. If those were independent the joint rate would be
     ~.0005, but they are not -- both derive from the same curve, so the realistic
     per-test rate sits between .0005 and .0228. Report the observed rate against
     that band rather than pretending to a clean binomial p.

  2. CONSISTENCY. This is the decisive one and needs no distributional assumption.
     A real effect makes the SAME frames localize on multiple models; noise
     scatters hits across different frames. Compare the observed repeat structure
     to what random reassignment of the same number of hits produces.

    uv run python experiments/localization_v2_aggregate.py
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"


def main():
    models = {}
    for p in sorted(glob.glob(str(RESULTS / "localization_v2_*.json"))):
        d = json.loads(Path(p).read_text())
        models[d["model"].split("/")[-1]] = d["frames"]
    if len(models) < 2:
        raise SystemExit("need results for several models")

    frames = sorted({f for m in models.values() for f in m})
    print(f"models={len(models)}  frames={len(frames)}  "
          f"tests={sum(len(m) for m in models.values())}\n")

    print("=" * 78)
    print("HITS PER MODEL")
    print("=" * 78)
    hits, n_tests = {}, 0
    for tag, fr in models.items():
        h = sorted(f for f, v in fr.items() if v["localizes"])
        sig = sum(1 for v in fr.values() if v["peak_z"] >= 2.0)
        hits[tag] = h
        n_tests += len(fr)
        print(f"  {tag:26s} localizes {len(h)}/{len(fr)}  {h or '[]'}"
              f"   (had signal: {sig}/{len(fr)})")

    total = sum(len(h) for h in hits.values())
    rate = total / n_tests
    print(f"\n  total {total}/{n_tests} tests = {rate:.3f}")
    print(f"  gate is one-sided z>=2 on BOTH shape and peak; the per-test null rate")
    print(f"  lies between .0005 (independent) and .0228 (perfectly dependent).")
    print(f"  observed {rate:.3f} vs that band -> "
          f"{'INSIDE the band: consistent with chance' if rate <= 0.0228 else 'ABOVE the band'}")

    print("\n" + "=" * 78)
    print("CONSISTENCY — do the SAME frames localize across models?")
    print("=" * 78)
    c = Counter(f for h in hits.values() for f in h)
    for f, k in c.most_common():
        n_avail = sum(1 for m in models.values() if f in m)
        print(f"  {f:14s} localizes on {k}/{n_avail} models")
    if not c:
        print("  no frame localizes on any model")
        return

    # Null: scatter the same number of hits per model uniformly over that model's
    # frames, and ask how often the top frame repeats as much as observed.
    rng = np.random.default_rng(0)
    obs_max = max(c.values())
    reps = 0
    for _ in range(20000):
        cc = Counter()
        for tag, h in hits.items():
            avail = list(models[tag])
            if h:
                cc.update(rng.choice(avail, size=len(h), replace=False))
        if cc and max(cc.values()) >= obs_max:
            reps += 1
    p_rep = reps / 20000
    print(f"\n  most-repeated frame appears on {obs_max} models")
    print(f"  P(a frame repeats >= {obs_max} times | hits scattered at random) = {p_rep:.3f}")
    verdict = ("NOT distinguishable from chance scatter" if p_rep > 0.05
               else "more consistent than chance — worth a second look")
    print(f"  => {verdict}")

    # power note: hits should concentrate in the largest lexicons if this is noise
    print(f"\n  lexicon sizes of the hit frames (noise favours the best-powered):")
    from turnstyle.frame_library import CANONICAL_FRAMES
    sizes = {f: len(CANONICAL_FRAMES[f]["data"]) for f in CANONICAL_FRAMES}
    allf = sorted(sizes, key=lambda f: -sizes[f])
    print(f"    all frames by size: "
          f"{', '.join(f'{f}={sizes[f]}' for f in allf[:5])} ...")
    for f in c:
        rank = allf.index(f) + 1 if f in allf else -1
        print(f"    {f:14s} n={sizes.get(f, '?')}  (rank {rank}/{len(allf)} by size)")

    out = RESULTS / "localization_v2_aggregate.json"
    out.write_text(json.dumps({"hits": hits, "total": total, "tests": n_tests,
                               "rate": rate, "repeat_counts": dict(c),
                               "p_repeat": p_rep}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
