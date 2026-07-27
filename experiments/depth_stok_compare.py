"""Does Mistral's localization advantage survive the tokenization control?

Per-model "localization" (mean curve spread) ranked Mistral-7B first at .469 and
Phi-4-mini last at .221, with no relation to parameter count.  But Mistral
fragments the frame lexicon at 1.47 subwords/word against ~1.11 elsewhere, and a
multi-subword word must have its meaning ASSEMBLED across layers -- which reads as
localization for a tokenizer reason.  Dropping Mistral flipped
corr(localization, embedding-availability) from -0.595 to +0.825, i.e. the entire
per-model effect was one model.

`depth_stability.py --single-token shared` refits every model on only those words
NO panel tokenizer splits, so all models see an identical word set.

Two things this script exists to get right:

  - The controlled run covers 10 frames (4 are skipped for too few surviving
    words), while the original covered 14.  Comparing a 10-frame mean against a
    14-frame mean would confound the control with frame selection, so the BEFORE
    number is recomputed over the same 10 frames.
  - The 4 skipped frames are not random: weight (12->3 words), potency (12->5),
    competence (19->7), shape (13->7) fragment most AND localize most
    (corr(retention, spread) = -0.569).  So the control is structurally unable to
    test the strongest-localizing frames, and that limit belongs in the result.

    uv run python experiments/depth_stok_compare.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"


def main():
    after = {}
    for p in sorted(glob.glob(str(RESULTS / "stok_*.json"))):
        d = json.loads(Path(p).read_text())
        after[d["model"].split("/")[-1]] = d
    if not after:
        raise SystemExit("no stok_*.json — run depth_stability --single-token shared first")

    before = {}
    for p in sorted(glob.glob(str(RESULTS / "depth_stability_*.json"))):
        d = json.loads(Path(p).read_text())
        before[d["model"].split("/")[-1]] = d

    # the frames the control could actually test, intersected across models
    tested = set.intersection(*[set(d["frames"]) for d in after.values()])
    tested = sorted(tested)
    print(f"controlled on {len(tested)} frames: {tested}")
    skipped = sorted({f for d in after.values() for f in d.get("skipped", {})})
    print(f"UNTESTABLE (too few single-token words): {skipped}")
    print("  ^ these fragment most AND localize most — the control cannot reach them\n")

    print("=" * 84)
    print("LOCALIZATION BEFORE vs AFTER, both means over the SAME frames")
    print("=" * 84)
    print(f"  {'model':26s} {'before':>8s} {'after':>8s} {'delta':>8s}   {'n words':>8s}")
    rows = []
    for tag in sorted(after, key=lambda t: -before.get(t, {}).get("mean_spread", 0)
                      if t in before else 0):
        if tag not in before:
            continue
        b = np.mean([before[tag]["frames"][f]["spread"] for f in tested
                     if f in before[tag]["frames"]])
        a = np.mean([after[tag]["frames"][f]["spread"] for f in tested])
        nw = int(np.mean([after[tag]["frames"][f].get("n_words", 0) for f in tested]))
        rows.append((tag, float(b), float(a)))
        print(f"  {tag:26s} {b:>8.3f} {a:>8.3f} {a - b:>+8.3f}   {nw:>8d}")

    rows.sort(key=lambda r: -r[2])
    print(f"\n  ranking AFTER control (most localized first):")
    for i, (t, b, a) in enumerate(rows, 1):
        print(f"    {i}. {t:26s} {a:.3f}")

    mist = next((r for r in rows if r[0].startswith("Mistral")), None)
    if mist:
        others = [r for r in rows if not r[0].startswith("Mistral")]
        lead_before = mist[1] - max(r[1] for r in others)
        lead_after = mist[2] - max(r[2] for r in others)
        print(f"\n  Mistral's lead over the next model: "
              f"{lead_before:+.3f} before -> {lead_after:+.3f} after")
        rank = [r[0] for r in rows].index(mist[0]) + 1
        if lead_after > 0.05:
            print(f"  => SURVIVES. Mistral is still #1 (rank {rank}) on an identical word")
            print(f"     set, so its localization is NOT a tokenization artifact.")
        elif lead_after <= 0:
            print(f"  => COLLAPSES. Mistral drops to rank {rank} once every model sees the")
            print(f"     same words: the per-model localization effect was tokenization.")
        else:
            print(f"  => WEAKENED but still #1 (rank {rank}); lead shrank "
                  f"{lead_before:.3f} -> {lead_after:.3f}. Partly tokenization.")

    b = np.array([r[1] for r in rows])
    a = np.array([r[2] for r in rows])
    print(f"\n  cross-model SPREAD of the localization statistic itself:")
    print(f"    before {b.max() - b.min():.3f}   after {a.max() - a.min():.3f}")
    print(f"    (shrinking means models look MORE alike once tokenization is held fixed)")
    print(f"  rank correlation before vs after = "
          f"{np.corrcoef(b.argsort().argsort(), a.argsort().argsort())[0, 1]:+.3f}")

    out = RESULTS / "depth_stok_compare.json"
    out.write_text(json.dumps({"tested_frames": tested, "untestable": skipped,
                               "rows": [{"model": t, "before": b_, "after": a_}
                                        for t, b_, a_ in rows]}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
