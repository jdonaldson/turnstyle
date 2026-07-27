"""Why do SOME frames localize? Because the rest are already in the embedding.

The threshold-free consolidation left a 3+3 split ([[peak_layer_not_load_bearing]]):
`concreteness`/`animacy`/`number` never localize on any of six models, while
`weight`/`potency`/`warmth` always do.  That split has a suspicious shape -- the
first three look like LEXICAL LOOKUPS, the second three like INFERRED properties,
and `weight` is independently known to be bilinear (felt heaviness ~ size x
density, [[frame_sweep_patterns]]).

Prediction, registered before looking: a frame the embedding already carries needs
no computation, so its curve starts high and stays flat; a frame that must be
COMPOSED starts low and rises to a peak.  So L0/peak should be strongly NEGATIVELY
correlated with spread.

Runs entirely off stored curves -- no model loads, no forward passes.

Two things to keep honest about the result:

  - It is not circular (spread is computed over L1..L_top and excludes L0) but it
    is not independent either: high L0 implies high L1 implies a high minimum,
    hence small spread. Part of the correlation is structural coupling, so read it
    as "the curve's shape is set by its starting level", not as a discovered law.
  - The FRAME-level and MODEL-level versions come apart sharply, which is the
    actual finding. See the second table.

    uv run python experiments/depth_lookup_vs_computed.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"

# the surviving lexical-early / relational-late groups, for the subsumption check
EARLY = {"opinion", "size", "shape", "number"}
LATE = {"concreteness", "time", "animacy", "certainty", "potency", "competence"}

FRAME_LOC = {
    "Mistral-7B-Instruct-v0.3": 0.469, "gemma-3n-E2B": 0.373,
    "Qwen2.5-1.5B-Instruct": 0.339, "SmolLM2-360M-Instruct": 0.275,
    "SmolLM2-1.7B-Instruct": 0.249, "Phi-4-mini-instruct": 0.221,
}
# mean subwords per frame-lexicon word, from the tokenizer comparison
FRAG = {
    "Mistral-7B-Instruct-v0.3": 1.47, "gemma-3n-E2B": 1.04,
    "Qwen2.5-1.5B-Instruct": 1.11, "SmolLM2-360M-Instruct": 1.12,
    "SmolLM2-1.7B-Instruct": 1.12, "Phi-4-mini-instruct": 1.11,
}


def load():
    M = {}
    for p in sorted(glob.glob(str(RESULTS / "depth_stability_*.json"))):
        d = json.loads(Path(p).read_text())
        M[d["model"].split("/")[-1]] = {k: v["curve"] for k, v in d["frames"].items()}
    g = RESULTS / "gemma3n_readout_sweep_alphatuned.json"
    if g.exists():
        d = json.loads(g.read_text())
        M["gemma-3n-E2B"] = {k: v["stream0"]["curve"] for k, v in d["results"].items()}
    return M


def curve(c):
    a = np.array([-9.0 if x is None else float(x) for x in c])
    a[a < -1] = np.nan
    return a


def main():
    M = load()
    frames = sorted({f for m in M.values() for f in m})

    print("=" * 82)
    print("FRAME LEVEL — is a flat frame simply one the embedding already has?")
    print("=" * 82)
    print(f"  {'frame':14s} {'L0/peak':>8s} {'spread':>8s} {'group':>7s}   reading")
    rows = []
    for f in frames:
        l0, pk, sp = [], [], []
        for t in M:
            if f not in M[t]:
                continue
            c = curve(M[t][f])
            l0.append(c[0])
            pk.append(np.nanmax(c))
            sp.append(np.nanmax(c[1:]) - np.nanmin(c[1:]))
        ratio = float(np.mean(l0) / np.mean(pk))
        rows.append((f, ratio, float(np.mean(sp))))
    rows.sort(key=lambda r: -r[1])
    for f, ratio, sp in rows:
        grp = "early" if f in EARLY else ("late" if f in LATE else "-")
        rd = "in the embedding" if ratio > 0.70 else (
            "must be COMPUTED" if ratio < 0.55 else "partly computed")
        print(f"  {f:14s} {ratio:>8.2f} {sp:>8.3f} {grp:>7s}   {rd}")

    x = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    print(f"\n  corr(L0/peak, spread) = {np.corrcoef(x, y)[0, 1]:+.3f}  (n={len(rows)} frames)")

    # Does this SUBSUME the surviving early/late ordering?
    e = [r[1] for r in rows if r[0] in EARLY]
    l = [r[1] for r in rows if r[0] in LATE]
    print(f"\n  mean L0/peak — claimed EARLY {np.mean(e):.2f}  vs  claimed LATE {np.mean(l):.2f}")
    print("  => the early/late gradient looks like a NOISY PROXY for this: the frames")
    print("     called 'early' are the ones the embedding already carries, so their")
    print("     (weak) peaks sit early by default.")

    print("\n" + "=" * 82)
    print("MODEL LEVEL — the same story does NOT hold, and Mistral is the outlier")
    print("=" * 82)
    print(f"  {'model':26s} {'frame loc':>9s} {'L0/peak':>8s} {'tok/word':>9s}")
    mr = []
    for t, fl in sorted(FRAME_LOC.items(), key=lambda kv: -kv[1]):
        if t not in M:
            continue
        v = [curve(M[t][f])[0] / np.nanmax(curve(M[t][f])) for f in M[t]]
        mr.append((t, fl, float(np.mean(v)), FRAG.get(t)))
        print(f"  {t:26s} {fl:>9.3f} {np.mean(v):>8.2f} {FRAG.get(t, float('nan')):>9.2f}")

    a = np.array([r[1] for r in mr])
    b = np.array([r[2] for r in mr])
    fg = np.array([r[3] for r in mr], float)
    print(f"\n  corr(model localization, L0/peak) = {np.corrcoef(a, b)[0, 1]:+.3f}")
    keep = [i for i, r in enumerate(mr) if r[0] != "Mistral-7B-Instruct-v0.3"]
    print(f"  ... excluding Mistral             = {np.corrcoef(a[keep], b[keep])[0, 1]:+.3f}"
          f"   <- collapses; it was one model")
    print(f"  corr(model localization, tok/word) = {np.corrcoef(a, fg)[0, 1]:+.3f}"
          f"   <- Mistral fragments most AND localizes most")

    print("\n  => FRAME-level localization is explained by embedding availability.")
    print("     MODEL-level localization is NOT -- and the leading remaining candidate")
    print("     is a tokenization artifact (heavier fragmentation -> the last subword's")
    print("     embedding carries less -> more must be assembled). That would also")
    print("     explain why a per-model localization scalar failed to replicate across")
    print("     lexicons: different lexicons fragment differently per model.")
    print("     CONTROLLED TEST: restrict to words single-token in ALL models, refit.")
    print("     Phi remains unexplained either way -- its L0/peak is unremarkable.")

    p = RESULTS / "depth_lookup_vs_computed.json"
    p.write_text(json.dumps({
        "frame_level": [{"frame": f, "l0_over_peak": r, "spread": s} for f, r, s in rows],
        "corr_frame": float(np.corrcoef(x, y)[0, 1]),
        "model_level": [{"model": t, "frame_loc": fl, "l0_over_peak": r, "tok_per_word": g}
                        for t, fl, r, g in mr],
        "corr_model": float(np.corrcoef(a, b)[0, 1]),
        "corr_model_ex_mistral": float(np.corrcoef(a[keep], b[keep])[0, 1]),
    }, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
