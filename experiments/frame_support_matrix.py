"""Which models support which frames? The surviving cross-model result.

The depth line closed with nothing: no reliable evidence that frames LOCALIZE
anywhere. What survived intact is the other half — frames are RECOVERABLE, and that
half is null-calibrated, so it can be stated cleanly.

Reads `localization_v2_<model>.json`, which carries both numbers per (frame, model):

  peak_r   best held-out CV r over layers — the magnitude
  peak_z   that value z-scored against 40 label permutations. Necessary because
           peak_r is a max over ~30 layers and therefore optimistically biased;
           the permutation null is taken the same way, so peak_z is the honest
           "is there signal here at all".

Supported = peak_z >= 2. A high peak_r with low peak_z means the lexicon is too
small for a max-over-layers statistic to clear its own null, not that the model
lacks the concept.

    uv run python experiments/frame_support_matrix.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turnstyle.frame_library import CANONICAL_FRAMES  # noqa: E402

RESULTS = Path(__file__).parent / "results"
Z_GATE = 2.0
SHORT = {"Mistral-7B-Instruct-v0.3": "Mistral7B", "Phi-4-mini-instruct": "Phi4mini",
         "Qwen2.5-1.5B-Instruct": "Qwen1.5B", "SmolLM2-1.7B-Instruct": "Smol1.7B",
         "SmolLM2-360M-Instruct": "Smol360M", "gemma-3n-E2B-it": "Gemma3n"}


def main():
    models = {}
    for p in sorted(glob.glob(str(RESULTS / "localization_v2_*.json"))):
        d = json.loads(Path(p).read_text())
        if "model" not in d:          # skips the aggregate roll-up, same prefix
            continue
        tag = d["model"].split("/")[-1]
        models[SHORT.get(tag, tag)] = d["frames"]
    order = ["Smol360M", "Qwen1.5B", "Smol1.7B", "Gemma3n", "Phi4mini", "Mistral7B"]
    order = [m for m in order if m in models] + [m for m in models if m not in order]
    frames = sorted({f for m in models.values() for f in m})

    print("=" * 92)
    print("FRAME SUPPORT MATRIX — peak held-out CV r;  ✗ = below its permutation null")
    print("=" * 92)
    print(f"  {'frame':14s} {'n':>3s} " + "".join(f"{m:>11s}" for m in order) + f"{'supported':>11s}")
    rows = []
    for f in frames:
        cells, n_ok = [], 0
        for m in order:
            v = models[m].get(f)
            if v is None:
                cells.append("—")
                continue
            ok = v["peak_z"] >= Z_GATE
            n_ok += ok
            cells.append(f"{v['peak_r']:.2f}" + ("" if ok else "✗"))
        n = len(CANONICAL_FRAMES.get(f, {}).get("data", {}))
        rows.append((f, n, n_ok))
        print(f"  {f:14s} {n:>3d} " + "".join(f"{c:>11s}" for c in cells)
              + f"{n_ok}/{len(order)}".rjust(11))

    print(f"\n  {'PER MODEL':14s} {'':>3s} " + "".join(
        f"{sum(1 for f in models[m] if models[m][f]['peak_z'] >= Z_GATE)}/{len(models[m])}".rjust(11)
        for m in order))
    print(f"  {'mean peak_r':14s} {'':>3s} " + "".join(
        f"{np.mean([v['peak_r'] for v in models[m].values()]):.3f}".rjust(11) for m in order))

    print("\n" + "=" * 92)
    print("READING")
    print("=" * 92)
    univ = [f for f, n, k in rows if k == len(order)]
    partial = [(f, n, k) for f, n, k in rows if 0 < k < len(order)]
    none = [f for f, n, k in rows if k == 0]
    print(f"  supported on ALL {len(order)} models: {len(univ)}/{len(frames)} — {univ}")
    if partial:
        print(f"  partial:")
        for f, n, k in sorted(partial, key=lambda r: r[2]):
            print(f"    {f:14s} {k}/{len(order)} models   (lexicon n={n})")
    if none:
        print(f"  supported NOWHERE: {none}")

    # is failure about the concept, or about lexicon size?
    ns = np.array([n for _, n, _ in rows], float)
    ks = np.array([k for _, _, k in rows], float)
    print(f"\n  corr(lexicon size, #models supporting) = {np.corrcoef(ns, ks)[0, 1]:+.3f}")
    print(f"  ^ a strong positive means non-support tracks POWER (too few words to")
    print(f"    clear a max-over-layers null), not the model lacking the concept.")

    out = RESULTS / "frame_support_matrix.json"
    out.write_text(json.dumps(
        {"z_gate": Z_GATE, "models": order,
         "frames": {f: {"n_words": n, "n_models_supported": k} for f, n, k in rows}},
        indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
