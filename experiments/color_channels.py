"""Which color channels does SmolLM2 encode best? Per-channel held-out recoverability
for both RGB and opponent (CIELAB) parameterizations, + naming the 2 dominant CCA
canonical dimensions by their correlation with each raw channel.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")
from epa_external_validation import collect_acts
from color_affect_frame import COLORS, _hex_to_lab


def _rgb(h):
    return np.array([int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])


def cv_r(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
    pred = cross_val_predict(est, X, y, cv=5)
    return float(np.corrcoef(pred, y)[0, 1])


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    cw = [c for c in COLORS if c not in ("fuchsia", "aqua")]
    RGB = np.array([_rgb(COLORS[w]) for w in cw])          # (N,3) 0..1
    LAB = np.array([_hex_to_lab(COLORS[w]) for w in cw])   # (N,3)
    chans = {"R": RGB[:, 0], "G": RGB[:, 1], "B": RGB[:, 2],
             "L*(light)": LAB[:, 0], "a*(red-grn)": LAB[:, 1], "b*(blu-yel)": LAB[:, 2]}
    print(f"colors={len(cw)}", flush=True)

    acts = collect_acts(mdl, tok, dev, cw)
    n_layers = acts[cw[0]].shape[0]

    print("\n=== per-channel best held-out CV r (across layers) ===")
    best = {}
    for layer in range(n_layers):
        X = np.array([acts[w][layer] for w in cw])
        for name, y in chans.items():
            r = cv_r(X, y)
            if r > best.get(name, (-9, -9))[0]:
                best[name] = (r, layer)
    for name in chans:
        r, l = best[name]
        print(f"  {name:12s} r={r:+.3f} @L{l}")

    # channel intercorrelations (so we know what's redundant in the sample)
    print("\n=== ground-truth channel correlations (this color sample) ===")
    names = list(chans)
    M = np.array([chans[n] for n in names])
    C = np.corrcoef(M)
    print("       " + " ".join(f"{n[:5]:>6s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n[:5]:>5s} " + " ".join(f"{C[i,j]:+.2f}" for j in range(len(names))))


if __name__ == "__main__":
    main()
