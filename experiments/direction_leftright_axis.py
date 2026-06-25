"""Do cross-lingual LEFT/RIGHT transfer on their OWN (egocentric) axis?

The compass test was the wrong axis (cos(LR,EW)~0). Here fit a dedicated LEFT-RIGHT
axis on English left/right variants, then project cross-lingual left/right onto IT,
layer-swept, per-word sign (left=-, right=+). This is the actual "how are cross-lingual
left/right captured" test.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/direction_leftright_axis.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from turnstyle.frame_library import _collect, _cv_r, _ridge_dir

_TMPL = "Move {w}."
LR_FIT = {"left": -1, "right": 1, "leftward": -1, "rightward": 1,
          "leftmost": -1, "rightmost": 1}
# held-out cross-lingual left/right (expected sign on the LR axis)
XL = [("izquierda", -1, "es"), ("derecha", +1, "es"),
      ("gauche", -1, "fr"), ("droite", +1, "fr"),
      ("links", -1, "de"), ("rechts", +1, "de")]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()

    fitw = list(LR_FIT)
    testw = [w for (w, _, _) in XL]
    fa = _collect(mdl, tok, dev, fitw, _TMPL, "last")
    ta = _collect(mdl, tok, dev, testw, _TMPL, "last")
    nL = fa[fitw[0]].shape[0]
    y = np.array([LR_FIT[w] for w in fitw], float)

    hdr = "  ".join(f"{w[:5]:>6}" for (w, _, _) in XL)
    print(f"LR-axis recoverability (in-fit CV r) + cross-lingual left/right on the LR axis")
    print(f"{'L':>3} | {'cv_r':>5} | {hdr} | xl-acc")
    best = (0.0, -1)
    for L in range(nL):
        X = np.array([fa[w][L] for w in fitw]); mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        r = _cv_r(X, y)
        d = _ridge_dir(Z, y)
        cells, ok = [], 0
        for (w, sgn, _l) in XL:
            p = float(((ta[w][L] - mu) / sd) @ d)
            hit = (p > 0) == (sgn > 0); ok += hit
            cells.append(f"{'+' if p>0 else '-'}{'ok' if hit else 'XX'}")
        if ok > best[0]:
            best = (ok, L)
        rr = f"{r:.2f}" if r == r else " nan"
        print(f"{L:>3} | {rr:>5} | " + "  ".join(f"{c:>6}" for c in cells)
              + f" | {ok}/6", flush=True)
    print(f"\nbest cross-lingual left/right on the LR axis: {best[0]}/6 @L{best[1]} "
          f"(chance 3/6). High => left/right ARE captured cross-lingually, just on their "
          f"OWN egocentric axis, not the compass.")


if __name__ == "__main__":
    main()
