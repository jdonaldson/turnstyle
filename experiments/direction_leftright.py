"""Cross-lingual LEFT/RIGHT — the test I had NOT run.

(1) Project cross-lingual left/right (es/fr/de) onto the compass-fit E-W axis,
    layer-swept, per-word sign (left expected west/-, right expected east/+).
(2) Egocentric vs allocentric: fit a dedicated LEFT-RIGHT axis on English left/right,
    and measure cos(LR axis, EW compass axis) per layer — are they the SAME axis or
    different representational systems? If different, "left=west" was a category error.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/direction_leftright.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np

from turnstyle.frame_library import _collect, _ridge_dir

_TMPL = "Move {w}."
S = 1 / math.sqrt(2)

EW_FIT = {"east": 1, "west": -1, "north": 0, "south": 0,
          "northeast": S, "northwest": -S, "southeast": S, "southwest": -S}
# dedicated egocentric axis fit (more anchors so the ridge is not 2-point)
LR_FIT = {"left": -1, "right": 1, "leftward": -1, "rightward": 1,
          "leftmost": -1, "rightmost": 1}

# cross-lingual left/right, expected E-W sign under the left=west/right=east convention
LR_TEST = [("left", -1, "en"), ("right", +1, "en"),
           ("izquierda", -1, "es"), ("derecha", +1, "es"),
           ("gauche", -1, "fr"), ("droite", +1, "fr"),
           ("links", -1, "de"), ("rechts", +1, "de")]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()

    ewf, lrf = list(EW_FIT), list(LR_FIT)
    testw = [w for (w, _, _) in LR_TEST]
    ew_acts = _collect(mdl, tok, dev, ewf, _TMPL, "last")
    lr_acts = _collect(mdl, tok, dev, lrf, _TMPL, "last")
    tst_acts = _collect(mdl, tok, dev, testw, _TMPL, "last")
    nL = ew_acts[ewf[0]].shape[0]
    ew_y = np.array([EW_FIT[w] for w in ewf], float)
    lr_y = np.array([LR_FIT[w] for w in lrf], float)

    hdr = "  ".join(f"{w[:5]:>6}" for (w, _, _) in LR_TEST)
    print(f"(1) cross-lingual left/right projected on the COMPASS E-W axis (sign; "
          f"want left=- right=+)")
    print(f"{'L':>3} | {hdr} | acc | cos(LR,EW)")
    for L in range(nL):
        Xe = np.array([ew_acts[w][L] for w in ewf]); mu, sd = Xe.mean(0), Xe.std(0) + 1e-6
        ew_dir = _ridge_dir((Xe - mu) / sd, ew_y)
        # dedicated LR axis (own standardization on the LR anchors)
        Xl = np.array([lr_acts[w][L] for w in lrf]); mul, sdl = Xl.mean(0), Xl.std(0) + 1e-6
        lr_dir = _ridge_dir((Xl - mul) / sdl, lr_y)
        cos = abs(float(lr_dir @ ew_dir) / ((np.linalg.norm(lr_dir) * np.linalg.norm(ew_dir)) or 1))
        cells, ok = [], 0
        for (w, sgn, _lang) in LR_TEST:
            z = (tst_acts[w][L] - mu) / sd
            p = float(z @ ew_dir)
            hit = (p > 0) == (sgn > 0)
            ok += hit
            cells.append(f"{'+' if p>0 else '-'}{'ok' if hit else 'XX':>2}")
        print(f"{L:>3} | " + "  ".join(f"{c:>6}" for c in cells)
              + f" | {ok}/8 | {cos:.2f}", flush=True)
    print("\nlow cos(LR,EW) => left/right is a DIFFERENT axis than east/west "
          "(egocentric vs allocentric) => 'left=west' was a category error, not just weak.")


if __name__ == "__main__":
    main()
