"""Payoff test: does SPATIAL CONTEXT (+ language-centering) recover cross-lingual left/right?

leftright_context.py showed left/right are context-gated polysemes: copular "It is right"
reads as CORRECT (lateral residual ~0), spatial "Turn to the right" activates the lateral
sense (cos(right,left) 0.65-0.75). So a context-free frame can't read them. Here we fit the
EN left/right direction in each template and measure cross-lingual sign transfer (6 langs,
per-language mean-centered), per layer. Prediction: spatial >> copular/Move.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/leftright_xlingual_spatial.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from turnstyle.frame_library import _collect, _ridge_dir

TMPLS = {"Move": "Move {w}.", "copular": "It is {w}.", "spatial": "Turn to the {w}."}
FIT = {"left": -1, "right": 1}
XL = {
    "es": [("izquierda", -1), ("derecha", +1)],
    "fr": [("gauche", -1), ("droite", +1)],
    "de": [("links", -1), ("rechts", +1)],
    "it": [("sinistra", -1), ("destra", +1)],
    "pt": [("esquerda", -1), ("direita", +1)],
    "nl": [("linker", -1), ("rechter", +1)],
}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()

    fit_words = list(FIT)
    xl_words = [w for items in XL.values() for (w, _) in items]
    acts = {n: _collect(mdl, tok, dev, fit_words + xl_words, t, pool="last")
            for n, t in TMPLS.items()}
    nL = acts["Move"][fit_words[0]].shape[0]
    y = np.array([FIT[w] for w in fit_words], float)
    nTot = sum(len(v) for v in XL.values())

    print(f"cross-lingual left/right sign acc (/{nTot}, chance {nTot // 2}, 6 langs, "
          f"language-centered) by template")
    print(f"{'L':>3} | {'Move':>6} {'copular':>8} {'spatial':>8}")
    best = {n: (0, -1) for n in TMPLS}
    for L in range(nL):
        cells = []
        for n in TMPLS:
            A = acts[n]
            Xf = np.array([A[w][L] for w in fit_words])
            mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
            d = _ridge_dir((Xf - mu) / sd, y)
            hit = 0
            for lang, items in XL.items():
                off = np.mean([A[w][L] for (w, _) in items], 0)   # per-language centering
                for (w, sgn) in items:
                    if (float(((A[w][L] - off) / sd) @ d) > 0) == (sgn > 0):
                        hit += 1
            cells.append(hit)
            if hit > best[n][0]:
                best[n] = (hit, L)
        print(f"{L:>3} | {cells[0]:>6} {cells[1]:>8} {cells[2]:>8}", flush=True)
    print("\nbest layer per template: " +
          "  ".join(f"{n}={h}/{nTot}@L{L}" for n, (h, L) in best.items()))


if __name__ == "__main__":
    main()
