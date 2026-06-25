"""Is the cross-lingual direction negative a LANGUAGE-SURFACE-OFFSET artifact?

The frame-family work established raw activation geometry is language-surface-DOMINATED:
a Spanish word's activation is mostly "this is Spanish", with the semantic direction a
low-variance residual. The prior transfer test standardized foreign words by the ENGLISH
anchor mean (mu_en), so a constant per-language offset vector rides on every foreign word.
On a 4-word compass set that offset can swamp the primary-axis SIGN.

Standard cross-lingual fix: per-language mean-centering — subtract the language's own
centroid so the surface offset cancels, leaving direction structure. Centering the 4
balanced compass words removes only their COMMON component (the offset); it cannot
manufacture correct signs (a no-signal set would still score chance), so it's a fair test.
Disjoint-anchor version below estimates the offset from NEUTRAL words to kill the circular
worry.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/direction_centered.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np

from turnstyle.frame_library import _collect, _ridge_dir

_TMPL = "Move {w}."
S = 1 / math.sqrt(2)

NS_FIT = {"north": 1, "south": -1, "east": 0, "west": 0,
          "northeast": S, "northwest": S, "southeast": -S, "southwest": -S}
EW_FIT = {"east": 1, "west": -1, "north": 0, "south": 0,
          "northeast": S, "northwest": -S, "southeast": S, "southwest": -S}
FIT_WORDS = list(NS_FIT)

# cross-lingual compass — primary axis only. 6 languages × 4 = 24 (was 12).
XL = {
    "es": [("norte", "ns", +1), ("sur", "ns", -1), ("este", "ew", +1), ("oeste", "ew", -1)],
    "fr": [("nord", "ns", +1), ("sud", "ns", -1), ("est", "ew", +1), ("ouest", "ew", -1)],
    "de": [("norden", "ns", +1), ("süden", "ns", -1), ("osten", "ew", +1), ("westen", "ew", -1)],
    "it": [("nord", "ns", +1), ("sud", "ns", -1), ("est", "ew", +1), ("ovest", "ew", -1)],
    "pt": [("norte", "ns", +1), ("sul", "ns", -1), ("leste", "ew", +1), ("oeste", "ew", -1)],
    "nl": [("noorden", "ns", +1), ("zuiden", "ns", -1), ("oosten", "ew", +1), ("westen", "ew", -1)],
}
# NOTE: it/de and pt/es share some compass surfaces (nord, oeste); kept — distinct contexts.
# neutral per-language anchors to estimate the language-surface offset (disjoint from compass)
NEUTRAL = {
    "es": ["cosa", "lugar", "tiempo", "parte", "manera"],
    "fr": ["chose", "endroit", "temps", "partie", "manière"],
    "de": ["sache", "ort", "zeit", "teil", "weise"],
    "it": ["cosa", "luogo", "tempo", "parte", "modo"],
    "pt": ["coisa", "lugar", "tempo", "parte", "maneira"],
    "nl": ["ding", "plaats", "tijd", "deel", "manier"],
}

# left/right egocentric axis (its OWN axis, fit on EN left/right). 6 languages × 2 = 12.
LR_FIT = {"left": -1, "right": 1}
LR_XL = {
    "es": [("izquierda", -1), ("derecha", +1)],
    "fr": [("gauche", -1), ("droite", +1)],
    "de": [("links", -1), ("rechts", +1)],
    "it": [("sinistra", -1), ("destra", +1)],
    "pt": [("esquerda", -1), ("direita", +1)],
    "nl": [("linker", -1), ("rechter", +1)],
}


def _acc(proj, items):
    return sum(1 for t in items
               for (w, ax, sgn) in [t if len(t) == 3 else (t[0], "lr", t[1])]
               if (proj[w][ax] > 0) == (sgn > 0))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()

    xl_words = [w for items in XL.values() for (w, _, _) in items]
    lr_words = [w for items in LR_XL.values() for (w, _) in items]
    neutral_words = [w for ws in NEUTRAL.values() for w in ws]

    A = _collect(mdl, tok, dev, FIT_WORDS, _TMPL, pool="last")
    AX = _collect(mdl, tok, dev, xl_words, _TMPL, pool="last")
    ALR = _collect(mdl, tok, dev, ["left", "right"] + lr_words, _TMPL, pool="last")
    AN = _collect(mdl, tok, dev, neutral_words, _TMPL, pool="last")
    nL = A[FIT_WORDS[0]].shape[0]
    ns_y = np.array([NS_FIT[w] for w in FIT_WORDS], float)
    ew_y = np.array([EW_FIT[w] for w in FIT_WORDS], float)
    lr_y = np.array([-1.0, 1.0])

    nC, nLR = sum(len(v) for v in XL.values()), sum(len(v) for v in LR_XL.values())
    print(f"compass cross-lingual sign acc (/{nC})   "
          f"| left/right (/{nLR})   [chance {nC // 2} / {nLR // 2}]")
    print(f"{'L':>3} | {'base':>5} {'selfC':>5} {'neutC':>5} "
          f"| {'lrBase':>6} {'lrSelfC':>7}")
    for L in range(nL):
        Xf = np.array([A[w][L] for w in FIT_WORDS])
        mu_en, sd = Xf.mean(0), Xf.std(0) + 1e-6
        nsd = _ridge_dir((Xf - mu_en) / sd, ns_y)
        ewd = _ridge_dir((Xf - mu_en) / sd, ew_y)

        # left/right direction fit on EN left/right
        Xlr = np.array([ALR["left"][L], ALR["right"][L]])
        mu_lr, sdl = Xlr.mean(0), Xlr.std(0) + 1e-6
        lrd = _ridge_dir((Xlr - mu_lr) / sdl, lr_y)

        def proj_compass(offset_fn):
            p = {}
            for lang, items in XL.items():
                off = offset_fn(lang, L)
                for (w, ax, _) in items:
                    z = (AX[w][L] - off) / sd
                    p[w] = {"ns": float(z @ nsd), "ew": float(z @ ewd)}
            return p

        base = proj_compass(lambda lang, L: mu_en)
        # self-centering: per-language compass centroid
        lang_mu = {lang: np.mean([AX[w][L] for (w, _, _) in items], 0)
                   for lang, items in XL.items()}
        selfc = proj_compass(lambda lang, L: lang_mu[lang])
        # neutral-centering: per-language neutral-word centroid (disjoint)
        neu_mu = {lang: np.mean([AN[w][L] for w in NEUTRAL[lang]], 0) for lang in NEUTRAL}
        neutc = proj_compass(lambda lang, L: neu_mu[lang])

        compass_items = [(w, ax, sgn) for items in XL.values() for (w, ax, sgn) in items]
        b = _acc(base, compass_items)
        sc = _acc(selfc, compass_items)
        nc = _acc(neutc, compass_items)

        # left/right: baseline (EN mean) vs self-centered (per-lang pair mean)
        def proj_lr(offset_fn):
            p = {}
            for lang, items in LR_XL.items():
                off = offset_fn(lang)
                for (w, _) in items:
                    p[w] = {"lr": float(((ALR[w][L] - off) / sdl) @ lrd)}
            return p
        lr_items = [(w, sgn) for items in LR_XL.values() for (w, sgn) in items]
        lrb = _acc(proj_lr(lambda lang: mu_lr), lr_items)
        lr_mu = {lang: np.mean([ALR[w][L] for (w, _) in items], 0)
                 for lang, items in LR_XL.items()}
        lrs = _acc(proj_lr(lambda lang: lr_mu[lang]), lr_items)

        print(f"{L:>3} | {b:>5} {sc:>5} {nc:>5} | {lrb:>6} {lrs:>7}", flush=True)
    print("\nbase=EN-mean (prior method)  selfC=per-lang compass centroid  "
          "neutC=per-lang neutral-word centroid (disjoint, non-circular)")


if __name__ == "__main__":
    main()
