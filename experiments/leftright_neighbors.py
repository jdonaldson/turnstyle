"""What is left/right NEAR, and how does it move through the layers / across languages?

Hypothesis for why egocentric left/right transfers weakly (unlike clean allocentric compass):
left/right are heavily POLYSEMOUS and the spatial-lateral sense competes with others —
  right  -> correct ("that's right") | law/entitlement (derecho/droit/Recht = law) | politics
  left   -> remaining ("what's left") | departed ("he left") | politics
The compass words have ~no such competition. So we expect each lateral word to sit near its
DISTRACTOR sense (correct / law) early, and only weakly/late drift toward its lateral PARTNER
and the spatial cluster — and the distractors diverge across languages (the law-cognate is
strong in es/fr/de, the correctness sense in en), explaining the cross-lingual weakness.

Method: a curated multilingual, multi-concept lexicon; per layer, mean-center the lexicon and
report (a) each lateral target's top-3 nearest neighbors and (b) its cosine to a few labelled
references: its lateral PARTNER, same-language CORRECT, same-language LAW, and compass EAST.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/leftright_neighbors.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from turnstyle.frame_library import _collect

_TMPL = "It is {w}."   # neutral frame; not a spatial "Move {w}." that biases toward direction

# (word, lang, concept)
LEX = [
    # English
    ("left", "en", "lateral"), ("right", "en", "lateral"),
    ("correct", "en", "correct"), ("wrong", "en", "correct"), ("true", "en", "correct"),
    ("law", "en", "law"), ("justice", "en", "law"), ("legal", "en", "law"),
    ("remaining", "en", "remain"), ("leftover", "en", "remain"), ("departed", "en", "remain"),
    ("liberal", "en", "politics"), ("conservative", "en", "politics"),
    ("north", "en", "compass"), ("south", "en", "compass"),
    ("east", "en", "compass"), ("west", "en", "compass"),
    ("up", "en", "vertical"), ("down", "en", "vertical"),
    ("hand", "en", "body"), ("thing", "en", "filler"), ("place", "en", "filler"),
    # Spanish  (derecho = law/straight; derecha = lateral-right)
    ("izquierda", "es", "lateral"), ("derecha", "es", "lateral"),
    ("correcto", "es", "correct"), ("derecho", "es", "law"),
    ("norte", "es", "compass"), ("sur", "es", "compass"),
    ("este", "es", "compass"), ("oeste", "es", "compass"), ("cosa", "es", "filler"),
    # French  (droit = law/straight; droite = lateral-right)
    ("gauche", "fr", "lateral"), ("droite", "fr", "lateral"),
    ("juste", "fr", "correct"), ("droit", "fr", "law"),
    ("nord", "fr", "compass"), ("sud", "fr", "compass"),
    ("est", "fr", "compass"), ("ouest", "fr", "compass"), ("chose", "fr", "filler"),
    # German  (Recht = law/right; rechts = lateral-right)
    ("links", "de", "lateral"), ("rechts", "de", "lateral"),
    ("richtig", "de", "correct"), ("recht", "de", "law"),
    ("norden", "de", "compass"), ("süden", "de", "compass"),
    ("osten", "de", "compass"), ("westen", "de", "compass"), ("sache", "de", "filler"),
]
WORDS = [w for w, _, _ in LEX]
LANG = {w: l for w, l, _ in LEX}
CONC = {w: c for w, _, c in LEX}

# the lateral targets we trace, and per-language reference words
TARGETS = ["right", "left", "derecha", "izquierda", "droite", "rechts"]
PARTNER = {"right": "left", "left": "right", "derecha": "izquierda", "izquierda": "derecha",
           "droite": "gauche", "rechts": "links"}
CORRECT = {"en": "correct", "es": "correcto", "fr": "juste", "de": "richtig"}
LAW = {"en": "law", "es": "derecho", "fr": "droit", "de": "recht"}
EAST = {"en": "east", "es": "este", "fr": "est", "de": "osten"}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()

    acts = _collect(mdl, tok, dev, WORDS, _TMPL, pool="last")
    nL = acts[WORDS[0]].shape[0]

    def layer_mat(L):
        X = np.array([acts[w][L] for w in WORDS])
        X = X - X.mean(0)                       # mean-center the lexicon
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        return X
    idx = {w: i for i, w in enumerate(WORDS)}

    def cos(L, a, b):
        X = layer_mat(L)
        return float(X[idx[a]] @ X[idx[b]])

    # ── Per-target reference cosines through the layers ───────────────────────
    for t in TARGETS:
        lg = LANG[t]
        print(f"\n=== {t} ({lg}) — cosine to references, per layer "
              f"(partner / correct / law / east) ===")
        print(f"{'L':>3} | {'partner':>8} {'correct':>8} {'law':>8} {'east':>8} | nearest-3")
        for L in range(nL):
            X = layer_mat(L)
            sims = X @ X[idx[t]]
            order = np.argsort(-sims)
            nn = [WORDS[j] for j in order if WORDS[j] != t][:3]
            cp = cos(L, t, PARTNER[t])
            cc = cos(L, t, CORRECT[lg])
            cl = cos(L, t, LAW[lg])
            ce = cos(L, t, EAST[lg])
            print(f"{L:>3} | {cp:>8.2f} {cc:>8.2f} {cl:>8.2f} {ce:>8.2f} | "
                  f"{', '.join(nn)}", flush=True)

    # ── Summary: at each layer, what concept is each lateral target nearest to? ─
    print("\n\n=== concept of the NEAREST neighbor, per layer (which sense wins) ===")
    print(f"{'L':>3} | " + " ".join(f"{t:>10}" for t in TARGETS))
    for L in range(nL):
        X = layer_mat(L)
        row = []
        for t in TARGETS:
            sims = X @ X[idx[t]]
            order = [j for j in np.argsort(-sims) if WORDS[j] != t]
            row.append(CONC[WORDS[order[0]]])
        print(f"{L:>3} | " + " ".join(f"{c:>10}" for c in row), flush=True)


if __name__ == "__main__":
    main()
