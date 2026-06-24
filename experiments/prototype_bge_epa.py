"""Prototype B: BGE-M3 as the basis for a cross-lingual EPA embedding.

Route B from the design discussion — use an aligned multilingual encoder so the
shared space (and thus cross-lingual EPA) comes for free, no Procrustes step. We
embed the EPA pole words in 4 languages with BGE-M3 (CLS-pooled + normalized =
its dense embedding), fit the EPA "head" (bipolar axes) on ENGLISH, and test:
  held-out  — leave-one-word-out pole sign (English): does BGE-M3 carry EPA?
  x-ling    — English-fit axis signs es/fr/de words: is the space shared?
  ortho     — |cos| between the three axes

Contrast with what we already have:
  fastText (static, English):  Eval 0.90 held-out, NO cross-lingual (separate spaces)
  SmolLM2  (contextual, L13):  Eval 1.00 held-out, 0.83 cross-lingual

Usage:  python experiments/prototype_bge_epa.py   (downloads BAAI/bge-m3 ~2.2GB once)
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import osgood_epa as OE

MODEL = "BAAI/bge-m3"
FACS = ["evaluation", "potency", "activity"]
LANGS = ["en", "es", "fr", "de"]


def embed_all():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).to(dev).eval()

    rows = []  # (vec, factor, lang, pole)
    for f in FACS:
        for lg in LANGS:
            for pol, words in (("hi", OE.EPA[f][lg]["hi"]), ("lo", OE.EPA[f][lg]["lo"])):
                for w in words:
                    enc = {k: v.to(dev) for k, v in tok(w, return_tensors="pt",
                                                        truncation=True).items()}
                    with torch.no_grad():
                        out = mdl(**enc)
                    v = F.normalize(out.last_hidden_state[:, 0], dim=-1)[0]  # BGE-M3 dense = CLS, norm
                    rows.append((v.float().cpu().numpy(), f, lg, 1 if pol == "hi" else -1))
    A = np.vstack([r[0] for r in rows])
    fac = np.array([r[1] for r in rows]); lang = np.array([r[2] for r in rows])
    pole = np.array([r[3] for r in rows])
    return A, fac, lang, pole


def _axis(hi, lo):
    d = hi.mean(0) - lo.mean(0); d = d / (np.linalg.norm(d) + 1e-12)
    return d, 0.5 * (hi.mean(0) + lo.mean(0)) @ d


def main():
    A, fac, lang, pole = embed_all()
    en = lang == "en"
    mu = A[en].mean(0); sd = A[en].std(0) + 1e-6
    Z = (A - mu) / sd
    print(f"BGE-M3 cross-lingual EPA prototype  (N={len(A)}, dim={A.shape[1]})\n")
    print(f"{'factor':11} {'held-out':>9} {'x-ling':>8}")
    dirs = {}
    for f in FACS:
        m = fac == f
        items = np.where(en & m)[0]
        ok = 0
        for i in items:
            keep = (en & m) & (np.arange(len(A)) != i)
            d, c = _axis(Z[keep & (pole == 1)], Z[keep & (pole == -1)])
            ok += int((Z[i] @ d - c > 0) == (pole[i] > 0))
        held = ok / len(items)
        d, c = _axis(Z[en & m & (pole == 1)], Z[en & m & (pole == -1)])
        dirs[f] = d
        non = (~en) & m
        xl = (((Z[non] @ d - c) > 0) == (pole[non] > 0)).mean()
        print(f"{f:11} {held:>9.2f} {xl:>8.2f}")
    print("\nfactor independence |cos|:")
    for a, b in itertools.combinations(FACS, 2):
        print(f"  {a[:4].title()}-{b[:4].title()}: {abs(float(dirs[a] @ dirs[b])):.2f}")
    print("\ncompare:  fastText x-ling = N/A (separate spaces) | SmolLM2 x-ling ~0.6-0.87")


if __name__ == "__main__":
    main()
