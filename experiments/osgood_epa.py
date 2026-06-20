"""Osgood's semantic differential (E-P-A) as a cross-lingual semantic frame.

Osgood (1957; cross-cultural, 1975) found connotative meaning organizes into three
universal bipolar factors: Evaluation (good-bad), Potency (strong-weak), Activity
(active-passive). This tests whether SmolLM2 encodes them, and crucially whether it
keeps them INDEPENDENT (Osgood's central claim — three *separate* factors).

Three consistency tests per axis:
  held-out — leave-one-word-out sign accuracy (English)
  x-ling   — English-fit axis signs ES/FR/DE pole words correctly
  ortho    — |cos| between the E, P, A directions (low = model keeps Osgood's
             factors separate; high = it collapses them)

Usage:  python experiments/osgood_epa.py [--collect]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_probe as PP
from turnstyle.semantic_frame import fit_axis_from_vectors

CACHE = "experiments/data/osgood_epa_acts.npz"

# factor -> {lang: {"hi": [...], "lo": [...]}}
EPA = {
    "evaluation": {
        "en": {"hi": ["good", "nice", "beautiful", "pleasant", "kind"],
               "lo": ["bad", "nasty", "ugly", "unpleasant", "cruel"]},
        "es": {"hi": ["bueno", "agradable", "hermoso", "placentero", "amable"],
               "lo": ["malo", "desagradable", "feo", "doloroso", "cruel"]},
        "fr": {"hi": ["bon", "agréable", "beau", "plaisant", "gentil"],
               "lo": ["mauvais", "désagréable", "laid", "pénible", "cruel"]},
        "de": {"hi": ["gut", "nett", "schön", "angenehm", "freundlich"],
               "lo": ["schlecht", "gemein", "hässlich", "unangenehm", "grausam"]}},
    "potency": {
        "en": {"hi": ["strong", "powerful", "big", "heavy", "hard"],
               "lo": ["weak", "powerless", "small", "light", "soft"]},
        "es": {"hi": ["fuerte", "poderoso", "grande", "pesado", "duro"],
               "lo": ["débil", "impotente", "pequeño", "ligero", "blando"]},
        "fr": {"hi": ["fort", "puissant", "grand", "lourd", "dur"],
               "lo": ["faible", "impuissant", "petit", "léger", "mou"]},
        "de": {"hi": ["stark", "mächtig", "groß", "schwer", "hart"],
               "lo": ["schwach", "machtlos", "klein", "leicht", "weich"]}},
    "activity": {
        "en": {"hi": ["active", "fast", "lively", "quick", "energetic"],
               "lo": ["passive", "slow", "calm", "sluggish", "lazy"]},
        "es": {"hi": ["activo", "rápido", "vivo", "ágil", "enérgico"],
               "lo": ["pasivo", "lento", "tranquilo", "perezoso", "inactivo"]},
        "fr": {"hi": ["actif", "rapide", "vif", "agile", "énergique"],
               "lo": ["passif", "lent", "calme", "paresseux", "inactif"]},
        "de": {"hi": ["aktiv", "schnell", "lebhaft", "flink", "energisch"],
               "lo": ["passiv", "langsam", "ruhig", "träge", "faul"]}},
}
TMPL = {"en": "It is very {a}.", "es": "Es muy {a}.",
        "fr": "C'est très {a}.", "de": "Es ist sehr {a}."}


def collect():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(PP.MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        PP.MODEL_ID, dtype=torch.float16).to(dev).eval()
    acts, fac, lang, pole, word = [], [], [], [], []
    for f, langs in EPA.items():
        for lg, poles in langs.items():
            for pol, words in poles.items():
                for a in words:
                    sent = TMPL[lg].format(a=a)
                    cs = sent.rfind(a); ce = cs + len(a)
                    enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
                    offs = enc.pop("offset_mapping")[0].tolist()
                    enc = {k: v.to(dev) for k, v in enc.items()}
                    with torch.no_grad():
                        out = mdl(**enc, output_hidden_states=True)
                    hs = torch.stack(out.hidden_states, 0)[:, 0].float().cpu().numpy()
                    tk = next((k for k, (s, e) in enumerate(offs) if e > cs and s < ce), None)
                    acts.append(hs[:, tk, :].astype(np.float16))
                    fac.append(f); lang.append(lg)
                    pole.append(1 if pol == "hi" else -1); word.append(a)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, acts=np.stack(acts), fac=np.array(fac),
                        lang=np.array(lang), pole=np.array(pole), word=np.array(word))
    print(f"saved {len(acts)} points → {CACHE}")


def analyze():
    d = np.load(CACHE, allow_pickle=True)
    A = d["acts"].astype(np.float32); fac = d["fac"]; lang = d["lang"]; pole = d["pole"]
    facs = ["evaluation", "potency", "activity"]
    nL = A.shape[1]
    print(f"N={len(fac)}  factors={facs}")
    print(f"{'L':>3}  " + "  ".join(f"{f[:4]}:held/xl" for f in facs) + "   ortho|cos| E-P E-A P-A")
    best = (0.0, -1)
    for L in range(nL):
        X = A[:, L, :]
        en = lang == "en"
        mu = X[en].mean(0); sd = X[en].std(0) + 1e-6
        Z = (X - mu) / sd
        dirs = {}
        cells = []
        xl_means = []
        for f in facs:
            m = fac == f
            hi = Z[en & m & (pole == 1)]; lo = Z[en & m & (pole == -1)]
            ax = fit_axis_from_vectors(f, "lo", "hi", L, hi, lo)
            d_ = ax.direction; dirs[f] = d_
            c = ax.center
            # held-out LOO sign (English)
            ehi = np.where(en & m & (pole == 1))[0]; elo = np.where(en & m & (pole == -1))[0]
            ok = tot = 0
            for idx, want in [(i, 1) for i in ehi] + [(i, -1) for i in elo]:
                keep = (en & m) & (np.arange(len(X)) != idx)
                h2 = Z[keep & (pole == 1)]; l2 = Z[keep & (pole == -1)]
                a2 = fit_axis_from_vectors(f, "lo", "hi", L, h2, l2)
                s = (Z[idx] @ a2.direction - a2.center)
                ok += int((s > 0) == (want > 0)); tot += 1
            held = ok / tot
            # cross-lingual sign
            non = (~en) & m
            sgn = (Z[non] @ d_ - c)
            xl = ((sgn > 0) == (pole[non] > 0)).mean()
            cells.append(f"{held:.2f}/{xl:.2f}")
            xl_means.append(xl)
        cos = {(a, b): abs(float(dirs[a] @ dirs[b])) for a, b in itertools.combinations(facs, 2)}
        mxl = float(np.mean(xl_means))
        if mxl > best[0]:
            best = (mxl, L)
        print(f"{L:>3}  " + "  ".join(f"{c:>11}" for c in cells) +
              f"   {cos[('evaluation','potency')]:.2f} {cos[('evaluation','activity')]:.2f} "
              f"{cos[('potency','activity')]:.2f}")
    print(f"\nbest cross-lingual layer L{best[1]} (mean x-ling sign = {best[0]:.2f})")
    print("held = leave-one-word-out sign (EN); xl = x-lingual sign (es/fr/de pooled);")
    print("ortho |cos|: low = model keeps Osgood's E/P/A factors independent")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect or not os.path.exists(CACHE):
        collect()
    analyze()
