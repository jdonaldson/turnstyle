"""Does SmolLM2 encode the adjective-ordering hierarchy as a cross-lingual
subjectivity axis?

Linguistics: adjective ordering (opinion → size → age → shape → color → origin →
material → noun) is predicted by SUBJECTIVITY — more subjective adjectives sit
farther from the noun (Scontras, Degen & Goodman 2017). The surface order varies
across languages, but the subjectivity/intrinsicness gradient is argued universal.

Test (the probe-local / reconstruct-global pattern):
  1. Fit a `BipolarAxis` from ONLY the two extremes — opinion (subjective, high) vs
     material (intrinsic, low). The 5 middle categories are HELD OUT.
  2. Recovery: do the held-out categories' mean projections rank in canonical order
     (Spearman of category-mean-subjectivity vs ordering rank)? English in-distribution.
  3. Transfer: does the ENGLISH-fit axis rank ES/FR/DE categories the same way —
     i.e., is the ordering hierarchy cross-lingual inside the model's geometry?

Fresh collection (these ordering-category adjectives aren't in prior caches).
Usage:  python experiments/subjectivity_order.py [--collect]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_probe as PP

CACHE = "experiments/data/subjectivity_order_acts.npz"

# ordering category -> canonical rank (1 = farthest from noun / most subjective)
RANK = {"opinion": 1, "size": 2, "age": 3, "shape": 4,
        "color": 5, "origin": 6, "material": 7}

# category -> {lang: [adjectives]}; attributive, common, translation-checked
LEX = {
    "opinion":  {"en": ["lovely", "nice", "ugly", "horrible"],
                 "es": ["encantador", "agradable", "feo", "horrible"],
                 "fr": ["charmant", "agréable", "laid", "horrible"],
                 "de": ["reizend", "nett", "hässlich", "schrecklich"]},
    "size":     {"en": ["big", "small", "huge", "tiny"],
                 "es": ["grande", "pequeño", "enorme", "diminuto"],
                 "fr": ["grand", "petit", "énorme", "minuscule"],
                 "de": ["groß", "klein", "riesig", "winzig"]},
    "age":      {"en": ["old", "new", "young", "ancient"],
                 "es": ["viejo", "nuevo", "joven", "antiguo"],
                 "fr": ["vieux", "nouveau", "jeune", "ancien"],
                 "de": ["alt", "neu", "jung", "uralt"]},
    "shape":    {"en": ["round", "square", "flat", "triangular"],
                 "es": ["redondo", "cuadrado", "plano", "triangular"],
                 "fr": ["rond", "carré", "plat", "triangulaire"],
                 "de": ["rund", "quadratisch", "flach", "dreieckig"]},
    "color":    {"en": ["red", "blue", "green", "black"],
                 "es": ["rojo", "azul", "verde", "negro"],
                 "fr": ["rouge", "bleu", "vert", "noir"],
                 "de": ["rot", "blau", "grün", "schwarz"]},
    "origin":   {"en": ["French", "Chinese", "Greek", "Roman"],
                 "es": ["francés", "chino", "griego", "romano"],
                 "fr": ["français", "chinois", "grec", "romain"],
                 "de": ["französisch", "chinesisch", "griechisch", "römisch"]},
    "material": {"en": ["metallic", "plastic", "golden", "wooden"],
                 "es": ["metálico", "plástico", "dorado", "de madera"],
                 "fr": ["métallique", "plastique", "doré", "en bois"],
                 "de": ["metallisch", "plastik", "golden", "hölzern"]},
}

# attributive template per language (read the adjective's last token)
TMPL = {"en": "It is a {a} object.", "es": "Es un objeto {a}.",
        "fr": "C'est un objet {a}.", "de": "Es ist ein {a} Objekt."}


def collect():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(PP.MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        PP.MODEL_ID, dtype=torch.float16).to(dev).eval()

    acts, cat, lang, word = [], [], [], []
    for c, langs in LEX.items():
        for lg, words in langs.items():
            for a in words:
                sent = TMPL[lg].format(a=a)
                content = a.split()[-1]
                cs = sent.rfind(content)
                ce = cs + len(content)
                enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
                offs = enc.pop("offset_mapping")[0].tolist()
                enc = {k: v.to(dev) for k, v in enc.items()}
                with torch.no_grad():
                    out = mdl(**enc, output_hidden_states=True)
                hs = torch.stack(out.hidden_states, 0)[:, 0].float().cpu().numpy()
                tk = next((k for k, (s, e) in enumerate(offs) if e > cs and s < ce), None)
                if tk is None:
                    continue
                acts.append(hs[:, tk, :].astype(np.float16))
                cat.append(c); lang.append(lg); word.append(a)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, acts=np.stack(acts), cat=np.array(cat),
                        lang=np.array(lang), word=np.array(word))
    print(f"saved {len(acts)} points → {CACHE}")


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx @ ry) / (np.sqrt(rx @ rx) * np.sqrt(ry @ ry) + 1e-12))


def analyze():
    d = np.load(CACHE, allow_pickle=True)
    A = d["acts"].astype(np.float32)
    cat = d["cat"]; lang = d["lang"]
    nL = A.shape[1]
    cats = sorted(RANK, key=RANK.get)
    held = [c for c in cats if c not in ("opinion", "material")]
    print(f"N={len(cat)}  fit axis on opinion(high)/material(low); "
          f"held-out middle = {held}")
    print(f"{'L':>3}  {'EN ρ(all)':>9} {'EN ρ(held)':>10}  {'ES ρ':>6} {'FR ρ':>6} {'DE ρ':>6}")
    best = (0.0, -1)
    for L in range(nL):
        X = A[:, L, :]
        en = lang == "en"
        mu = X[en].mean(0); sd = X[en].std(0) + 1e-6
        Z = (X - mu) / sd
        hi = Z[en & (cat == "opinion")].mean(0)
        lo = Z[en & (cat == "material")].mean(0)
        dirn = hi - lo
        dirn = dirn / (np.linalg.norm(dirn) + 1e-12)
        proj = Z @ dirn                      # subjectivity score (high = subjective)

        def rho(mask, subset=cats):
            cs, ys = [], []
            for c in subset:
                m = mask & (cat == c)
                if m.any():
                    cs.append(proj[m].mean()); ys.append(RANK[c])
            return _spearman(np.array(cs), np.array(ys))

        en_all = rho(en); en_held = rho(en, held)
        es = rho(lang == "es"); fr = rho(lang == "fr"); de = rho(lang == "de")
        # expect strong NEGATIVE rho (subjectivity decreases with rank)
        if abs((es + fr + de) / 3) > best[0]:
            best = (abs((es + fr + de) / 3), L)
        print(f"{L:>3}  {en_all:>9.2f} {en_held:>10.2f}  {es:>6.2f} {fr:>6.2f} {de:>6.2f}")
    print(f"\nbest cross-lingual layer L{best[1]} (mean|ρ| es/fr/de = {best[0]:.2f})")
    print("ρ near -1 = subjectivity axis recovers the canonical ordering hierarchy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect or not os.path.exists(CACHE):
        collect()
    analyze()
