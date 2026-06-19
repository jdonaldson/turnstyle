"""Cross-language proof for the adjective-polarity primitive.

If polarity is a SEMANTIC direction (not lexical/English), an English-trained
probe should pole adjectives in other languages correctly — "el más viejo",
"le plus grand", "am teuersten" — even though it never saw those tokens. That
is the property that makes it worth shipping as a primitive: one calibration,
many languages.

Clean design: a shared intensifier template ("... very {adj}." / muy / très /
sehr) so ONLY the language varies. Train the probe on English activations, test
per-language / per-axis pole accuracy. Age is included to check the SmolLM2
"old" collapse replicates cross-lingually (it should, if it's semantic).

Usage:  python experiments/cross_language.py [--collect]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_harness as H
import pole_generalize as G

CACHE = "experiments/data/cross_language_acts.npz"

# (axis, adjective, pole)  pole +1 = HIGH end ("more of attribute")
LANGS = {
    "en": {
        "tmpl": "In comparison, this {n} is very {adj}.",
        "nouns": ["object", "item", "car"],
        "adj": [("size", "big", +1), ("size", "small", -1), ("size", "tall", +1),
                ("size", "short", -1), ("size", "long", +1), ("size", "wide", +1),
                ("size", "narrow", -1), ("weight", "heavy", +1), ("weight", "light", -1),
                ("speed", "fast", +1), ("speed", "slow", -1), ("temp", "hot", +1),
                ("temp", "cold", -1), ("value", "expensive", +1), ("value", "cheap", -1),
                ("value", "rich", +1), ("value", "poor", -1), ("quality", "good", +1),
                ("quality", "bad", -1), ("intensity", "strong", +1), ("intensity", "weak", -1),
                ("intensity", "bright", +1), ("intensity", "dark", -1), ("age", "new", +1),
                ("age", "old", -1), ("mood", "happy", +1), ("mood", "sad", -1)],
    },
    "es": {
        "tmpl": "En comparación, este {n} es muy {adj}.",
        "nouns": ["objeto", "coche", "libro"],
        "adj": [("size", "grande", +1), ("size", "pequeño", -1), ("size", "alto", +1),
                ("size", "bajo", -1), ("size", "largo", +1), ("size", "ancho", +1),
                ("size", "estrecho", -1), ("weight", "pesado", +1), ("weight", "ligero", -1),
                ("speed", "rápido", +1), ("speed", "lento", -1), ("temp", "caliente", +1),
                ("temp", "frío", -1), ("value", "caro", +1), ("value", "barato", -1),
                ("value", "rico", +1), ("value", "pobre", -1), ("quality", "bueno", +1),
                ("quality", "malo", -1), ("intensity", "fuerte", +1), ("intensity", "débil", -1),
                ("intensity", "brillante", +1), ("intensity", "oscuro", -1), ("age", "nuevo", +1),
                ("age", "viejo", -1), ("mood", "feliz", +1), ("mood", "triste", -1)],
    },
    "fr": {
        "tmpl": "En comparaison, cet {n} est très {adj}.",
        "nouns": ["objet", "livre", "vélo"],
        "adj": [("size", "grand", +1), ("size", "petit", -1), ("size", "haut", +1),
                ("size", "long", +1), ("size", "large", +1), ("size", "étroit", -1),
                ("weight", "lourd", +1), ("weight", "léger", -1), ("speed", "rapide", +1),
                ("speed", "lent", -1), ("temp", "chaud", +1), ("temp", "froid", -1),
                ("value", "cher", +1), ("value", "riche", +1), ("value", "pauvre", -1),
                ("quality", "bon", +1), ("quality", "mauvais", -1), ("intensity", "fort", +1),
                ("intensity", "faible", -1), ("intensity", "brillant", +1), ("intensity", "sombre", -1),
                ("age", "neuf", +1), ("age", "vieux", -1), ("mood", "heureux", +1),
                ("mood", "triste", -1)],
    },
    "de": {
        "tmpl": "Im Vergleich ist dieses {n} sehr {adj}.",
        "nouns": ["Objekt", "Auto", "Buch"],
        "adj": [("size", "groß", +1), ("size", "klein", -1), ("size", "hoch", +1),
                ("size", "lang", +1), ("size", "breit", +1), ("size", "schmal", -1),
                ("weight", "schwer", +1), ("weight", "leicht", -1), ("speed", "schnell", +1),
                ("speed", "langsam", -1), ("temp", "heiß", +1), ("temp", "kalt", -1),
                ("value", "teuer", +1), ("value", "billig", -1), ("value", "reich", +1),
                ("value", "arm", -1), ("quality", "gut", +1), ("quality", "schlecht", -1),
                ("intensity", "stark", +1), ("intensity", "schwach", -1), ("intensity", "hell", +1),
                ("intensity", "dunkel", -1), ("age", "neu", +1), ("age", "alt", -1),
                ("mood", "glücklich", +1), ("mood", "traurig", -1)],
    },
}


def collect():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pole_probe as PP

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(PP.MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        PP.MODEL_ID, dtype=torch.float16).to(device).eval()

    acts, poles, langs, axes, words = [], [], [], [], []
    for lang, spec in LANGS.items():
        for axis, adj, pole in spec["adj"]:
            for n in spec["nouns"]:
                sent = spec["tmpl"].format(n=n, adj=adj)
                cs = sent.rfind(adj)
                ce = cs + len(adj)
                enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
                offs = enc.pop("offset_mapping")[0].tolist()
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = mdl(**enc, output_hidden_states=True)
                hs = torch.stack(out.hidden_states, 0)[:, 0].float().cpu().numpy()
                tk = None
                for k, (s, e) in enumerate(offs):
                    if e > cs and s < ce:
                        tk = k
                if tk is None:
                    continue
                acts.append(hs[:, tk, :].astype(np.float16))
                poles.append(H.HIGH if pole > 0 else H.LOW)
                langs.append(lang)
                axes.append(axis)
                words.append(adj)
        print(f"  {lang} done ({len(acts)})", end="\r", flush=True)
    print()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, acts=np.stack(acts), poles=np.array(poles),
                        langs=np.array(langs), axes=np.array(axes),
                        words=np.array(words))
    print(f"saved {len(acts)} occurrences → {CACHE}")


def analyze():
    d = np.load(CACHE, allow_pickle=True)
    A = d["acts"].astype(np.float32)
    y = (d["poles"] == H.HIGH).astype(int)
    langs = d["langs"]
    axes = d["axes"]

    en = langs == "en"
    nL = A.shape[1]
    print("Train probe on ENGLISH 'very {adj}' acts → test other languages.\n")
    print(f"{'L':>3}  " + "  ".join(f"{lg:>6}" for lg in ["es", "fr", "de"]))
    best = (-1.0, -1)
    for L in range(nL):
        sc, clf = G._fit(A[en, L, :], y[en])
        row = []
        for lg in ["es", "fr", "de"]:
            m = langs == lg
            row.append(clf.score(sc.transform(A[m, L, :]), y[m]))
        avg = float(np.mean(row))
        if avg > best[0]:
            best = (avg, L)
        print(f"{L:>3}  " + "  ".join(f"{v:6.2f}" for v in row))
    Lb = best[1]
    print(f"\nbest transfer layer L{Lb} (avg es/fr/de = {best[0]:.2f})")

    # per-axis at best layer, excluding age vs including
    sc, clf = G._fit(A[en, Lb, :], y[Lb == Lb] if False else y[en])
    print(f"\nper-language @L{Lb}, split by axis (age = known SmolLM2 collapse):")
    for lg in ["es", "fr", "de"]:
        m = langs == lg
        pred = clf.predict(sc.transform(A[m, Lb, :]))
        truth = y[m]
        ax = axes[m]
        non_age = ax != "age"
        overall = (pred == truth).mean()
        no_age = (pred[non_age] == truth[non_age]).mean()
        age_m = ax == "age"
        age_acc = (pred[age_m] == truth[age_m]).mean() if age_m.any() else float("nan")
        print(f"  {lg}: all={overall:.2f}  non-age={no_age:.2f}  age={age_acc:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect or not os.path.exists(CACHE):
        collect()
    analyze()
