"""Model-measurement probe: does a dyf tree recover cross-lingual concept structure?

Generalizes the polarity capability (one linear axis) to a DESCRIPTIVE measurement
of the geometry a model imposes on a multi-level concept space. Two decisive
questions, at a scale (~1k points) where the prior dyf×turnstyle falsifications
(N≈200, "kNN beats dyf-leaf") no longer apply:

  (a) dyf vs flat  — build_dyf_tree + cut_tree_to_labels(K) vs KMeans(K): does the
      tree recover the known category taxonomy better than flat clustering? (the
      descriptive analog of the falsified calibration test — no predictive baseline
      to lose to; the deliverable is structural fidelity.)
  (b) concept vs language — is the geometry organized by MEANING (animal/fruit/...,
      pooled across languages) or by SURFACE (which language)? High category-NMI +
      low language-NMI = a language-agnostic concept geometry (the cross-lingual
      "much more structure" payoff that the polarity result hinted at).

Probe set: a concrete-noun taxonomy (deep, translation-stable) + scalar-adjective
axes (the polarity seed), in EN/ES/FR/DE, read in a neutral shared template so the
adjective/noun domain split is not template-confounded (a trap the prior arc hit).

Usage:  python experiments/concept_geometry.py [--collect] [--layer L]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_probe as PP   # MODEL_ID + collection helpers

CACHE = "experiments/data/concept_geometry_acts.npz"

# ── concrete-noun taxonomy (category -> {lang: [items]}), translation-stable ──
NOUNS = {
    "animal":   {"en": ["dog", "cat", "horse", "bird", "fish", "cow", "lion", "bear"],
                 "es": ["perro", "gato", "caballo", "pájaro", "pez", "vaca", "león", "oso"],
                 "fr": ["chien", "chat", "cheval", "oiseau", "poisson", "vache", "lion", "ours"],
                 "de": ["Hund", "Katze", "Pferd", "Vogel", "Fisch", "Kuh", "Löwe", "Bär"]},
    "fruit":    {"en": ["apple", "banana", "orange", "grape", "lemon", "pear", "cherry"],
                 "es": ["manzana", "plátano", "naranja", "uva", "limón", "pera", "cereza"],
                 "fr": ["pomme", "banane", "orange", "raisin", "citron", "poire", "cerise"],
                 "de": ["Apfel", "Banane", "Orange", "Traube", "Zitrone", "Birne", "Kirsche"]},
    "body":     {"en": ["hand", "foot", "head", "eye", "arm", "leg", "nose"],
                 "es": ["mano", "pie", "cabeza", "ojo", "brazo", "pierna", "nariz"],
                 "fr": ["main", "pied", "tête", "bras", "jambe", "nez", "genou"],
                 "de": ["Hand", "Fuß", "Kopf", "Auge", "Arm", "Bein", "Nase"]},
    "vehicle":  {"en": ["car", "train", "boat", "plane", "bicycle", "truck"],
                 "es": ["coche", "tren", "barco", "avión", "bicicleta", "camión"],
                 "fr": ["voiture", "train", "bateau", "avion", "vélo", "camion"],
                 "de": ["Auto", "Zug", "Boot", "Flugzeug", "Fahrrad", "Lastwagen"]},
    "clothing": {"en": ["shirt", "shoe", "hat", "coat", "dress", "glove"],
                 "es": ["camisa", "zapato", "sombrero", "abrigo", "vestido", "guante"],
                 "fr": ["chemise", "chaussure", "chapeau", "manteau", "robe", "gant"],
                 "de": ["Hemd", "Schuh", "Hut", "Mantel", "Kleid", "Handschuh"]},
    "furniture":{"en": ["table", "chair", "bed", "desk", "shelf", "sofa"],
                 "es": ["mesa", "silla", "cama", "escritorio", "estante", "sofá"],
                 "fr": ["table", "chaise", "lit", "bureau", "étagère", "canapé"],
                 "de": ["Tisch", "Stuhl", "Bett", "Schreibtisch", "Regal", "Sofa"]},
    "tool":     {"en": ["hammer", "knife", "saw", "drill", "wrench"],
                 "es": ["martillo", "cuchillo", "sierra", "taladro", "llave"],
                 "fr": ["marteau", "couteau", "scie", "perceuse", "tournevis"],
                 "de": ["Hammer", "Messer", "Säge", "Bohrer", "Zange"]},
}

# ── scalar-adjective axes (the polarity seed) -> {lang: [words]} ──────────────
ADJ = {
    "size":  {"en": ["big", "small", "tall", "short", "wide"],
              "es": ["grande", "pequeño", "alto", "bajo", "ancho"],
              "fr": ["grand", "petit", "haut", "court", "large"],
              "de": ["groß", "klein", "hoch", "kurz", "breit"]},
    "temp":  {"en": ["hot", "cold", "warm", "cool"],
              "es": ["caliente", "frío", "templado", "fresco"],
              "fr": ["chaud", "froid", "tiède", "frais"],
              "de": ["heiß", "kalt", "warm", "kühl"]},
    "value": {"en": ["expensive", "cheap", "rich", "poor"],
              "es": ["caro", "barato", "rico", "pobre"],
              "fr": ["cher", "économique", "riche", "pauvre"],
              "de": ["teuer", "billig", "reich", "arm"]},
    "speed": {"en": ["fast", "slow", "quick", "rapid"],
              "es": ["rápido", "lento", "veloz", "ligero"],
              "fr": ["rapide", "lent", "vif", "prompt"],
              "de": ["schnell", "langsam", "flink", "rasch"]},
}

# neutral carriers — grammatical for any part of speech, so the noun/adj domain
# split is semantic, not template-driven
_TEMPLATES = [
    "Here is a word: {w}.",
    "I am thinking about {w}.",
    "The word is {w}.",
    "Consider this: {w}.",
]


def _iter_items():
    for cat, langs in NOUNS.items():
        for lang, words in langs.items():
            for w in words:
                yield ("noun", cat, lang, w)
    for axis, langs in ADJ.items():
        for lang, words in langs.items():
            for w in words:
                yield ("adj", axis, lang, w)


def collect_activations():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(PP.MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        PP.MODEL_ID, dtype=torch.float16).to(device).eval()

    acts, domain, category, language, word = [], [], [], [], []
    items = list(_iter_items())
    for n, (dom, cat, lang, w) in enumerate(items):
        for tmpl in _TEMPLATES:
            sent = tmpl.format(w=w)
            cs = sent.rfind(w)
            ce = cs + len(w)
            enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = mdl(**enc, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, 0)[:, 0].float().cpu().numpy()
            tk = next((k for k, (s, e) in enumerate(offs) if e > cs and s < ce), None)
            if tk is None:
                continue
            acts.append(hs[:, tk, :].astype(np.float16))
            domain.append(dom); category.append(cat)
            language.append(lang); word.append(w)
        print(f"  [{n+1}/{len(items)}] {cat:9s} {lang} {w:14s} N={len(acts)}",
              end="\r", flush=True)
    print()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, acts=np.stack(acts), domain=np.array(domain),
                        category=np.array(category), language=np.array(language),
                        word=np.array(word))
    print(f"saved {len(acts)} points → {CACHE}")


# ── measurement ──────────────────────────────────────────────────────────────

def _nmi_ari(labels, truth):
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    return (normalized_mutual_info_score(truth, labels),
            adjusted_rand_score(truth, labels))


def analyze(layer: int):
    import dyf
    from sklearn.cluster import KMeans

    d = np.load(CACHE, allow_pickle=True)
    A = d["acts"].astype(np.float32)[:, layer, :]
    cat = d["category"]; lang = d["language"]; dom = d["domain"]
    N = len(cat)
    cats = sorted(set(cat.tolist()))
    K = len(cats)                       # 7 noun cats + 4 adj axes = 11
    print(f"N={N} points, L{layer}, {K} categories, langs={sorted(set(lang.tolist()))}")

    # standardize (dyf pca + kmeans both benefit; keeps the comparison fair)
    Az = (A - A.mean(0)) / (A.std(0) + 1e-6)

    # dyf tree → cut to K labels
    tree = dyf.build_dyf_tree(Az, max_depth=8, num_bits=4, min_leaf_size=4)
    dyf_lab = np.asarray(dyf.cut_tree_to_labels(tree, N, K, embeddings=Az))

    km_lab = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(Az)

    print(f"\n(a) dyf vs flat — recover the {K}-category taxonomy:")
    for name, lab in (("dyf_tree", dyf_lab), ("kmeans ", km_lab)):
        nmi, ari = _nmi_ari(lab, cat)
        print(f"    {name}  category NMI={nmi:.3f}  ARI={ari:.3f}")

    print("\n(b) concept vs language — what drives the geometry?")
    for name, lab in (("dyf_tree", dyf_lab), ("kmeans ", km_lab)):
        cnmi, _ = _nmi_ari(lab, cat)
        lnmi, _ = _nmi_ari(lab, lang)
        verdict = "SEMANTIC" if cnmi > 2 * lnmi else ("surface" if lnmi > cnmi else "mixed")
        print(f"    {name}  category-NMI={cnmi:.3f}  language-NMI={lnmi:.3f}  → {verdict}")

    print("\n(coarse) recover the noun/adjective domain (2-way):")
    dyf2 = np.asarray(dyf.cut_tree_to_labels(tree, N, 2, embeddings=Az))
    for name, lab in (("dyf_tree", dyf2),
                      ("kmeans ", KMeans(2, n_init=10, random_state=0).fit_predict(Az))):
        nmi, _ = _nmi_ari(lab, dom)
        print(f"    {name}  domain NMI={nmi:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()
    if args.collect or not os.path.exists(CACHE):
        collect_activations()
    analyze(args.layer)
