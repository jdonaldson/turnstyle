"""Does the 'purpose' rung carve PURPOSE categories, or is it VERB-semantic structure?

Purpose modifiers are deverbal gerunds (cooking/racing/hunting), so the apparent purpose
frame may just be the model's verb/event space. Test: cluster a BROAD, unbiased gerund set
spanning known verb-semantic classes (Levin/VerbNet-style) UNSUPERVISED, and check whether
the emergent clusters recover verb classes (high NMI/ARI vs the seeded classes) rather than
some 'object purpose' organization.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")

# ~10 verb-semantic classes (seed labels for scoring; clustering is unsupervised)
CLASSES = {
    "motion": "running walking jumping flying swimming climbing crawling".split(),
    "ingestion": "eating drinking chewing swallowing tasting sipping".split(),
    "creation": "building writing painting drawing sculpting knitting".split(),
    "destruction": "breaking smashing crushing burning tearing shattering".split(),
    "communication": "talking shouting whispering arguing singing chatting".split(),
    "cognition": "thinking remembering learning calculating reasoning imagining".split(),
    "cleaning": "cleaning washing scrubbing mopping sweeping wiping".split(),
    "cooking": "cooking baking frying roasting boiling grilling".split(),
    "contact": "hitting kicking punching slapping stabbing poking".split(),
    "perception": "seeing hearing watching listening observing smelling".split(),
}
TMPL = "the act of {w}"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def collect(mdl, tok, dev, words, layer):
    import torch
    out = []
    for w in words:
        sent = TMPL.format(w=w); cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        out.append(hs[layer][0, idxs[-1]].float().cpu().numpy())
    return np.array(out)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, normalized_mutual_info_score,
                                 adjusted_rand_score)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    nL = mdl.config.num_hidden_layers

    words = [w for c in CLASSES for w in CLASSES[c]]
    labels = [c for c in CLASSES for _ in CLASSES[c]]
    K_true = len(CLASSES)

    print(f"{len(words)} gerunds, {K_true} seeded verb classes, template {TMPL!r}\n")
    for L in sorted({int(d * nL) for d in (0.25, 0.4, 0.55, 0.7)}):
        X = collect(mdl, tok, dev, words, L)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        lam = np.linalg.svd(Xs - Xs.mean(0), compute_uv=False) ** 2
        pr = float(lam.sum() ** 2 / (lam ** 2).sum())
        # unsupervised k-means at the seeded K; recovery vs verb classes
        km = KMeans(n_clusters=K_true, n_init=10, random_state=0).fit(Xs)
        nmi = normalized_mutual_info_score(labels, km.labels_)
        ari = adjusted_rand_score(labels, km.labels_)
        # best k by silhouette
        sils = {k: silhouette_score(Xs, KMeans(n_clusters=k, n_init=10, random_state=0)
                                    .fit_predict(Xs)) for k in range(3, 13)}
        bestk = max(sils, key=sils.get)
        print(f"L{L:<2d} PR={pr:5.1f}  NMI(verb-class)={nmi:.2f} ARI={ari:.2f}  "
              f"best_k(silhouette)={bestk}")

    # detailed cluster contents at the most separable layer (mid)
    L = int(0.4 * nL)
    X = collect(mdl, tok, dev, words, L)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    km = KMeans(n_clusters=K_true, n_init=10, random_state=0).fit(Xs)
    print(f"\nclusters @L{L} (k={K_true}, unsupervised):")
    for c in range(K_true):
        members = [words[i] for i in range(len(words)) if km.labels_[i] == c]
        # dominant seeded class in this cluster
        seeds = [labels[i] for i in range(len(words)) if km.labels_[i] == c]
        dom = max(set(seeds), key=seeds.count) if seeds else "-"
        print(f"  [{dom:13s}] {' '.join(members)}")


if __name__ == "__main__":
    main()
