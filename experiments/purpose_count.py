"""How many verb/purpose classes does the model natively carve?

The previous run was semi-circular (10 balanced designed classes -> silhouette found 10).
Here: a BROAD, UNLABELED gerund set (no pre-binning), sweep k widely, and find the natural
granularity via silhouette (higher=better) + Davies-Bouldin (lower=better). Report the
peak k, the coarse superclasses (small k), and the finer clusters at the peak — "how many"
is resolution-dependent, so we show the hierarchy, not one number. Linguistic anchors:
Levin ~50 classes, VerbNet ~270; the model's *linearly-separable* granularity is coarser.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")

# ~120 diverse gerunds, deliberately UNGROUPED (one flat list, no balancing)
GERUNDS = ("running walking jumping flying swimming climbing crawling sprinting jogging "
           "skipping marching hiking dancing leaping gliding "
           "eating drinking chewing swallowing tasting sipping gulping biting licking "
           "building writing painting drawing sculpting knitting sewing carving molding "
           "designing composing baking cooking frying roasting boiling grilling steaming "
           "breaking smashing crushing burning tearing shattering demolishing ripping "
           "cracking exploding "
           "talking shouting whispering arguing singing chatting yelling speaking "
           "explaining preaching mumbling "
           "thinking remembering learning calculating reasoning imagining understanding "
           "memorizing planning analyzing "
           "seeing hearing watching listening observing smelling staring glancing "
           "hitting kicking punching slapping stabbing poking pushing pulling grabbing "
           "throwing catching "
           "cleaning washing scrubbing mopping sweeping wiping bathing brushing polishing "
           "crying laughing smiling frowning weeping sighing "
           "sleeping breathing yawning sneezing coughing snoring "
           "buying selling trading paying lending borrowing "
           "healing nursing teaching helping guiding rescuing "
           "driving sailing rowing cycling flying-planes parking").split()
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
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    nL = mdl.config.num_hidden_layers
    L = int(0.55 * nL)                       # mid-stack, where verb classes peaked (L13)

    words = sorted(set(GERUNDS))
    X = collect(mdl, tok, dev, words, L)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    lam = np.linalg.svd(Xs - Xs.mean(0), compute_uv=False) ** 2
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    print(f"{len(words)} unlabeled gerunds @L{L}; PR(effective dim)={pr:.1f}\n")

    print("  k   silhouette  davies-bouldin")
    sils = {}
    for k in range(2, 25):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
        s = silhouette_score(Xs, km); db = davies_bouldin_score(Xs, km)
        sils[k] = s
        mark = ""
        print(f"  {k:2d}    {s:.3f}       {db:.2f}{mark}")
    bestk = max(sils, key=sils.get)
    print(f"\n  -> silhouette-optimal k = {bestk}")

    for k in (4, bestk):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        print(f"\nclusters @k={k}:")
        for c in range(k):
            members = [words[i] for i in range(len(words)) if km.labels_[i] == c]
            print(f"  ({len(members):2d}) {' '.join(members[:14])}"
                  + (" ..." if len(members) > 14 else ""))


if __name__ == "__main__":
    main()
