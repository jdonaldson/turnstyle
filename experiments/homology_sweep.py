"""
homology_sweep.py — dyf x ripser: the composed topology instrument, per layer.

The capstone of the interpretability arc. Per layer over a ~830-word pool
(750 common nouns + planted categories, all unprivileged):

  1. dyf tree (build_dyf_tree) -> flat clusters (cut_tree_to_labels).
     TREE PURITY curve: per planted category, fraction of members landing in
     their majority cluster — dyf-instrumented "when does each category become
     a coherent region" (replaces shape_zoo's ad-hoc tree metric).
  2. Per discovered cluster (6 <= size <= 80): ripser H1 max persistence — the
     LOOP detector, no hypothesized ordering or period.
  3. HONESTY: within-cluster per-dimension shuffle null (preserves membership
     and marginal spread, destroys inter-dimension arrangement — the
     CLAUDE.md rule: preserve clustering, break only the claimed structure).
     Loop score = z of observed H1 persistence vs B=10 null shuffles.

Blind validation targets: the weekday cluster should be DISCOVERED by dyf and
its H1-z should rise into L26-30 (the known ring assembly); kinship (a
contractible 2D grid) should stay at null; months/zodiac intermediate.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/homology_sweep.py [model_id]
Writes experiments/results/homology_sweep.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from time_ring import collect_last
from who_moved import load_vocab, PLANTED
from shape_zoo import TAXO_PATH, KIN

import dyf
from ripser import ripser

TEMPLATE = "They talked about {w}."
N_CLUSTERS = 40
B_NULL = 10
Z_LOOP = 3.0

CATEGORIES = dict(PLANTED)
CATEGORIES["taxonomy"] = list(TAXO_PATH)
CATEGORIES["kinship"] = list(KIN)
CATEGORIES["zodiac"] = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
                        "Libra", "Scorpio", "Sagittarius", "Capricorn",
                        "Aquarius", "Pisces"]


def h1_max_persistence(X, rng=None):
    """Max H1 persistence of a point set (euclidean on given coords)."""
    d = X.shape[0]
    sq = (X ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    dgm = ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
    if len(dgm) == 0:
        return 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    pers = pers[np.isfinite(pers)]
    return float(pers.max()) if len(pers) else 0.0


def loop_z(X, rng):
    """Observed H1 max persistence vs within-cluster per-dimension shuffle null."""
    obs = h1_max_persistence(X)
    null = []
    for _ in range(B_NULL):
        Xs = X.copy()
        for j in range(X.shape[1]):
            rng.shuffle(Xs[:, j])
        null.append(h1_max_persistence(Xs))
    mu, sd = float(np.mean(null)), float(np.std(null)) + 1e-9
    return obs, (obs - mu) / sd


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    words = sorted(set(load_vocab()) | {w for ws in CATEGORIES.values() for w in ws})
    n = len(words)
    widx = {w: i for i, w in enumerate(words)}
    cat_idx = {c: [widx[w] for w in ws] for c, ws in CATEGORIES.items()}
    print(f"{n} words, {len(CATEGORIES)} planted categories, model {mid}", flush=True)

    acts = collect_last(mdl, tok, dev, words, TEMPLATE)
    n_layers = acts[words[0]].shape[0]
    print("collected; sweeping layers...", flush=True)

    # PCA to 100 dims per layer for the topology (ripser/null cost + distance
    # concentration in 4096-d); dyf gets the same reduced space for consistency.
    from numpy.linalg import svd
    results = {"model": mid, "n_words": n, "layers": {}}
    rng = np.random.default_rng(0)
    for L in range(n_layers):
        X = np.array([acts[w][L] for w in words], dtype=np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        Xc = X - X.mean(0)
        U, S, _ = svd(Xc, full_matrices=False)
        # float32 REQUIRED: dyf_rs's DensityClassifier has a typed PyO3 f32
        # signature — float64 makes every fit throw, which build_dyf_tree
        # swallows into a single-leaf tree (the everything-in-one-cluster bug).
        Xr = (U[:, :100] * S[:100]).astype(np.float32)

        tree = dyf.build_dyf_tree(Xr, max_depth=4, num_bits=3, min_leaf_size=4)
        labels = np.asarray(dyf.cut_tree_to_labels(tree, n, N_CLUSTERS, embeddings=Xr))

        # tree purity per category
        purity = {}
        for c, idxs in cat_idx.items():
            lab = labels[idxs]
            purity[c] = round(float(np.bincount(lab).max() / len(lab)), 2)

        # per-cluster H1 with null
        loops = []
        tracked = {}
        for cl in np.unique(labels):
            members = np.where(labels == cl)[0]
            if not (6 <= len(members) <= 80):
                continue
            obs, z = loop_z(Xr[members], rng)
            entry = {"size": int(len(members)), "H1": round(obs, 3), "z": round(z, 1),
                     "members": [words[i] for i in members[:14]]}
            if z >= Z_LOOP:
                loops.append(entry)
            for c, idxs in cat_idx.items():
                if len(set(members) & set(idxs)) >= max(4, len(idxs) // 2):
                    tracked[c] = {"z": round(z, 1), "H1": round(obs, 3),
                                  "contained": len(set(members) & set(idxs)),
                                  "cluster_size": int(len(members))}
        loops.sort(key=lambda e: -e["z"])
        results["layers"][L] = {"purity": purity, "loop_clusters": loops[:4],
                                "tracked": tracked}
        tstr = " ".join(f"{c}:z={v['z']}" for c, v in sorted(tracked.items()))
        print(f"L{L:2d} purity(mean {np.mean(list(purity.values())):.2f}) "
              f"loops(z>={Z_LOOP}): {len(loops)}  | {tstr}", flush=True)
        if loops:
            top = loops[0]
            print(f"     top loop z={top['z']} size={top['size']}: "
                  f"{', '.join(top['members'][:10])}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "homology_sweep.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
