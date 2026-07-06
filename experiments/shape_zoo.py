"""
shape_zoo.py — hunt geometric structures more complex than a ring, and where
they live in the stack. Tests the complexity->depth hypothesis: lines peak
early (L1-4), rings late (L28-30); do 2D grids, 3D solids, and trees peak
later still / in a characteristic band?

Three structure classes, each with a domain the model surely knows:

  A. 2D GRID — kinship (gender x generation). Fit each axis as a ridge scalar
     per layer; grid_score = min(recov_gender, recov_gen) * (1 - |cos|). High
     only when BOTH axes recover AND are orthogonal. Compare the layer where
     the GRID (product) is sharpest vs where each 1D axis alone peaks — the
     direct test of "product structure assembles later than its factors".

  B. 3D SOLID — color (CIELAB L*, a*, b*). We showed hue is a ring; does the
     model carry a full 3D color space? Fit L*/a*/b* ridge directions per
     layer; solid_score = min recoverability * mean(1 - |cos|) over the 3 pairs.

  C. TREE — biological taxonomy (leaf nouns with known tree path-distances).
     Per layer: spearman(activation distances, tree path distances) + an
     ULTRAMETRICITY fraction (of triangles, is the top-2 distances near-equal
     — the tree signature). A non-manifold structure class.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/shape_zoo.py [model_id]
Writes experiments/results/shape_zoo.json.
"""
from __future__ import annotations
import json, os, sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from time_ring import collect_last
from ordering_frames import cv_r, ridge_dir

# ── A. kinship grid: (gender +1 F / -1 M, generation +2 gp .. -2 gc) ──────────
KIN = {
    "grandmother": (+1, +2), "grandfather": (-1, +2),
    "mother": (+1, +1), "father": (-1, +1), "aunt": (+1, +1), "uncle": (-1, +1),
    "sister": (+1, 0), "brother": (-1, 0),
    "daughter": (+1, -1), "son": (-1, -1), "niece": (+1, -1), "nephew": (-1, -1),
    "granddaughter": (+1, -2), "grandson": (-1, -2),
}
KIN_T = "They visited their {w}."

# ── C. taxonomy tree: leaf -> path from root (living>kingdom>class>...) ────────
TAXO_PATH = {
    "dog": ("animal", "mammal", "carnivore"), "cat": ("animal", "mammal", "carnivore"),
    "horse": ("animal", "mammal", "ungulate"), "whale": ("animal", "mammal", "cetacean"),
    "eagle": ("animal", "bird", "raptor"), "robin": ("animal", "bird", "songbird"),
    "penguin": ("animal", "bird", "flightless"),
    "salmon": ("animal", "fish", "salmonid"), "shark": ("animal", "fish", "cartilaginous"),
    "oak": ("plant", "tree", "deciduous"), "pine": ("plant", "tree", "conifer"),
    "rose": ("plant", "flower", "shrub"), "tulip": ("plant", "flower", "bulb"),
    "fern": ("plant", "nonflowering", "spore"),
}
TAXO_T = "They looked at the {w}."


def tree_dist(a, b):
    pa, pb = TAXO_PATH[a], TAXO_PATH[b]
    shared = 0
    for x, y in zip(pa, pb):
        if x == y:
            shared += 1
        else:
            break
    return (len(pa) - shared) + (len(pb) - shared)


def zdist(acts, words, L):
    X = np.array([acts[w][L] for w in words], dtype=np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    sq = (X ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)), X


def upper(M):
    return M[np.triu_indices(M.shape[0], 1)]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def axis(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return ridge_dir((X - mu) / sd, np.asarray(y, float))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model {mid} on {dev}", flush=True)
    results = {"model": mid}

    # ── A. kinship grid ──────────────────────────────────────────────────────
    kw = list(KIN)
    gender = np.array([KIN[w][0] for w in kw], float)
    gen = np.array([KIN[w][1] for w in kw], float)
    acts = collect_last(mdl, tok, dev, kw, KIN_T)
    nL = acts[kw[0]].shape[0]
    grid = []
    for L in range(nL):
        X = np.array([acts[w][L] for w in kw])
        rg = cv_r(X, gender); rn = cv_r(X, gen)
        dg, dn = axis(X, gender), axis(X, gen)
        cos = abs(float(dg @ dn))
        grid.append({"gender_r": round(rg, 3), "gen_r": round(rn, 3),
                     "cos": round(cos, 3),
                     "grid_score": round(min(rg, rn) * (1 - cos), 3)})
    gs = [g["grid_score"] for g in grid]
    gpk = int(np.argmax(gs))
    gender_pk = int(np.argmax([g["gender_r"] for g in grid]))
    gen_pk = int(np.argmax([g["gen_r"] for g in grid]))
    results["kinship_grid"] = {"by_layer": grid, "grid_peak_L": gpk,
                               "gender_peak_L": gender_pk, "gen_peak_L": gen_pk,
                               "peak": grid[gpk]}
    print(f"\n=== A. kinship GRID ===")
    print(f"  gender axis peaks L{gender_pk} (r={grid[gender_pk]['gender_r']}), "
          f"generation axis peaks L{gen_pk} (r={grid[gen_pk]['gen_r']})")
    print(f"  GRID (both⊥) peaks L{gpk}: {grid[gpk]}")

    # ── B. color 3D solid ──────────────────────────────────────────────────────
    from color_ring import COLOR_WORDS, CHROMA_MIN
    from color_affect_frame import _hex_to_lab
    from matplotlib.colors import CSS4_COLORS
    cw = [w for w in COLOR_WORDS if w in CSS4_COLORS]
    lab = {w: _hex_to_lab(CSS4_COLORS[w].lstrip("#")) for w in cw}
    Ls = np.array([lab[w][0] for w in cw]); As = np.array([lab[w][1] for w in cw])
    Bs = np.array([lab[w][2] for w in cw])
    acts = collect_last(mdl, tok, dev, cw, "It is a {w} object.")
    nLc = acts[cw[0]].shape[0]
    solid = []
    for L in range(nLc):
        X = np.array([acts[w][L] for w in cw])
        rs = {"L": cv_r(X, Ls), "a": cv_r(X, As), "b": cv_r(X, Bs)}
        ds = {k: axis(X, v) for k, v in (("L", Ls), ("a", As), ("b", Bs))}
        coss = [abs(float(ds[x] @ ds[y])) for x, y in (("L", "a"), ("L", "b"), ("a", "b"))]
        solid.append({**{f"{k}_r": round(v, 3) for k, v in rs.items()},
                      "mean_cos": round(float(np.mean(coss)), 3),
                      "solid_score": round(min(rs.values()) * (1 - float(np.mean(coss))), 3)})
    ss = [s["solid_score"] for s in solid]
    spk = int(np.argmax(ss))
    results["color_solid"] = {"by_layer": solid, "solid_peak_L": spk, "peak": solid[spk]}
    print(f"\n=== B. color 3D SOLID (L*,a*,b* mutually ⊥) ===")
    print(f"  SOLID peaks L{spk}: {solid[spk]}")

    # ── C. taxonomy tree ────────────────────────────────────────────────────────
    tw = list(TAXO_PATH)
    T = np.array([[tree_dist(a, b) for b in tw] for a in tw], float)
    acts = collect_last(mdl, tok, dev, tw, TAXO_T)
    nLt = acts[tw[0]].shape[0]
    tree = []
    for L in range(nLt):
        D, _ = zdist(acts, tw, L)
        r = spearman(upper(D), upper(T))
        # ultrametricity: for each triangle, top-2 side lengths near-equal?
        um = []
        for i, j, k in combinations(range(len(tw)), 3):
            s = sorted([D[i, j], D[i, k], D[j, k]])
            um.append(1 - (s[2] - s[1]) / (s[2] + 1e-9))
        tree.append({"tree_r": round(r, 3), "ultrametric": round(float(np.mean(um)), 3)})
    tr = [t["tree_r"] for t in tree]
    tpk = int(np.argmax(tr))
    results["taxonomy_tree"] = {"by_layer": tree, "tree_peak_L": tpk, "peak": tree[tpk]}
    print(f"\n=== C. taxonomy TREE ===")
    print(f"  tree-distance fit peaks L{tpk}: {tree[tpk]}")

    # ── the complexity->depth ladder ───────────────────────────────────────────
    print(f"\n=== complexity -> depth ladder (of {nL} layers) ===")
    print(f"  1D line   (size/age)     early  L1-4      [prior]")
    print(f"  1D ring   (weekday)      late   L28-30    [prior]")
    print(f"  2D grid   (kinship)      L{gpk}   score {grid[gpk]['grid_score']}")
    print(f"  3D solid  (color)        L{spk}   score {solid[spk]['solid_score']}")
    print(f"  tree      (taxonomy)     L{tpk}   r {tree[tpk]['tree_r']} "
          f"ultra {tree[tpk]['ultrametric']}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "shape_zoo.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
