"""
number_helix_followup.py — what actually closes the L10 number loop?

number_helix.py refuted decade congruence at face value: number-word geometry
is LINE-dominated (RSA 0.7-0.8 mid-stack); at L10 the congruence ratio is ns
(z=-0.7) and the mod-10 semi-partial is dead (p=.28). The z=22.7 H1 loop
(homology_sweep, cluster two..twelve + town) needs another closure mechanism.
Suspects:
  B. surface morphology bridging (twelve~two "tw"; eleven~seven "even")
  C. the "town" intruder bridging the cluster
  D. bent-line (horseshoe) closure that only exists in the sweep's GLOBAL
     823-word standardize+PCA-100 basis
Plus one loose end: the small mod-10 semi-partial at L3-7 (number template)
was never tested for DECADE SPECIFICITY vs control periods.

Stage 1 (cheap, 21 words, local basis):
  - per layer: H1 shuffle-z for {two..twelve,town} and {two..twelve};
  - per layer: semi-partial (beyond line) for periods 5/7/10/12;
  - morphology ranks: (seven,eleven) among all gap-4 pairs; (one,eleven),
    (two,twelve) among gap-10 pairs;
  - at the best-z layer: the CLOSURE EDGE (pairs at the H1 birth radius) +
    3-NN table.
Stage 2 (823-word pool re-collect, the sweep's exact basis, L8-12):
  - H1 z for the cluster with/without town, closure edge + 3-NN at L10,
    within-cluster line RSA (is it still a line in the global basis?).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/number_helix_followup.py [model_id]
Writes experiments/results/number_helix_followup.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from time_ring import collect_last
from who_moved import load_vocab
from ripser import ripser

TEMPLATE = "They talked about {w}."
WORDS20 = ["one", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
           "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
CLUSTER = ["two", "three", "four", "five", "six", "seven", "eight", "nine",
           "ten", "eleven", "twelve", "town"]   # the sweep's L10 loop members
B_NULL = 20


def pdist(X):
    sq = (X ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))


def upper(M):
    return M[np.triu_indices_from(M, k=1)]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def h1_top(D):
    """(birth, death, persistence) of the max-persistence H1 class."""
    dgm = ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
    if len(dgm) == 0:
        return 0.0, 0.0, 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    ok = np.isfinite(pers)
    if not ok.any():
        return 0.0, 0.0, 0.0
    i = int(np.argmax(np.where(ok, pers, -1)))
    return float(dgm[i, 0]), float(dgm[i, 1]), float(pers[i])


def h1_z(X, rng):
    _, _, obs = h1_top(pdist(X))
    null = []
    for _ in range(B_NULL):
        Xs = X.copy()
        for j in range(X.shape[1]):
            rng.shuffle(Xs[:, j])
        null.append(h1_top(pdist(Xs))[2])
    mu, sd = float(np.mean(null)), float(np.std(null)) + 1e-9
    return obs, (obs - mu) / sd


def closure_report(names, X):
    """Birth/death of the top H1 class, the edges at the birth radius, 3-NN."""
    D = pdist(X)
    b, d, p = h1_top(D)
    n = len(names)
    edges = [(names[i], names[j], round(float(D[i, j]), 2))
             for i in range(n) for j in range(i + 1, n)
             if abs(D[i, j] - b) <= 0.005 * max(b, 1e-9)]
    nn = {}
    for i in range(n):
        order = np.argsort(D[i])
        nn[names[i]] = [(names[k], round(float(D[i, k]), 1)) for k in order[1:4]]
    return {"birth": round(b, 2), "death": round(d, 2), "pers": round(p, 2),
            "birth_edges": edges, "nn": nn}


def semi_partial(Du, t, P):
    dt = np.abs(t[:, None] - t[None, :]).astype(float)
    th = 2 * np.pi * t / P
    c, s = np.cos(th), np.sin(th)
    circ = np.sqrt((c[:, None] - c[None, :]) ** 2 + (s[:, None] - s[None, :]) ** 2)
    lin = upper(dt)
    A = np.stack([lin, np.ones_like(lin)], 1)
    beta, *_ = np.linalg.lstsq(A, Du, rcond=None)
    return spearman(Du - A @ beta, upper(circ))


def rank_of(pair, pairs, D, wi):
    ds = sorted(float(D[wi[a], wi[b]]) for a, b in pairs)
    d0 = float(D[wi[pair[0]], wi[pair[1]]])
    return 1 + sum(1 for v in ds if v < d0), len(pairs)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    results: dict = {"model": mid}

    # ---- Stage 1: local basis --------------------------------------------
    pool = WORDS20 + ["town"]
    acts = collect_last(mdl, tok, dev, pool, TEMPLATE)
    n_layers = acts[pool[0]].shape[0]
    t20 = np.arange(1, 21, dtype=float)
    rng = np.random.default_rng(0)
    gap4 = [(WORDS20[a - 1], WORDS20[a + 3]) for a in range(1, 17)]
    gap10 = [(WORDS20[a - 1], WORDS20[a + 9]) for a in range(1, 11)]
    wi = {w: i for i, w in enumerate(pool)}

    print("=== stage 1: local basis (21 words) ===", flush=True)
    print("  L | z(+town) z(-town) | sp P5 P7 P10 P12 | rank(7,11)/gap4 "
          "rank(1,11) rank(2,12)/gap10", flush=True)
    s1 = {}
    best = (-1e9, None)
    for L in range(n_layers):
        X = np.array([acts[w][L] for w in pool], dtype=np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        Xc = np.array([X[wi[w]] for w in CLUSTER])
        Xc2 = np.array([X[wi[w]] for w in CLUSTER[:-1]])
        _, z1 = h1_z(Xc, rng)
        _, z2 = h1_z(Xc2, rng)
        X20 = X[:20]
        Du = upper(pdist(X20))
        sp = {P: round(semi_partial(Du, t20, P), 3) for P in (5, 7, 10, 12)}
        D = pdist(X)
        r4 = rank_of(("seven", "eleven"), gap4, D, wi)
        r10a = rank_of(("one", "eleven"), gap10, D, wi)
        r10b = rank_of(("two", "twelve"), gap10, D, wi)
        s1[L] = {"z_town": round(z1, 1), "z_no_town": round(z2, 1), "sp": sp,
                 "rank_7_11_gap4": r4, "rank_1_11_gap10": r10a,
                 "rank_2_12_gap10": r10b}
        if z1 > best[0]:
            best = (z1, L)
        print(f" {L:2d} | {z1:7.1f} {z2:7.1f} | {sp[5]:+.2f} {sp[7]:+.2f} "
              f"{sp[10]:+.2f} {sp[12]:+.2f} | {r4[0]:2d}/{r4[1]} "
              f"{r10a[0]:2d}/{r10a[1]} {r10b[0]:2d}/{r10b[1]}", flush=True)

    Lb = best[1]
    X = np.array([acts[w][Lb] for w in pool], dtype=np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    rep = closure_report(CLUSTER, np.array([X[wi[w]] for w in CLUSTER]))
    print(f"\nlocal best z at L{Lb}: birth={rep['birth']} death={rep['death']}")
    print(f"  birth edges: {rep['birth_edges']}")
    for w, ns in rep["nn"].items():
        print(f"  {w:9s} -> {ns}")
    results["stage1"] = {"layers": s1, "best_layer": Lb, "closure": rep}

    # ---- Stage 2: the sweep's global basis -------------------------------
    print("\n=== stage 2: global 823-word standardize+PCA-100 basis ===",
          flush=True)
    words = sorted(set(load_vocab()) | set(WORDS20) | {"town"})
    print(f"collecting {len(words)} words...", flush=True)
    acts_g = collect_last(mdl, tok, dev, words, TEMPLATE)
    gwi = {w: i for i, w in enumerate(words)}
    from numpy.linalg import svd
    s2 = {}
    vals = np.arange(2, 13, dtype=float)      # two..twelve
    for L in range(8, 13):
        X = np.array([acts_g[w][L] for w in words], dtype=np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        Xc = X - X.mean(0)
        U, S, _ = svd(Xc, full_matrices=False)
        Xr = (U[:, :100] * S[:100]).astype(np.float32)
        Xcl = np.array([Xr[gwi[w]] for w in CLUSTER])
        Xcl2 = Xcl[:-1]
        _, z1 = h1_z(Xcl, rng)
        _, z2 = h1_z(Xcl2, rng)
        Dn = pdist(Xcl2)
        line_rsa = spearman(upper(Dn),
                            upper(np.abs(vals[:, None] - vals[None, :])))
        rep = closure_report(CLUSTER, Xcl)
        s2[L] = {"z_town": round(z1, 1), "z_no_town": round(z2, 1),
                 "line_rsa_no_town": round(line_rsa, 3), "closure": rep}
        print(f" L{L}: z(+town)={z1:.1f} z(-town)={z2:.1f} "
              f"line_rsa={line_rsa:.2f} birth_edges={rep['birth_edges']}",
              flush=True)
        if L == 10:
            for w, ns in rep["nn"].items():
                print(f"   {w:9s} -> {ns}", flush=True)
    results["stage2"] = s2

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "number_helix_followup.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
