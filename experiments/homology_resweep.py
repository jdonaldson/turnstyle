"""
homology_resweep.py — the persistence-floored homology spectrometer + re-audit.

Fixes homology_sweep.py's degenerate metric (see the warning there) and upgrades
it into a homology-COMPLEXITY instrument:

  metric fix   a topological class only counts if
                 (a) persistence >= FLOOR * median-NN-distance of the cluster
                     (scale-relative absolute floor — kills the tiny Rips blips
                     that z-scoring alone promoted), AND
                 (b) z >= Z_MIN vs a B=50 within-cluster per-dim shuffle null
                     (computed only for floor-passing candidates).
  spectrometer maxdim=2: count floor-passing H1 classes (n1) and H2 classes
               (n2); classify each cluster on the complexity ladder
                 contractible < loop (n1=1) < multi-loop (n1>=2)
                 < cavity (n2>=1) < torus-signature (n1>=2 AND n2>=1);
               confirmed H1 additionally gets a TORSION check (Z/2 vs Z/3
               coefficients — a persistence mismatch flags Mobius/Klein-like
               structure).
  calibration  synthetic point clouds FIRST (no model): noisy circle n=12,
               noisy torus n=84, noisy sphere n=200, Gaussian blob, and the
               exact failure mode that fooled the old metric — an unevenly
               sampled line (dense head, sparse tail). The instrument must
               pass the first three and fail the last two or the sweep aborts.

Re-audit targets (all claimed by the old sweep under the broken z): months
L4/L14-17, professions L13, orgs L16/L28, landscape L32, taxonomy+geo L18,
kinship z=4-7, number L10 (already adjudicated: artifact).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/homology_resweep.py [model_id]
Writes experiments/results/homology_resweep.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from time_ring import collect_last
from homology_sweep import CATEGORIES, TEMPLATE, N_CLUSTERS

import dyf
from ripser import ripser

FLOOR_H1 = 1.0     # confirmed loop: pers >= FLOOR * median NN distance.
                   # Calibration: real rings/torus cycles run ~2x medNN while
                   # noisy SPHERES carry incidental H1 up to ~0.7x (which the
                   # shuffle-z cannot screen — destroying any manifold makes
                   # the null tiny). 1.0 splits them.
FLOOR_H1_WEAK = 0.5  # suggestive band [0.5, 1.0) -> "weak-loop", never counts
                     # toward multi-loop / torus-sig
FLOOR_H2 = 0.25    # H2 persistences run smaller (torus H2 at n=84 is ~0.4)
MIN_N_H2 = 30      # a closed 2-surface needs points to triangulate; 12-point
                   # sets throw floor-passing H2 blips (calibration v2)
B_NULL = 50
B_NULL_H2 = 20
Z_MIN = 3.0
TORSION_TOL = 0.05  # |pers(Z/2) - pers(Z/3)| > tol * medNN -> torsion flag


def pdist(X):
    sq = (X ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))


def med_nn(D):
    E = D + np.eye(len(D)) * 1e18
    return float(np.median(E.min(1)))


def persistences(D, maxdim, coeff=2):
    """Per-dimension sorted persistence lists (finite classes only)."""
    dg = ripser(D, distance_matrix=True, maxdim=maxdim, coeff=coeff)["dgms"]
    out = []
    for k in range(1, maxdim + 1):
        d = dg[k]
        if len(d) == 0:
            out.append(np.array([]))
            continue
        p = d[:, 1] - d[:, 0]
        out.append(np.sort(p[np.isfinite(p)])[::-1])
    return out


def null_max_pers(X, maxdim, B, rng):
    """Null distribution of max persistence per dim (per-dim shuffle)."""
    mx = [[] for _ in range(maxdim)]
    for _ in range(B):
        Xs = X.copy()
        for j in range(X.shape[1]):
            rng.shuffle(Xs[:, j])
        ps = persistences(pdist(Xs), maxdim)
        for k in range(maxdim):
            mx[k].append(ps[k][0] if len(ps[k]) else 0.0)
    return [np.array(v) for v in mx]


def spectro(X, rng, full_null=True):
    """Classify one cluster on the complexity ladder. Returns a dict."""
    D = pdist(X)
    mnn = med_nn(D)
    if mnn < 1e-9:  # duplicate points (e.g. identical embeddings at L0)
        return {"mnn": 0.0, "H1_ratio": 0.0, "H2_ratio": 0.0, "n1": 0,
                "n2": 0, "z1": None, "z2": None, "torsion": False,
                "cls": "degenerate"}
    p1, p2 = persistences(D, 2)
    n1 = int((p1 >= FLOOR_H1 * mnn).sum()) if len(p1) else 0
    n1w = int((p1 >= FLOOR_H1_WEAK * mnn).sum()) if len(p1) else 0
    n2 = int((p2 >= FLOOR_H2 * mnn).sum()) if (len(p2)
                                               and len(X) >= MIN_N_H2) else 0
    r = {"mnn": round(mnn, 2),
         "H1_ratio": round(float(p1[0] / mnn), 2) if len(p1) else 0.0,
         "H2_ratio": round(float(p2[0] / mnn), 2) if len(p2) else 0.0,
         "n1": n1, "n2": n2, "z1": None, "z2": None, "torsion": False}
    if (n1w or n2) and full_null:
        null1, null2 = null_max_pers(X, 2, B_NULL_H2 if n2 else B_NULL, rng)
        if n1w:
            r["z1"] = round(float((p1[0] - null1.mean())
                                  / (null1.std() + 1e-9)), 1)
            if r["z1"] < Z_MIN:
                n1 = n1w = r["n1"] = 0
        if n2:
            r["z2"] = round(float((p2[0] - null2.mean())
                                  / (null2.std() + 1e-9)), 1)
            if r["z2"] < Z_MIN:
                n2 = r["n2"] = 0
        if n1:  # torsion check on confirmed loops only
            q1 = persistences(D, 1, coeff=3)[0]
            top2, top3 = float(p1[0]), float(q1[0]) if len(q1) else 0.0
            r["torsion"] = bool(abs(top2 - top3) > TORSION_TOL * mnn)
    # dominance rule (calibration v2): closed 2-surfaces carry big INCIDENTAL
    # H1 (noisy sphere: 1.7x medNN, z-passing) — but their H2 dwarfs it,
    # while a genuine torus is the reverse (H1 3.2 vs H2 0.4). When H2
    # outranks H1, the loops are surface artifacts.
    if n2 >= 1 and r["H2_ratio"] > r["H1_ratio"]:
        r["cls"] = "cavity"
    elif n1 >= 2 and n2 >= 1:
        r["cls"] = "torus-sig"
    elif n2 >= 1:
        r["cls"] = "cavity" if n1 == 0 else "loop+cavity"
    elif n1 >= 2:
        r["cls"] = "multi-loop"
    elif n1 == 1:
        r["cls"] = "loop"
    elif n1w >= 1:
        r["cls"] = "weak-loop"
    else:
        r["cls"] = "contractible"
    if r["torsion"]:
        r["cls"] += "+torsion"
    return r


# ---------------------------------------------------------------- calibration
def calibrate(rng):
    def embed(P, dim=100, noise=0.2):
        """Isometric embed into dim-d; `noise` = TOTAL relative displacement
        (per-dim noise is scaled by 1/sqrt(dim) — unscaled per-dim noise in
        100-d buried a 12-point ring in calibration v1)."""
        P = (P - P.mean(0)) / (P.std() + 1e-9)
        Q = np.zeros((len(P), dim))
        Q[:, :P.shape[1]] = P
        R = np.linalg.qr(rng.normal(size=(dim, dim)))[0]
        return ((Q @ R) + (noise / np.sqrt(dim))
                * rng.normal(size=(len(P), dim))).astype(np.float32)

    def circle(n, noise=0.25):
        th = (np.arange(n) + rng.uniform(-0.15, 0.15, n)) * 2 * np.pi / n
        return embed(np.stack([np.cos(th), np.sin(th)], 1), noise=noise)

    u = rng.uniform(0, 2 * np.pi, 84); v = rng.uniform(0, 2 * np.pi, 84)
    torus = embed(np.stack([np.cos(u), np.sin(u),
                            0.6 * np.cos(v), 0.6 * np.sin(v)], 1), noise=0.2)

    g = rng.normal(size=(200, 3))
    sphere = embed(g / np.linalg.norm(g, axis=1, keepdims=True), noise=0.2)

    blob = embed(rng.normal(size=(40, 5)), noise=0.0)

    # the number-word failure mode: dense head, sparse tail, one line
    t = np.concatenate([np.linspace(0, 1, 9), [1.45, 1.9, 2.35]])
    uneven = embed(t[:, None] * np.ones((1, 2)), noise=0.2)

    cases = {"circle_n12": (circle(12), "loop", True),
             "circle_n7": (circle(7), "loop", False),   # weekday case: info
             "torus_n84": (torus, "torus-sig", True),
             "sphere_n200": (sphere, "cavity", True),
             "blob_n40": (blob, "contractible", True),
             "uneven_line_n12": (uneven, "contractible", True)}
    print("=== synthetic calibration ===", flush=True)
    ok = True
    for name, (X, want, gates) in cases.items():
        r = spectro(X, rng)
        got = r["cls"].replace("+torsion", "")
        passed = (got == want) or (want == "torus-sig"
                                   and got in ("torus-sig", "multi-loop"))
        if gates:
            ok &= passed
        print(f"  {name:16s} want={want:12s} got={r['cls']:12s} "
              f"H1r={r['H1_ratio']:.2f} H2r={r['H2_ratio']:.2f} "
              f"n1={r['n1']} n2={r['n2']} z1={r['z1']} z2={r['z2']} "
              f"{'OK' if passed else ('FAIL' if gates else 'info')}",
              flush=True)
    return ok


# ---------------------------------------------------------------- main sweep
def main():
    rng = np.random.default_rng(0)
    if not calibrate(rng):
        print("\nCALIBRATION FAILED — not running the model sweep.")
        sys.exit(1)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from who_moved import load_vocab
    mid = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    words = sorted(set(load_vocab()) | {w for ws in CATEGORIES.values()
                                        for w in ws})
    n = len(words)
    widx = {w: i for i, w in enumerate(words)}
    cat_idx = {c: [widx[w] for w in ws] for c, ws in CATEGORIES.items()}
    print(f"\n{n} words, model {mid}; collecting...", flush=True)
    acts = collect_last(mdl, tok, dev, words, TEMPLATE)
    n_layers = acts[words[0]].shape[0]

    from numpy.linalg import svd
    results = {"model": mid, "n_words": n, "floors": [FLOOR_H1, FLOOR_H2],
               "layers": {}}
    ladder_rank = {"contractible": 0, "weak-loop": 1, "loop": 2,
                   "multi-loop": 3, "cavity": 4, "loop+cavity": 4,
                   "torus-sig": 5}
    for L in range(n_layers):
        X = np.array([acts[w][L] for w in words], dtype=np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        Xc = X - X.mean(0)
        U, S, _ = svd(Xc, full_matrices=False)
        Xr = (U[:, :100] * S[:100]).astype(np.float32)

        tree = dyf.build_dyf_tree(Xr, max_depth=4, num_bits=3, min_leaf_size=4)
        labels = np.asarray(dyf.cut_tree_to_labels(tree, n, N_CLUSTERS,
                                                   embeddings=Xr))
        found, tracked = [], {}
        for cl in np.unique(labels):
            members = np.where(labels == cl)[0]
            if not (6 <= len(members) <= 80):
                continue
            r = spectro(Xr[members], rng)
            r["size"] = int(len(members))
            r["members"] = [words[i] for i in members[:14]]
            base = r["cls"].replace("+torsion", "")
            if base != "contractible":
                found.append(r)
            for c, idxs in cat_idx.items():
                if len(set(members) & set(idxs)) >= max(4, len(idxs) // 2):
                    tracked[c] = {k: r[k] for k in
                                  ("cls", "H1_ratio", "H2_ratio", "z1", "z2",
                                   "size")}
        found.sort(key=lambda e: (-ladder_rank.get(
            e["cls"].replace("+torsion", ""), 0), -e["H1_ratio"]))
        results["layers"][L] = {"found": found[:6], "tracked": tracked}
        tstr = " ".join(f"{c}:{v['cls']}({v['H1_ratio']:.1f})"
                        for c, v in sorted(tracked.items())
                        if v["cls"] != "contractible")
        print(f"L{L:2d} nontrivial={len(found):2d}  {tstr}", flush=True)
        for e in found[:3]:
            print(f"     {e['cls']:12s} n1={e['n1']} n2={e['n2']} "
                  f"H1r={e['H1_ratio']:.2f} H2r={e['H2_ratio']:.2f} "
                  f"z1={e['z1']} z2={e['z2']} size={e['size']}: "
                  f"{', '.join(e['members'][:8])}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "homology_resweep.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
