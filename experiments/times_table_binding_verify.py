"""Verify that the L0->L1 operand-position emergence is real binding,
not an artifact of regularised probe variance.

Three independent tests, all from the cached hidden_states.npz:

1. SWAP-DISTANCE  Per layer, compare ||h(a,b) - h(b,a)|| to ||h(a,b) - h(c,d)||
                  for random (c,d). If position is bound, the swap distance is
                  substantial; if only the set is encoded, swap distance is ~0.

2. PROBE ORTHOGONALITY  Train W_a (predict operand_a) and W_b (predict operand_b)
                  at each layer. For each digit k, compute cos(W_a[k], W_b[k]).
                  - cos ~ +1 : same direction encodes "set contains k" (NOT bound)
                  - cos ~  0 : independent role-specific directions (BOUND)
                  - cos ~ -1 : antipodal — also implies binding, with shared axis

3. VARIANCE DECOMPOSITION  Partition the 100 pairs into 45 swap-buckets
                  (each containing (a,b) + (b,a) for a≠b) plus 10 a=a singletons.
                  Report within-bucket variance / total variance. A nonzero
                  ratio means the network distinguishes (a,b) from (b,a) within
                  the same {a,b} set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
STATES = ROOT / "hidden_states.npz"

RNG = np.random.default_rng(0)


def by_layer(H, layer, a, b, n_layers):
    out = {}
    for l in range(n_layers):
        m = layer == l
        out[l] = dict(H=H[m], a=a[m], b=b[m])
    return out


def swap_distance_test(per_layer, n_layers):
    print("\n[1] SWAP-DISTANCE TEST")
    print("    d_swap = mean ||h(a,b) - h(b,a)|| / ||h(a,b)||  (over a!=b)")
    print("    d_rand = mean ||h(a,b) - h(c,d)|| / ||h(a,b)||  (random other pair)")
    print(f"    {'layer':>6}  {'d_swap':>8}  {'d_rand':>8}  {'ratio':>8}")

    for l in range(n_layers):
        H = per_layer[l]["H"]
        a = per_layer[l]["a"]
        b = per_layer[l]["b"]
        # index by (a,b)
        idx = {(int(a[i]), int(b[i])): i for i in range(len(a))}

        d_swap, d_rand = [], []
        for i in range(len(a)):
            ai, bi = int(a[i]), int(b[i])
            if ai == bi:
                continue
            j = idx[(bi, ai)]  # swap partner
            norm_i = np.linalg.norm(H[i]) + 1e-8
            d_swap.append(np.linalg.norm(H[i] - H[j]) / norm_i)
            # random different pair
            k = i
            while k == i or k == j:
                k = int(RNG.integers(0, len(a)))
            d_rand.append(np.linalg.norm(H[i] - H[k]) / norm_i)

        ds, dr = float(np.mean(d_swap)), float(np.mean(d_rand))
        print(f"    L{l:<5d}  {ds:8.4f}  {dr:8.4f}  {ds/dr:8.4f}")


def probe_orthogonality_test(per_layer, n_layers):
    print("\n[2] PROBE ORTHOGONALITY")
    print("    cos(W_a[k], W_b[k]) averaged over digits k=0..9 at each layer.")
    print("    +1 => set encoding (same direction for op-a=k and op-b=k)")
    print("     0 => independent role-specific directions (binding)")
    print(f"    {'layer':>6}  {'mean cos':>10}  {'min cos':>10}  {'max cos':>10}")

    for l in range(n_layers):
        H = per_layer[l]["H"]
        a = per_layer[l]["a"]
        b = per_layer[l]["b"]

        clf_a = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=4000))
        clf_b = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=4000))
        clf_a.fit(H, a)
        clf_b.fit(H, b)

        Wa = clf_a.named_steps["logisticregression"].coef_  # (10, 128)
        Wb = clf_b.named_steps["logisticregression"].coef_
        # align rows by class id (clf.classes_ is sorted ascending: 0..9)
        cls_a = clf_a.named_steps["logisticregression"].classes_
        cls_b = clf_b.named_steps["logisticregression"].classes_
        order_a = np.argsort(cls_a)
        order_b = np.argsort(cls_b)
        Wa, Wb = Wa[order_a], Wb[order_b]

        cos = (Wa * Wb).sum(axis=1) / (
            np.linalg.norm(Wa, axis=1) * np.linalg.norm(Wb, axis=1) + 1e-8
        )
        print(f"    L{l:<5d}  {cos.mean():10.4f}  {cos.min():10.4f}  {cos.max():10.4f}")


def variance_decomposition(per_layer, n_layers):
    print("\n[3] VARIANCE DECOMPOSITION")
    print("    Within-set variance / total variance.")
    print("    Higher => position info present (ordered pairs separated within set).")
    print(f"    {'layer':>6}  {'within/total':>14}")

    for l in range(n_layers):
        H = per_layer[l]["H"]
        a = per_layer[l]["a"]
        b = per_layer[l]["b"]

        # bucket by unordered set {a,b}
        buckets = {}
        for i in range(len(a)):
            key = tuple(sorted((int(a[i]), int(b[i]))))
            buckets.setdefault(key, []).append(i)

        # only multi-element buckets contribute within-variance
        within_sum = 0.0
        n_within = 0
        for key, ixs in buckets.items():
            if len(ixs) < 2:
                continue
            X = H[ixs]
            within_sum += ((X - X.mean(0)) ** 2).sum()
            n_within += X.size  # n_rows * n_dims
        total = ((H - H.mean(0)) ** 2).sum()

        ratio = within_sum / total
        print(f"    L{l:<5d}  {ratio:14.4f}")


def main():
    data = np.load(STATES)
    H, a, b, layer = data["H"], data["a"], data["b"], data["layer"]
    n_layers = int(layer.max()) + 1
    print(f"Loaded {H.shape[0]} states, {n_layers} layers x {(layer==0).sum()} pairs")

    per_layer = by_layer(H, layer, a, b, n_layers)
    swap_distance_test(per_layer, n_layers)
    probe_orthogonality_test(per_layer, n_layers)
    variance_decomposition(per_layer, n_layers)


if __name__ == "__main__":
    main()
