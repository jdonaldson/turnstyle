"""Per-layer information-gain curve for times-table features.

Fits a regularised linear probe per (layer, feature) and reports 5-fold
CV decodability.  Goal: see whether the L3 annihilator is the biggest
single-layer information-gain event, or whether other features (operand
position, product value) gain more.

Run after times_table_trace.py has populated hidden_states.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
STATES = ROOT / "hidden_states.npz"

N_SPLITS = 5
N_SEEDS = 5  # average over seeds to stabilise n=100 CV estimate


def cv_acc(X, y):
    accs = []
    for seed in range(N_SEEDS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000),
            )
            clf.fit(X[tr], y[tr])
            accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))


def cv_r2(X, y):
    r2s = []
    for seed in range(N_SEEDS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            reg.fit(X[tr], y[tr])
            r2s.append(reg.score(X[te], y[te]))
    return float(np.mean(r2s))


def majority(y):
    _, cnts = np.unique(y, return_counts=True)
    return float(cnts.max() / cnts.sum())


def main():
    data = np.load(STATES)
    H, a, b, prod, layer = (
        data["H"], data["a"], data["b"], data["product"], data["layer"],
    )
    n_layers = int(layer.max()) + 1
    n_pairs = int((layer == 0).sum())
    print(f"Loaded: {n_layers} layers x {n_pairs} pairs, hidden_dim={H.shape[1]}\n")

    cls_feats = {
        "zero_product":   (a * b == 0).astype(int),
        "product_lt_10":  (a * b < 10).astype(int),
        "a_eq_b":         (a == b).astype(int),
        "operand_a":      a,
        "operand_b":      b,
        "min_ab":         np.minimum(a, b),
        "max_ab":         np.maximum(a, b),
    }
    reg_feats = {
        "product":        prod.astype(float),
        "sum_ab":         (a + b).astype(float),
    }

    headers = "  ".join(f" L{l}  " for l in range(n_layers))
    print(f"{'feature':18}  chance  {headers}  Δmax  argmax(Δ_step)")
    print("-" * (20 + 8 + n_layers * 7 + 22))

    rows = []
    for name, y in cls_feats.items():
        per = []
        for l in range(n_layers):
            m = layer == l
            per.append(cv_acc(H[m], y[m]))
        per = np.array(per)
        deltas = np.diff(per)
        step = int(np.argmax(deltas)) if deltas.size else 0
        rows.append((name, majority(y[layer == 0]), per, step, deltas))

    for name, ch, per, step, deltas in rows:
        cells = "  ".join(f"{x:.3f}" for x in per)
        dmax = per.max() - per[0]
        max_step_delta = deltas.max() if deltas.size else 0.0
        print(f"{name:18}  {ch:.2f}    {cells}  {dmax:+.2f}  "
              f"L{step}->L{step+1} (+{max_step_delta:.2f})")

    print()
    print(f"{'reg feature (R²)':18}  baseline  {headers}  Δmax  argmax(Δ_step)")
    print("-" * (20 + 10 + n_layers * 7 + 22))
    for name, y in reg_feats.items():
        per = []
        for l in range(n_layers):
            m = layer == l
            per.append(cv_r2(H[m], y[m]))
        per = np.array(per)
        deltas = np.diff(per)
        step = int(np.argmax(deltas)) if deltas.size else 0
        cells = "  ".join(f"{x:+.3f}" for x in per)
        dmax = per.max() - per[0]
        max_step_delta = deltas.max() if deltas.size else 0.0
        print(f"{name:18}   0.00     {cells}  {dmax:+.2f}  "
              f"L{step}->L{step+1} (+{max_step_delta:.2f})")


if __name__ == "__main__":
    main()
