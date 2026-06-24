"""Late-layer sorting: which OUTPUT features become decodable per layer?

The earlier information-gain sweep tracked input-side features
(operand a, operand b, min, max, zero).  But the late layers face the
task of emitting the next character — so they have to sort hidden
states by *what they're about to write*, not by what they're seeing.

Output-side features for a × b prompts:
  - first_digit_of_product  — what character to emit next at the '=' slot
  - product_length          — 1-digit vs 2-digit answer (controls newline timing)
  - ones_digit_of_product   — the second character (only meaningful for 2-digit)
  - tens_digit_of_product   — same as first-digit for 2-digit, 0 for 1-digit
  - is_a_squared            — diagonal pairs (a==b)
  - is_commutative_pair     — pairs with a swap partner (a != b)

The expectation: input-side features (operand_a, min, max) peak early and
drop with depth; output-side features rise with depth.  L5 sorts by what
it's about to write.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STATES = Path(__file__).parent / "data" / "nanogpt_times_table" / "hidden_states.npz"

N_SPLITS = 5
N_SEEDS = 5


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


def majority(y):
    _, cnts = np.unique(y, return_counts=True)
    return float(cnts.max() / cnts.sum())


def main():
    data = np.load(STATES)
    H, a, b, prod, layer = (
        data["H"], data["a"], data["b"], data["product"], data["layer"],
    )
    n_layers = int(layer.max()) + 1
    print(f"Loaded {n_layers} layers x {(layer==0).sum()} pairs\n")

    # Input-side features (compare against)
    in_feats = {
        "operand_a":       a,
        "operand_b":       b,
        "min(a,b)":        np.minimum(a, b),
        "zero":            (a * b == 0).astype(int),
    }
    # Output-side features
    first_dig = np.array([int(str(int(p))[0]) for p in prod])
    ones_dig  = (prod % 10).astype(int)
    tens_dig  = (prod // 10).astype(int)
    out_feats = {
        "first_digit_of_product":  first_dig,
        "product_length(1v2)":     (prod >= 10).astype(int),
        "tens_digit_of_product":   tens_dig,
        "ones_digit_of_product":   ones_dig,
    }

    layers = list(range(n_layers))
    print(f"{'feature':30}  chance  " + "  ".join(f" L{l}  " for l in layers))
    print("-" * 80)

    print("INPUT-SIDE features")
    print("-" * 80)
    for name, y in in_feats.items():
        ch = majority(y[layer == 0])
        per = [cv_acc(H[layer == l], y[layer == l]) for l in layers]
        cells = "  ".join(f"{v:.3f}" for v in per)
        delta_max = max(per) - per[0]
        delta_layer = int(np.argmax(per))
        print(f"  {name:28}  {ch:.2f}    {cells}    "
              f"peak L{delta_layer}  Δfromear={per[delta_layer]-per[0]:+.2f}  "
              f"Δfromend={per[-1]-per[delta_layer]:+.2f}")

    print()
    print("OUTPUT-SIDE features")
    print("-" * 80)
    for name, y in out_feats.items():
        ch = majority(y[layer == 0])
        per = [cv_acc(H[layer == l], y[layer == l]) for l in layers]
        cells = "  ".join(f"{v:.3f}" for v in per)
        peak = int(np.argmax(per))
        print(f"  {name:28}  {ch:.2f}    {cells}    "
              f"peak L{peak}  Δfromear={per[peak]-per[0]:+.2f}  "
              f"Δfromend={per[-1]-per[peak]:+.2f}")


if __name__ == "__main__":
    main()
