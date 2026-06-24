"""Interpret what L0 organizes the 100 (a,b) =-token states by.

Loads the saved (600, 128) hidden states, filters to L0 (post-block-0 at
the =-token, one vector per (a,b) pair in 0..9^2), and asks which simple
properties of (a, b, a*b) are linearly decodable. Reports per-property
1-NN classification accuracy and Spearman correlation of pairwise
similarity in L0 with similarity in each candidate property.

The goal: name what L0 is "about" — operands? trailing digits? product?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
STATES = ROOT / "hidden_states.npz"

LAYER = 0  # change to 1..5 to compare


def main():
    d = np.load(STATES)
    mask = d["layer"] == LAYER
    H = d["H"][mask]  # (100, 128)
    a = d["a"][mask]
    b = d["b"][mask]
    p = d["product"][mask]
    print(f"L{LAYER} states: {H.shape}")

    # Candidate properties of each (a,b) pair
    props = {
        "a (first operand)": a,
        "b (second operand)": b,
        "a * b (product)": p,
        "a + b (sum)": a + b,
        "min(a,b)": np.minimum(a, b),
        "max(a,b)": np.maximum(a, b),
        "trailing digit of product (p % 10)": p % 10,
        "leading digit of product": np.where(p == 0, 0, p // 10**np.floor(np.log10(np.clip(p, 1, None))).astype(int)),
        "is product < 10": (p < 10).astype(int),
        "is a == 0 or b == 0": ((a == 0) | (b == 0)).astype(int),
        "a parity (mod 2)": a % 2,
        "b parity (mod 2)": b % 2,
        "product parity (mod 2)": p % 2,
        "a mod 5": a % 5,
        "b mod 5": b % 5,
    }

    print(f"\n{'Property':45s}  {'5-fold LogReg acc':>20s}")
    print("-" * 70)
    results = []
    for name, y in props.items():
        n_classes = len(np.unique(y))
        if n_classes < 2:
            continue
        chance = float(np.bincount(y).max()) / len(y)
        try:
            scores = cross_val_score(
                LogisticRegression(max_iter=2000, C=1.0),
                H,
                y,
                cv=5,
                scoring="accuracy",
            )
            acc = scores.mean()
        except Exception as e:
            acc = float("nan")
            print(f"  ({name}: {e})")
        lift = acc - chance
        results.append((name, acc, chance, lift, n_classes))
        print(f"{name:45s}  {acc:6.1%}  (chance {chance:5.1%}, lift {lift:+.1%})")

    # RSA: how much does pairwise cosine similarity in H reflect each property's similarity?
    print("\nRSA: Spearman r(cosine(H_i,H_j), property_sim(i,j))")
    print("-" * 70)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / np.clip(norms, 1e-9, None)
    sim_H = Hn @ Hn.T
    iu = np.triu_indices_from(sim_H, k=1)
    cos = sim_H[iu]

    for name, y in props.items():
        if y.ndim != 1:
            continue
        # "Similarity" for a property: 1 if equal, 0 otherwise (categorical)
        eq = (y[:, None] == y[None, :]).astype(float)
        rho, _ = spearmanr(cos, eq[iu])
        print(f"{name:45s}  rho = {rho:+.3f}")


if __name__ == "__main__":
    main()
