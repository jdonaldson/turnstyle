"""Find the next degeneracy after the zero annihilator.

For each candidate degeneracy (identity, square, small-product, etc.),
compute Fisher's LDA d' between the special-case pairs and the rest at
every layer.  A large d' means the layer has a direction that cleanly
routes that case away from the general population.  The peak layer for
each degeneracy tells us when (if ever) the network builds a dedicated
channel for it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
          / "hidden_states.npz")


def fisher_d_prime(H: np.ndarray, mask: np.ndarray) -> float:
    """Fisher's LDA d' for a binary class mask in feature space H."""
    if mask.sum() < 2 or (~mask).sum() < 2:
        return 0.0
    H0 = H[~mask]
    H1 = H[mask]
    Sw = np.cov(H0.T) + np.cov(H1.T)
    diff = H1.mean(0) - H0.mean(0)
    w = np.linalg.pinv(Sw) @ diff
    w = w / (np.linalg.norm(w) + 1e-8)
    p0 = H0 @ w
    p1 = H1 @ w
    d = abs(p1.mean() - p0.mean()) / np.sqrt(
        (p0.var() + p1.var()) / 2 + 1e-8
    )
    return float(d)


def main():
    data = np.load(STATES)
    H, a, b, prod, layer = (
        data["H"], data["a"], data["b"], data["product"], data["layer"],
    )
    n_layers = int(layer.max()) + 1

    degeneracies = {
        "zero  (a*b == 0)":
            (a * b == 0),
        "identity  (a==1 or b==1, non-zero)":
            (((a == 1) | (b == 1)) & (a * b != 0)),
        "square  (a == b, non-zero)":
            ((a == b) & (a != 0)),
        "small product  (1 <= a*b < 10)":
            ((a * b >= 1) & (a * b < 10)),
        "round product  (a*b % 10 == 0, non-zero)":
            (((prod % 10) == 0) & (prod != 0)),
        "operand is 5  (a==5 or b==5)":
            ((a == 5) | (b == 5)),
        "odd product  (a*b is odd)":
            ((prod % 2) == 1),
        "two-digit answer  (a*b >= 10)":
            (a * b >= 10),
        "first digit == 1 of answer":
            np.array([str(int(p))[0] == "1" for p in prod]),
        "first digit == 2 of answer":
            np.array([str(int(p))[0] == "2" for p in prod]),
        "first digit == 6 of answer":
            np.array([str(int(p))[0] == "6" for p in prod]),
    }

    print(f"{'degeneracy':45}  n  " +
          "  ".join(f"L{l}".center(6) for l in range(n_layers)) +
          "   peak")
    print("-" * 100)
    for name, mask in degeneracies.items():
        n = int(mask.sum())
        if n < 3:
            continue
        per_layer = []
        for l in range(n_layers):
            m_layer = layer == l
            per_layer.append(fisher_d_prime(H[m_layer], mask[m_layer]))
        peak_layer = int(np.argmax(per_layer))
        peak_val = per_layer[peak_layer]
        cells = "  ".join(f"{d:5.2f}" for d in per_layer)
        print(f"{name:45}  {n:2d}  {cells}   L{peak_layer} ({peak_val:.2f})")


if __name__ == "__main__":
    main()
