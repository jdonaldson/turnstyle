"""Is operand-is-5 really special, or do all operand identities have
comparable Fisher d'?  Compute LDA separation for 'one of the operands
== k' across k=0..9 to see which digits get dedicated channels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
          / "hidden_states.npz")


def fisher_d_prime(H: np.ndarray, mask: np.ndarray) -> float:
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
    H, a, b, layer = data["H"], data["a"], data["b"], data["layer"]
    n_layers = int(layer.max()) + 1

    print(f"{'mask':30}  " +
          "  ".join(f"L{l}".center(6) for l in range(n_layers)) +
          "    peak")
    print("-" * 90)

    rows = []
    for k in range(10):
        mask = (a == k) | (b == k)
        per_layer = []
        for l in range(n_layers):
            m_layer = layer == l
            per_layer.append(fisher_d_prime(H[m_layer], mask[m_layer]))
        peak = int(np.argmax(per_layer))
        rows.append((k, mask.sum(), per_layer, peak))

    for k, n, per_layer, peak in rows:
        cells = "  ".join(f"{d:5.2f}" for d in per_layer)
        marker = "   <--" if k in (0, 1, 5) else ""
        print(f"operand == {k}  (n={int(n)})           {cells}    L{peak} "
              f"({per_layer[peak]:.2f}){marker}")

    print()
    print("Peak d' by operand at each layer:")
    print(f"{'layer':>6}  " + "  ".join(f"k={k}".center(7) for k in range(10)))
    for l in range(n_layers):
        cells = "  ".join(f"{rows[k][2][l]:6.2f}" for k in range(10))
        print(f"  L{l}    {cells}")


if __name__ == "__main__":
    main()
