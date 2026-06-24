"""All 100 (a, b) pairs with the layer at which they commit.

Commit = first layer where the model's logit-lens argmax matches the
correct next character AND stays correct through L5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import STOI, load  # noqa: E402

STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
          / "hidden_states.npz")


def main():
    model = load("cpu")
    data = np.load(STATES)
    a = data["a"]; b = data["b"]; prod = data["product"]; l_arr = data["layer"]
    H = torch.from_numpy(data["H"]).float()
    with torch.no_grad():
        logits = model.head(model.ln_f(H))
        probs = torch.softmax(logits, dim=-1).numpy()
    true_ids = np.array([STOI[str(int(p))[0]] for p in prod])
    correct = probs.argmax(axis=1) == true_ids
    n_layers = int(l_arr.max()) + 1

    table = np.full((10, 10), -1, dtype=int)
    for a_v in range(10):
        for b_v in range(10):
            per_layer = []
            for l in range(n_layers):
                m = (l_arr == l) & (a == a_v) & (b == b_v)
                per_layer.append(bool(correct[m].item()))
            for l in range(n_layers):
                if all(per_layer[l:]):
                    table[a_v, b_v] = l
                    break

    # 10×10 grid by commit layer
    print("Commit-layer grid  (rows = a, cols = b):")
    print("       " + "    ".join(f"b={b}" for b in range(10)))
    for a_v in range(10):
        cells = []
        for b_v in range(10):
            l = table[a_v, b_v]
            cells.append(f" L{l} " if l >= 0 else " --")
        print(f" a={a_v}  " + " ".join(cells))

    # Same grid but showing product instead of cell label, for reference
    print()
    print("Product grid  (rows = a, cols = b):")
    print("       " + "    ".join(f"b={b}" for b in range(10)))
    for a_v in range(10):
        cells = [f"{a_v*b_v:3d} " for b_v in range(10)]
        print(f" a={a_v}  " + " ".join(cells))

    # Grouped list by commit layer
    print()
    print("Pairs grouped by commit layer (sorted by product within each group):")
    for l in range(n_layers):
        pairs = [(a_v, b_v) for a_v in range(10) for b_v in range(10)
                 if table[a_v, b_v] == l]
        if not pairs:
            continue
        pairs.sort(key=lambda x: (x[0] * x[1], x[0], x[1]))
        print(f"\n  L{l} commits ({len(pairs)} pairs):")
        for i in range(0, len(pairs), 8):
            chunk = pairs[i:i + 8]
            print("    " + "    ".join(
                f"{av}*{bv}={av*bv:2d}" for av, bv in chunk
            ))


if __name__ == "__main__":
    main()
