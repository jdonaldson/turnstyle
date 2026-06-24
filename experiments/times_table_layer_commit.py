"""When are answers available?  Apply head(ln_f(h)) at every layer's
'=' hidden state and check whether the argmax is the right next
character.  Per-layer accuracy + per-difficulty breakdown tells us
whether the computation is coasting in the late layers or doing real
work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import ITOS, load  # noqa: E402


STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
          / "hidden_states.npz")


def main():
    model = load("cpu")
    data = np.load(STATES)
    a = data["a"]; b = data["b"]; prod = data["product"]
    l_arr = data["layer"]
    H_np = data["H"]
    n_layers = int(l_arr.max()) + 1

    # Logit-lens at every (prompt, layer) cell.
    H = torch.from_numpy(H_np).float()
    with torch.no_grad():
        H_normed = model.ln_f(H)
        logits = model.head(H_normed)
    pred = logits.argmax(dim=-1).numpy()
    pred_chars = np.array([ITOS[int(p)] for p in pred])
    true_chars = np.array([str(int(p))[0] for p in prod])
    correct = (pred_chars == true_chars)

    print("Per-layer next-character accuracy (logit-lens):")
    for l in range(n_layers):
        m = l_arr == l
        acc = correct[m].mean()
        print(f"  L{l}: {acc:.3f}")

    print()
    print("Per-difficulty breakdown:")
    print(f"{'group':38}  n  " +
          "  ".join(f"L{l}".center(6) for l in range(n_layers)))
    groups = [
        ("zero  (a*b == 0)", (a * b) == 0),
        ("identity  (×1, non-zero)",
         ((a == 1) | (b == 1)) & ((a * b) != 0)),
        ("small  (2..9, no ×0 ×1)",
         ((a * b) >= 2) & ((a * b) < 10) & (a != 0) & (b != 0)
         & (a != 1) & (b != 1)),
        ("medium  (10..19)",
         ((a * b) >= 10) & ((a * b) < 20)),
        ("large  (≥ 20)",  (a * b) >= 20),
    ]
    for name, pair_mask in groups:
        n_per_pair = int((pair_mask & (l_arr == 0)).sum())
        cells = []
        for l in range(n_layers):
            m = (l_arr == l) & pair_mask
            cells.append(f"{correct[m].mean():.3f}" if m.sum() else "  -  ")
        print(f"{name:38} {n_per_pair:3d}  " +
              "  ".join(c.center(6) for c in cells))

    # When does each prompt commit?  First layer where it's correct AND
    # stays correct through L5.
    print()
    print("Layer where each prompt commits (first correct, stays correct):")
    commit_layers = []
    for a_v in range(10):
        for b_v in range(10):
            pair_correct = []
            for l in range(n_layers):
                m = (l_arr == l) & (a == a_v) & (b == b_v)
                pair_correct.append(bool(correct[m].item()) if m.sum() else False)
            # first layer where it's correct and stays correct
            commit = None
            for l in range(n_layers):
                if all(pair_correct[l:]):
                    commit = l
                    break
            commit_layers.append((a_v, b_v, a_v * b_v, commit))

    from collections import Counter
    layer_dist = Counter([c[3] for c in commit_layers])
    print(f"  Distribution: {dict(sorted(layer_dist.items(), key=lambda x: (x[0] is None, x[0] or 0)))}")

    # Hardest pairs (commit latest)
    by_commit = sorted(commit_layers, key=lambda x: (x[3] is None, x[3] or 0), reverse=True)
    print()
    print("Pairs that commit latest (the genuinely hard cases):")
    for a_v, b_v, p, c in by_commit[:15]:
        print(f"  {a_v}*{b_v}={p:2d}  commit at L{c}")


if __name__ == "__main__":
    main()
