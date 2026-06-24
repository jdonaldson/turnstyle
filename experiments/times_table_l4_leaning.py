"""At L4, for each pair, is the correct digit already the strongest
direction in the residual?  If yes for ALL pairs (including L5-commit
ones), L5 is a pure amplifier. If no for L5-commit pairs, L5 is doing
selection too.

For each pair, compute h_L4 · (head[d] − centroid) for d=0..9.
Then ask: rank of correct digit's projection among the 10 digits.
  rank=1 → correct is already strongest at L4 (amplifier interpretation).
  rank>1 → correct is NOT strongest at L4 (L5 must select, not just amplify).

Stratify by commit class.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import STOI, load  # noqa: E402

STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
          / "hidden_states.npz")


def main():
    model = load("cpu")
    digit_ids = [STOI[str(d)] for d in range(10)]
    W = model.head.weight.detach().cpu().numpy()  # (vocab, H)
    W_d = W[digit_ids]                             # (10, H)
    centroid = W_d.mean(axis=0)
    W_d_c = W_d - centroid                          # (10, H)

    data = np.load(STATES)
    H_all = data["H"]
    l_arr = data["layer"]
    prod = data["product"]

    # Per-pair commit class (same logic as block-decomp script)
    import torch
    H_t = torch.from_numpy(H_all).float()
    with torch.no_grad():
        logits = model.head(model.ln_f(H_t)).numpy()
    pred = logits.argmax(axis=1)
    true_ids = np.array([STOI[str(int(p))[0]] for p in prod])
    correct = pred == true_ids
    n_pairs = 100
    n_layers = 6
    correct_grid = correct.reshape(n_pairs, n_layers)  # (100, 6)

    commit = np.full(n_pairs, -1, dtype=int)
    for i in range(n_pairs):
        for L in range(n_layers):
            if correct_grid[i, L:].all():
                commit[i] = L
                break

    # h_L4 per pair
    L4_mask = l_arr == 4
    H_L4 = H_all[L4_mask]                    # (100, 128)
    prod_L4 = prod[L4_mask]
    true_first_L4 = np.array([int(str(int(p))[0]) for p in prod_L4])

    # Projection of each pair's h_L4 onto each digit's centered row
    # Shape: (100, 10)
    projs = H_L4 @ W_d_c.T

    rank_correct = np.zeros(n_pairs, dtype=int)
    for i in range(n_pairs):
        # Higher projection = closer alignment. Rank=1 means highest.
        order = np.argsort(-projs[i])
        rank_correct[i] = int(np.where(order == true_first_L4[i])[0][0]) + 1

    # Stratify by commit class
    from collections import Counter
    print("Distribution of correct-digit rank at L4 (rank=1 means correct "
          "is the strongest direction):")
    print()
    classes = {
        "L2_commit (n=6)":   commit == 2,
        "L3_commit (n=19)":  commit == 3,
        "L4_commit (n=49)":  commit == 4,
        "L5_commit (n=26)":  commit == 5,
    }
    for name, mask in classes.items():
        ranks = rank_correct[mask]
        c = Counter(ranks.tolist())
        rank1 = c.get(1, 0)
        rank2 = c.get(2, 0)
        rank3 = c.get(3, 0)
        mean_rank = float(ranks.mean())
        n = int(mask.sum())
        print(f"  {name:20}   rank=1: {rank1}/{n}   "
              f"rank=2: {rank2}/{n}   rank=3: {rank3}/{n}   "
              f"mean rank: {mean_rank:.2f}")
        rank_dist = sorted(c.items())
        print(f"  {' ' * 20}   full distribution: {rank_dist}")
        print()

    # Also show: for L5_commit pairs, how far is correct from being top-1?
    print("L5_commit pairs — correct-digit projection vs strongest "
          "(non-correct) at L4:")
    print(f"  {'pair':>6}  {'correct':>8}  {'proj_correct':>12}  "
          f"{'top_wrong':>10}  {'proj_top_wrong':>14}  {'gap':>8}")
    L5_idx = np.where(commit == 5)[0]
    for i in L5_idx[:15]:
        p_arr = projs[i].copy()
        c_d = true_first_L4[i]
        p_c = float(p_arr[c_d])
        p_arr[c_d] = -1e9
        top_w = int(p_arr.argmax())
        p_w = float(projs[i, top_w])
        a_val = i // 10; b_val = i % 10
        gap = p_c - p_w
        print(f"  {a_val}*{b_val:<2d}  d={c_d}     {p_c:+12.3f}  "
              f"d={top_w}        {p_w:+14.3f}  {gap:+8.3f}")


if __name__ == "__main__":
    main()
