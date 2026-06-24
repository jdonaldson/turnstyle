"""Does L5 actively suppress the wrong-winner-at-L4, or just boost correct?

For the 14 L5-commit pairs where L4's strongest digit direction was WRONG:
  - Compute Δh_L5 · wrong_rank_1_digit_dir (the dir that "won" at L4).
  - Compute Δh_L5 · correct_digit_dir.
  - If wrong_dir projection is NEGATIVE: L5 actively suppresses.
  - If ≈ 0: L5 just boosts correct, lets wrong fade by relative magnitude.
  - If positive: L5 boosts both (correct wins because magnitude larger).

Also done for all 100 pairs and stratified by commit class.
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
    digit_ids = [STOI[str(d)] for d in range(10)]
    W = model.head.weight.detach().cpu().numpy()
    W_d = W[digit_ids]
    centroid = W_d.mean(axis=0)
    W_d_c = W_d - centroid

    data = np.load(STATES)
    H_all = data["H"]
    l_arr = data["layer"]
    prod = data["product"]
    true_first = np.array([int(str(int(p))[0]) for p in prod[l_arr == 0]])

    H_L4 = H_all[l_arr == 4]
    H_L5 = H_all[l_arr == 5]
    Delta = H_L5 - H_L4   # (100, 128)

    # Projections of Δh_L5 onto each digit direction (unit-norm)
    digit_norms = np.linalg.norm(W_d_c, axis=1, keepdims=True)
    digit_unit = W_d_c / (digit_norms + 1e-9)   # (10, 128)
    projs_delta = Delta @ digit_unit.T          # (100, 10) — onto unit digit dirs

    # L4 ranks of digits (head-axis projection, sign-aware)
    projs_L4 = H_L4 @ W_d_c.T                   # (100, 10) — not unit-norm
    rank_correct = np.zeros(100, dtype=int)
    rank1_digit = np.zeros(100, dtype=int)
    rank2_digit = np.zeros(100, dtype=int)
    for i in range(100):
        order = np.argsort(-projs_L4[i])
        rank_correct[i] = int(np.where(order == true_first[i])[0][0]) + 1
        rank1_digit[i] = int(order[0])
        rank2_digit[i] = int(order[1])

    # Commit class
    H_t = torch.from_numpy(H_all).float()
    with torch.no_grad():
        logits = model.head(model.ln_f(H_t)).numpy()
    true_ids = np.array([STOI[str(int(p))[0]] for p in prod])
    correct_per_state = (logits.argmax(axis=1) == true_ids).reshape(100, 6)
    commit = np.full(100, -1, dtype=int)
    for i in range(100):
        for L in range(6):
            if correct_per_state[i, L:].all():
                commit[i] = L
                break

    L5_commit_idx = np.where(commit == 5)[0]
    wrong_rank = L5_commit_idx[rank_correct[L5_commit_idx] > 1]
    right_rank = L5_commit_idx[rank_correct[L5_commit_idx] == 1]

    print(f"L5_commit n={len(L5_commit_idx)}: "
          f"correct-rank=1 at L4: {len(right_rank)}, rank>1: {len(wrong_rank)}")
    print()

    # ── (1) For the 14 wrong-rank pairs: projections on correct vs wrong-winner ──
    print("=" * 90)
    print("(1) L5_commit pairs with WRONG rank-1 at L4 (n=14)")
    print("=" * 90)
    print(f"  {'pair':>6}  {'correct':>8}  {'L4 wrong':>9}  "
          f"{'proj_corr':>10}  {'proj_wrong':>11}  {'proj_diff':>10}")
    for i in wrong_rank:
        a = i // 10; b = i % 10
        c_d = true_first[i]
        w_d = rank1_digit[i]
        p_c = projs_delta[i, c_d]
        p_w = projs_delta[i, w_d]
        print(f"  {a}*{b:<2d}  d={c_d}     d={w_d}        "
              f"{p_c:+10.3f}  {p_w:+11.3f}  {p_c - p_w:+10.3f}")
    print()
    print(f"  Mean proj on correct (wrong-rank pairs): "
          f"{projs_delta[wrong_rank, true_first[wrong_rank]].mean():+.3f}")
    print(f"  Mean proj on L4-wrong-winner (wrong-rank pairs): "
          f"{projs_delta[wrong_rank, rank1_digit[wrong_rank]].mean():+.3f}")

    # ── (2) Mean Δh_L5 projections per "role" (correct, runner-up, third) ──
    print()
    print("=" * 90)
    print("(2) Mean Δh_L5 projection onto each role digit, by commit class")
    print("=" * 90)
    classes = {
        "all (n=100)": np.arange(100),
        "L2_commit (n=6)": np.where(commit == 2)[0],
        "L3_commit (n=19)": np.where(commit == 3)[0],
        "L4_commit (n=49)": np.where(commit == 4)[0],
        "L5_commit_RIGHT (n=12)": right_rank,
        "L5_commit_WRONG (n=14)": wrong_rank,
    }
    print(f"  {'class':>26}  {'proj_correct':>13}  {'proj_rank1':>11}  "
          f"{'proj_rank2':>11}  {'mean_other':>11}")
    for name, idx in classes.items():
        if len(idx) == 0:
            continue
        p_c = projs_delta[idx, true_first[idx]].mean()
        p_r1 = projs_delta[idx, rank1_digit[idx]].mean()
        p_r2 = projs_delta[idx, rank2_digit[idx]].mean()
        # mean of all 10 minus the correct (background)
        all_proj = projs_delta[idx]
        # Exclude correct digit per row
        mask = np.ones((len(idx), 10), dtype=bool)
        for k, ii in enumerate(idx):
            mask[k, true_first[ii]] = False
        bg = all_proj[mask].reshape(len(idx), 9).mean(axis=1).mean()
        print(f"  {name:>26}  {p_c:+13.3f}  {p_r1:+11.3f}  "
              f"{p_r2:+11.3f}  {bg:+11.3f}")

    # ── (3) Per-digit "background" projection: for each digit d, the mean
    # Δh_L5 · d_dir across pairs whose correct digit is NOT d ──
    print()
    print("=" * 90)
    print("(3) Mean Δh_L5 projection on each digit dir, conditioned on correctness")
    print("=" * 90)
    print(f"  {'d':>3}  {'when correct (mean)':>20}  {'when NOT correct (mean)':>23}  "
          f"{'n_correct':>10}  {'n_not_correct':>13}")
    for d in range(10):
        is_d = true_first == d
        not_d = true_first != d
        m_when_corr = projs_delta[is_d, d].mean() if is_d.any() else 0.0
        m_when_wrong = projs_delta[not_d, d].mean()
        n_c = int(is_d.sum())
        n_nc = int(not_d.sum())
        print(f"  {d:>3}  {m_when_corr:+20.3f}  {m_when_wrong:+23.3f}  "
              f"{n_c:>10d}  {n_nc:>13d}")
    print()
    print("Interpretation: 'when correct' >> 'when not correct' implies L5 boosts.")
    print("'when not correct' negative implies L5 suppresses that digit when wrong.")


if __name__ == "__main__":
    main()
