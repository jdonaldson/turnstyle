"""What is L5 actually computing that L4 didn't?

Two probes:

  (1) L5 attention pattern, stratified by L4-rank.
      L5_commit pairs split into:
        rank=1 at L4 (12/26) — h_L4 already has correct as strongest
        rank>1 at L4 (14/26) — h_L4 has wrong digit as strongest
      Compare per-head L5 attention weights from =-token. If L5
      re-gathers operand info for the wrong-rank pairs but not the
      right-rank ones, L5 attention is doing pair-conditional repair.
      If L5 attention is the same regardless of rank, the work is in
      L5's MLP only.

  (2) Linear decodability of first digit from h_L4 vs head readout.
      Head readout at L4 gets the correct first digit for 86/100 pairs
      (=14 wrong + 12 weak L5-commit). Train a fresh 10-way LogReg
      probe on h_L4 → first_digit (5-fold CV). If probe accuracy is
      ~100%, the correct-digit info is already LINEARLY available at
      L4 — L5's MLP just rotates/translates so the head can read it.
      If probe accuracy is ~86%, the info has to be computed
      nonlinearly by L5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import STOI, encode, load  # noqa: E402
from times_table_block_decomp import capture_decomposition  # noqa: E402

STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
          / "hidden_states.npz")


def cv_acc(X, y, n_splits=5, seeds=3):
    accs = []
    for seed in range(seeds):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
        for tr, te in kf.split(X, y):
            s = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=4000, C=1.0).fit(
                s.transform(X[tr]), y[tr]
            )
            accs.append(m.score(s.transform(X[te]), y[te]))
    return float(np.mean(accs))


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

    # commit class per pair (logit-lens on cached states)
    H_t = torch.from_numpy(H_all).float()
    with torch.no_grad():
        logits = model.head(model.ln_f(H_t)).numpy()
    pred = logits.argmax(axis=1)
    true_ids = np.array([STOI[str(int(p))[0]] for p in prod])
    correct = (pred == true_ids).reshape(100, 6)
    commit = np.full(100, -1, dtype=int)
    for i in range(100):
        for L in range(6):
            if correct[i, L:].all():
                commit[i] = L
                break

    H_L4 = H_all[l_arr == 4]   # (100, 128)
    H_L5 = H_all[l_arr == 5]

    # L4 rank of correct digit
    projs_L4 = H_L4 @ W_d_c.T  # (100, 10)
    rank_correct = np.zeros(100, dtype=int)
    for i in range(100):
        order = np.argsort(-projs_L4[i])
        rank_correct[i] = int(np.where(order == true_first[i])[0][0]) + 1

    L5_commit_idx = np.where(commit == 5)[0]
    L5_rank1 = L5_commit_idx[rank_correct[L5_commit_idx] == 1]
    L5_rankgt1 = L5_commit_idx[rank_correct[L5_commit_idx] > 1]
    print(f"L5_commit n={len(L5_commit_idx)}: "
          f"rank=1 at L4: {len(L5_rank1)}, rank>1: {len(L5_rankgt1)}")

    # ── (1) L5 attention pattern, stratified ──
    print("\n=" * 50)
    print("(1) L5 attention pattern at =-token, stratified by L4-rank")
    print("=" * 50)
    dec = capture_decomposition(model, device="cpu")
    weights = dec["weights"]  # (100, n_L, n_head, T)
    # L5 attention weights at =-pos (already keyed there in capture)
    L5_w = weights[:, 5]  # (100, n_head, T=4)
    print(f"Positions: 0=a, 1=*, 2=b, 3==")
    print()

    def summary(name, idx):
        w = L5_w[idx]  # (n, n_head, 4)
        per_head_avg = w.mean(axis=0)  # (n_head, 4)
        head_avg = per_head_avg.mean(axis=0)
        print(f"  {name} (n={len(idx)}):")
        print(f"    Mean attention from =-token (heads averaged):")
        print(f"      a={head_avg[0]:.3f}  *={head_avg[1]:.3f}  "
              f"b={head_avg[2]:.3f}  ==​{head_avg[3]:.3f}")
        print(f"    Per-head breakdown:")
        for h in range(per_head_avg.shape[0]):
            print(f"      head {h}: a={per_head_avg[h,0]:.3f}  "
                  f"*={per_head_avg[h,1]:.3f}  "
                  f"b={per_head_avg[h,2]:.3f}  "
                  f"==​{per_head_avg[h,3]:.3f}")

    summary("L5_rank1 (correct already strongest at L4)", L5_rank1)
    print()
    summary("L5_rank>1 (wrong digit was strongest at L4)", L5_rankgt1)
    print()
    # All other pairs for comparison
    other_idx = np.array([i for i in range(100) if i not in L5_commit_idx])
    summary("All non-L5-commit pairs (n=74)", other_idx)

    # ── (2) Linear decodability vs head readout ──
    print()
    print("=" * 50)
    print("(2) Linear decodability of first digit from h_L4 vs h_L5")
    print("=" * 50)
    print()
    # All 100 pairs
    print("All 100 pairs:")
    acc_L4 = cv_acc(H_L4, true_first)
    acc_L5 = cv_acc(H_L5, true_first)
    head_L4_acc = (rank_correct == 1).mean()
    head_L5_pred = logits[l_arr == 5].argmax(axis=1)
    head_L5_acc = (head_L5_pred == true_ids[l_arr == 5]).mean()
    print(f"  head readout acc at L4:        {head_L4_acc:.1%}")
    print(f"  head readout acc at L5:        {head_L5_acc:.1%}")
    print(f"  linear probe acc on h_L4:      {acc_L4:.1%}   "
          f"(5-fold CV, 3 seeds)")
    print(f"  linear probe acc on h_L5:      {acc_L5:.1%}")
    print()
    print("  → if linear-on-L4 ≈ 100% but head-on-L4 is lower, "
          "L5 is a rotation/translation.")
    print("  → if linear-on-L4 ≈ head-on-L4, L5 must compute new info.")

    # Restrict to L5_commit pairs (n=26) for stratified read
    print()
    print("L5_commit pairs only (n=26) — head got these right ONLY because "
          "of L5's work:")
    H_L4_sub = H_L4[L5_commit_idx]
    H_L5_sub = H_L5[L5_commit_idx]
    y_sub = true_first[L5_commit_idx]
    # For very small n, CV doesn't make sense; use k=min(5, n_classes_min).
    # Run train-test split as an indicator.
    # Stratified k-fold needs each class in each fold; many classes have 1
    # sample. Skip CV here — show simple decoder accuracy on the
    # whole-set fit (in-sample) plus the head readout at L4/L5.
    head_at_L4_sub = (rank_correct[L5_commit_idx] == 1).mean()
    head_at_L5_sub = (
        logits[l_arr == 5].argmax(axis=1)[L5_commit_idx]
        == true_ids[l_arr == 5][L5_commit_idx]
    ).mean()
    # Fit on whole 100, score on L5_commit subset.
    s = StandardScaler().fit(H_L4)
    m_L4 = LogisticRegression(max_iter=4000, C=1.0).fit(s.transform(H_L4),
                                                         true_first)
    sub_L4_acc = m_L4.score(s.transform(H_L4_sub), y_sub)
    s2 = StandardScaler().fit(H_L5)
    m_L5 = LogisticRegression(max_iter=4000, C=1.0).fit(s2.transform(H_L5),
                                                         true_first)
    sub_L5_acc = m_L5.score(s2.transform(H_L5_sub), y_sub)
    print(f"  head readout acc at L4:                {head_at_L4_sub:.1%}")
    print(f"  head readout acc at L5:                {head_at_L5_sub:.1%}")
    print(f"  linear probe (trained on all) on h_L4: {sub_L4_acc:.1%}")
    print(f"  linear probe (trained on all) on h_L5: {sub_L5_acc:.1%}")


if __name__ == "__main__":
    main()
