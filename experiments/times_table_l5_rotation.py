"""Extract and analyze L5's basis-change rotation.

Three views:

  (1) SVD of Δh_L5 across pairs. Δh_L5 = h_L5 − h_L4 is what block 5
      writes to the residual stream at the =-position. Stack across the
      100 pairs to a (100, 128) matrix. Singular values reveal the
      effective dimensionality of L5's writes:
        - If S[0] ≫ S[1], L5 writes in essentially one direction.
        - If S[0..9] dominate and S[10:] are tiny, L5 writes in a ~10-d
          subspace (one direction per digit class).
        - If singular values are spread, L5 writes pair-specifically.

  (2) Linear approximation of the full block. Fit Ridge h_L5 ≈ M·h_L4 + c.
      M is 128×128. Decompose M = I + B where B = M − I is the "added
      transformation" beyond residual identity. SVD of B shows how much
      of L5's effect is identity (pass-through) vs structured rotation.
      Cosine angles of M's input/output singular vectors tell us whether
      M is close to a rotation (unitary) or general linear map.

  (3) Direction alignment. For each pair, compute cosine of Δh_L5 with
      the correct-digit row direction. If cos is uniformly high (~0.7+)
      across all 100 pairs regardless of which digit they need, L5 is
      writing a "pair-conditioned vector aimed at the correct row" with
      stable angular alignment. Combined with the +6 uniform magnitude
      from before, this would mean L5's write is approximately
      `α · correct_digit_dir[pair]` — a per-pair selection with normalized
      magnitude.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import STOI, load  # noqa: E402

STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
          / "hidden_states.npz")


def main():
    model = load("cpu")
    digit_ids = [STOI[str(d)] for d in range(10)]
    W = model.head.weight.detach().cpu().numpy()
    W_d = W[digit_ids]            # (10, H)
    centroid = W_d.mean(axis=0)
    W_d_c = W_d - centroid

    data = np.load(STATES)
    H_all = data["H"]
    l_arr = data["layer"]
    prod = data["product"]
    true_first = np.array([int(str(int(p))[0]) for p in prod[l_arr == 0]])

    H_L4 = H_all[l_arr == 4]   # (100, 128)
    H_L5 = H_all[l_arr == 5]
    Delta = H_L5 - H_L4         # (100, 128) — block-5 contribution

    print(f"Delta shape: {Delta.shape}")
    print(f"Mean ‖Delta‖: {np.linalg.norm(Delta, axis=1).mean():.3f}")
    print(f"Mean ‖h_L4‖:  {np.linalg.norm(H_L4, axis=1).mean():.3f}")
    print(f"Mean ‖h_L5‖:  {np.linalg.norm(H_L5, axis=1).mean():.3f}")

    # ── (1) SVD of Δh_L5 ──
    U, S, Vt = np.linalg.svd(Delta, full_matrices=False)
    print()
    print("=" * 80)
    print("(1) SVD of Δh_L5 across 100 pairs")
    print("=" * 80)
    print(f"  Top 15 singular values:")
    for i, s in enumerate(S[:15]):
        print(f"    σ{i:>2}: {s:8.3f}    cumulative: "
              f"{(S[:i+1]**2).sum() / (S**2).sum() * 100:5.1f}% of variance")
    print(f"  Effective rank (90% variance): "
          f"{int((np.cumsum(S**2) / (S**2).sum() < 0.90).sum() + 1)}")
    print(f"  Effective rank (99% variance): "
          f"{int((np.cumsum(S**2) / (S**2).sum() < 0.99).sum() + 1)}")

    # Alignment of top singular vectors with head digit directions
    print()
    print("  Alignment of top-10 right singular vectors with digit rows:")
    print(f"  {'σ':>4}  " + "  ".join(f"d{d}".center(7) for d in range(10))
          + f"  {'|max|':>6}  {'argmax_d':>9}")
    for i in range(10):
        v = Vt[i]
        cosines = W_d_c @ v / (np.linalg.norm(W_d_c, axis=1)
                                 * np.linalg.norm(v) + 1e-9)
        cells = "  ".join(f"{c:+.2f}".center(7) for c in cosines)
        amax = int(np.argmax(np.abs(cosines)))
        print(f"  σ{i:<2}  {cells}  {abs(cosines).max():.2f}    d={amax}")

    # ── (3) Per-pair direction alignment ──
    print()
    print("=" * 80)
    print("(3) Per-pair Δh_L5 alignment with correct-digit direction")
    print("=" * 80)
    correct_dir = W_d_c[true_first]   # (100, H)
    delta_norm = Delta / (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-9)
    correct_norm = correct_dir / (np.linalg.norm(correct_dir, axis=1,
                                                  keepdims=True) + 1e-9)
    cos_per_pair = (delta_norm * correct_norm).sum(axis=1)

    # Stratify by commit class
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

    print(f"  {'class':>20}  {'mean cos':>10}  {'std':>8}  {'min':>8}  "
          f"{'max':>8}")
    classes = {
        "all (n=100)": np.arange(100),
        "L2_commit (n=6)": np.where(commit == 2)[0],
        "L3_commit (n=19)": np.where(commit == 3)[0],
        "L4_commit (n=49)": np.where(commit == 4)[0],
        "L5_commit (n=26)": np.where(commit == 5)[0],
    }
    for name, idx in classes.items():
        c = cos_per_pair[idx]
        print(f"  {name:>20}  {c.mean():+10.3f}  {c.std():8.3f}  "
              f"{c.min():+8.3f}  {c.max():+8.3f}")

    # ── (2) Linear fit h_L5 = M·h_L4 + c via Ridge ──
    print()
    print("=" * 80)
    print("(2) Linear fit h_L5 ≈ M · h_L4 + c (per output dim, Ridge α=1)")
    print("=" * 80)
    # Fit each output dim separately; with 128 features and 100 samples,
    # Ridge regularization is essential.
    M_cols = []
    intercepts = []
    s_in = StandardScaler().fit(H_L4)
    H_L4_s = s_in.transform(H_L4)
    for d in range(128):
        m = Ridge(alpha=1.0, fit_intercept=True).fit(H_L4_s, H_L5[:, d])
        M_cols.append(m.coef_ / s_in.scale_)
        intercepts.append(float(m.intercept_ - m.coef_ @ (s_in.mean_
                                                            / s_in.scale_)))
    M = np.stack(M_cols, axis=0)   # (128 out, 128 in)
    c_vec = np.array(intercepts)

    # Quality of fit
    H_L5_hat = H_L4 @ M.T + c_vec
    residual = H_L5 - H_L5_hat
    print(f"  Mean ‖residual‖:        {np.linalg.norm(residual, axis=1).mean():.4f}")
    print(f"  Mean ‖h_L5‖:            {np.linalg.norm(H_L5, axis=1).mean():.4f}")
    print(f"  Mean ‖residual‖/‖h_L5‖: "
          f"{np.linalg.norm(residual, axis=1).mean() / np.linalg.norm(H_L5, axis=1).mean():.4f}")
    print(f"  Mean cos(h_L5, h_L5_hat) per pair: "
          f"{((H_L5 * H_L5_hat).sum(axis=1) / (np.linalg.norm(H_L5, axis=1) * np.linalg.norm(H_L5_hat, axis=1) + 1e-9)).mean():.4f}")

    # SVD of M
    U_M, S_M, Vt_M = np.linalg.svd(M, full_matrices=False)
    print()
    print(f"  Top 15 singular values of M (h_L5 ≈ M·h_L4 + c):")
    for i in range(15):
        print(f"    σ{i:>2}: {S_M[i]:8.3f}")
    print(f"  Bottom 5: {S_M[-5:]}")

    # SVD of B = M - I
    B = M - np.eye(128)
    U_B, S_B, Vt_B = np.linalg.svd(B, full_matrices=False)
    print()
    print(f"  Top 15 singular values of B = M − I (the 'added rotation'):")
    for i in range(15):
        print(f"    σ{i:>2}: {S_B[i]:8.3f}    cumulative: "
              f"{(S_B[:i+1]**2).sum() / (S_B**2).sum() * 100:5.1f}% of variance")

    # Is M close to a rotation? Check M @ M.T spectrum
    MMT = M @ M.T
    eigs = np.sort(np.linalg.eigvalsh(MMT))[::-1]
    print()
    print(f"  Eigenvalues of M·M^T (if rotation: all = 1):")
    print(f"    top 5:    {eigs[:5]}")
    print(f"    bottom 5: {eigs[-5:]}")
    print(f"    Mean: {eigs.mean():.3f}    Std: {eigs.std():.3f}")

    # Alignment of top singular vectors of B with digit rows
    print()
    print("  Alignment of B's top-10 right singular vectors with digit rows:")
    print(f"  {'σ':>4}  " + "  ".join(f"d{d}".center(7) for d in range(10))
          + f"  {'|max|':>6}  {'argmax':>8}")
    for i in range(10):
        v = Vt_B[i]
        cosines = W_d_c @ v / (np.linalg.norm(W_d_c, axis=1)
                                 * np.linalg.norm(v) + 1e-9)
        cells = "  ".join(f"{c:+.2f}".center(7) for c in cosines)
        amax = int(np.argmax(np.abs(cosines)))
        print(f"  σ{i:<2}  {cells}  {abs(cosines).max():.2f}    d={amax}")


if __name__ == "__main__":
    main()
