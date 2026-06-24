"""Does the zero trajectory carry meaning beyond 'zero detected'?

Three probes at L3 (where the zero state-norm peak sits):

  1) Within the 19 zero-product pairs, can we recover (a) which position
     was zero (a vs b vs both), and (b) which digit the non-zero operand
     was? If yes, the zero bulge has internal structure — it isn't a
     flat 'I noticed a zero' bit.

  2) Is there a *single* 'zero direction' that explains the bulge? Take
     mean(zero) - mean(non-zero) at L3, project every pair onto it, and
     check (i) how separable zero / non-zero are along that single axis,
     and (ii) whether ablating it kills the zero structure.

  3) Does the multiplicative identity (1×n) get a similar bulge? Or is
     zero uniquely loud? Compares state norms and per-transition update
     norms across {zero pairs, identity pairs (a=1 or b=1, excluding any
     that overlap zero), squares (a=b), general non-trivial}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
STATES = ROOT / "hidden_states.npz"


def main():
    d = np.load(STATES)
    H = d["H"]
    a_all = d["a"]
    b_all = d["b"]
    p_all = d["product"]
    layer = d["layer"]
    n_layers = int(layer.max()) + 1

    # reshape to (100, 6, 128) by pair index a*10+b
    pid = a_all * 10 + b_all
    H_p = np.zeros((100, n_layers, 128), dtype=H.dtype)
    A = np.zeros(100, dtype=int)
    B = np.zeros(100, dtype=int)
    P = np.zeros(100, dtype=int)
    for i in range(len(H)):
        H_p[pid[i], layer[i]] = H[i]
        A[pid[i]] = a_all[i]
        B[pid[i]] = b_all[i]
        P[pid[i]] = p_all[i]

    L = 3  # the peak-bulge layer
    H_L = H_p[:, L, :]  # (100, 128)
    is_zero = P == 0
    Hz = H_L[is_zero]  # (19, 128)

    # --- Probe 1: sub-structure within zero pairs ---
    print("=" * 70)
    print(f"Probe 1: structure within zero-product states at L{L} (n=19)")
    print("=" * 70)

    pos_of_zero = np.array(
        [0 if (A[i] == 0 and B[i] == 0) else (1 if A[i] == 0 else 2) for i in np.where(is_zero)[0]]
    )  # 0=both, 1=a is zero, 2=b is zero
    nonzero_digit = np.array(
        [max(A[i], B[i]) for i in np.where(is_zero)[0]]
    )  # the digit that wasn't zero (0 if both zero)

    print(f"\npos-of-zero distribution: both={np.sum(pos_of_zero==0)}  a-is-zero={np.sum(pos_of_zero==1)}  b-is-zero={np.sum(pos_of_zero==2)}")
    print(f"nonzero-digit distribution: {dict(zip(*np.unique(nonzero_digit, return_counts=True)))}")

    # Use leave-one-out CV since n=19 is small
    from sklearn.model_selection import LeaveOneOut  # type: ignore
    loo = LeaveOneOut()
    for name, y in [("pos-of-zero (3-class)", pos_of_zero), ("non-zero digit (10-class)", nonzero_digit)]:
        n_classes = len(np.unique(y))
        chance = float(np.bincount(y).max()) / len(y)
        try:
            scores = cross_val_score(
                LogisticRegression(max_iter=2000, C=1.0), Hz, y, cv=loo
            )
            acc = scores.mean()
        except Exception as e:
            print(f"  {name}: failed ({e})")
            continue
        print(f"  {name:30s}  LOO acc {acc:.1%}  (chance {chance:.1%}, n_classes={n_classes}, lift {acc-chance:+.1%})")

    # --- Probe 2: single 'zero direction' ---
    print("\n" + "=" * 70)
    print(f"Probe 2: is there a single 'zero direction' at L{L}?")
    print("=" * 70)

    mu_z = H_L[is_zero].mean(axis=0)
    mu_n = H_L[~is_zero].mean(axis=0)
    direction = mu_z - mu_n
    direction = direction / np.linalg.norm(direction)
    proj = H_L @ direction  # (100,)

    z_proj = proj[is_zero]
    n_proj = proj[~is_zero]
    print(f"  projection mean: zero {z_proj.mean():+.3f} ± {z_proj.std():.3f}")
    print(f"  projection mean: nz   {n_proj.mean():+.3f} ± {n_proj.std():.3f}")
    print(f"  d' (separability) = (mu_z - mu_n) / sqrt(0.5*(var_z + var_n)) = {(z_proj.mean()-n_proj.mean())/np.sqrt(0.5*(z_proj.var()+n_proj.var())):.2f}")
    # 1-NN classification along the single axis
    thresh = 0.5 * (z_proj.mean() + n_proj.mean())
    pred_zero = proj > thresh if z_proj.mean() > n_proj.mean() else proj < thresh
    print(f"  threshold classifier acc on single direction: {(pred_zero == is_zero).mean():.1%}")

    # Ablate the direction: project out
    H_L_ablated = H_L - (H_L @ direction)[:, None] * direction[None, :]
    Hz_abl = H_L_ablated[is_zero]

    # Repeat sub-structure probes on ablated states
    print(f"\n  after ablating the 'zero direction', within-zero probes:")
    for name, y in [("pos-of-zero (3-class)", pos_of_zero), ("non-zero digit (10-class)", nonzero_digit)]:
        try:
            scores = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), Hz_abl, y, cv=loo)
            acc = scores.mean()
        except Exception as e:
            print(f"    {name}: failed ({e})")
            continue
        print(f"    {name:30s}  LOO acc {acc:.1%}")

    # Also: can we still detect zero/non-zero from the ABLATED states?
    try:
        scores = cross_val_score(
            LogisticRegression(max_iter=2000, C=1.0), H_L_ablated, is_zero.astype(int), cv=5
        )
        print(f"\n  zero/non-zero detection from ablated states: {scores.mean():.1%}")
    except Exception as e:
        print(f"\n  zero detection on ablated: {e}")

    # --- Probe 3: does 1×n (identity-like) get a similar bulge? ---
    print("\n" + "=" * 70)
    print("Probe 3: zero pairs vs identity pairs vs squares vs general")
    print("=" * 70)

    # Groups (disjoint)
    is_identity = ((A == 1) | (B == 1)) & ~is_zero  # 1×n or n×1, n>0
    is_square = (A == B) & ~is_zero & ~is_identity  # n×n, n>1 (1×1 is identity)
    is_general = ~is_zero & ~is_identity & ~is_square

    print(f"  group sizes: zero={is_zero.sum()}  identity={is_identity.sum()}  squares={is_square.sum()}  general={is_general.sum()}")

    state_norms = np.linalg.norm(H_p, axis=2)  # (100, 6)
    deltas = H_p[:, 1:, :] - H_p[:, :-1, :]
    update_norms = np.linalg.norm(deltas, axis=2)  # (100, 5)

    print(f"\n  Mean ‖h‖ per layer per group:")
    print(f"  {'layer':6s}  {'zero':>7s}  {'identity':>9s}  {'square':>7s}  {'general':>8s}")
    for ll in range(n_layers):
        print(
            f"  L{ll}     "
            f" {state_norms[is_zero, ll].mean():6.2f}   "
            f" {state_norms[is_identity, ll].mean():7.2f}   "
            f" {state_norms[is_square, ll].mean():6.2f}   "
            f" {state_norms[is_general, ll].mean():7.2f}"
        )

    print(f"\n  Mean L{L-1}→L{L} update magnitude per group:")
    t = L - 1
    print(
        f"  zero     {update_norms[is_zero, t].mean():.3f}\n"
        f"  identity {update_norms[is_identity, t].mean():.3f}\n"
        f"  square   {update_norms[is_square, t].mean():.3f}\n"
        f"  general  {update_norms[is_general, t].mean():.3f}"
    )

    # cosine: does identity group share direction with zero group at L3?
    mu_id = H_L[is_identity].mean(axis=0)
    mu_sq = H_L[is_square].mean(axis=0)
    mu_gn = H_L[is_general].mean(axis=0)

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

    # Compare each group's *deviation* from the general mean
    z_dev = mu_z - mu_gn
    id_dev = mu_id - mu_gn
    sq_dev = mu_sq - mu_gn

    print(f"\n  Direction of group bulge at L{L} (cos w.r.t. 'general' mean):")
    print(f"  cos(zero-dev, identity-dev) = {cos(z_dev, id_dev):+.3f}")
    print(f"  cos(zero-dev, square-dev)   = {cos(z_dev, sq_dev):+.3f}")
    print(f"  cos(identity-dev, square-dev) = {cos(id_dev, sq_dev):+.3f}")
    print(f"  ‖zero-dev‖     = {np.linalg.norm(z_dev):.3f}")
    print(f"  ‖identity-dev‖ = {np.linalg.norm(id_dev):.3f}")
    print(f"  ‖square-dev‖   = {np.linalg.norm(sq_dev):.3f}")


if __name__ == "__main__":
    main()
