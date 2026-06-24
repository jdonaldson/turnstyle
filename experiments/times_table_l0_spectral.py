"""Do the L0 =-token cluster's spectral features capture min/max/zero/set?

Hypothesis: at L0 the operand info is encoded as an UNORDERED set (probes
say min/max are 94-96% decodable, position is +10-14pp). Geometrically
this means the cluster is a quotient of the {0..9}x{0..9} grid by the
swap action -> a 55-point triangular sub-lattice. If the cluster
genuinely encodes that quotient, the top Laplacian eigenvectors should
BE min/max (or rotations thereof) and PCA spectrum should look 2D-ish.

Method (per operator: mul, add, sub):
  1. Load (100, 128) =-token states at L0 (one per (a,b) in 0..9^2)
  2. PCA spectrum:    eff70 = smallest k with cum-var >= 0.7
                      anisotropy = lambda_1 / sum(lambda)
  3. Laplacian spectrum:
        kNN graph (k=10), symmetric-normalized Laplacian
        top-5 non-trivial eigenvectors u_1..u_5
        regress each u_i onto candidate features:
            CANDS = a, b, min, max, sum, |diff|, signed_diff,
                    prod, zero_flag, lt10_flag
        report best R^2 and feature per u_i
  4. dyf-style ratios:  e12 = lambda_2 / lambda_1  (cycle if > 0.80)
                        gap2 = lambda_3 / lambda_2 (2D plateau sharpness)

Prediction:
  mul + add  - top-2 Laplacian eigvecs encode min/max (or sum/diff
               rotation, same span); PCA effective dim ~ 2-3
  sub        - top eigvec encodes signed (a-b); 1D-leaning structure
  zero pairs - identifiable as a substructure (high anisotropy slice)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import laplacian  # type: ignore
from sklearn.neighbors import kneighbors_graph  # type: ignore

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
FILES = {
    "mul": (ROOT / "hidden_states.npz", "product"),
    "add": (ROOT / "hidden_states_add.npz", "sum"),
    "sub": (ROOT / "hidden_states_sub.npz", "diff"),
}
LAYER = 0
K_NN = 10
N_EIG = 6  # u_0 trivial + 5 informative
N_PCA = 10


def load_layer(path: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    m = d["layer"] == layer
    return d["H"][m], d["a"][m], d["b"][m]


def candidates(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "a": a.astype(float),
        "b": b.astype(float),
        "min": np.minimum(a, b).astype(float),
        "max": np.maximum(a, b).astype(float),
        "sum": (a + b).astype(float),
        "abs_diff": np.abs(a - b).astype(float),
        "signed_diff": (a - b).astype(float),
        "prod": (a * b).astype(float),
        "zero_flag": ((a == 0) | (b == 0)).astype(float),
        "lt10_flag": (a * b < 10).astype(float),
    }


def r2(y: np.ndarray, x: np.ndarray) -> float:
    """R^2 of y predicted from x via OLS with intercept."""
    x = x - x.mean()
    y = y - y.mean()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    beta = (x @ y) / (x @ x)
    y_hat = beta * x
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = (y * y).sum()
    return 1.0 - ss_res / ss_tot


def pca_spectrum(H: np.ndarray) -> tuple[np.ndarray, float, int]:
    Hc = H - H.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Hc, compute_uv=False)
    lam = s**2
    lam_norm = lam / lam.sum()
    anisotropy = float(lam_norm[0])
    cum = np.cumsum(lam_norm)
    eff70 = int(np.searchsorted(cum, 0.70) + 1)
    return lam_norm[:N_PCA], anisotropy, eff70


def laplacian_eigs(H: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric-normalized Laplacian on kNN(k) graph; smallest N_EIG eigs."""
    A = kneighbors_graph(H, n_neighbors=k, mode="connectivity",
                         include_self=False)
    A = (A + A.T).maximum(A.T)  # symmetrize
    L = laplacian(A, normed=True)
    L_dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
    w, v = np.linalg.eigh(L_dense)
    return w[:N_EIG], v[:, :N_EIG]


def main():
    print(f"{'op':>4}  {'eff70':>6} {'anis':>6} | "
          f"{'lam1':>5} {'lam2':>5} {'lam3':>5} {'lam4':>5} {'lam5':>5} | "
          f"{'e12':>5} {'gap2':>5}")
    print("-" * 78)
    spectra = {}
    eigs = {}
    feats = {}
    for op, (path, _) in FILES.items():
        H, a, b = load_layer(path, LAYER)
        lam, ani, eff = pca_spectrum(H)
        w, v = laplacian_eigs(H, K_NN)
        # w[0] ~ 0 (trivial); use w[1:] for ratios
        w_nz = w[1:]
        e12 = w_nz[1] / w_nz[0] if w_nz[0] > 1e-12 else float("nan")
        gap2 = w_nz[2] / w_nz[1] if w_nz[1] > 1e-12 else float("nan")
        print(f"{op:>4}  {eff:>6d} {ani:>6.3f} | "
              + " ".join(f"{x:>5.3f}" for x in lam[:5])
              + f" | {e12:>5.2f} {gap2:>5.2f}")
        spectra[op] = (lam, ani, eff)
        eigs[op] = (w, v)
        feats[op] = candidates(a, b)

    print("\nPCA scree (first 10 components, fraction-of-variance):")
    for op in FILES:
        lam = spectra[op][0]
        print(f"  {op}: " + " ".join(f"{x:.3f}" for x in lam))

    print("\nLaplacian eigenvalues (smallest 6, ~0 first):")
    for op in FILES:
        w = eigs[op][0]
        print(f"  {op}: " + " ".join(f"{x:.4f}" for x in w))

    # Regress each non-trivial Laplacian eigenvector onto each candidate.
    print("\nLaplacian eigvec -> candidate feature R^2 (best per eigvec):")
    cand_names = list(candidates(np.array([0]), np.array([0])).keys())
    for op in FILES:
        w, v = eigs[op]
        f = feats[op]
        print(f"\n  [{op}] (u_0 trivial; reporting u_1..u_5)")
        header = "    eigvec  " + " ".join(f"{n:>11s}" for n in cand_names) + "   best"
        print(header)
        for i in range(1, N_EIG):
            row = []
            scores = {}
            for name in cand_names:
                s = r2(v[:, i], f[name])
                scores[name] = s
                row.append(f"{s:>11.3f}")
            best = max(scores.items(), key=lambda kv: kv[1])
            print(f"    u_{i}     " + " ".join(row)
                  + f"   {best[0]}={best[1]:.2f}")

    # Joint two-eigvec span: do (u_1, u_2) jointly recover (min, max)?
    print("\nJoint span (u_1, u_2) R^2 vs feature pairs:")
    pairs = [("min", "max"), ("sum", "abs_diff"), ("sum", "signed_diff"),
             ("a", "b")]
    for op in FILES:
        w, v = eigs[op]
        f = feats[op]
        U = v[:, 1:3]  # (n, 2)
        Uc = U - U.mean(axis=0, keepdims=True)
        for fa, fb in pairs:
            Y = np.stack([f[fa], f[fb]], axis=1)
            Yc = Y - Y.mean(axis=0, keepdims=True)
            beta, *_ = np.linalg.lstsq(Uc, Yc, rcond=None)
            Y_hat = Uc @ beta
            ss_res = ((Yc - Y_hat) ** 2).sum()
            ss_tot = (Yc**2).sum()
            r2_joint = 1.0 - ss_res / ss_tot
            print(f"  {op}  (u1,u2) -> ({fa},{fb}): R^2 = {r2_joint:.3f}")

    # Zero-pair substructure: anisotropy restricted to zero-pair rows.
    print("\nZero-pair substructure (anisotropy of zero-flag subset):")
    for op, (path, _) in FILES.items():
        H, a, b = load_layer(path, LAYER)
        zmask = (a == 0) | (b == 0)
        Hz = H[zmask]
        _, ani_z, eff_z = pca_spectrum(Hz)
        Hnz = H[~zmask]
        _, ani_nz, eff_nz = pca_spectrum(Hnz)
        print(f"  {op}  zero (n={zmask.sum()}): anis={ani_z:.3f} eff70={eff_z} "
              f"| nonzero (n={(~zmask).sum()}): anis={ani_nz:.3f} eff70={eff_nz}")


if __name__ == "__main__":
    main()
