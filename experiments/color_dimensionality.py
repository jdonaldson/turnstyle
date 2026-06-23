"""How many dimensions does COLOR occupy in SmolLM2?

Perceptually color is 3D (CIELAB L*,a*,b*). This asks how many of those the model
encodes as *separate* axes, three ways:
  1. Reference: participation ratio (effective dim) of the 104-color Lab cloud — how
     many perceptual dims my color SAMPLE actually spans (a,b can be correlated).
  2. Held-out CCA(color activations, Lab): up to 3 canonical correlations, 5-fold,
     vs a label-permutation null. #(components above null) = #color dims encoded.
  3. Participation ratio of the color-activation cloud (descriptive; includes
     non-color lexical variance, so an upper bound on "color dims").
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")
from epa_external_validation import collect_acts          # reuse rig
from color_affect_frame import COLORS, _hex_to_lab        # reuse lexicon + Lab


def participation_ratio(X):
    """Effective dimensionality: (Σλ)² / Σλ² of the covariance eigenvalues."""
    Xc = X - X.mean(0)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def cca_canon_corrs(X, Y, n_comp=3, k_pca=15, seed=0):
    """Held-out 5-fold canonical correlations between X and Y (Y is 3D Lab)."""
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_corrs = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=min(k_pca, len(tr) - 1)).fit(sc.transform(X[tr]))
        Xtr = pca.transform(sc.transform(X[tr]))
        Xte = pca.transform(sc.transform(X[te]))
        cca = CCA(n_components=n_comp, max_iter=2000).fit(Xtr, Y[tr])
        Xc, Yc = cca.transform(Xte, Y[te])
        corrs = []
        for j in range(n_comp):
            c = np.corrcoef(Xc[:, j], Yc[:, j])[0, 1]
            corrs.append(abs(c) if np.isfinite(c) else 0.0)
        fold_corrs.append(corrs)
    return np.mean(fold_corrs, 0)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    cw = [c for c in COLORS if c not in ("fuchsia", "aqua")]
    LAB = np.array([_hex_to_lab(COLORS[w]) for w in cw])
    print(f"colors={len(cw)}", flush=True)

    # (1) reference: effective dim of the color SAMPLE's perceptual geometry
    LABz = (LAB - LAB.mean(0)) / LAB.std(0)
    print(f"\nReference — participation ratio of Lab cloud (z-scored): "
          f"{participation_ratio(LABz):.2f} / 3.00")
    R = np.corrcoef(LAB.T)
    print(f"  Lab channel correlations: L-a={R[0,1]:+.2f} L-b={R[0,2]:+.2f} a-b={R[1,2]:+.2f}")

    acts = collect_acts(mdl, tok, dev, cw)
    n_layers = acts[cw[0]].shape[0]

    # (2) held-out CCA(activations, Lab) per layer + permutation null
    print("\n=== held-out CCA canonical correlations (color acts vs Lab) ===")
    print("  layer    cc1    cc2    cc3   | null(cc1 cc2 cc3)   PR(acts)")
    rng = np.random.default_rng(0)
    for layer in range(n_layers):
        X = np.array([acts[w][layer] for w in cw])
        cc = cca_canon_corrs(X, LAB)
        # permutation null: shuffle Lab rows, average a few shuffles
        nulls = []
        for s in range(3):
            perm = rng.permutation(len(cw))
            nulls.append(cca_canon_corrs(X, LAB[perm], seed=s))
        null = np.mean(nulls, 0)
        pr = participation_ratio(X)
        print(f"  L{layer:<2d}   {cc[0]:.3f}  {cc[1]:.3f}  {cc[2]:.3f}  | "
              f"{null[0]:.2f}  {null[1]:.2f}  {null[2]:.2f}      {pr:5.1f}")


if __name__ == "__main__":
    main()
