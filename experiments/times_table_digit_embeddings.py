"""Are digit embeddings log-spaced?

Hypothesis: the token embeddings (wte rows) for '0'..'9' lie on a curve
where arc length from digit 0 correlates with log(d+1), not with d.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import STOI, load  # noqa: E402


def main():
    model = load("cpu")
    W = model.wte.weight.detach().numpy()
    digit_ids = [STOI[str(d)] for d in range(10)]
    W_d = W[digit_ids]
    print(f"Digit embedding shape: {W_d.shape}")
    print(f"Mean norm: {np.linalg.norm(W_d, axis=1).mean():.3f}")

    pca = PCA(n_components=3)
    proj = pca.fit_transform(W_d)
    print(f"PCA explained variance ratio: "
          f"{[f'{v:.3f}' for v in pca.explained_variance_ratio_]}")

    print("\nDigit positions in PCA space:")
    print(f"{'digit':>5}  {'PC1':>8}  {'PC2':>8}  {'PC3':>8}")
    for d in range(10):
        print(f"  {d}     {proj[d, 0]:8.3f}  {proj[d, 1]:8.3f}  "
              f"{proj[d, 2]:8.3f}")

    # Arc length walking digits in numeric order 0,1,...,9
    arc = [0.0]
    for d in range(1, 10):
        arc.append(arc[-1] + float(np.linalg.norm(proj[d] - proj[d - 1])))
    arc = np.array(arc)
    print("\nArc length walking 0→1→...→9 in PCA-3 space:")
    print(f"{'digit':>5}  {'arc':>8}  {'log(d+1)':>10}  {'d':>5}")
    for d in range(10):
        print(f"  {d}     {arc[d]:8.3f}  {np.log(d+1):10.3f}  {d:5d}")

    d_vals = np.arange(10)
    r_linear = np.corrcoef(arc, d_vals)[0, 1]
    r_log = np.corrcoef(arc, np.log(d_vals + 1))[0, 1]
    print(f"\nPearson correlation:")
    print(f"  arc-length vs d (linear):   r = {r_linear:.3f}")
    print(f"  arc-length vs log(d+1):     r = {r_log:.3f}")

    # Also: try total pairwise distance.  If embeddings lie on a 1D log curve,
    # then dist(d_i, d_j) should scale with |log(i+1) - log(j+1)|.
    dist_matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            dist_matrix[i, j] = float(np.linalg.norm(W_d[i] - W_d[j]))

    log_dist = np.zeros((10, 10))
    lin_dist = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            log_dist[i, j] = abs(np.log(i + 1) - np.log(j + 1))
            lin_dist[i, j] = abs(i - j)

    iu = np.triu_indices(10, k=1)
    r_emb_d = float(np.corrcoef(dist_matrix[iu], lin_dist[iu])[0, 1])
    r_emb_logd = float(np.corrcoef(dist_matrix[iu], log_dist[iu])[0, 1])
    print(f"\nPairwise full-128d embedding distance correlations:")
    print(f"  emb_dist vs |i-j| (linear):              r = {r_emb_d:.3f}")
    print(f"  emb_dist vs |log(i+1)-log(j+1)|:         r = {r_emb_logd:.3f}")

    print("\nThe stronger correlation tells us which scale the embedding is on.")


if __name__ == "__main__":
    main()
