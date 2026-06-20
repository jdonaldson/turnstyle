"""Visualize the semantic frame: raw activations (surface-organized) vs frame
coordinates (meaning-organized). The contrast IS the finding.

Fits bipolar axes from the cached ENGLISH adjective activations (cross_language,
no model run), projects all four languages into the semantic-coordinate space,
and UMAPs both the raw 2048-d activations and the 9-d semantic coordinates —
each colored by language and by axis. Surface suppression (drop top-3 PCs) is
applied to the axis directions.

Output: a 2x3 PNG (raw|coords × language|axis, + a named-axis scatter).
Usage:  python experiments/viz_semantic_frame.py [--layer L]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp"))
sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import cross_language as CL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap


def fit_coords(acts, lang, axis, pole, surface_suppress=3):
    """English-fit bipolar axes (shared standardization + surface suppression),
    return (coords[N, n_axes], axis_names, directions, centers)."""
    en = lang == "en"
    mean = acts[en].mean(0)
    scale = acts[en].std(0) + 1e-6
    Z = (acts - mean) / scale
    names = sorted(set(axis.tolist()))
    raw = []
    for ax in names:
        m = en & (axis == ax)
        hi = Z[m & (pole == 1)].mean(0)
        lo = Z[m & (pole == -1)].mean(0)
        raw.append(hi - lo)
    raw = np.vstack(raw)
    if surface_suppress:
        zc = Z[en] - Z[en].mean(0)
        _, _, Vt = np.linalg.svd(zc, full_matrices=False)
        surf = Vt[:surface_suppress]
        raw = raw - (raw @ surf.T) @ surf
    dirs = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12)
    centers = []
    coords = np.zeros((len(acts), len(names)))
    for i, ax in enumerate(names):
        m = en & (axis == ax)
        hi = Z[m & (pole == 1)].mean(0)
        lo = Z[m & (pole == -1)].mean(0)
        c = 0.5 * (hi + lo) @ dirs[i]
        centers.append(c)
        coords[:, i] = Z @ dirs[i] - c
    return coords, names, dirs, np.array(centers)


def _scatter(ax, xy, labels, title, order=None):
    cats = order or sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
    for i, c in enumerate(cats):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=14, color=cmap(i), label=str(c), alpha=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6, markerscale=1.2, loc="best", framealpha=0.6)


def main(layer):
    d = np.load(CL.CACHE, allow_pickle=True)
    acts = d["acts"].astype(np.float32)[:, layer, :]
    lang = d["langs"]; axis = d["axes"]; pole = (d["poles"] == 1).astype(int) * 2 - 1

    coords, names, _, _ = fit_coords(acts, lang, axis, pole)

    reducer = lambda X: umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X)
    raw2d = reducer((acts - acts.mean(0)) / (acts.std(0) + 1e-6))
    sem2d = reducer((coords - coords.mean(0)) / (coords.std(0) + 1e-6))

    fig, axx = plt.subplots(2, 3, figsize=(15, 9))
    _scatter(axx[0, 0], raw2d, lang, f"RAW activations @L{layer} — by language")
    _scatter(axx[0, 1], raw2d, axis, "RAW activations — by axis")
    # named-axis scatter: the two most-populated axes, by language
    pop = sorted(names, key=lambda a: -(axis == a).sum())[:2]
    ai, aj = names.index(pop[0]), names.index(pop[1])
    _scatter(axx[0, 2], coords[:, [ai, aj]], lang,
             f"SEMANTIC coords: {pop[0]} vs {pop[1]} — by language")
    axx[0, 2].axhline(0, color="gray", lw=0.5); axx[0, 2].axvline(0, color="gray", lw=0.5)
    axx[0, 2].set_xticks([]); axx[0, 2].set_yticks([])
    _scatter(axx[1, 0], sem2d, lang, "SEMANTIC coords (UMAP) — by language")
    _scatter(axx[1, 1], sem2d, axis, "SEMANTIC coords (UMAP) — by axis")
    # pole structure on the same UMAP
    hi = pole == 1
    axx[1, 2].scatter(sem2d[hi, 0], sem2d[hi, 1], s=14, marker="^", color="C3",
                      label="high pole", alpha=0.8)
    axx[1, 2].scatter(sem2d[~hi, 0], sem2d[~hi, 1], s=14, marker="v", color="C0",
                      label="low pole", alpha=0.8)
    axx[1, 2].set_title("SEMANTIC coords — by pole", fontsize=10)
    axx[1, 2].set_xticks([]); axx[1, 2].set_yticks([])
    axx[1, 2].legend(fontsize=7)

    fig.suptitle("Semantic frame: raw activations are surface-organized; "
                 "frame coordinates are meaning-organized (English-fit, projects es/fr/de)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "semantic_frame_viz.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=11)
    args = ap.parse_args()
    main(args.layer)
