"""Resolve the orthogonal-vs-transfer tension in SemanticFrame axis fitting.

Non-orthogonal contrastive axes transfer cross-lingually but are correlated (weak
joint clustering); QR-orthogonal axes cluster cleanly but lose transfer. Hypothesis
(from concept_geometry): surface/language is the DOMINANT variance, so the right move
is to suppress the surface subspace before fitting — that should decorrelate the axes
AND improve transfer (a word from any language then projects consistently).

Head-to-head, fully from the cached cross_language activations (pole-labeled,
EN/ES/FR/DE, all layers — no model run). Two metrics per method:
  transfer  — fit axes on ENGLISH, project ES/FR/DE words, fraction signed correctly
  cluster   — axis-NMI vs language-NMI of KMeans on the projected coordinates
The winner maximizes BOTH (high transfer, axis-NMI > language-NMI).

Methods: raw contrastive | QR-orthogonal | drop-top-K-PC | per-language-center |
within-axis-whiten (LDA-style).

Usage:  python experiments/semantic_axis_fit.py [--layer L]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import cross_language as CL
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi


def _load(layer):
    d = np.load(CL.CACHE, allow_pickle=True)
    return (d["acts"].astype(np.float32)[:, layer, :],
            (d["poles"] == 1).astype(int), d["langs"], d["axes"])


def _axis_dirs(Xtr, pole_tr, axis_tr, axes):
    """Per-axis contrastive direction (high−low) in the given (preprocessed) space."""
    dirs, centers = {}, {}
    for ax in axes:
        m = axis_tr == ax
        hi = Xtr[m & (pole_tr == 1)].mean(0)
        lo = Xtr[m & (pole_tr == 0)].mean(0)
        d = hi - lo
        d = d / (np.linalg.norm(d) + 1e-12)
        dirs[ax] = d
        centers[ax] = 0.5 * (hi + lo) @ d
    return dirs, centers


def _metrics(X, pole, lang, axis, axes, prep):
    """prep(Xtrain)→(transform fn, fitted on EN). Fit axes on EN, eval all langs."""
    en = lang == "en"
    transform, = (prep(X[en]),)            # returns a function X→X'
    Xt = transform(X)
    # standardize in transformed space (fit on EN)
    mu = Xt[en].mean(0); sd = Xt[en].std(0) + 1e-6
    Z = (Xt - mu) / sd
    dirs, centers = _axis_dirs(Z[en], pole[en], axis[en], axes)
    # transfer: non-EN words, project on own axis, sign == pole?
    non = ~en
    correct = 0
    for i in np.where(non)[0]:
        s = Z[i] @ dirs[axis[i]] - centers[axis[i]]
        correct += int((s > 0) == (pole[i] == 1))
    transfer = correct / non.sum()
    # cluster: coordinate matrix over all axes
    coords = np.vstack([[Z[i] @ dirs[a] - centers[a] for a in axes]
                        for i in range(len(Z))])
    cz = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    lab = KMeans(len(axes), n_init=10, random_state=0).fit_predict(cz)
    return transfer, nmi(axis, lab), nmi(lang, lab)


# ── preprocessing strategies (each returns a transform fitted on EN acts) ─────

def prep_raw(Xen):
    return lambda X: X


def prep_qr(Xen):
    # handled specially (orthogonalizes the DIRECTIONS, not the space) — see runner
    return lambda X: X


def prep_drop_pc(k):
    def make(Xen):
        mu = Xen.mean(0)
        _, _, Vt = np.linalg.svd(Xen - mu, full_matrices=False)
        drop = Vt[:k]                       # top-k surface directions
        P = np.eye(Xen.shape[1]) - drop.T @ drop
        return lambda X: (X - mu) @ P
    return make


def prep_lang_center(langs):
    def make(Xen):
        return lambda X: X                  # centering applied in runner (needs labels)
    return make


def prep_whiten(Xen):
    # whiten by within-EN covariance (suppress the dominant shared variance)
    mu = Xen.mean(0)
    cov = np.cov((Xen - mu).T) + 1e-2 * np.eye(Xen.shape[1])
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
    return lambda X: (X - mu) @ W


def run(layer):
    X, pole, lang, axis = _load(layer)
    axes = sorted(set(axis.tolist()))
    print(f"L{layer}  N={len(X)}  axes={len(axes)}  langs={sorted(set(lang.tolist()))}")
    print(f"{'method':18s} {'transfer':>9} {'axis-NMI':>9} {'lang-NMI':>9}  {'verdict':>10}")

    def show(name, transfer, anmi, lnmi):
        v = "MEANING" if anmi > lnmi else "surface"
        print(f"{name:18s} {transfer:>9.3f} {anmi:>9.3f} {lnmi:>9.3f}  {v:>10}")

    # raw
    show("raw_contrastive", *_metrics(X, pole, lang, axis, axes, prep_raw))
    # qr: fit raw dirs then orthonormalize
    show("qr_orthogonal", *_qr_metrics(X, pole, lang, axis, axes))
    # drop top-K PCs
    for k in (1, 3, 10, 30):
        show(f"drop_top{k}PC", *_metrics(X, pole, lang, axis, axes, prep_drop_pc(k)))
    # per-language centering
    show("lang_center", *_langcenter_metrics(X, pole, lang, axis, axes))
    # within-EN whitening
    show("whiten", *_metrics(X, pole, lang, axis, axes, prep_whiten))


def _qr_metrics(X, pole, lang, axis, axes):
    en = lang == "en"
    mu = X[en].mean(0); sd = X[en].std(0) + 1e-6
    Z = (X - mu) / sd
    raw, _ = _axis_dirs(Z[en], pole[en], axis[en], axes)
    D = np.vstack([raw[a] for a in axes])
    Q, _ = np.linalg.qr(D.T)
    O = Q.T[:len(axes)]
    for i, a in enumerate(axes):
        if raw[a] @ O[i] < 0:
            O[i] = -O[i]
    dirs = {a: O[i] for i, a in enumerate(axes)}
    centers = {a: 0.5 * (Z[en][(axis[en] == a) & (pole[en] == 1)].mean(0)
                         + Z[en][(axis[en] == a) & (pole[en] == 0)].mean(0)) @ dirs[a]
               for a in axes}
    non = ~en
    correct = sum(int((Z[i] @ dirs[axis[i]] - centers[axis[i]] > 0) == (pole[i] == 1))
                  for i in np.where(non)[0])
    coords = np.vstack([[Z[i] @ dirs[a] - centers[a] for a in axes] for i in range(len(Z))])
    cz = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    lab = KMeans(len(axes), n_init=10, random_state=0).fit_predict(cz)
    return correct / non.sum(), nmi(axis, lab), nmi(lang, lab)


def _langcenter_metrics(X, pole, lang, axis, axes):
    Xc = X.copy().astype(float)
    for lg in set(lang.tolist()):
        m = lang == lg
        Xc[m] = X[m] - X[m].mean(0)
    return _metrics(Xc, pole, lang, axis, axes, prep_raw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=11)
    args = ap.parse_args()
    run(args.layer)
