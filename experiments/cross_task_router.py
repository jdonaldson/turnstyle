#!/usr/bin/env python3
"""Cross-task routing — Q2 first step.

Tests whether SmolLM2 hidden states (L8..L23) cleanly separate the 5 cached
BBH task types. If yes, the activation-routing premise (Level 3 of
``docs/composition_and_activation_routing.md``) holds on this model and we
should expand to novel prompts. If no, the premise is wrong and we need to
reconsider.

Reuses the cache from capability_probe.py — every example already has hidden
states at L8/12/16/18/20/22/L23 via last-token pooling. No model loading,
no generation.

Comparisons per layer:
  - LogReg (5-fold CV)         : linear separability ceiling
  - dyf-leaf majority routing  : how cleanly tasks cluster in PCA-LSH leaves
  - random baseline            : 1 / num_tasks

Tasks (all from cache):
  - penguins_in_a_table              → table_reasoning
  - tracking_shuffled_objects_three  → state_tracking
  - object_counting                  → counting
  - navigate                         → navigation
  - web_of_lies                      → boolean_propagation

Run:
    /Users/jdonaldson/Projects/dyf/.venv/bin/python \\
        experiments/cross_task_router.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dyf import build_dyf_tree, cut_tree_to_labels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

EXPERIMENT_DIR = Path(__file__).parent
CACHE_PATH = EXPERIMENT_DIR / "capability_probe_data.npz"

TASKS = [
    "penguins_in_a_table",
    "tracking_shuffled_objects_three_objects",
    "object_counting",
    "navigate",
    "web_of_lies",
]
SHORT_NAMES = {
    "penguins_in_a_table": "penguins",
    "tracking_shuffled_objects_three_objects": "tracking",
    "object_counting": "counting",
    "navigate": "navigate",
    "web_of_lies": "wol",
}

LAYER_SWEEP = ["l8", "l12", "l16", "l18", "l20", "l22", "l23"]
K_SWEEP = [8, 16, 32, 64]   # cluster counts for dyf-leaf
TREE_MAX_DEPTH = 6
TREE_NUM_BITS = 4
TREE_MIN_LEAF = 4
N_FOLDS = 5
RNG_SEED = 0


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_all_tasks(cache: np.lib.npyio.NpzFile, layer: str):
    """Stack hidden states from every task; return (X, y, task_names)."""
    X_list = []
    y_list = []
    for ti, task in enumerate(TASKS):
        H = cache[f"{task}__hidden_{layer}"].astype(np.float32)
        X_list.append(H)
        y_list.append(np.full(H.shape[0], ti, dtype=np.int64))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


# ──────────────────────────────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────────────────────────────

def logreg_cv_acc(X, y, n_folds=N_FOLDS):
    """5-fold stratified CV with multinomial LogReg."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        accs.append(clf.score(X[test_idx], y[test_idx]))
    return float(np.mean(accs)), float(np.std(accs))


def dyf_leaf_majority_acc(X, y, n_clusters):
    """Build dyf tree, route by majority task in leaf (LOO within leaf)."""
    N = X.shape[0]
    tree = build_dyf_tree(
        X,
        max_depth=TREE_MAX_DEPTH,
        num_bits=TREE_NUM_BITS,
        min_leaf_size=TREE_MIN_LEAF,
    )
    leaf_labels = cut_tree_to_labels(
        tree, n_points=N, n_clusters=n_clusters, embeddings=X,
    )
    K = int(leaf_labels.max()) + 1
    n_classes = int(y.max()) + 1

    # Per-leaf class counts
    leaf_counts = np.zeros((K, n_classes), dtype=np.int64)
    for i in range(N):
        leaf_counts[leaf_labels[i], y[i]] += 1
    leaf_size = leaf_counts.sum(axis=1)

    # LOO within leaf: subtract own class from leaf counts, pick argmax
    correct = 0
    for i in range(N):
        L = leaf_labels[i]
        if leaf_size[L] <= 1:
            # singleton leaf — fall back to global majority
            pred = int(np.bincount(y, minlength=n_classes).argmax())
        else:
            counts = leaf_counts[L].copy()
            counts[y[i]] -= 1
            pred = int(counts.argmax())
        if pred == y[i]:
            correct += 1
    return correct / N, K


def class_purity(X, y, n_clusters):
    """Mean per-leaf max-class fraction (cluster purity ignoring LOO)."""
    N = X.shape[0]
    tree = build_dyf_tree(
        X, max_depth=TREE_MAX_DEPTH, num_bits=TREE_NUM_BITS, min_leaf_size=TREE_MIN_LEAF,
    )
    leaf_labels = cut_tree_to_labels(
        tree, n_points=N, n_clusters=n_clusters, embeddings=X,
    )
    K = int(leaf_labels.max()) + 1
    n_classes = int(y.max()) + 1
    leaf_counts = np.zeros((K, n_classes), dtype=np.int64)
    for i in range(N):
        leaf_counts[leaf_labels[i], y[i]] += 1
    leaf_size = leaf_counts.sum(axis=1)
    nonempty = leaf_size > 0
    purity = (leaf_counts.max(axis=1)[nonempty] / leaf_size[nonempty]).mean()
    return float(purity), int(nonempty.sum())


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────

def fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main():
    print(f"Loading cache: {CACHE_PATH}")
    cache = np.load(CACHE_PATH, allow_pickle=True)
    print(f"  {len(cache.files)} keys")
    print(f"  tasks: {len(TASKS)}")
    print(f"  layers: {LAYER_SWEEP}")
    print()

    # Sanity check: per-task counts
    print("Task counts:")
    for task in TASKS:
        n = cache[f"{task}__hidden_l18"].shape[0]
        print(f"  {SHORT_NAMES[task]:>9}  N={n}")
    total = sum(cache[f"{t}__hidden_l18"].shape[0] for t in TASKS)
    n_classes = len(TASKS)
    print(f"  total N = {total}, baseline (uniform) = {100/n_classes:.1f}%")
    print()

    # ── Layer × method sweep ─────────────────────────────────────────
    print(f"=== Layer sweep — multi-class task prediction ({n_classes} classes) ===")
    header = f"  {'layer':>5}  {'logreg':>9}  " + "  ".join(f"dyf-K={K:>2d}" for K in K_SWEEP)
    print(header)

    rows = []  # (layer, lr_acc, lr_std, {K: dyf_acc})
    for layer in LAYER_SWEEP:
        X, y = load_all_tasks(cache, layer)
        lr_mean, lr_std = logreg_cv_acc(X, y)
        dyf_accs = {}
        for K in K_SWEEP:
            acc, _ = dyf_leaf_majority_acc(X, y, n_clusters=K)
            dyf_accs[K] = acc
        rows.append((layer, lr_mean, lr_std, dyf_accs))
        cells = "  ".join(f"{fmt_pct(dyf_accs[K]):>7}" for K in K_SWEEP)
        print(f"  {layer.upper():>5}  {fmt_pct(lr_mean):>5}±{lr_std*100:.1f}  {cells}")

    # ── Best layer details ───────────────────────────────────────────
    best_layer = max(rows, key=lambda r: r[1])[0]
    print()
    print(f"=== Best layer: {best_layer.upper()} ===")
    X, y = load_all_tasks(cache, best_layer)

    # Confusion matrix from a single CV split (just for visualization)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    y_pred_all = np.zeros_like(y)
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        y_pred_all[test_idx] = clf.predict(X[test_idx])
    cm = confusion_matrix(y, y_pred_all)
    print("LogReg 5-fold CV confusion matrix (rows=true, cols=pred):")
    short = [SHORT_NAMES[t] for t in TASKS]
    print(f"  {'':>10}  " + "  ".join(f"{s:>9}" for s in short))
    for i, name in enumerate(short):
        cells = "  ".join(f"{cm[i, j]:>9d}" for j in range(len(short)))
        n_total = cm[i].sum()
        acc_i = cm[i, i] / max(n_total, 1)
        print(f"  {name:>10}  " + cells + f"   ({100*acc_i:5.1f}% recall)")

    # Per-class precision & recall
    print()
    print("Per-class:")
    for i, name in enumerate(short):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"  {name:>10}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    # ── dyf-leaf purity for the best layer ──────────────────────────
    print()
    print(f"=== dyf-leaf class purity at {best_layer.upper()} ===")
    print(f"  {'K':>4}  {'leaves':>7}  {'purity':>7}  {'majority-acc':>13}")
    for K in K_SWEEP:
        purity, n_leaves = class_purity(X, y, n_clusters=K)
        acc, _ = dyf_leaf_majority_acc(X, y, n_clusters=K)
        print(f"  {K:>4d}  {n_leaves:>7d}  {fmt_pct(purity):>7}  {fmt_pct(acc):>13}")


if __name__ == "__main__":
    main()
