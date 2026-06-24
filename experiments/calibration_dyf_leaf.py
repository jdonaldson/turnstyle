#!/usr/bin/env python3
"""dyf-leaf as confidence calibrator (Q3 / Proposal A).

For each (task, layer, K):
  - Build a dyf PCA-LSH tree on hidden states; cut to K leaves
  - LOO leaf-mate accuracy: leaf_conf[i] = mean(baseline_correct[j])
    over j != i in the same leaf
  - kNN baseline (cheapest ablation): same number of LOO neighbors
    chosen by Euclidean kNN in raw hidden-state space, no tree
  - Random baseline: constant = baseline accuracy

Ask: does leaf_conf predict baseline_correct better than knn_conf?
A floor-raising result is a small lift across many (task, layer) cells,
with the same lift direction across tasks.

Metrics:
  - Spearman(conf, baseline_correct) — calibration signal strength
  - AUC of accuracy@coverage curve — value if used as abstention rule
  - Single-threshold portability: does threshold τ chosen on task A
    transfer to task B?

Run with the dyf venv:
    /Users/jdonaldson/Projects/dyf/.venv/bin/python \
        experiments/calibration_dyf_leaf.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from dyf import build_dyf_tree, cut_tree_to_labels


EXPERIMENT_DIR = Path(__file__).parent
CACHE_PATH = EXPERIMENT_DIR / "capability_probe_data.npz"

TASKS = [
    "penguins_in_a_table",
    "tracking_shuffled_objects_three_objects",
    "object_counting",
    "navigate",
    "web_of_lies",
]
LAYER_SWEEP = ["l8", "l12", "l16", "l18", "l20", "l22", "l23"]
K_SWEEP = [8, 12]
TREE_MAX_DEPTH = 4
TREE_NUM_BITS = 3
TREE_MIN_LEAF = 4


def load_task(cache, task: str, layer: str) -> dict:
    H = cache[f"{task}__hidden_{layer}"].astype(np.float32)
    y = cache[f"{task}__baseline__correct"].astype(np.int8)
    return {"H": H, "y": y, "N": H.shape[0]}


def leaf_conf_loo(leaf_labels: np.ndarray, y: np.ndarray, global_mean: float) -> np.ndarray:
    """LOO leaf-mate accuracy. Tiny leaves fall back to global mean."""
    N = len(y)
    K = int(leaf_labels.max()) + 1
    size = np.bincount(leaf_labels, minlength=K).astype(np.float64)
    sum_y = np.zeros(K, dtype=np.float64)
    for i in range(N):
        sum_y[leaf_labels[i]] += y[i]
    conf = np.empty(N, dtype=np.float64)
    for i in range(N):
        L = leaf_labels[i]
        n_others = size[L] - 1
        if n_others <= 0:
            conf[i] = global_mean
        else:
            conf[i] = (sum_y[L] - y[i]) / n_others
    return conf


def knn_conf_loo(H: np.ndarray, y: np.ndarray, leaf_labels: np.ndarray) -> np.ndarray:
    """kNN baseline using the same per-query neighborhood SIZE as the leaf,
    but neighbors picked by Euclidean distance in raw H, not by tree.

    Same number of LOO neighbors as leaf_conf_loo, so directly comparable.
    """
    N = H.shape[0]
    K = int(leaf_labels.max()) + 1
    leaf_size = np.bincount(leaf_labels, minlength=K).astype(np.int64)
    # Pairwise squared distances (N × N)
    sq = ((H * H).sum(axis=1, keepdims=True)
          + (H * H).sum(axis=1, keepdims=True).T
          - 2 * H @ H.T)
    np.fill_diagonal(sq, np.inf)  # exclude self
    conf = np.empty(N, dtype=np.float64)
    global_mean = float(y.mean())
    for i in range(N):
        k = int(leaf_size[leaf_labels[i]]) - 1
        if k <= 0:
            conf[i] = global_mean
            continue
        idx = np.argpartition(sq[i], k - 1)[:k]
        conf[i] = float(y[idx].mean())
    return conf


def acc_at_coverage_auc(conf: np.ndarray, y: np.ndarray) -> float:
    """Area under the accuracy@coverage curve.

    Sort examples by conf descending. Retain top fraction c (c ∈ (0,1]).
    Accuracy@coverage = mean(y[retained]). AUC = mean over c grid.

    Random baseline (constant accuracy) gives AUC = mean(y).
    Perfect oracle gives AUC = 1.0 if any y == 1 (sort all correct first).
    """
    order = np.argsort(-conf)  # high conf first
    y_sorted = y[order].astype(np.float64)
    cum_correct = np.cumsum(y_sorted)
    n = len(y)
    # accuracy at coverage k/n is cum_correct[k-1] / k
    accs = cum_correct / np.arange(1, n + 1)
    return float(accs.mean())


def report_cell(task: str, layer: str, K: int, data: dict, verbose: bool = False) -> dict:
    H, y = data["H"], data["y"]
    N = data["N"]
    global_mean = float(y.mean())

    tree = build_dyf_tree(
        H, max_depth=TREE_MAX_DEPTH, num_bits=TREE_NUM_BITS, min_leaf_size=TREE_MIN_LEAF,
    )
    leaf_labels = cut_tree_to_labels(tree, n_points=N, n_clusters=K, embeddings=H)

    leaf_conf = leaf_conf_loo(leaf_labels, y, global_mean)
    knn_conf = knn_conf_loo(H, y, leaf_labels)

    rho_leaf, _ = spearmanr(leaf_conf, y)
    rho_knn, _ = spearmanr(knn_conf, y)
    auc_leaf = acc_at_coverage_auc(leaf_conf, y)
    auc_knn = acc_at_coverage_auc(knn_conf, y)
    auc_random = global_mean

    leaf_sizes = np.bincount(leaf_labels).tolist()
    return {
        "task": task, "layer": layer, "K": K, "N": N,
        "baseline_acc": global_mean,
        "rho_leaf": float(rho_leaf) if rho_leaf == rho_leaf else 0.0,
        "rho_knn": float(rho_knn) if rho_knn == rho_knn else 0.0,
        "auc_leaf": auc_leaf,
        "auc_knn": auc_knn,
        "auc_random": auc_random,
        "leaf_sizes": leaf_sizes,
    }


def main():
    print(f"Loading cache: {CACHE_PATH}")
    cache = np.load(CACHE_PATH, allow_pickle=True)
    print(f"  {len(cache.files)} keys")
    print(f"  layers: {LAYER_SWEEP}")
    print(f"  K:      {K_SWEEP}")
    print()

    rows = []
    for task in TASKS:
        print(f"=== {task} ===")
        data0 = load_task(cache, task, "l8")
        print(f"  N={data0['N']}  baseline_acc={100*float(data0['y'].mean()):.1f}%")
        print(f"  {'layer':>5} {'K':>3}  {'ρ_leaf':>7} {'ρ_knn':>7} {'Δρ':>7}  "
              f"{'AUC_leaf':>9} {'AUC_knn':>9} {'AUC_rand':>9}  {'ΔAUC':>7}")
        t0 = time.time()
        for layer in LAYER_SWEEP:
            data = load_task(cache, task, layer)
            for K in K_SWEEP:
                r = report_cell(task, layer, K, data)
                rows.append(r)
                d_rho = r["rho_leaf"] - r["rho_knn"]
                d_auc = r["auc_leaf"] - r["auc_knn"]
                print(f"  {layer.upper():>5} {K:>3}  "
                      f"{r['rho_leaf']:+7.3f} {r['rho_knn']:+7.3f} {d_rho:+7.3f}  "
                      f"{r['auc_leaf']:>9.3f} {r['auc_knn']:>9.3f} {r['auc_random']:>9.3f}  "
                      f"{d_auc:+7.3f}")
        print(f"  ({time.time() - t0:.1f}s)")
        print()

    # ── Aggregate: mean ΔAUC and Δρ across tasks per (layer, K) ─────
    print("=" * 78)
    print(f"  Aggregate ΔAUC = AUC_leaf - AUC_knn (mean across {len(TASKS)} tasks)")
    print("=" * 78)
    print(f"  {'layer':>5}  " + "  ".join(f"K={K:>2d}" for K in K_SWEEP))
    for layer in LAYER_SWEEP:
        cells = []
        for K in K_SWEEP:
            sel = [r for r in rows if r["layer"] == layer and r["K"] == K]
            cells.append(f"{np.mean([r['auc_leaf'] - r['auc_knn'] for r in sel]):+.4f}")
        print(f"  {layer.upper():>5}  " + "  ".join(f"{c:>9}" for c in cells))
    print()
    print("=" * 78)
    print(f"  Aggregate Δρ = ρ_leaf - ρ_knn (mean across {len(TASKS)} tasks)")
    print("=" * 78)
    print(f"  {'layer':>5}  " + "  ".join(f"K={K:>2d}" for K in K_SWEEP))
    for layer in LAYER_SWEEP:
        cells = []
        for K in K_SWEEP:
            sel = [r for r in rows if r["layer"] == layer and r["K"] == K]
            cells.append(f"{np.mean([r['rho_leaf'] - r['rho_knn'] for r in sel]):+.4f}")
        print(f"  {layer.upper():>5}  " + "  ".join(f"{c:>9}" for c in cells))

    # ── Vs random baseline ──────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"  Aggregate AUC_leaf - AUC_random (mean across tasks) — calibration lift over coin flip")
    print("=" * 78)
    print(f"  {'layer':>5}  " + "  ".join(f"K={K:>2d}" for K in K_SWEEP))
    for layer in LAYER_SWEEP:
        cells = []
        for K in K_SWEEP:
            sel = [r for r in rows if r["layer"] == layer and r["K"] == K]
            cells.append(f"{np.mean([r['auc_leaf'] - r['auc_random'] for r in sel]):+.4f}")
        print(f"  {layer.upper():>5}  " + "  ".join(f"{c:>9}" for c in cells))

    # ── Per-task best layer ─────────────────────────────────────────
    print()
    print(f"  Per-task best (layer, K) by ΔAUC over kNN")
    print(f"  {'task':<45}  {'layer':>5}  {'K':>3}  {'AUC_leaf':>9} {'AUC_knn':>9}  {'ΔAUC':>7}")
    for task in TASKS:
        sel = [r for r in rows if r["task"] == task]
        best = max(sel, key=lambda r: r["auc_leaf"] - r["auc_knn"])
        d = best["auc_leaf"] - best["auc_knn"]
        print(f"  {task:<45}  {best['layer'].upper():>5}  {best['K']:>3}  "
              f"{best['auc_leaf']:>9.3f} {best['auc_knn']:>9.3f}  {d:+7.3f}")

    # Save raw rows for downstream analysis
    out = EXPERIMENT_DIR / "calibration_dyf_leaf_results.npz"
    np.savez(
        out,
        tasks=np.array([r["task"] for r in rows]),
        layers=np.array([r["layer"] for r in rows]),
        Ks=np.array([r["K"] for r in rows]),
        rho_leaf=np.array([r["rho_leaf"] for r in rows]),
        rho_knn=np.array([r["rho_knn"] for r in rows]),
        auc_leaf=np.array([r["auc_leaf"] for r in rows]),
        auc_knn=np.array([r["auc_knn"] for r in rows]),
        auc_random=np.array([r["auc_random"] for r in rows]),
        baseline_acc=np.array([r["baseline_acc"] for r in rows]),
    )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
