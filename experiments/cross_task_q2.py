#!/usr/bin/env python3
"""Q2 — cross-task routing via hidden-state classification.

Tests whether tasks that should route to the same turnstyle category
cluster together in hidden-state space, even when their surface templates
are completely different.

Loads hidden states from two caches:
  - capability_probe_data.npz  (5 tasks from Phase 0)
  - q2_cache.npz               (10 new tasks from q2_extract.py)

Two classification targets:
  1. **task_id**: 15-class per-task (expects 100% — surface giveaway)
  2. **category**: 8-class turnstyle category (the real Q2 question)

If category accuracy is high at L22/L23 but lower at L8, that confirms
the activation-routing premise: late layers encode task semantics beyond
surface form.

Run:
    /Users/jdonaldson/Projects/dyf/.venv/bin/python \
        experiments/cross_task_q2.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dyf import build_dyf_tree, cut_tree_to_labels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

EXPERIMENT_DIR = Path(__file__).parent
PHASE0_CACHE = EXPERIMENT_DIR / "capability_probe_data.npz"
Q2_CACHE = EXPERIMENT_DIR / "q2_cache.npz"

# Task → turnstyle category mapping
# Categories based on which turnstyle solver handles the task
TASK_CATEGORY = {
    # From Phase 0 cache
    "penguins_in_a_table":                        "table_qa",
    "object_counting":                             "counting",
    "tracking_shuffled_objects_three_objects":      "tracking",
    "navigate":                                    "navigation",
    "web_of_lies":                                 "boolean",
    # From Q2 cache
    "reasoning_about_colored_objects":              "counting",
    "tracking_shuffled_objects_five_objects":       "tracking",
    "tracking_shuffled_objects_seven_objects":      "tracking",
    "boolean_expressions":                         "boolean",
    "multistep_arithmetic_two":                    "arithmetic",
    "date_understanding":                          "date",
    "disambiguation_qa":                           "passthrough",
    "snarks":                                      "passthrough",
    "causal_judgement":                            "passthrough",
    "movie_recommendation":                        "passthrough",
}

# Phase 0 tasks (hidden states stored as {task}__hidden_l{L})
PHASE0_TASKS = [
    "penguins_in_a_table",
    "tracking_shuffled_objects_three_objects",
    "object_counting",
    "navigate",
    "web_of_lies",
]

Q2_TASKS = [
    "reasoning_about_colored_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "boolean_expressions",
    "multistep_arithmetic_two",
    "date_understanding",
    "disambiguation_qa",
    "snarks",
    "causal_judgement",
    "movie_recommendation",
]

LAYERS = ["l8", "l22", "l23"]
K_SWEEP = [16, 32, 64]
TREE_MAX_DEPTH = 6
TREE_NUM_BITS = 4
TREE_MIN_LEAF = 4
N_FOLDS = 5
RNG_SEED = 0


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_combined(p0_cache, q2_cache, layer: str):
    """Stack hidden states across all 15 tasks, return X, task_ids, cat_ids."""
    X_list = []
    task_ids = []
    cat_ids = []

    all_tasks = PHASE0_TASKS + Q2_TASKS
    categories = sorted(set(TASK_CATEGORY.values()))
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    for ti, task in enumerate(all_tasks):
        key = f"{task}__hidden_{layer}"
        if task in PHASE0_TASKS:
            H = p0_cache[key].astype(np.float32)
        else:
            H = q2_cache[key].astype(np.float32)
        X_list.append(H)
        task_ids.extend([ti] * H.shape[0])
        cat = TASK_CATEGORY[task]
        cat_ids.extend([cat_to_idx[cat]] * H.shape[0])

    X = np.vstack(X_list)
    task_ids = np.array(task_ids, dtype=np.int64)
    cat_ids = np.array(cat_ids, dtype=np.int64)
    return X, task_ids, cat_ids, all_tasks, categories


# ──────────────────────────────────────────────────────────────────────
# Classifiers
# ──────────────────────────────────────────────────────────────────────

def logreg_cv(X, y, n_folds=N_FOLDS):
    """Stratified CV, returns (mean_acc, std_acc, y_pred from all folds)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    y_pred = np.zeros_like(y)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        y_pred[test_idx] = preds
        accs.append(float((preds == y[test_idx]).mean()))
    return float(np.mean(accs)), float(np.std(accs)), y_pred


def dyf_leaf_majority(X, y, n_clusters):
    """dyf-leaf LOO majority vote accuracy."""
    N = X.shape[0]
    n_classes = int(y.max()) + 1
    tree = build_dyf_tree(
        X, max_depth=TREE_MAX_DEPTH, num_bits=TREE_NUM_BITS, min_leaf_size=TREE_MIN_LEAF,
    )
    leaf_labels = cut_tree_to_labels(tree, n_points=N, n_clusters=n_clusters, embeddings=X)
    K = int(leaf_labels.max()) + 1
    leaf_counts = np.zeros((K, n_classes), dtype=np.int64)
    for i in range(N):
        leaf_counts[leaf_labels[i], y[i]] += 1
    leaf_size = leaf_counts.sum(axis=1)

    correct = 0
    for i in range(N):
        L = leaf_labels[i]
        if leaf_size[L] <= 1:
            pred = int(np.bincount(y, minlength=n_classes).argmax())
        else:
            counts = leaf_counts[L].copy()
            counts[y[i]] -= 1
            pred = int(counts.argmax())
        if pred == y[i]:
            correct += 1
    return correct / N


# ──────────────────────────────────────────────────────────────────────
# Leave-one-task-out (the real Q2 test)
# ──────────────────────────────────────────────────────────────────────

def leave_one_task_out(X, task_ids, cat_ids, all_tasks, categories):
    """Train on N-1 tasks, predict held-out task's category.

    This is the strongest test: can we route a task we've never seen
    to the right turnstyle category based purely on hidden-state similarity
    to tasks in the same category?
    """
    n_tasks = len(all_tasks)
    results = []
    for held_out in range(n_tasks):
        cat = TASK_CATEGORY[all_tasks[held_out]]
        # Skip if this task is the only representative of its category
        same_cat_tasks = [t for t in range(n_tasks) if t != held_out
                          and TASK_CATEGORY[all_tasks[t]] == cat]
        if not same_cat_tasks:
            results.append((all_tasks[held_out], cat, None, "singleton"))
            continue

        train_mask = task_ids != held_out
        test_mask = task_ids == held_out
        X_train, y_train = X[train_mask], cat_ids[train_mask]
        X_test, y_test = X[test_mask], cat_ids[test_mask]

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = float((preds == y_test).mean())

        # What category was most predicted?
        from collections import Counter
        pred_dist = Counter(preds.tolist())
        top_pred = categories[pred_dist.most_common(1)[0][0]]

        results.append((all_tasks[held_out], cat, acc, top_pred))
    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main():
    print(f"Loading Phase 0 cache: {PHASE0_CACHE}")
    p0 = np.load(PHASE0_CACHE, allow_pickle=True)
    print(f"  {len(p0.files)} keys")

    print(f"Loading Q2 cache: {Q2_CACHE}")
    q2 = np.load(Q2_CACHE, allow_pickle=True)
    print(f"  {len(q2.files)} keys")
    print()

    # ── Per-task counts ──────────────────────────────────────────────
    categories = sorted(set(TASK_CATEGORY.values()))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    all_tasks = PHASE0_TASKS + Q2_TASKS

    print(f"Tasks: {len(all_tasks)}, Categories: {len(categories)}")
    print(f"Categories: {categories}")
    print()
    for cat in categories:
        tasks = [t for t in all_tasks if TASK_CATEGORY[t] == cat]
        print(f"  {cat:>12}  ({len(tasks)} tasks): {', '.join(t[:25] for t in tasks)}")
    print()

    # ── Layer sweep ──────────────────────────────────────────────────
    print("=" * 78)
    print("Layer sweep — task_id (15-class) vs category (8-class)")
    print("=" * 78)
    header = f"  {'layer':>5}  {'task-LR':>9}  {'cat-LR':>9}  " + \
             "  ".join(f"cat-dyf-K={K}" for K in K_SWEEP)
    print(header)

    best_cat_acc = 0
    best_cat_layer = None

    for layer in LAYERS:
        X, task_ids, cat_ids, _, _ = load_combined(p0, q2, layer)

        task_mean, task_std, _ = logreg_cv(X, task_ids)
        cat_mean, cat_std, y_pred_cat = logreg_cv(X, cat_ids)

        dyf_cells = []
        for K in K_SWEEP:
            dyf_acc = dyf_leaf_majority(X, cat_ids, n_clusters=K)
            dyf_cells.append(f"{fmt_pct(dyf_acc):>10}")

        print(f"  {layer.upper():>5}  {fmt_pct(task_mean):>5}±{task_std*100:.1f}"
              f"  {fmt_pct(cat_mean):>5}±{cat_std*100:.1f}"
              f"  {'  '.join(dyf_cells)}")

        if cat_mean > best_cat_acc:
            best_cat_acc = cat_mean
            best_cat_layer = layer

    # ── Detailed results at best layer ───────────────────────────────
    print()
    print(f"=== Best category layer: {best_cat_layer.upper()} ({fmt_pct(best_cat_acc)}) ===")
    X, task_ids, cat_ids, all_tasks_list, cats = load_combined(p0, q2, best_cat_layer)
    _, _, y_pred = logreg_cv(X, cat_ids)

    cm = confusion_matrix(cat_ids, y_pred)
    print(f"\nCategory confusion matrix (rows=true, cols=pred):")
    short_cats = [c[:10] for c in cats]
    print(f"  {'':>12}  " + "  ".join(f"{s:>10}" for s in short_cats))
    for i, name in enumerate(short_cats):
        cells = "  ".join(f"{cm[i, j]:>10d}" for j in range(len(short_cats)))
        n_total = cm[i].sum()
        acc_i = cm[i, i] / max(n_total, 1)
        print(f"  {name:>12}  " + cells + f"   ({100*acc_i:5.1f}%)")

    print(f"\nPer-category classification report:")
    print(classification_report(cat_ids, y_pred, target_names=cats, digits=3))

    # ── Leave-one-task-out ───────────────────────────────────────────
    print("=" * 78)
    print(f"Leave-one-task-out (train on 14 tasks, predict held-out category)")
    print(f"Layer: {best_cat_layer.upper()}")
    print("=" * 78)

    for layer in LAYERS:
        X, task_ids, cat_ids, tasks_list, cats = load_combined(p0, q2, layer)
        results = leave_one_task_out(X, task_ids, cat_ids, tasks_list, cats)
        print(f"\n  {layer.upper()}:")
        print(f"  {'task':<45}  {'true-cat':>12}  {'acc':>7}  {'pred-cat':>12}")
        n_correct = 0
        n_tested = 0
        for task_name, true_cat, acc, pred_cat in results:
            if acc is None:
                print(f"  {task_name:<45}  {true_cat:>12}  {'skip':>7}  {'(singleton)':>12}")
            else:
                correct_str = "✓" if acc > 0.5 else "✗"
                print(f"  {task_name:<45}  {true_cat:>12}  {fmt_pct(acc):>7}  {pred_cat:>12}  {correct_str}")
                n_tested += 1
                if acc > 0.5:
                    n_correct += 1
        if n_tested > 0:
            print(f"  Task-level accuracy: {n_correct}/{n_tested} = {100*n_correct/n_tested:.0f}%")


if __name__ == "__main__":
    main()
