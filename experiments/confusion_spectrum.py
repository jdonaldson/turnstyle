#!/usr/bin/env python3
"""Confusion spectrum — does SmolLM2 have a detectable "I'm confused" signal?

Three questions:
  1. Binary confusion probe: does "baseline correct" transfer across tasks?
     (leave-one-task-out — the critical cross-template test)
  2. Confusion margin: is there a gradient, not just a binary signal?
     (probe logit distribution for correct vs wrong examples)
  3. Failure mode geometry: do different failure modes cluster in hidden space?
     (PCA of wrong-only examples, colored by source task)

Uses Phase 0 cache only (5 tasks, N=1131, baseline correct/wrong per example).

Run:
    /Users/jdonaldson/Projects/dyf/.venv/bin/python \
        experiments/confusion_spectrum.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA


# ──────────────────────────────────────────────────────────────────────
# Config
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
SHORT = {
    "penguins_in_a_table": "penguins",
    "tracking_shuffled_objects_three_objects": "tracking",
    "object_counting": "counting",
    "navigate": "navigate",
    "web_of_lies": "wol",
}

LAYERS = ["l8", "l12", "l16", "l18", "l20", "l22", "l23"]
N_FOLDS = 5
RNG_SEED = 42


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_all(cache, layer: str):
    """Stack hidden states + baseline correct labels across all tasks."""
    X_list, y_list, task_ids = [], [], []
    for ti, task in enumerate(TASKS):
        H = cache[f"{task}__hidden_{layer}"].astype(np.float32)
        correct = cache[f"{task}__baseline__correct"].astype(np.int8)
        X_list.append(H)
        y_list.append(correct)
        task_ids.extend([ti] * len(correct))
    return np.vstack(X_list), np.concatenate(y_list), np.array(task_ids)


# ──────────────────────────────────────────────────────────────────────
# Q1: Binary confusion probe — does the signal transfer?
# ──────────────────────────────────────────────────────────────────────

def within_task_cv(cache, layer: str):
    """Per-task 5-fold CV — does the signal exist within each task?"""
    results = []
    for task in TASKS:
        H = cache[f"{task}__hidden_{layer}"].astype(np.float32)
        y = cache[f"{task}__baseline__correct"].astype(np.int8)

        # Skip if fewer than 2 classes present
        if len(np.unique(y)) < 2:
            results.append((task, float("nan"), float("nan"), len(y)))
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
        accs, aucs = [], []
        for train_idx, test_idx in skf.split(H, y):
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(H[train_idx], y[train_idx])
            accs.append(clf.score(H[test_idx], y[test_idx]))
            proba = clf.predict_proba(H[test_idx])[:, 1]
            if len(np.unique(y[test_idx])) > 1:
                aucs.append(roc_auc_score(y[test_idx], proba))
        results.append((task, float(np.mean(accs)), float(np.mean(aucs)) if aucs else float("nan"), len(y)))
    return results


def leave_one_task_out(cache, layer: str):
    """Train on 4 tasks, predict held-out. THE critical cross-template test."""
    X, y, task_ids = load_all(cache, layer)
    results = []
    for held_out in range(len(TASKS)):
        train_mask = task_ids != held_out
        test_mask = task_ids == held_out
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test)
        proba = clf.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, proba)
        else:
            auc = float("nan")

        # Majority baseline: predict the most common class
        majority = float((y_test == np.bincount(y_test).argmax()).mean())

        results.append((TASKS[held_out], acc, auc, majority, len(y_test)))
    return results


# ──────────────────────────────────────────────────────────────────────
# Q2: Confusion margin — is there a gradient?
# ──────────────────────────────────────────────────────────────────────

def confusion_margin_analysis(cache, layer: str):
    """Train on all data, examine probe logit distribution."""
    X, y, task_ids = load_all(cache, layer)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X, y)

    # Raw logits (before sigmoid)
    logits = X @ clf.coef_.T + clf.intercept_  # (N, 1)
    logits = logits.ravel()

    print(f"\n  Confusion margin at {layer.upper()} (positive = model predicts correct):")
    print(f"  {'task':>12}  {'correct_mean':>12}  {'correct_std':>11}  {'wrong_mean':>10}  {'wrong_std':>9}  {'gap':>8}")

    for ti, task in enumerate(TASKS):
        mask = task_ids == ti
        correct_mask = mask & (y == 1)
        wrong_mask = mask & (y == 0)

        c_mean = logits[correct_mask].mean() if correct_mask.sum() > 0 else float("nan")
        c_std = logits[correct_mask].std() if correct_mask.sum() > 1 else float("nan")
        w_mean = logits[wrong_mask].mean() if wrong_mask.sum() > 0 else float("nan")
        w_std = logits[wrong_mask].std() if wrong_mask.sum() > 1 else float("nan")
        gap = c_mean - w_mean

        print(f"  {SHORT[task]:>12}  {c_mean:>+12.3f}  {c_std:>11.3f}  {w_mean:>+10.3f}  {w_std:>9.3f}  {gap:>+8.3f}")

    # Overall percentile distribution
    print(f"\n  Logit percentiles (all examples):")
    for pct in [5, 25, 50, 75, 95]:
        val = np.percentile(logits, pct)
        # What fraction of each class falls below this threshold?
        frac_correct = (logits[y == 1] < val).mean()
        frac_wrong = (logits[y == 0] < val).mean()
        print(f"    P{pct:02d}: logit={val:+.3f}  correct_below={frac_correct:.2f}  wrong_below={frac_wrong:.2f}")

    return logits, y, task_ids


# ──────────────────────────────────────────────────────────────────────
# Q3: Failure mode geometry
# ──────────────────────────────────────────────────────────────────────

def failure_geometry(cache, layer: str):
    """PCA of wrong-only examples — do failure modes cluster by task?"""
    X, y, task_ids = load_all(cache, layer)

    wrong_mask = y == 0
    X_wrong = X[wrong_mask]
    tasks_wrong = task_ids[wrong_mask]
    n_wrong = X_wrong.shape[0]

    print(f"\n  Failure mode geometry at {layer.upper()} (N_wrong={n_wrong}):")

    # PCA to 2D for interpretability
    pca = PCA(n_components=min(10, n_wrong, X_wrong.shape[1]))
    X_pca = pca.fit_transform(X_wrong)
    var_explained = pca.explained_variance_ratio_

    print(f"  PCA variance explained: {', '.join(f'{v:.3f}' for v in var_explained[:5])}")
    print(f"  Top-5 cumulative: {var_explained[:5].sum():.3f}")

    # Within-cluster vs between-cluster separation in PCA space
    # Use first 5 PCA components
    X_5d = X_pca[:, :5]
    centroids = {}
    for ti in range(len(TASKS)):
        mask = tasks_wrong == ti
        if mask.sum() > 0:
            centroids[ti] = X_5d[mask].mean(axis=0)

    # Between-cluster distances
    print(f"\n  Between-failure-mode distances (5d PCA, Euclidean):")
    task_list = sorted(centroids.keys())
    header = f"  {'':>12}  " + "  ".join(f"{SHORT[TASKS[t]]:>9}" for t in task_list)
    print(header)
    for ti in task_list:
        cells = []
        for tj in task_list:
            if ti == tj:
                # Within-cluster spread
                mask = tasks_wrong == ti
                spread = np.sqrt(((X_5d[mask] - centroids[ti]) ** 2).sum(axis=1).mean())
                cells.append(f"({spread:.1f})")
            else:
                dist = np.sqrt(((centroids[ti] - centroids[tj]) ** 2).sum())
                cells.append(f"{dist:.1f}")
        print(f"  {SHORT[TASKS[ti]]:>12}  " + "  ".join(f"{c:>9}" for c in cells))

    # Can we classify failure mode (which task) from wrong-only hidden states?
    if n_wrong >= 20 and len(np.unique(tasks_wrong)) >= 2:
        skf = StratifiedKFold(n_splits=min(N_FOLDS, min(np.bincount(tasks_wrong[tasks_wrong >= 0]))),
                              shuffle=True, random_state=RNG_SEED)
        accs = []
        for train_idx, test_idx in skf.split(X_wrong, tasks_wrong):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X_wrong[train_idx], tasks_wrong[train_idx])
            accs.append(clf.score(X_wrong[test_idx], tasks_wrong[test_idx]))
        print(f"\n  Wrong-only task classification ({N_FOLDS}-fold CV): {100*np.mean(accs):.1f}%")
        print(f"  (If <100%: failures overlap geometrically — similar confusion states)")
        print(f"  (If 100%: failures are template-bound — each task fails differently)")

    # Also: among wrong examples, are some "more wrong" than others?
    # Use distance to the correct-example centroid as a confusion depth metric
    correct_mask = y == 1
    X_correct_all = X[correct_mask]
    correct_centroid = X_correct_all.mean(axis=0)
    wrong_dists = np.sqrt(((X_wrong - correct_centroid) ** 2).sum(axis=1))
    correct_dists = np.sqrt(((X_correct_all - correct_centroid) ** 2).sum(axis=1))

    print(f"\n  Distance to 'correct' centroid:")
    print(f"    Correct examples: mean={correct_dists.mean():.1f} ± {correct_dists.std():.1f}")
    print(f"    Wrong examples:   mean={wrong_dists.mean():.1f} ± {wrong_dists.std():.1f}")
    print(f"    (Gap = {wrong_dists.mean() - correct_dists.mean():+.1f})")

    # Per-task breakdown
    print(f"\n  Per-task distance to correct centroid:")
    for ti, task in enumerate(TASKS):
        c_mask = (task_ids == ti) & (y == 1)
        w_mask = (task_ids == ti) & (y == 0)
        if c_mask.sum() > 0 and w_mask.sum() > 0:
            c_dists = np.sqrt(((X[c_mask] - correct_centroid) ** 2).sum(axis=1))
            w_dists = np.sqrt(((X[w_mask] - correct_centroid) ** 2).sum(axis=1))
            print(f"    {SHORT[task]:>12}  correct={c_dists.mean():.1f}±{c_dists.std():.1f}"
                  f"  wrong={w_dists.mean():.1f}±{w_dists.std():.1f}"
                  f"  gap={w_dists.mean()-c_dists.mean():+.1f}")


# ──────────────────────────────────────────────────────────────────────
# Q4: Per-tier confusion — which solver would have helped?
# ──────────────────────────────────────────────────────────────────────

TASK_TIERS = {
    "penguins_in_a_table": ["sql", "knowledge_poll", "logit_poll", "baseline"],
    "tracking_shuffled_objects_three_objects": ["sql", "logit_poll", "baseline"],
    "object_counting": ["sql", "baseline"],
    "navigate": ["ir", "baseline"],
    "web_of_lies": ["ir", "baseline"],
}

def tier_confusion_analysis(cache, layer: str):
    """For wrong examples: could another tier have saved them?

    Classifies wrong examples by their 'rescue profile':
      - rescuable: another tier got it right
      - hopeless: no tier got it right

    Then probes whether rescuable vs hopeless separates in hidden space.
    """
    print(f"\n  Tier confusion analysis at {layer.upper()}:")

    all_X_wrong = []
    all_rescue = []  # 1 = rescuable by some tier, 0 = hopeless
    all_task_ids = []

    for ti, task in enumerate(TASKS):
        H = cache[f"{task}__hidden_{layer}"].astype(np.float32)
        baseline_correct = cache[f"{task}__baseline__correct"].astype(np.int8)

        # Collect per-tier correct arrays
        tiers = TASK_TIERS[task]
        tier_correct = {}
        for tier in tiers:
            tier_correct[tier] = cache[f"{task}__{tier}__correct"].astype(np.int8)

        wrong_mask = baseline_correct == 0
        n_wrong = wrong_mask.sum()

        # For each wrong example: is there any non-baseline tier that got it right?
        non_baseline_tiers = [t for t in tiers if t != "baseline"]
        rescuable = np.zeros(n_wrong, dtype=np.int8)
        rescue_tier = []

        wrong_indices = np.where(wrong_mask)[0]
        for j, idx in enumerate(wrong_indices):
            for tier in non_baseline_tiers:
                if tier_correct[tier][idx]:
                    rescuable[j] = 1
                    rescue_tier.append(tier)
                    break
            else:
                rescue_tier.append("none")

        n_rescuable = int(rescuable.sum())
        n_hopeless = n_wrong - n_rescuable

        # Rescue tier distribution
        from collections import Counter
        tier_dist = Counter(rescue_tier)
        dist_str = ", ".join(f"{t}={c}" for t, c in tier_dist.most_common())

        print(f"    {SHORT[task]:>12}  wrong={n_wrong}  rescuable={n_rescuable}  hopeless={n_hopeless}  rescue_by=[{dist_str}]")

        all_X_wrong.append(H[wrong_mask])
        all_rescue.append(rescuable)
        all_task_ids.extend([ti] * n_wrong)

    X_wrong = np.vstack(all_X_wrong)
    rescue = np.concatenate(all_rescue)
    task_ids = np.array(all_task_ids)

    n_rescuable = int(rescue.sum())
    n_hopeless = len(rescue) - n_rescuable
    print(f"\n    Total: rescuable={n_rescuable} ({100*n_rescuable/len(rescue):.0f}%), hopeless={n_hopeless} ({100*n_hopeless/len(rescue):.0f}%)")

    # Probe: can we separate rescuable from hopeless?
    if n_rescuable >= 10 and n_hopeless >= 10:
        # All-data CV
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
        accs, aucs = [], []
        for train_idx, test_idx in skf.split(X_wrong, rescue):
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(X_wrong[train_idx], rescue[train_idx])
            accs.append(clf.score(X_wrong[test_idx], rescue[test_idx]))
            proba = clf.predict_proba(X_wrong[test_idx])[:, 1]
            if len(np.unique(rescue[test_idx])) > 1:
                aucs.append(roc_auc_score(rescue[test_idx], proba))
        print(f"    Rescuable vs hopeless probe (5-fold CV): acc={100*np.mean(accs):.1f}%  AUC={np.mean(aucs):.3f}")
        print(f"    Majority baseline: {100*max(n_rescuable, n_hopeless)/len(rescue):.1f}%")

        # Leave-one-task-out for rescuable probe
        print(f"\n    Leave-one-task-out (rescuable vs hopeless):")
        for held_out in range(len(TASKS)):
            train_mask = task_ids != held_out
            test_mask = task_ids == held_out
            if test_mask.sum() == 0 or len(np.unique(rescue[test_mask])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(X_wrong[train_mask], rescue[train_mask])
            acc = clf.score(X_wrong[test_mask], rescue[test_mask])
            auc = roc_auc_score(rescue[test_mask], clf.predict_proba(X_wrong[test_mask])[:, 1])
            majority = float((rescue[test_mask] == np.bincount(rescue[test_mask]).argmax()).mean())
            print(f"      {SHORT[TASKS[held_out]]:>12}  acc={100*acc:.1f}%  AUC={auc:.3f}  majority={100*majority:.1f}%")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main():
    print(f"Loading cache: {CACHE_PATH}")
    cache = np.load(CACHE_PATH, allow_pickle=True)
    print(f"  {len(cache.files)} keys")
    print()

    # ── Q1: Within-task confusion probe ──────────────────────────────
    print("=" * 78)
    print("Q1: Within-task confusion probe (5-fold CV)")
    print("    Does SmolLM2 know when it will fail, within a template?")
    print("=" * 78)

    for layer in LAYERS:
        results = within_task_cv(cache, layer)
        if layer == LAYERS[0]:
            print(f"  {'layer':>5}  ", end="")
            for task, _, _, _ in results:
                print(f"  {SHORT[task]:>12}", end="")
            print()
        print(f"  {layer.upper():>5}  ", end="")
        for _, acc, auc, n in results:
            if np.isnan(acc):
                print(f"  {'n/a':>12}", end="")
            else:
                print(f"  {100*acc:5.1f}/{100*auc:5.1f}", end="")
        print("   (acc/AUC)")

    # ── Q1b: Leave-one-task-out ──────────────────────────────────────
    print()
    print("=" * 78)
    print("Q1b: Leave-one-task-out confusion probe")
    print("     Does the 'confused' signal transfer across templates?")
    print("=" * 78)

    for layer in LAYERS:
        results = leave_one_task_out(cache, layer)
        print(f"\n  {layer.upper()}:")
        print(f"  {'task':>12}  {'acc':>7}  {'AUC':>7}  {'majority':>9}  {'Δmaj':>7}  {'N':>5}")
        total_gain = 0
        n_tested = 0
        for task, acc, auc, majority, n in results:
            delta = acc - majority
            total_gain += delta
            n_tested += 1
            auc_str = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
            print(f"  {SHORT[task]:>12}  {fmt_pct(acc):>7}  {auc_str:>7}  {fmt_pct(majority):>9}  {100*delta:>+6.1f}%  {n:>5}")
        print(f"  {'mean Δmaj':>12}  {100*total_gain/n_tested:>+6.1f}%")

    # ── Q2: Confusion margin ─────────────────────────────────────────
    print()
    print("=" * 78)
    print("Q2: Confusion margin — is there a gradient?")
    print("=" * 78)

    best_layer = "l22"  # use a representative layer
    logits, y, task_ids = confusion_margin_analysis(cache, best_layer)

    # ── Q3: Failure mode geometry ────────────────────────────────────
    print()
    print("=" * 78)
    print("Q3: Failure mode geometry — do different confusions cluster?")
    print("=" * 78)

    failure_geometry(cache, best_layer)

    # ── Q4: Tier rescue analysis ─────────────────────────────────────
    print()
    print("=" * 78)
    print("Q4: Tier confusion — rescuable vs hopeless failures")
    print("=" * 78)

    tier_confusion_analysis(cache, best_layer)


if __name__ == "__main__":
    main()
