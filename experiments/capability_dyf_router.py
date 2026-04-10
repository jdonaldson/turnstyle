#!/usr/bin/env python3
"""Capability dyf-leaf router experiment (Q1).

Tests whether a dyf PCA-LSH tree built on L18 hidden states can route
each example to the right per-task solver tier better than the production
sequential fallback chain.

Reuses cached hidden states + tier outcomes from capability_probe.py
(Phase 0 cache: experiments/capability_probe_data.npz). No model loading,
no generation — pure numpy + dyf.

Comparison columns (per task, per K):
  - sequential : production fallback order, walk-on-not-answered
  - random     : averaged over N random tier orderings
  - dyf-leaf   : rank tiers by per-leaf success rate (LOO within leaf),
                 walk-on-not-answered
  - oracle     : any tier correct → correct

Web_of_lies is dropped: IR tier answers 0% on SmolLM2, leaving only
baseline (single tier = no routing question).

Run with the dyf venv (which has dyf 0.8.0 + dyf_rs 0.7.0 installed):
    /Users/jdonaldson/Projects/dyf/.venv/bin/python \
        experiments/capability_dyf_router.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from dyf import build_dyf_tree, cut_tree_to_labels


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

EXPERIMENT_DIR = Path(__file__).parent
CACHE_PATH = EXPERIMENT_DIR / "capability_probe_data.npz"

# Same task / tier definitions as capability_probe.py.
# web_of_lies dropped (only baseline tier is functional on SmolLM2).
TASK_TIERS: dict[str, list[str]] = {
    "penguins_in_a_table": ["sql", "knowledge_poll", "logit_poll", "baseline"],
    "tracking_shuffled_objects_three_objects": ["sql", "logit_poll", "baseline"],
    "object_counting": ["sql", "baseline"],
    "navigate": ["ir", "baseline"],
}

LAYER_SWEEP = ["l8", "l12", "l16", "l18", "l20", "l22", "l23"]
DEFAULT_LAYER = "l18"       # used for verbose first-N output
K_SWEEP = [4, 8, 12]        # cluster counts to compare
TREE_MAX_DEPTH = 4
TREE_NUM_BITS = 3
TREE_MIN_LEAF = 4
RANDOM_SEEDS = 200          # for random baseline averaging
VERBOSE_EXAMPLES = 3        # per-task verbose first-N output


# ──────────────────────────────────────────────────────────────────────
# Cache loading
# ──────────────────────────────────────────────────────────────────────

def load_task(cache: np.lib.npyio.NpzFile, task: str, layer: str = DEFAULT_LAYER) -> dict:
    """Pull hidden states + per-tier correct/answered arrays for a task."""
    tiers = TASK_TIERS[task]
    H = cache[f"{task}__hidden_{layer}"].astype(np.float32)  # (N, D)
    N = H.shape[0]
    correct = np.zeros((N, len(tiers)), dtype=np.int8)
    answered = np.zeros((N, len(tiers)), dtype=np.int8)
    for ti, tier in enumerate(tiers):
        correct[:, ti] = cache[f"{task}__{tier}__correct"]
        answered[:, ti] = cache[f"{task}__{tier}__answered"]
    return {"H": H, "N": N, "tiers": tiers, "correct": correct, "answered": answered, "layer": layer}


# ──────────────────────────────────────────────────────────────────────
# Routing strategies
# ──────────────────────────────────────────────────────────────────────

def walk_order(order: np.ndarray, answered_row: np.ndarray, correct_row: np.ndarray) -> int:
    """Walk a tier order; return correct (0/1) of first-answered tier.

    If no tier answered, returns 0.
    """
    for t in order:
        if answered_row[t]:
            return int(correct_row[t])
    return 0


def sequential_acc(data: dict) -> float:
    order = np.arange(len(data["tiers"]))  # production order = TASK_TIERS order
    N = data["N"]
    return sum(walk_order(order, data["answered"][i], data["correct"][i])
               for i in range(N)) / N


def random_acc(data: dict, n_seeds: int = RANDOM_SEEDS) -> tuple[float, float]:
    """Average accuracy over many random tier orderings."""
    N, T = data["correct"].shape
    rng = np.random.default_rng(0)
    accs = np.zeros(n_seeds)
    for s in range(n_seeds):
        order = rng.permutation(T)
        accs[s] = sum(walk_order(order, data["answered"][i], data["correct"][i])
                      for i in range(N)) / N
    return float(accs.mean()), float(accs.std())


def oracle_acc(data: dict) -> float:
    return float((data["correct"].max(axis=1) > 0).mean())


def dyf_route(data: dict, n_clusters: int, verbose_n: int = 0) -> tuple[float, np.ndarray, np.ndarray]:
    """Build dyf tree, route each example by per-leaf tier success (LOO within leaf).

    Returns: (accuracy, leaf_labels, chosen_tier_per_example)
    """
    H = data["H"]
    correct = data["correct"]
    answered = data["answered"]
    N, T = correct.shape

    # Build tree on full data; LOO within leaf handles train/test leakage.
    tree = build_dyf_tree(
        H,
        max_depth=TREE_MAX_DEPTH,
        num_bits=TREE_NUM_BITS,
        min_leaf_size=TREE_MIN_LEAF,
    )
    leaf_labels = cut_tree_to_labels(
        tree, n_points=N, n_clusters=n_clusters, embeddings=H,
    )
    K = int(leaf_labels.max()) + 1

    # Aggregate per-leaf per-tier success counts
    leaf_size = np.bincount(leaf_labels, minlength=K).astype(np.float64)
    leaf_correct = np.zeros((K, T), dtype=np.float64)
    for i in range(N):
        leaf_correct[leaf_labels[i]] += correct[i]

    # Global per-tier rate for tiny-leaf fallback
    global_rate = correct.mean(axis=0)
    seq_rank = np.arange(T)  # sequential position = tier index in TASK_TIERS

    chosen = np.zeros(N, dtype=int)
    route_correct = np.zeros(N, dtype=np.int8)

    for i in range(N):
        L = leaf_labels[i]
        n_others = leaf_size[L] - 1
        if n_others <= 0:
            rates = global_rate
        else:
            # LOO within leaf: subtract this example from its leaf
            rates = (leaf_correct[L] - correct[i]) / n_others

        # Rank tiers by rate desc, ties broken by sequential order
        keys = list(zip(-rates, seq_rank, range(T)))
        keys.sort()
        ranked = [k[2] for k in keys]

        picked_correct = 0
        picked_tier = ranked[0]
        for t in ranked:
            if answered[i, t]:
                picked_tier = t
                picked_correct = int(correct[i, t])
                break
        chosen[i] = picked_tier
        route_correct[i] = picked_correct

        if i < verbose_n:
            tier_names = data["tiers"]
            rate_str = "  ".join(f"{tier_names[t]}={rates[t]:.2f}" for t in range(T))
            ranked_str = ",".join(tier_names[t] for t in ranked)
            print(f"  ex {i:3d}: leaf={L:2d}  rates=[{rate_str}]")
            print(f"          ranked={ranked_str}  picked={tier_names[picked_tier]}  correct={picked_correct}")

    acc = float(route_correct.mean())
    return acc, leaf_labels, chosen


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────

def fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main():
    print(f"Loading cache: {CACHE_PATH}")
    cache = np.load(CACHE_PATH, allow_pickle=True)
    print(f"  {len(cache.files)} keys")
    print(f"  layer sweep: {LAYER_SWEEP}")
    print(f"  K sweep:     {K_SWEEP}")
    print()

    # all_rows: (task, layer, K, seq, rand, dyf, oracle)
    all_rows = []

    for task in TASK_TIERS:
        # Sequential / oracle / random are layer-independent — compute once.
        data_default = load_task(cache, task, layer=DEFAULT_LAYER)
        N = data_default["N"]
        tiers = data_default["tiers"]
        seq = sequential_acc(data_default)
        oracle = oracle_acc(data_default)
        rand_mean, rand_std = random_acc(data_default)
        gap = oracle - seq

        print(f"=== {task}  (N={N}, tiers={tiers}) ===")
        print(f"  sequential={fmt_pct(seq)}  random={fmt_pct(rand_mean)}±{100*rand_std:.1f}  oracle={fmt_pct(oracle)}  gap={100*gap:+.1f}pp")

        # Verbose pass at default layer, K=8 (illustrates one cell)
        print(f"  -- verbose first {VERBOSE_EXAMPLES} examples ({DEFAULT_LAYER.upper()}, K=8) --")
        t0 = time.time()
        _, leaf_labels, _ = dyf_route(data_default, n_clusters=8, verbose_n=VERBOSE_EXAMPLES)
        elapsed = time.time() - t0
        leaf_dist = np.bincount(leaf_labels)
        print(f"  {DEFAULT_LAYER.upper()} K=8: leaves={len(leaf_dist)} dist={leaf_dist.tolist()} build+route={elapsed:.2f}s")

        # Layer × K sweep
        print(f"  -- layer × K sweep (Δseq pp) --")
        header = f"  {'layer':>5}  " + "  ".join(f"K={K:>2d}" for K in K_SWEEP)
        print(header)
        for layer in LAYER_SWEEP:
            data = load_task(cache, task, layer=layer)
            cells = []
            for K in K_SWEEP:
                acc, _, _ = dyf_route(data, n_clusters=K)
                all_rows.append((task, layer, K, seq, rand_mean, acc, oracle))
                cells.append(f"{100*(acc - seq):+5.1f}")
            print(f"  {layer.upper():>5}  " + "  ".join(f"{c:>4}" for c in cells))
        print()

    # ── Aggregate: mean Δseq across tasks per (layer, K) ─────────────────
    print("=" * 78)
    print(f"  Aggregate Δseq (mean across {len(TASK_TIERS)} tasks, percentage points)")
    print("=" * 78)
    header = f"  {'layer':>5}  " + "  ".join(f"K={K:>2d}" for K in K_SWEEP)
    print(header)
    for layer in LAYER_SWEEP:
        cells = []
        for K in K_SWEEP:
            rows = [r for r in all_rows if r[1] == layer and r[2] == K]
            delta = np.mean([(r[5] - r[3]) for r in rows]) * 100
            cells.append(f"{delta:+5.1f}")
        print(f"  {layer.upper():>5}  " + "  ".join(f"{c:>4}" for c in cells))

    # Best (layer, K) per task
    print()
    print(f"  Per-task best (layer, K) by Δseq")
    print(f"  {'task':<45}  {'layer':>5}  {'K':>3}  {'seq':>7}  {'dyf*':>7}  {'oracle':>7}  {'Δseq':>7}  {'closure':>9}")
    for task in TASK_TIERS:
        rows = [r for r in all_rows if r[0] == task]
        best = max(rows, key=lambda r: r[5] - r[3])  # max Δseq
        _, layer, K, seq, _, dyf, oracle = best
        delta = dyf - seq
        gap = oracle - seq
        closure = delta / max(gap, 1e-9) * 100 if gap > 0 else 0.0
        print(f"  {task:<45}  {layer.upper():>5}  {K:>3d}  {fmt_pct(seq):>7}  {fmt_pct(dyf):>7}  {fmt_pct(oracle):>7}  {100*delta:+6.1f}pp  {closure:+8.0f}%")

    # Best layer overall (averaged over K)
    print()
    print(f"  Best layer (mean Δseq across tasks, averaged over K sweep)")
    layer_scores = {}
    for layer in LAYER_SWEEP:
        rows = [r for r in all_rows if r[1] == layer]
        layer_scores[layer] = np.mean([(r[5] - r[3]) for r in rows]) * 100
    for layer, score in sorted(layer_scores.items(), key=lambda x: -x[1]):
        print(f"    {layer.upper():>5}: Δseq={score:+5.2f}pp")


if __name__ == "__main__":
    main()
