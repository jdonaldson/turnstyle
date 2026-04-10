# dyf-leaf router (Q1: within-task tier routing)

**Date**: 2026-04-07 | **Model**: SmolLM2-1.7B-Instruct | **Mode**: no-regex

## Question

Can a dyf PCA-LSH leaf-based router pick the right per-example tier from a
fixed per-task fallback chain better than walking the chain sequentially?

Replaces the LogReg-per-(task, tier) probe approach in `capability_probe.py`
Phase 1 — single tree per task, leaf-conditional tier ranking with
LOO-within-leaf to avoid leakage.

## Setup

- Cache: `experiments/capability_probe_data.npz` (84 keys, ~53 MB)
- Hidden states: per-task last-token pooling at L8/12/16/18/20/22/**L23** (added post-hoc)
- Tree: `build_dyf_tree(max_depth=4, num_bits=3, min_leaf_size=4)`
- Cut: `cut_tree_to_labels(tree, n_points=N, n_clusters=K, embeddings=H)` (dyf 0.8.0)
- Sweep: layer × K, K ∈ {4, 8, 12}
- Tasks: penguins, tracking_shuffled_3, object_counting, navigate
  (web_of_lies dropped — IR tier 0% answered, leaving only baseline)
- Comparisons: sequential / random (200 seeds) / dyf-leaf / oracle

## Results

### Aggregate Δseq (mean across 4 tasks, percentage points)

| layer | K=4 | K=8 | K=12 |
|---|---|---|---|
| L8 | +0.9 | +1.2 | +1.0 |
| L12 | -0.0 | -1.4 | -1.2 |
| L16 | -0.8 | +0.5 | -1.9 |
| L18 | +0.3 | -2.2 | -0.1 |
| L20 | +0.6 | -0.3 | +0.0 |
| **L22** | **+0.7** | **+1.5** | **+1.3** |
| L23 | +0.7 | +0.5 | +0.4 |

**Best layer (mean across K)**:
- L22: +1.17pp
- L8: +1.00pp
- **L23: +0.49pp** ← does not continue the L20→L22 trend
- L20: +0.09pp
- L18: -0.69pp
- L16: -0.71pp
- L12: -0.86pp

### Per-task best

| task | seq | oracle | gap | best (layer, K) | dyf | Δseq | closure |
|---|---|---|---|---|---|---|---|
| penguins | 94.4% | 95.8% | +1.4 | (any) | 94.4% | +0.0pp | 0% |
| tracking_shuffled_3 | 30.0% | 61.5% | +31.6 | (L22, 12) | 37.7% | **+7.7pp** | 24% |
| object_counting | 27.1% | 33.6% | +6.5 | (L18, 12) | 29.6% | +2.4pp | 38% |
| navigate | 79.8% | 86.2% | +6.5 | (any) | 79.8% | +0.0pp | 0% |

## Findings

1. **L22 wins, not L23-24.** The activation-routing doc claims SmolLM2 task
   type lives at L23-24 (1-indexed → L22-23 0-indexed), but for *within-task
   tier routing*, L22 (0-indexed) is the peak and L23 is slightly worse.
   The L20 → L22 ascent does not continue.
2. **L8 is competitive** (+1.00pp). Structural / formatting features (boundary
   detection, segment shape) have routing-relevant signal even at shallow
   layers, consistent with the L8 split-probe finding.
3. **Most tasks have no headroom to route.** Penguins gap is +1.4pp; navigate
   and object_counting are +6.5pp each. Only tracking_shuffled (+31.6pp gap)
   is meaningfully routable, and even there closure is 24%.
4. **Cluster-size sensitivity is mild.** K=4/8/12 swing only ±1pp at the best
   layers. tracking_shuffled prefers K=12; object_counting also K=12;
   navigate/penguins are insensitive.

## Interpretation

Within-task tier routing has limited headroom on this set because the
sequential fallback chain is *already close to oracle on 3 of 4 tasks*. The
only task where routing can pay off (tracking_shuffled, +31.6pp gap) is also
the only task where we see real lift (+7.7pp, 24% closure). This is a
generalizable observation: dyf-leaf routing can only buy gap-shaped headroom,
and the gap shape varies wildly by task.

Two structural conclusions:

- **Q1 is mostly answered.** A unified within-task router would buy ~+1pp
  aggregate at L22. Worth it if cheap, but it does not justify training
  separate per-tier probes.
- **Q2 (cross-task routing) is the bigger lever.** Replacing regex
  `_detect_task` with a hidden-state probe affects every task simultaneously.
  Even +1pp per task aggregates faster than within-task tier routing.

## Cache extension

`experiments/capability_extend_cache.py` adds L23 hidden states to the
existing cache via pure forward passes (no generation). 2.2 min wall on MPS
for all 5 tasks. SmolLM2 has 24 transformer layers (indices 0-23), so L23 is
the deepest available — there is no L24.

## Files

- `experiments/capability_dyf_router.py` — sweeps layer × K, prints all 4 columns
- `experiments/capability_extend_cache.py` — extends cache with L23
- `experiments/capability_probe_data.npz` — 84 keys, gitignored
- `experiments/capability_probe_phase0.md` — feasibility gate (passed)
- `experiments/capability_probe_phase1.md` — LogReg per-tier probes (mostly negative gain)

## Next

Pivot to Q2 (cross-task routing): collect ~100-200 multi-label prompts
spanning ~7 turnstyle categories, build dyf-leaf on L22-23, compare against
regex `_detect_task` baseline. See `docs/composition_and_activation_routing.md`
Level 3 Phase A.
