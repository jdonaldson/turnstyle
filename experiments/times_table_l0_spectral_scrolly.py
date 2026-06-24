"""Scroll-driven version of the layer-by-layer Laplacian animation.

Each scroll step = one layer (L0..L5).  As the section enters the
viewport, Plotly.animate() smoothly morphs the visible figure to
that layer's frame.  Operator switcher at the top swaps which figure
is sticky; the current scroll position determines which layer the
new figure jumps to.

The narrative text contains *clickable color pills* — words like
[max] [sum] [zero_flag] that, when clicked, recolor every point in
the figure by that feature.  This lets the reader probe the
"which axis is the cluster organizing around at this layer?"
question interactively.

Builds on:
  - times_table_l0_spectral_animation.py: per-op animated Plotly figure
  - times_table_story.py:                 scrollama + sticky-viz pattern

Usage:
    uv run python experiments/times_table_l0_spectral_scrolly.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import kneighbors_graph

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
OUT = ROOT / "l0_spectral_scrolly.html"

FILES = {
    "mul": ROOT / "hidden_states.npz",
    "add": ROOT / "hidden_states_add.npz",
    "sub": ROOT / "hidden_states_sub.npz",
}
N_LAYERS = 6
K_NN = 10
N_DIM = 3  # 2 -> classic u_1×u_2 plane; 3 -> u_1×u_2×u_3 cube
OPS = list(FILES.keys())

SIGN_CLASS_HEX = {-1: "#1f77b4", 0: "#7f7f7f", 1: "#d62728"}

# Per-op default coloring.  Add's sign_class is degenerate (only (0,0)
# has answer = 0), so zero_flag is the natural informative default.
OP_DEFAULT_FEATURE = {
    "mul": "sign_class",
    "add": "zero_flag",
    "sub": "sign_class",
}


# ─── Feature metadata (JS-side colorscale config) ──────────────────


FEATURE_META = {
    "a":           {"scale": "Viridis", "cmin": 0,  "cmax": 9,  "label": "a"},
    "b":           {"scale": "Viridis", "cmin": 0,  "cmax": 9,  "label": "b"},
    "min":         {"scale": "Viridis", "cmin": 0,  "cmax": 9,  "label": "min(a, b)"},
    "max":         {"scale": "Viridis", "cmin": 0,  "cmax": 9,  "label": "max(a, b)"},
    "sum":         {"scale": "Viridis", "cmin": 0,  "cmax": 18, "label": "a + b"},
    "abs_diff":    {"scale": "Viridis", "cmin": 0,  "cmax": 9,  "label": "|a − b|"},
    "signed_diff": {"scale": "RdBu_r",  "cmin": -9, "cmax": 9,  "label": "a − b"},
    "prod":        {"scale": "Viridis", "cmin": 0,  "cmax": 81, "label": "a × b"},
    "log_min":     {"scale": "Viridis", "cmin": 0,  "cmax": 2.31, "label": "log(min + 1)"},
    "log_max":     {"scale": "Viridis", "cmin": 0,  "cmax": 2.31, "label": "log(max + 1)"},
    "log_sum":     {"scale": "Viridis", "cmin": 0,  "cmax": 2.95, "label": "log(a + b + 1)"},
    "log_prod":    {"scale": "Viridis", "cmin": 0,  "cmax": 4.41, "label": "log(a × b + 1)"},
    "zero_flag":   {"scale": "Reds",    "cmin": 0,  "cmax": 1,  "label": "is zero pair?"},
    "sign_class":  {"kind": "categorical", "label": "answer sign class"},
    "multichar_answer": {"scale": "Reds", "cmin": 0, "cmax": 1,
                         "label": "answer is multi-character"},
    "predicted_char": {"kind": "categorical",
                       "label": "predicted next char at ="},
}

# Hex palette for the predicted_char categorical feature.
# 11 classes: digits 0-9 plus "-" (sub negatives only).
PRED_CHAR_HEX = {
    "0": "#7f7f7f",  # grey, matches sign_class=0
    "1": "#440154",  # Viridis-ish dark purple
    "2": "#482878",
    "3": "#3e4989",
    "4": "#31688e",
    "5": "#26828e",
    "6": "#1f9e89",
    "7": "#35b779",
    "8": "#6ece58",
    "9": "#b5de2b",
    "-": "#d62728",  # red — visually distinct from digits
}


# ─── Layer narratives, with clickable [feature] tags ───────────────


# Use the pattern [feature] (e.g. [max], [sum], [b], [signed_diff],
# [zero_flag], [sign_class]) inside the body strings.  The build step
# converts them into clickable pills.


LAYER_TEXT = {
    "mul": {
        0: ("L0 — Magnitude composite cluster",
            "<p>For mul, L0 is the (min, max) sub-lattice — operands "
            "encoded as an unordered set.  Swap pairs (a, b) and "
            "(b, a) collapse to within 8% of random-pair distance, so "
            "100 raw pairs project to ~55 distinct spots.</p>"
            "<p><strong>u_1</strong> is a magnitude composite: [prod] "
            "R²=0.86, [min] 0.84, [sum] 0.78, [log_prod] 0.69.  "
            "<strong>u_2</strong> picks up [zero_flag] (R²=0.62) — "
            "zero pairs sit along one edge of the cluster.  "
            "<strong>u_3</strong> carries [abs_diff] (R²=0.56) and "
            "[max] (0.36).</p>"
            "<p>The (u_1, u_2) plane is the classic V-shape: joint "
            "R² onto ([min], [max]) = 0.63.  Zero arm runs up u_2, "
            "main cluster runs right along u_1.</p>"
            "<p>Extremes of u_1 are already digit-grouped: leftmost "
            "10 are all zero pairs (predict <code>0</code>); rightmost "
            "10 are the largest products (54, 56, 63, 64, 72) which "
            "predict digits <strong>4–7</strong>.  Loosely clustered "
            "by digit, sharpening downstream.</p>"
            "<p>Click [multichar_answer] to see which 58 pairs will "
            "need a tens-digit prediction at L5 — at L0 they're "
            "spread through the magnitude axis, not yet grouped.  "
            "Click [log_prod] for the log-magnitude axis: u_2 lifts "
            "from R²=0.00 against raw [prod] to R²=0.23 against "
            "log_prod, the early signature of L0's role as the "
            "log encoder (causally: block 0's MLP decodes "
            "log(a+1)/log(b+1) at R²=0.99 by L1).</p>"),
        1: ("L1 — (min, max) grid rotates into (u_1, u_3)",
            "<p>L1 keeps <strong>u_1</strong> as the magnitude axis: "
            "[sum] R²=0.63, [prod] 0.62, [min] 0.60 — all roughly "
            "equal weight.  But <strong>u_2 empties out</strong>: no "
            "feature R² above 0.06 here.  <strong>u_3</strong> picks "
            "up [abs_diff] (R²=0.49) and [max] (0.47).</p>"
            "<p>The (min, max) grid that lived in (u_1, u_2) at L0 "
            "has <strong>rotated into (u_1, u_3)</strong>: joint R² = "
            "0.71, stronger than L0's 0.63.  Rotate the 3D view to "
            "put u_3 horizontal and the operand grid pops out "
            "clearly.</p>"
            "<p>What's going on: block 1's attention dispatched the "
            "operator (causal evidence — ≥91% operator-flip rate via "
            "activation patching across operators).  But that dispatch "
            "hasn't produced operator-specific computation yet.  "
            "Block 1's MLP is in a preparation state — operands "
            "log-encoded, operator known, no commitment.  "
            "Zero-detection, which was on u_2 at L0, has faded; it "
            "comes back at L2's annihilator phase.</p>"),
        2: ("L2 — Annihilator phase: zero detection emerges",
            "<p>At L2 zero detection shows up across the cluster: "
            "[zero_flag] R²=0.36 on <strong>u_2</strong>, R²=0.52 on "
            "<strong>u_3</strong>.  Magnitude still leads "
            "<strong>u_1</strong> ([sum] 0.51, [prod] 0.48, [min] "
            "0.45) but weaker than L0/L1.</p>"
            "<p>Joint R²(u_1, u_2) → ([min], [max]) drops to 0.42 — "
            "L0's V geometry is dissolving.  Block 2 attention is "
            "pulling operand information into a zero-vs-nonzero "
            "split.</p>"
            "<p>Note on what's hidden: in the raw per-layer Laplacian "
            "(unaligned), <em>u_1 IS zero_flag at L2 with R²=1.0</em>. "
            "Procrustes alignment to L0 has rotated that direction "
            "into u_2/u_3 to maintain smooth animation continuity.  "
            "What you see in the plot — the 19 zero pairs starting "
            "to peel into a sub-cluster in (u_2, u_3) — is that same "
            "Fiedler-cut event, just visible on different axes.</p>"),
        3: ("L3 — Annihilator gutter consolidated",
            "<p>[zero_flag] dominates <strong>u_3</strong> (R²=0.56) "
            "and is strong on <strong>u_2</strong> (R²=0.42).  "
            "<strong>u_1</strong> retains partial magnitude ([sum] "
            "0.49, [prod] 0.47, [min] 0.42) but is weakening.</p>"
            "<p>This is the <em>annihilator gutter</em> layer — a "
            "single 1-D direction (raw u_1 here) cleanly separates "
            "the 19 zero pairs from the 81 non-zero pairs with "
            "d'=8.5.  In aligned coords that gutter has rotated into "
            "u_3.</p>"
            "<p>Look at (u_2, u_3): zero pairs cluster in one corner; "
            "non-zero pairs form the bulk.  u_1's extremes still "
            "cluster by digit but loosely (top 10 predict 3–5, "
            "bottom 10 predict 1–7).</p>"),
        4: ("L4 — u_1 collapses, digit clustering survives",
            "<p><strong>u_1 has lost feature meaning here</strong> — "
            "top loading is [min] R²=0.04, [prod] 0.03.  The cluster "
            "has rotated nearly 90° from L0's frame; Procrustes is "
            "preserving the L0-aligned u_1 direction but no clean "
            "feature lives there anymore.</p>"
            "<p>[zero_flag] dominates <strong>u_3</strong> (R²=0.56) "
            "and <strong>u_2</strong> (R²=0.42).  Cluster geometry "
            "is mostly in the (u_2, u_3) plane.</p>"
            "<p>Despite u_1 being feature-empty, its extremes are "
            "<strong>pure single-digit clusters</strong>: top 10 by "
            "u_1 all predict <code>2</code>, bottom 10 all predict "
            "<code>1</code>.  The digit organization survives the "
            "eigenvector reordering — what changed is which axis "
            "carries it.  L4's MLP injects +1.57 along the correct-"
            "digit direction for typical (non-zero) pairs.</p>"),
        5: ("L5 — Digit corners are pure",
            "<p>Same structure as L4: <strong>u_1</strong> is "
            "feature-empty ([prod] R²=0.08), <strong>u_2/u_3</strong> "
            "carry [zero_flag] (R²=0.44, 0.53).</p>"
            "<p><strong>The digit corners are pure single-digit "
            "clusters:</strong></p>"
            "<ul>"
            "<li>Top 10 by u_1 all predict <code>2</code>; bottom 10 "
            "all predict <code>1</code></li>"
            "<li>Top 10 by u_3 all predict <code>0</code>; bottom 10 "
            "all predict <code>3</code></li>"
            "</ul>"
            "<p>Every extreme of the cluster is a single-digit pure "
            "region — daylight between digit groups.  Click "
            "[predicted_char] to see all 10 digit clusters at once.</p>"
            "<p>L5's MLP writes +6 on the correct digit row and "
            "−0.7 on each of the other 9.  Effective rank ~10 "
            "matches the 10 digit rows of the unembedding head.  The "
            "digit organization started loosely at L0 (extremes "
            "contained 4-value digit ranges) and tightened into pure "
            "single-digit corners by L5.</p>"),
    },
    "add": {
        0: ("L0 — Pre-operator (min, max) triangle",
            "<p>Same starting geometry as mul — L0 is operator-agnostic.  "
            "Joint R² of (u_1, u_2) onto ([min], [max]) = 0.79 for add, "
            "the strongest of any operator.</p>"
            "<p>u_1 loads on [prod] (R²=0.85), [min] (0.82), [sum] "
            "(0.79).  u_2 picks up [abs_diff] (R²=0.48) and [max] "
            "(0.36).  For add, [sign_class] only marks <em>one</em> "
            "point grey — (0, 0), the single pair where a + b = 0 — "
            "so coloring by sign_class makes everything else red.  "
            "Try [zero_flag] instead to see the 19 pairs where one "
            "operand is zero (those sit along the upper arm of the V).</p>"),
        1: ("L1 — Symmetric magnitude basis preserved",
            "<p>Add's L1 looks similar to L0 because addition's "
            "combination rule (a + b) is symmetric in (a, b) — "
            "no need to break operand symmetry yet.  u_1 ≈ [max] "
            "(0.81), [sum] (0.69), and the (min, max) joint R² is "
            "still 0.56.</p>"
            "<p>Caveat: the cross-operator dispatcher fires at block 1 "
            "(causal via patching), but within add's single-op view it "
            "doesn't appear as a basis flip — the dispatch keeps add's "
            "computation symmetric.</p>"),
        2: ("L2 — Shared zero detection",
            "<p>u_1 ≈ [zero_flag] (R² = 0.99).  Same Fiedler cut as mul "
            "and sub.  Add will <em>not</em> use this — addition's "
            "answer is [sum], not a zero/non-zero classification.  But "
            "the substrate is here.</p>"
            "<p>u_2 carries [max] (R²=0.55) and [sum] (0.55) — "
            "magnitude already on the second axis, ready to take over "
            "at L3.</p>"),
        3: ("L3 — Commit to <code>sum</code>",
            "<p>u_1 shifts to [sum] (R²=0.71); add's leading mode now "
            "tracks the answer directly.  [zero_flag] drops to u_2 "
            "(R²=0.36) — secondary, not primary.</p>"
            "<p>This is add's operator-specific commit: skip the zero "
            "detection branch, go straight to the magnitude axis.</p>"),
        4: ("L4 — Refinement on the magnitude axis",
            "<p>u_1 still on [sum] (R²=0.71).  Pairs spread along a "
            "1D-leaning ramp — answers 0..18 ordered along the "
            "principal coordinate.  Recolor by [prod] to see the "
            "secondary spread (correlated with sum, but distinct).</p>"),
        5: ("L5 — Linear contrastive digit selector",
            "<p>u_1 ≈ [sum] / [prod] composite (~0.6).  Add's L5 looks "
            "more continuous than mul's because [sum] doesn't quantize "
            "into discrete commit classes — answers form a 1D ramp, "
            "not a simplex.  Color by [max] to see the secondary "
            "spread on u_2.</p>"
            "<p>Recolor by [predicted_char] to see the digit-class "
            "structure.  Add's prediction distribution is heavily skewed: "
            "47 of 100 pairs have answer ≥ 10, so the first emitted char "
            "is <code>1</code> (the tens digit).  Pairs predicting "
            "<code>1</code> form one big cluster; the rest spread across "
            "<code>0</code>–<code>9</code>.</p>"),
    },
    "sub": {
        0: ("L0 — Pre-operator (min, max) cluster",
            "<p>Same starting geometry as mul / add — sub's L0 has "
            "<em>no signed information at all</em>: [signed_diff] "
            "has R² = 0.000 on every eigenvector.  Sign emerges "
            "downstream.</p>"
            "<p>u_1 loads on [prod] (0.86), [min] (0.85), [sum] "
            "(0.79).  u_2 picks up [zero_flag] (0.52).  Recolor by "
            "[sign_class]: positive (red) and negative (blue) "
            "answers are completely overlapping — the model hasn't "
            "decided yet which is which.  (a, b) and (b, a) "
            "collapse to nearly the same L0 point even though "
            "they'll produce opposite-sign answers downstream.</p>"
            "<p>Click [multichar_answer] to highlight the 45 "
            "pairs where a &lt; b — these all need a leading "
            "<code>−</code> at the output.  At L0 they're spread "
            "through the cluster by magnitude, indistinguishable "
            "from positive-answer pairs.  Sign emerges at L3+.</p>"),
        1: ("L1 — Sub diverges: u_1 locks onto operand b",
            "<p>This is the dramatic L1 case.  Sub IS the only "
            "anti-commutative operator (a − b ≠ b − a), so its "
            "downstream machinery has to know which operand is the "
            "subtractor.  At L1, u_1 = [b] with R² = 0.78 — the "
            "subtractor has been isolated as a coordinate.</p>"
            "<p>[a] only loads at R²=0.00; [max] at 0.42.  Sub's L1 "
            "is selectively pulling out one operand's identity.  "
            "Compare to mul/add at L1, where neither [a] nor [b] "
            "individually dominates — only their symmetric "
            "combinations do.</p>"
            "<p>[signed_diff] is still R²=0.07 here; the sign answer "
            "itself emerges later.</p>"),
        2: ("L2 — Zero detection overlaid on the b-axis",
            "<p>u_1 = [b] still (R²=0.69), but [zero_flag] is "
            "building up on u_2 (R²=0.17) and other axes.  The shared "
            "L2 zero-detection event is happening, but sub's b-axis "
            "from L1 hasn't been overwritten yet.</p>"
            "<p>[signed_diff] has crept up to R²=0.17 on u_2 — sign "
            "information is starting to appear.</p>"),
        3: ("L3 — Commit to <code>signed_diff</code>",
            "<p>u_1 = [signed_diff] with R² = 0.76.  Joint R² for "
            "([min], [max]) collapses <strong>0.65 → 0.05</strong>.  "
            "The leading plane has shifted decisively off the "
            "symmetric basis.  u_2 picks up [zero_flag] (R²=0.76).</p>"
            "<p>This is sub's operator-specific commit.  Recolor by "
            "[sign_class] — the three answer-sign clusters are now "
            "visible.</p>"),
        4: ("L4 — Three answer-sign clusters separate",
            "<p>Refinement of the [signed_diff] axis (R²=0.71).  "
            "u_2 picks up [max] (0.36) and [abs_diff] (0.22) — "
            "distance from the a=b diagonal stretches each arm.  "
            "Positive, zero, and negative answer classes tighten "
            "into distinct clusters.</p>"),
        5: ("L5 — Three-vertex simplex = three predicted characters",
            "<p>Between/within cluster ratio = <strong>10.5×</strong>.  "
            "The three [sign_class] classes form a 2-simplex — the "
            "neural-collapse / ETF pattern.  u_1 ≈ [signed_diff] "
            "(R²=0.74); u_2 ≈ [abs_diff] (R²=0.24).</p>"
            "<p>Recolor by [predicted_char] for the semantic kicker: "
            "the three vertices correspond <em>exactly</em> to three "
            "predicted-next-character classes — the model emits "
            "<code>−</code> at the negative vertex (45 pairs), "
            "<code>0</code> at the zero vertex (10 pairs), and digits "
            "<code>1</code>–<code>9</code> at the positive vertex (45 "
            "pairs).  Sub is the only operator that ever emits a "
            "non-digit; mul and add never do.</p>"
            "<p>The 10 pairs at the zero vertex are the "
            "<strong>diagonal pairs</strong>: (0,0), (1,1), (2,2), "
            "…, (9,9), since a − b = 0 iff a = b.  At L0 these "
            "diagonals were scattered through the main cluster by "
            "magnitude; L3 onward they consolidate into this "
            "distinct vertex.  Click [multichar_answer] to see "
            "the complementary 45 pairs (a &lt; b) at the negative "
            "vertex — their predicted first char is "
            "<code>−</code>.</p>"
            "<p>Within the positive vertex, individual digit identity "
            "is in u_3+: L5 has effective rank ~10 (matching the 10 "
            "digit rows of the unembedding head), with the higher "
            "eigenvectors carrying digit-specific structure.  In 3D "
            "you can rotate to see whether u_3 starts pulling out "
            "that digit structure.</p>"),
    },
}


# ─── Spectral helpers ──────────────────────────────────────────────


def load_stack(path):
    """Load (n_layers, n_pairs, d) tensor with pairs in canonical
    lexicographic (a, b) order so all three operators share the same
    point ordering — required for Procrustes alignment across ops."""
    d = np.load(path)
    layers_arr = d["layer"]
    sel0 = layers_arr == 0
    base_a, base_b = d["a"][sel0], d["b"][sel0]
    sort_idx = np.lexsort((base_b, base_a))
    base_a, base_b = base_a[sort_idx], base_b[sort_idx]
    n_pairs = len(base_a)
    H_stack = np.zeros((N_LAYERS, n_pairs, d["H"].shape[1]))
    for L in range(N_LAYERS):
        sel = layers_arr == L
        aL, bL, HL = d["a"][sel], d["b"][sel], d["H"][sel]
        order = []
        for ai, bi in zip(base_a, base_b):
            order.append(int(np.where((aL == ai) & (bL == bi))[0][0]))
        H_stack[L] = HL[np.array(order)]
    return H_stack, base_a, base_b


def lap_eigs(H, n_dim=N_DIM, k=K_NN):
    """Top n_dim non-trivial Laplacian eigenvectors of kNN graph.
    Returns shape (n_pairs, n_dim)."""
    A = kneighbors_graph(H, n_neighbors=k, mode="connectivity",
                        include_self=False)
    A = (A + A.T).maximum(A.T)
    L = laplacian(A, normed=True)
    Ld = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
    _, v = np.linalg.eigh(Ld)
    # Skip eigenvector 0 (trivial constant, eigenvalue ~0).
    return v[:, 1:1 + n_dim].copy()


def procrustes_R(X, Y):
    """Orthogonal d×d R such that X @ R best matches Y (Frobenius).
    Works for any dimension — used in 2D and 3D."""
    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def compute_joint_coords(H_stacks_per_op, n_dim=N_DIM):
    """Pool all ops at each layer and compute the joint Laplacian.

    This is the projection that makes operator-population separation
    visible: 100% 5-fold 1-NN op-recovery at L0, decaying to ~chance
    by L5.  Different from per-op compute_aligned_coords (which is for
    within-op structure).

    H_stacks_per_op: ordered dict {op: (n_layers, n_pairs, d)}.
    Returns: (n_layers, n_pairs * n_ops, n_dim), Procrustes-aligned
    cumulatively across layers for smooth animation.
    """
    ops = list(H_stacks_per_op.keys())
    n_layers, n_pairs, _ = H_stacks_per_op[ops[0]].shape
    n_total = n_pairs * len(ops)
    coords = np.zeros((n_layers, n_total, n_dim))

    for L in range(n_layers):
        H_joint = np.concatenate(
            [H_stacks_per_op[op][L] for op in ops], axis=0)
        X = lap_eigs(H_joint, n_dim=n_dim)
        X = X - X.mean(axis=0)
        scale = np.linalg.norm(X) / np.sqrt(n_dim * n_total) + 1e-12
        coords[L] = X / scale

    # Procrustes across layers (smooth animation)
    for L in range(1, n_layers):
        R = procrustes_R(coords[L], coords[L - 1])
        coords[L] = coords[L] @ R
    return coords


def split_joint_coords(joint_coords, n_pairs, ops=("mul", "add", "sub")):
    """Slice (n_layers, n_pairs * n_ops, n_dim) back into per-op chunks."""
    return {op: joint_coords[:, i * n_pairs:(i + 1) * n_pairs, :]
            for i, op in enumerate(ops)}


def compute_aligned_coords(H_stack, ref=None, n_dim=N_DIM):
    """Per-layer Laplacian projection in n_dim with alignment passes:

    1. Each layer is centered and total-Frobenius-normalized to
       sqrt(n_dim * n_pairs) so per-layer scale is consistent.
    2. Within-op: Procrustes-align L=1..5 cumulatively to L=L-1, so
       there's no spurious rotation between consecutive frames.
    3. Cross-op (if ref given): Procrustes-align L=0 to the reference
       and propagate the same orthogonal R to every layer.

    Returns shape (n_layers, n_pairs, n_dim).
    """
    n_layers, n_pairs, _ = H_stack.shape
    coords = np.zeros((n_layers, n_pairs, n_dim))

    # Pass 1
    for L in range(n_layers):
        X = lap_eigs(H_stack[L], n_dim=n_dim)
        X = X - X.mean(axis=0)
        scale = np.linalg.norm(X) / np.sqrt(n_dim * n_pairs) + 1e-12
        coords[L] = X / scale

    # Pass 2: within-op
    for L in range(1, n_layers):
        R = procrustes_R(coords[L], coords[L - 1])
        coords[L] = coords[L] @ R

    # Pass 3: cross-op
    if ref is not None:
        R0 = procrustes_R(coords[0], ref)
        for L in range(n_layers):
            coords[L] = coords[L] @ R0

    return coords


def answer_sign_class(op, a, b):
    if op == "mul":
        return np.where((a * b) == 0, 0, 1)
    if op == "add":
        return np.where((a + b) == 0, 0, 1)
    if op == "sub":
        return np.sign(a - b).astype(int)
    raise ValueError(op)


_PRED_CACHE_PATH = ROOT / "predicted_chars.json"


def load_predictions():
    """Return {op: {(a, b) -> predicted_char}} from the cached JSON
    written by times_table_predict_chars.py.  Returns empty dict if
    the cache doesn't exist (predicted_char feature will be unavailable)."""
    if not _PRED_CACHE_PATH.exists():
        return {}
    raw = json.loads(_PRED_CACHE_PATH.read_text())
    out = {}
    for op, info in raw.items():
        out[op] = {}
        for key, ch in info["predicted"].items():
            ai, bi = [int(x) for x in key.split(",")]
            out[op][(ai, bi)] = ch
    return out


PREDS = load_predictions()


def _multichar_answer(op, a, b):
    """Binary: is the predicted answer multi-character?
       mul: a*b >= 10 (predicted char is the tens digit, then ones digit)
       add: a+b >= 10 (predicted char is '1', then ones digit)
       sub: a < b  (predicted char is '-', then digit)
       else: single-character answer (predicted char IS the answer)."""
    if op == "mul":
        return (a * b >= 10).astype(int)
    if op == "add":
        return (a + b >= 10).astype(int)
    if op == "sub":
        return (a < b).astype(int)
    raise ValueError(op)


def compute_features(op, a, b):
    feats = {
        "a":           a.astype(float).tolist(),
        "b":           b.astype(float).tolist(),
        "min":         np.minimum(a, b).astype(float).tolist(),
        "max":         np.maximum(a, b).astype(float).tolist(),
        "sum":         (a + b).astype(float).tolist(),
        "abs_diff":    np.abs(a - b).astype(float).tolist(),
        "signed_diff": (a - b).astype(float).tolist(),
        "prod":        (a * b).astype(float).tolist(),
        "log_min":     np.log(np.minimum(a, b) + 1).astype(float).tolist(),
        "log_max":     np.log(np.maximum(a, b) + 1).astype(float).tolist(),
        "log_sum":     np.log(a + b + 1).astype(float).tolist(),
        "log_prod":    np.log(a * b + 1).astype(float).tolist(),
        "zero_flag":   ((a == 0) | (b == 0)).astype(int).tolist(),
        "sign_class":  answer_sign_class(op, a, b).astype(int).tolist(),
        "multichar_answer": _multichar_answer(op, a, b).astype(int).tolist(),
    }
    if op in PREDS:
        feats["predicted_char"] = [PREDS[op][(int(ai), int(bi))]
                                   for ai, bi in zip(a, b)]
    return feats


def hex_for_sign_class(op, a, b):
    cls = answer_sign_class(op, a, b)
    return [SIGN_CLASS_HEX[int(c)] for c in cls]


def hex_for_predicted_char(op, a, b):
    if op not in PREDS:
        return None
    return [PRED_CHAR_HEX[PREDS[op][(int(ai), int(bi))]]
            for ai, bi in zip(a, b)]


# ─── Per-operator scroll figure (single-trace frames) ──────────────


def build_scroll_figure(op, coords, a, b):
    """Single-trace-per-frame; 2D or 3D depending on N_DIM.

    coords has shape (n_layers, n_pairs, N_DIM).
    """
    hover = [f"({ai}, {bi})" for ai, bi in zip(a, b)]
    default_feat = OP_DEFAULT_FEATURE[op]
    if default_feat == "sign_class":
        init_colors = hex_for_sign_class(op, a, b)
    else:
        feats = compute_features(op, a, b)
        init_colors = feats[default_feat]

    # Global axis ranges spanning all layers, with padding.
    flat = coords.reshape(-1, N_DIM)
    pad = 0.10
    ranges = []
    for d in range(N_DIM):
        lo, hi = flat[:, d].min(), flat[:, d].max()
        spread = hi - lo
        ranges.append([lo - pad * spread, hi + pad * spread])

    def make_marker(colors):
        m = dict(size=7 if N_DIM == 3 else 11,
                 color=colors,
                 line=dict(width=0.4, color="black"))
        meta = FEATURE_META[default_feat]
        if meta.get("kind") != "categorical":
            m["colorscale"] = meta["scale"]
            m["cmin"] = meta["cmin"]
            m["cmax"] = meta["cmax"]
            m["showscale"] = True
            m["colorbar"] = dict(
                title=dict(text=meta["label"], font=dict(size=11)),
                len=0.6, thickness=12, x=1.05, y=0.5,
            )
        return m

    def make_trace(L, colors):
        if N_DIM == 3:
            return go.Scatter3d(
                x=coords[L, :, 0].tolist(),
                y=coords[L, :, 1].tolist(),
                z=coords[L, :, 2].tolist(),
                mode="markers",
                marker=make_marker(colors),
                text=hover, hovertemplate="(%{text})<extra></extra>",
                showlegend=False,
            )
        return go.Scatter(
            x=coords[L, :, 0].tolist(),
            y=coords[L, :, 1].tolist(),
            mode="markers",
            marker=make_marker(colors),
            text=hover, hovertemplate="(%{text})<extra></extra>",
            showlegend=False,
        )

    frames = [go.Frame(data=[make_trace(L, init_colors)], name=f"L{L}")
              for L in range(N_LAYERS)]

    if N_DIM == 3:
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title="u_1", range=ranges[0],
                           backgroundcolor="white", gridcolor="#e6e6e6",
                           zeroline=False, showspikes=False),
                yaxis=dict(title="u_2", range=ranges[1],
                           backgroundcolor="white", gridcolor="#e6e6e6",
                           zeroline=False, showspikes=False),
                zaxis=dict(title="u_3", range=ranges[2],
                           backgroundcolor="white", gridcolor="#e6e6e6",
                           zeroline=False, showspikes=False),
                aspectmode="cube",
                camera=dict(eye=dict(x=1.55, y=1.55, z=1.20)),
                bgcolor="white",
            ),
            height=620, autosize=True,
            margin=dict(l=0, r=0, t=10, b=0),
        )
    else:
        layout = go.Layout(
            xaxis=dict(title="u_1 (normalized)", range=ranges[0],
                       showgrid=True, gridcolor="#f0f0f0", zeroline=False),
            yaxis=dict(title="u_2 (normalized)", range=ranges[1],
                       showgrid=True, gridcolor="#f0f0f0", zeroline=False),
            plot_bgcolor="white",
            height=560, autosize=True,
            margin=dict(l=70, r=110, t=20, b=60),
        )

    fig = go.Figure(
        data=[make_trace(0, init_colors)],
        frames=frames,
        layout=layout,
    )
    return fig


def compute_hex_colors_for_feature(op, feature, values):
    """Pre-compute hex colors per pair for a given (op, feature).
    Used by canvas-based minimaps that don't have access to Plotly's
    colorscale machinery in JS."""
    import plotly.colors as pc
    meta = FEATURE_META[feature]
    if meta.get("kind") == "categorical":
        if feature == "sign_class":
            return [SIGN_CLASS_HEX[int(v)] for v in values]
        if feature == "predicted_char":
            return [PRED_CHAR_HEX[v] for v in values]
        return ["#888"] * len(values)
    cmin, cmax = meta["cmin"], meta["cmax"]
    scale_name = meta["scale"]
    out = []
    for v in values:
        if cmax > cmin:
            norm = (float(v) - cmin) / (cmax - cmin)
        else:
            norm = 0.5
        norm = max(0.0, min(1.0, norm))
        rgb = pc.sample_colorscale(scale_name, [norm])[0]
        out.append(rgb)
    return out


def compute_all_hex_colors(feature_data):
    """Returns {op: {feature: [hex strings per pair]}} for canvas use."""
    out = {}
    for op, feats in feature_data.items():
        out[op] = {}
        for feat_name, vals in feats.items():
            if feat_name not in FEATURE_META:
                continue
            out[op][feat_name] = compute_hex_colors_for_feature(
                op, feat_name, vals)
    return out


def compute_minimap_ranges(coords_by_op):
    """Per-op u_1/u_2 ranges across all layers, for canvas axes.
    Returns {op: {x: [lo, hi], y: [lo, hi]}}."""
    out = {}
    for op, coords in coords_by_op.items():
        x = coords[:, :, 0].flatten()
        y = coords[:, :, 1].flatten()
        xpad = 0.05 * (x.max() - x.min())
        ypad = 0.05 * (y.max() - y.min())
        out[op] = {
            "x": [float(x.min() - xpad), float(x.max() + xpad)],
            "y": [float(y.min() - ypad), float(y.max() + ypad)],
        }
    return out


def build_k_sensitivity_image():
    """Show the L0 mul Laplacian projection at four values of k:
    k=5 (graph disconnected), k=10 (stable / what we use),
    k=20 (still connected but starting to blur), k=50 (over-connected,
    structure washed out).  Makes the k-choice tradeoff visible."""
    import io
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.sparse.csgraph import connected_components as cc

    H_stack, a, b = load_stack(FILES["mul"])
    H = H_stack[0]

    zero_m = (a == 0) | (b == 0)
    diag_m = (a == b) & ~zero_m
    other_m = ~zero_m & ~diag_m

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), facecolor="white")
    for ax, k in zip(axes, [5, 10, 20, 50]):
        A = kneighbors_graph(H, n_neighbors=k, mode="connectivity",
                            include_self=False)
        A = (A + A.T).maximum(A.T)
        n_cc, _ = cc(A, directed=False)
        L = laplacian(A, normed=True)
        Ld = L.toarray()
        _, v = np.linalg.eigh(Ld)
        u1, u2 = v[:, 1], v[:, 2]

        ax.scatter(u1[other_m], u2[other_m], c="#1f77b4", s=22, lw=0)
        ax.scatter(u1[diag_m], u2[diag_m], c="#2ca02c", s=32, lw=0.3,
                   edgecolor="k")
        ax.scatter(u1[zero_m], u2[zero_m], c="#d62728", s=32, lw=0.3,
                   edgecolor="k")
        comp_note = (f"{n_cc} components — broken" if n_cc > 1
                     else "connected ✓")
        ax.set_title(f"k = {k}  ({comp_note})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#fafafa")
        for s in ax.spines.values():
            s.set_color("#ccc")

    fig.suptitle(
        "How k affects the L0 projection (mul) — V-shape stable for k = 8..15",
        y=1.04, fontsize=12,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_knn_graph_image():
    """Show the actual kNN graph on mul L0: 100 nodes (pairs),
    edges between each node and its 10 nearest neighbors in 128-dim
    space, laid out using the Laplacian's own 2D projection so the
    cluster shape and graph topology are both visible."""
    import io
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H_stack, a, b = load_stack(FILES["mul"])
    H = H_stack[0]
    lap = lap_eigs(H, n_dim=2)

    A = kneighbors_graph(H, n_neighbors=K_NN, mode="connectivity",
                        include_self=False)
    A = (A + A.T).maximum(A.T).tocoo()

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor="white")
    ax.set_facecolor("#fafafa")

    # Edges first, behind the points
    for i, j in zip(A.row, A.col):
        if i < j:
            ax.plot(
                [lap[i, 0], lap[j, 0]],
                [lap[i, 1], lap[j, 1]],
                color="#a0a4ac", linewidth=0.45, alpha=0.55, zorder=1,
            )

    zero_m = (a == 0) | (b == 0)
    diag_m = (a == b) & ~zero_m
    other_m = ~zero_m & ~diag_m
    ax.scatter(lap[other_m, 0], lap[other_m, 1], c="#1f77b4", s=55,
               lw=0.4, edgecolor="k", zorder=2,
               label="off-diagonal interior")
    ax.scatter(lap[diag_m, 0], lap[diag_m, 1], c="#2ca02c", s=70,
               lw=0.4, edgecolor="k", zorder=2,
               label="a = b (diagonal arm)")
    ax.scatter(lap[zero_m, 0], lap[zero_m, 1], c="#d62728", s=70,
               lw=0.4, edgecolor="k", zorder=2,
               label="min = 0 (zero arm)")

    for label, (av, bv) in [("(0,0)", (0, 0)),
                            ("(0,9)", (0, 9)),
                            ("(9,9)", (9, 9))]:
        idx = np.where((a == av) & (b == bv))[0]
        if len(idx):
            i = idx[0]
            ax.annotate(
                label, xy=(lap[i, 0], lap[i, 1]),
                xytext=(lap[i, 0] + 0.011, lap[i, 1] + 0.008),
                fontsize=11, fontweight="bold",
            )

    ax.set_title(
        "kNN graph (k = 10) on mul L0 hidden states — "
        "100 nodes, ~1000 edges",
        fontsize=12,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_method_comparison_image():
    """Generate the UMAP / PCA / Laplacian three-panel comparison on
    mul L0 as a base64-embedded PNG.  Used in the intro section to
    motivate the Laplacian choice over UMAP/PCA."""
    import io
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap as umap_lib

    H_stack, a, b = load_stack(FILES["mul"])
    H = H_stack[0]

    Hc = H - H.mean(axis=0)
    U, S, _ = np.linalg.svd(Hc, full_matrices=False)
    pca = U[:, :2] * S[:2]
    um = umap_lib.UMAP(n_components=2, random_state=42).fit_transform(H)
    lap = lap_eigs(H, n_dim=2)

    zero_m = (a == 0) | (b == 0)
    diag_m = (a == b) & ~zero_m
    offdiag_m = (a != b) & ~zero_m & (np.abs(a - b) >= 3)
    mid_m = ~zero_m & ~diag_m & ~offdiag_m

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor="white")

    def panel(ax, X, title):
        ax.scatter(X[offdiag_m, 0], X[offdiag_m, 1], c="#1f77b4", s=20, lw=0)
        ax.scatter(X[mid_m, 0], X[mid_m, 1], c="#7f7f7f", s=20, lw=0)
        ax.scatter(X[diag_m, 0], X[diag_m, 1], c="#2ca02c", s=28, lw=0.3,
                   edgecolor="k")
        ax.scatter(X[zero_m, 0], X[zero_m, 1], c="#d62728", s=28, lw=0.3,
                   edgecolor="k")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#fafafa")
        for spine in ax.spines.values():
            spine.set_color("#ccc")

    panel(axes[0], um, "UMAP  (n_neighbors=15, min_dist=0.1)")
    panel(axes[1], pca, "PCA  (PC1 × PC2)")
    panel(axes[2], lap, "Laplacian  (u₁ × u₂)")
    fig.suptitle(
        "Same 100 mul L0 hidden states, three projections",
        y=1.02, fontsize=13,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_og_image_matplotlib(joint_per_op, feature_data, out_path):
    """Render the stacked-3D view as a 1200x630 PNG for OG/social previews.

    Same data as the live Plotly figure (joint Laplacian, all 3 ops ×
    all 6 layers), rendered via matplotlib 3D for reliable static export.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    BG = "#252a31"
    GRID = "#3d4148"
    TEXT = "#cfd3da"
    OP_COLOR = {"mul": "#1f77b4", "add": "#ff7f0e", "sub": "#2ca02c"}
    OP_MARKER = {"mul": "o", "add": "s", "sub": "D"}

    sample = joint_per_op["mul"]
    per_layer_spread = np.array([
        np.linalg.norm(sample[L] - sample[L].mean(axis=0), axis=1).max()
        for L in range(N_LAYERS)
    ])
    z_spacing = float(per_layer_spread.max() * 2.0)

    fig = plt.figure(figsize=(12, 6.3), facecolor=BG, dpi=100)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)

    for op, coords in joint_per_op.items():
        n_pairs = coords.shape[1]
        all_xs, all_ys, all_zs = [], [], []
        for i in range(n_pairs):
            xs = coords[:, i, 0]
            ys = coords[:, i, 1]
            zs = np.arange(N_LAYERS) * z_spacing
            ax.plot(xs, ys, zs, color=OP_COLOR[op], alpha=0.16, lw=0.7)
            all_xs.append(xs); all_ys.append(ys); all_zs.append(zs)
        all_xs = np.concatenate(all_xs)
        all_ys = np.concatenate(all_ys)
        all_zs = np.concatenate(all_zs)
        ax.scatter(all_xs, all_ys, all_zs, c=OP_COLOR[op],
                   marker=OP_MARKER[op], s=10, alpha=0.65,
                   edgecolors="none", label=op)

    ax.set_zticks([L * z_spacing for L in range(N_LAYERS)])
    ax.set_zticklabels([f"L{L}" for L in range(N_LAYERS)],
                       color=TEXT, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.14, 0.16, 0.19, 1.0))
        pane._axinfo["grid"]["color"] = GRID
    ax.set_xlabel("u₁", color=TEXT, fontsize=11, labelpad=-12)
    ax.set_ylabel("u₂", color=TEXT, fontsize=11, labelpad=-12)
    ax.tick_params(colors=TEXT)
    ax.view_init(elev=12, azim=42)
    ax.set_box_aspect((1.2, 1.2, 1.4))

    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02),
                    ncol=3, framealpha=0.85, facecolor=BG,
                    edgecolor="#444", fontsize=12,
                    labelcolor=TEXT, markerscale=2.0)
    for txt in leg.get_texts():
        txt.set_color(TEXT)

    fig.text(0.5, 0.96,
             "NanoGPT arithmetic — Laplacian eigenvector trajectories",
             color="white", ha="center", fontsize=18, weight="bold")
    fig.text(0.5, 0.92,
             "All three operators across six layers · joint graph "
             "Laplacian · jdonaldson.github.io/nanogpt-arithmetic-viz",
             color=TEXT, ha="center", fontsize=11)

    fig.savefig(out_path, dpi=100, facecolor=BG)
    plt.close(fig)


def build_stacked_3d_figure(joint_per_op, feature_data):
    """Standalone 3D scatter: all 3 ops × all 6 layers, layer on z-axis.

    Each operator becomes one trace (toggleable via legend).  All 600
    points per op are plotted at the joint-Laplacian (u_1, u_2) coords,
    with z = layer * spacing so each layer becomes a visible slice.
    """
    OP_COLORS_PY = {"mul": "#1f77b4", "add": "#ff7f0e", "sub": "#2ca02c"}
    OP_SYMBOLS_PY = {"mul": "circle", "add": "square", "sub": "diamond"}

    # Auto-compute spacing: ~2x the per-layer cluster diameter
    sample_coords = joint_per_op["mul"]
    per_layer_spread = np.array([
        np.linalg.norm(sample_coords[L] - sample_coords[L].mean(axis=0),
                       axis=1).max()
        for L in range(N_LAYERS)
    ])
    z_spacing = float(per_layer_spread.max() * 2.0)

    fig = go.Figure()

    for op, coords in joint_per_op.items():
        a = feature_data[op]["a"]
        b = feature_data[op]["b"]
        n_pairs = len(a)

        # Interleave per-pair so each pair becomes a 6-point polyline
        # (L0 → L1 → ... → L5).  Use None separators to break the
        # polyline between pairs.
        xs, ys, zs, hover = [], [], [], []
        for i in range(n_pairs):
            for L in range(N_LAYERS):
                xs.append(float(coords[L, i, 0]))
                ys.append(float(coords[L, i, 1]))
                zs.append(L * z_spacing)
                hover.append(f"{op}  ({int(a[i])}, {int(b[i])})  L{L}")
            # Break polyline before next pair
            xs.append(None); ys.append(None); zs.append(None)
            hover.append("")

        base_hex = OP_COLORS_PY[op]
        rr = int(base_hex[1:3], 16)
        gg = int(base_hex[3:5], 16)
        bb = int(base_hex[5:7], 16)
        line_rgba = f"rgba({rr}, {gg}, {bb}, 0.32)"

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines+markers",
            name=op,
            marker=dict(
                size=2.8,
                color=base_hex,
                symbol=OP_SYMBOLS_PY[op],
                opacity=0.75,
                line=dict(width=0),
            ),
            line=dict(color=line_rgba, width=1.8),
            connectgaps=False,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    # Layer labels along z-axis (light text for dark background)
    layer_annotations = [
        dict(
            x=0, y=0, z=L * z_spacing,
            text=f"<b>L{L}</b>",
            showarrow=False,
            font=dict(size=14, color="#e0e0e0"),
            xanchor="center",
        )
        for L in range(N_LAYERS)
    ]

    AX_LABEL = "#cfd3da"  # light grey for axis titles/ticks
    GRID = "#3d4148"      # subtle grid on dark
    WALL = "#1f232a"      # scene walls slightly darker than scene bg
    SCENE_BG = "#252a31"  # main scene background

    fig.update_layout(
        title=dict(
            text=("All operators × all layers — joint Laplacian, "
                  "stacked by layer on z-axis"),
            x=0.02, font=dict(size=15, color=AX_LABEL),
        ),
        scene=dict(
            xaxis=dict(title=dict(text="u_1", font=dict(color=AX_LABEL)),
                       backgroundcolor=WALL, gridcolor=GRID,
                       zeroline=False, tickfont=dict(color=AX_LABEL),
                       showspikes=False),
            yaxis=dict(title=dict(text="u_2", font=dict(color=AX_LABEL)),
                       backgroundcolor=WALL, gridcolor=GRID,
                       zeroline=False, tickfont=dict(color=AX_LABEL),
                       showspikes=False),
            zaxis=dict(title=dict(text="layer (L0 → L5)", font=dict(color=AX_LABEL)),
                       tickmode="array",
                       tickvals=[L * z_spacing for L in range(N_LAYERS)],
                       ticktext=[f"L{L}" for L in range(N_LAYERS)],
                       backgroundcolor=WALL, gridcolor=GRID,
                       zeroline=False, tickfont=dict(color=AX_LABEL),
                       showspikes=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.6),
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.7)),
            bgcolor=SCENE_BG,
            annotations=layer_annotations,
        ),
        height=720, autosize=True,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="top", y=1.02,
                    xanchor="right", x=1.0,
                    title=dict(text="operator (click to toggle)",
                               font=dict(color=AX_LABEL)),
                    font=dict(color=AX_LABEL),
                    bgcolor="rgba(37, 42, 49, 0.7)"),
        paper_bgcolor=SCENE_BG,
    )
    return fig


def featurize_text(html: str) -> str:
    """Convert [feature] into clickable color pills.  Recognizes only
    feature names in FEATURE_META, leaves other bracketed text alone."""
    import re
    pattern = re.compile(r"\[(" + "|".join(FEATURE_META.keys()) + r")\]")

    def repl(m):
        feat = m.group(1)
        label = FEATURE_META[feat].get("label", feat)
        return (f'<span class="color-pill" data-feature="{feat}" '
                f'title="Color by {label}">{feat}</span>')

    return pattern.sub(repl, html)


# ─── HTML page template ────────────────────────────────────────────


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>NanoGPT arithmetic — Laplacian eigenvector evolution</title>
<meta name="description" content="How a 1.2M-parameter char-level NanoGPT organizes arithmetic across its 6 layers, projected via graph Laplacian eigenvectors. Interactive visualization with scroll-driven layer animation.">
<meta property="og:type" content="article">
<meta property="og:title" content="NanoGPT arithmetic — Laplacian eigenvector evolution">
<meta property="og:description" content="How a 1.2M-parameter char-level NanoGPT organizes arithmetic across its 6 layers, projected via graph Laplacian eigenvectors. Interactive 3D visualization.">
<meta property="og:image" content="https://jdonaldson.github.io/nanogpt-arithmetic-viz/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:url" content="https://jdonaldson.github.io/nanogpt-arithmetic-viz/">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="NanoGPT arithmetic — Laplacian eigenvector evolution">
<meta name="twitter:description" content="How a 1.2M-parameter char-level NanoGPT organizes arithmetic across its 6 layers. Interactive 3D visualization.">
<meta name="twitter:image" content="https://jdonaldson.github.io/nanogpt-arithmetic-viz/og.png">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://unpkg.com/scrollama@3.2.0"></script>
<style>
  :root {{ --accent: #2643d4; --pill-bg: #eef2ff; --pill-bd: #d4ddf5; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue",
                 Arial, sans-serif;
    color: #222; background: #fafafa; margin: 0; line-height: 1.55;
  }}
  header {{
    max-width: 1400px; margin: 0 auto; padding: 36px 32px 4px;
  }}
  header h1 {{ font-size: 26px; margin: 0 0 4px; }}
  header .sub {{ color: #555; font-size: 14px; max-width: 800px; }}
  .controls {{
    max-width: 1400px; margin: 12px auto 8px; padding: 0 32px;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    position: sticky; top: 0; z-index: 50;
    background: rgba(250, 250, 250, 0.97);
    backdrop-filter: blur(6px);
    padding-top: 10px; padding-bottom: 10px;
    border-bottom: 1px solid #e6e6e6;
  }}
  .controls span.label {{ color: #555; font-size: 13px; }}
  .controls button.op {{
    border: 1px solid #ccc; background: white; padding: 6px 16px;
    border-radius: 4px; cursor: pointer; font-size: 13.5px;
    font-family: inherit; color: #333;
  }}
  .controls button.op.active {{
    background: var(--accent); color: white; border-color: var(--accent);
  }}
  .controls .sep {{
    width: 1px; height: 24px; background: #ccc; margin: 0 4px;
  }}
  .controls .status {{
    margin-left: auto; color: #666; font-size: 12.5px;
  }}
  .controls .status b {{ color: var(--accent); }}
  .scrolly {{
    display: grid; grid-template-columns: 1fr 1.3fr; gap: 32px;
    max-width: 1500px; margin: 0 auto; padding: 0 32px;
  }}
  .story-col {{ padding-top: 4vh; }}
  .step {{
    min-height: 85vh; padding: 32px 0 35vh 0;
    opacity: 0.38; transition: opacity 0.4s ease;
  }}
  .step.is-active {{ opacity: 1; }}
  .step h2 {{
    border-bottom: 1px solid #e3e3e3; padding-bottom: 8px;
    font-size: 21px; color: #1a1a1a; margin-top: 0;
  }}
  .step p {{ font-size: 15px; }}
  .step .layer-tag {{
    display: inline-block;
    background: var(--accent); color: white;
    font-size: 11px; padding: 2px 8px; border-radius: 3px;
    margin-bottom: 8px; letter-spacing: 0.5px;
  }}
  .color-pill {{
    display: inline-block;
    padding: 0 8px;
    border-radius: 3px;
    background: var(--pill-bg);
    border: 1px solid var(--pill-bd);
    color: var(--accent);
    font-size: 13px;
    font-family: SF Mono, Menlo, Monaco, monospace;
    cursor: pointer;
    transition: all 0.12s ease;
    margin: 0 1px;
    user-select: none;
  }}
  .color-pill:hover {{
    background: var(--accent); color: white; border-color: var(--accent);
  }}
  .color-pill.active {{
    background: var(--accent); color: white; border-color: var(--accent);
    box-shadow: 0 1px 3px rgba(38, 67, 212, 0.3);
  }}
  .viz-col {{
    position: sticky; top: 60px; height: calc(100vh - 80px);
    display: flex; flex-direction: row; align-items: stretch;
    padding: 12px 0;
    gap: 8px;
  }}
  .viz-main {{
    flex: 1; min-width: 0;
    display: flex; flex-direction: column;
    justify-content: center;
  }}
  .minimaps {{
    width: 110px; flex-shrink: 0;
    background: white; border: 1px solid #e6e6e6; border-radius: 6px;
    padding: 6px;
    display: flex; flex-direction: column;
    justify-content: center;
  }}
  .minimaps .minimap-row {{
    display: flex; flex-direction: column; gap: 5px;
  }}
  .minimaps .minimap-row[data-op] {{ display: none; }}
  .minimaps .minimap-row.visible {{ display: flex; }}
  .minimap-cell {{
    position: relative;
    cursor: pointer;
    width: 100%;
  }}
  .minimap-cell canvas {{
    display: block;
    width: 100%; height: 70px;
    border: 2px solid #ddd;
    border-radius: 3px;
    background: white;
    transition: border-color 0.15s;
  }}
  .minimap-cell:hover canvas {{ border-color: #aaa; }}
  .minimap-cell.active canvas {{
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
  }}
  .minimap-cell .lbl {{
    position: absolute;
    top: 3px; left: 4px;
    font-size: 9px;
    color: #666;
    font-weight: 700;
    background: rgba(255,255,255,0.85);
    padding: 0 3px;
    border-radius: 2px;
  }}
  .viz-fig {{
    display: none; width: 100%;
    background: white; border: 1px solid #e6e6e6; border-radius: 6px;
    padding: 8px;
    max-height: calc(100vh - 100px); overflow-y: auto;
  }}
  .viz-fig.visible {{ display: block; }}
  .stacked-section {{
    max-width: 1400px; margin: 8px auto 0;
    padding: 16px 32px 24px;
  }}
  .stacked-section--hero {{
    margin-top: 0;
    padding-top: 8px;
  }}
  .stacked-section h2 {{
    font-size: 20px; margin: 0 0 8px; color: #1a1a1a;
  }}
  .stacked-section p {{
    font-size: 14.5px; color: #555;
    max-width: 950px; margin: 0 0 14px;
  }}
  .stacked-section p.diving-in {{
    margin-top: 16px;
    padding: 12px 16px;
    background: #eef2ff;
    border-left: 4px solid var(--accent);
    border-radius: 4px;
    color: #333;
    font-size: 14px;
  }}
  .stacked-fig {{
    background: #252a31; border: 1px solid #1a1d22; border-radius: 6px;
    padding: 0;
    overflow: hidden;
  }}
  .intro-section {{
    max-width: 1100px; margin: 8px auto 0;
    padding: 16px 32px 24px;
  }}
  .intro-section h2 {{
    font-size: 22px; margin: 0 0 14px; color: #1a1a1a;
  }}
  .intro-section h3 {{
    font-size: 17px; margin: 28px 0 10px; color: #1a1a1a;
    border-bottom: 1px solid #eee; padding-bottom: 5px;
  }}
  .intro-section p {{
    font-size: 14.5px; color: #333; line-height: 1.6;
    margin: 0 0 14px;
  }}
  .intro-section ol, .intro-section ul {{
    font-size: 14px; color: #333; line-height: 1.65;
    margin: 0 0 16px;
    padding-left: 24px;
  }}
  .intro-section ol li, .intro-section ul li {{ margin-bottom: 6px; }}
  .method-image-wrap {{
    margin: 18px 0 22px;
    text-align: center;
  }}
  .method-image {{
    max-width: 100%; height: auto;
    border: 1px solid #e6e6e6; border-radius: 6px;
    background: white; padding: 8px;
  }}
  .method-image-wrap .caption {{
    font-size: 13px; color: #555;
    max-width: 850px; margin: 10px auto 0;
    text-align: left;
    line-height: 1.5;
  }}
  footer {{
    max-width: 1400px; margin: 0 auto;
    padding: 60px 32px 80px; color: #888; font-size: 13px;
  }}
  code {{ background: #f0f0f4; padding: 1px 5px; border-radius: 3px;
          font-size: 13px; }}
  @media (max-width: 920px) {{
    .scrolly {{ grid-template-columns: 1fr; }}
    .viz-col {{ position: sticky; height: 70vh; top: 56px; }}
  }}
</style>
</head><body>

<header>
  <h1>Laplacian Eigenvector Evolution — NanoGPT arithmetic</h1>
  <p class="sub">
    Operator activity in the model is highly structured across all
    six layers — the overview plot below shows how cleanly the
    three operators separate at L0 and how their stacks weave
    together by L5.  To understand <em>why</em> that structure
    emerges, we have to dive into each operator one layer at a
    time: that's what the scroll-driven views below the overview
    are for.
  </p>
</header>

<section class="stacked-section stacked-section--hero">
  <h2>Overview: all operators across all layers</h2>
  <p>
    Joint Laplacian projection (one eigendecomposition per layer over
    all 300 pooled (a, b, op) hidden states) with layer index on the
    z-axis (L0 at the bottom, L5 at the top).  Each pair traces a
    six-point polyline up through the layers; one trace per operator.
    Click <strong>mul / add / sub</strong> in the legend to toggle.
    Rotate to see how L0's clean three-cluster separation
    (1-NN op-recovery = 100%) decays as the trajectories weave
    together at higher layers (~51% by L5).
  </p>
  <div id="fig-stacked3d" class="stacked-fig">{fig_stacked_html}</div>
</section>

<section class="intro-section">
  <h2>How these plots work — graph Laplacian eigenvectors</h2>

  <p>
    Every projection on this page comes from <strong>eigenvectors of
    the symmetric graph Laplacian</strong> of a kNN graph built from
    the model's own hidden states.  This section walks the bridge
    from "embeddings are vectors" to "we have a graph to take a
    Laplacian of."
  </p>

  <h3>What we start with</h3>
  <p>
    At each layer L0..L5 the model produces a hidden state for the
    <code>=</code>-token of each (a, b) input — a vector in 128-dim
    space.  For one operator that's 100 vectors per layer; pooled
    across mul / add / sub it's 300.  The question is what
    <em>shape</em> these vectors form.
  </p>

  <h3>From embeddings to a graph</h3>
  <p>
    The Laplacian is an operator on a <em>graph</em>, not on a set
    of vectors.  So step one is converting the vectors into a graph.
    We do it the simplest way that works:
  </p>
  <ul>
    <li><strong>Each hidden state vector becomes a node.</strong>
        100 pairs per (operator, layer) → 100 nodes.</li>
    <li><strong>For every node, find its <code>k = 10</code> nearest
        neighbors</strong> by Euclidean distance in the 128-dim
        embedding space, and add an edge to each one.</li>
    <li><strong>Symmetrize</strong>: if A had B as a neighbor but B
        didn't have A (kNN isn't symmetric in general), add the edge
        anyway.</li>
  </ul>
  <p>
    What we get is an unweighted graph where edges mean "spatially
    close in the embedding space."  The graph's topology mirrors
    the cluster shape: chains of edges trace continuous regions,
    bottlenecks mark cluster boundaries, sparsely-connected
    components mark distinct sub-clusters.  Here's the actual
    graph for mul L0 — 100 nodes and ~1000 edges, laid out at the
    Laplacian-derived positions so you can see both the graph and
    the cluster shape at once:
  </p>

  <div class="method-image-wrap">
    <img src="data:image/png;base64,{knn_image_b64}"
         alt="kNN graph on mul L0 hidden states"
         class="method-image">
    <p class="caption">
      The graph naturally traces the cluster's three regions —
      red nodes (zero arm) cluster along one edge of the V, green
      nodes (a = b diagonal) along the other, blue off-diagonal
      interior fills the rest.  Edges between regions are sparse;
      edges <em>within</em> each region are dense.  That's the
      structure the Laplacian picks up.
    </p>
  </div>

  <h3>From graph to Laplacian</h3>
  <p>
    With the graph in hand, two matrices fall out:
  </p>
  <ul>
    <li><strong>Adjacency matrix A</strong> (n × n):
        <code>A[i, j] = 1</code> if nodes i and j are connected,
        else 0.</li>
    <li><strong>Degree matrix D</strong> (diagonal):
        <code>D[i, i]</code> = number of edges at node i.</li>
  </ul>
  <p>
    The symmetric-normalized Laplacian is
    <code>L<sub>sym</sub> = I − D<sup>−½</sup> A D<sup>−½</sup></code>.
    Its eigenvalues 0 = λ<sub>0</sub> &lt; λ<sub>1</sub> ≤
    λ<sub>2</sub> ≤ ... measure how much each corresponding
    eigenvector "wiggles" across the graph.  Low eigenvalues mean
    the eigenvector assigns nearly-equal values to neighboring
    nodes — they're <em>smooth functions on the graph</em>.  High
    eigenvalues mean rapid oscillation between neighbors.
  </p>
  <p>
    The trivial eigenvector u<sub>0</sub> is the constant function
    (everyone gets the same value).  The next few —
    u<sub>1</sub>, u<sub>2</sub>, u<sub>3</sub> — are the
    <strong>smoothest non-constant functions on the graph</strong>.
    For a graph that captures manifold structure (like ours), those
    smoothest functions align with the manifold's principal
    coordinate directions.  Using them as 2D or 3D coordinates
    gives us a faithful layout of the cluster's geometry.
  </p>

  <h3>Choosing k</h3>
  <p>
    The kNN parameter <code>k</code> is the one real knob in this
    method.  Too small and the graph fragments into disconnected
    components — the Laplacian's lowest eigenvalues become a wall
    of zeros that don't carry structure.  Too large and the graph
    becomes overly dense, neighbors-of-neighbors get smeared
    together, and the eigenvectors lose specificity to the
    cluster's geometry.
  </p>
  <p>
    Empirically on mul L0 (n = 100 hidden states), the graph
    connectivity and the eigenvector quality move together:
  </p>
  <div class="method-image-wrap">
    <img src="data:image/png;base64,{k_image_b64}"
         alt="Laplacian projection at k = 5, 10, 20, 50"
         class="method-image">
    <p class="caption">
      At k = 5 the graph has 2 disconnected components and the
      eigenvectors are uninterpretable.  At k = 10 it's connected
      and the V-shape is sharp.  At k = 20 it's still connected
      and reasonable.  At k = 50 the graph is so dense that the
      V geometry is washed into a smear — every point is "close"
      to most others.  The cluster shape is <em>stable</em> for
      k ∈ [8, 15]; we use <strong>k = 10</strong>.
    </p>
  </div>
  <p>
    The good rule of thumb is <strong>k ≈ √n</strong>: enough
    neighbors to connect the graph reliably, few enough to keep
    local structure.  For n = 100 that's exactly 10.  For the
    pooled overview (n = 300 across operators) √n ≈ 17 would also
    be a defensible choice; we stay at k = 10 to keep the method
    identical across views, and the joint projection is stable
    against the choice (the three-operator separation at L0 is
    robust to k ∈ [10, 20]).
  </p>

  <h3>Why not PCA / UMAP / t-SNE?</h3>
  <div class="method-image-wrap">
    <img src="data:image/png;base64,{method_image_b64}"
         alt="UMAP, PCA, and Laplacian projections of mul L0 hidden states"
         class="method-image">
    <p class="caption">
      Same 100 mul L0 hidden states, three projections.  UMAP
      shreds the cluster into three disjoint blobs (zero pairs
      separated, (9,9) corner cluster isolated, everything else
      compressed).  PCA is a featureless smear because variance is
      roughly isotropic in the top components.  The Laplacian
      recovers the V-shape — the triangular (min, max) sub-lattice
      that's the genuine geometry of the L0 cluster.
    </p>
  </div>
  <p>
    PCA looks at variance directions but ignores graph/manifold
    structure.  UMAP and t-SNE optimize for cluster
    <em>detection</em> with a parameter (<code>min_dist</code>)
    that explicitly pushes neighbors apart so clusters look
    visually distinct — great for "is this clustered?" questions,
    ruinous for "what is the cluster shaped like?"  The graph
    Laplacian's smoothest eigenvectors don't bend the data to any
    objective — they're just the natural coordinates implied by
    the kNN graph.  Deterministic, no random seed, no parameter
    knobs beyond <code>k</code>.
  </p>

  <h3>Pipeline per (operator, layer)</h3>
  <ol>
    <li>100 hidden states (128-dim) → kNN graph,
        <code>k = 10</code>, symmetrized.</li>
    <li>Symmetric-normalized Laplacian
        <code>L<sub>sym</sub> = I − D<sup>−½</sup> A D<sup>−½</sup></code>.</li>
    <li>Eigendecompose; take u<sub>1</sub>, u<sub>2</sub>, u<sub>3</sub>
        for the three smallest non-trivial eigenvalues (skip
        u<sub>0</sub>, the trivial constant).</li>
    <li>Center, normalize total Frobenius norm to
        <code>√(d·n)</code> so scale is consistent across layers.</li>
    <li><strong>Procrustes-align</strong> each layer's projection
        to the previous one (optimal orthogonal rotation +
        reflection that minimizes
        <code>‖X<sub>L</sub>R − X<sub>L−1</sub>‖<sub>F</sub></code>),
        so the animation has no spurious frame flips between layers.</li>
    <li>For the overview, run steps 1–2 once on the <em>pooled</em>
        300 (a, b, op) hidden states per layer — that's the
        "joint Laplacian" that produces the three-cluster
        separation visible at L0.  For per-operator views, run
        per operator independently and Procrustes-align each
        operator's L0 to mul's L0 so single-op views start in
        a comparable orientation.</li>
  </ol>

  <p class="diving-in">
    <strong>Diving in →</strong> Pick an operator below to drill
    into its layer-by-layer geometry.  The scroll-driven panel
    shows the per-operator Laplacian at every layer with
    click-to-recolor feature pills in the prose.
  </p>
</section>

<div class="controls">
  <span class="label">Operator:</span>
  <button id="btn-mul" class="op active" data-op="mul">× multiplication</button>
  <button id="btn-add" class="op" data-op="add">+ addition</button>
  <button id="btn-sub" class="op" data-op="sub">− subtraction</button>
  <div class="sep"></div>
  <span class="label">Quick recolor:</span>
  <span class="color-pill" data-feature="sign_class">sign_class</span>
  <span class="color-pill" data-feature="max">max</span>
  <span class="color-pill" data-feature="min">min</span>
  <span class="color-pill" data-feature="sum">sum</span>
  <span class="color-pill" data-feature="prod">prod</span>
  <span class="color-pill" data-feature="zero_flag">zero_flag</span>
  <span class="color-pill" data-feature="multichar_answer">multichar_answer</span>
  <span class="color-pill" data-feature="signed_diff">signed_diff</span>
  <span class="color-pill" data-feature="log_prod">log_prod</span>
  <span class="color-pill" data-feature="predicted_char">predicted_char</span>
  <span class="status">
    Showing <b id="status-op">mul</b> · <b id="status-layer">L0</b> · color by <b id="status-feature">sign_class</b>
  </span>
</div>

<div class="scrolly">
  <div class="story-col">
    {steps_html}
  </div>
  <div class="viz-col">
    <div class="minimaps">
      {minimaps_html}
    </div>
    <div class="viz-main">
      <div id="fig-mul" class="viz-fig visible">{fig_mul_html}</div>
      <div id="fig-add" class="viz-fig">{fig_add_html}</div>
      <div id="fig-sub" class="viz-fig">{fig_sub_html}</div>
    </div>
  </div>
</div>

<footer>
  Per-layer Laplacian (u_1, u_2) of the <code>=</code>-token kNN graph
  (k=10) on 100 (a, b) pairs ∈ {{0..9}}².  Sign-aligned + Procrustes-
  aligned across consecutive layers, unit-variance normalized for visual
  continuity.  Joint Laplacian (overview) pools all 300 (a, b, op)
  hidden states per layer in a single eigendecomposition.  Click
  <span class="color-pill" style="cursor:default;">feature</span> tags
  inline to recolor.
</footer>

<script>
const FEATURE_META = {feature_meta_json};
const FEATURE_DATA = {feature_data_json};
const LAYER_DATA = {layer_data_json};
const SIGN_CLASS_HEX = {sign_class_hex_json};
const PRED_CHAR_HEX = {pred_char_hex_json};
const OP_DEFAULT_FEATURE = {op_default_json};
const HEX_COLORS = {hex_colors_json};       // [op][feature] -> [hex per pair]
const MINIMAP_RANGES = {minimap_ranges_json}; // [op] -> {{x: [lo, hi], y: [lo, hi]}}
const JOINT_LAYER_DATA = {joint_layer_data_json}; // [op].u1/u2/u3 = per-layer arrays from joint Laplacian (used by all-ops overlay)

let currentOp = "mul";
let currentLayer = 0;
let currentFeature = OP_DEFAULT_FEATURE[currentOp];
let userPickedFeature = false;  // becomes true once user clicks a pill

function plotDiv(op) {{
  return document.querySelector('#fig-' + op + ' .js-plotly-plot');
}}

// ─── Minimap rendering ──────────────────────────────────────────────

function renderMinimap(canvas, op, layer, hexColors) {{
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0) return;  // hidden
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const x = LAYER_DATA[op].u1[layer];
  const y = LAYER_DATA[op].u2[layer];
  const rng = MINIMAP_RANGES[op];
  const xs = (w - 12) / (rng.x[1] - rng.x[0]);
  const ys = (h - 14) / (rng.y[1] - rng.y[0]);

  for (let i = 0; i < x.length; i++) {{
    const cx = (x[i] - rng.x[0]) * xs + 6;
    const cy = h - ((y[i] - rng.y[0]) * ys + 6) - 2;
    ctx.fillStyle = hexColors[i] || '#888';
    ctx.beginPath();
    ctx.arc(cx, cy, 2.2, 0, 2 * Math.PI);
    ctx.fill();
  }}
}}

function renderAllMinimaps(op) {{
  const hexColors = HEX_COLORS[op] && HEX_COLORS[op][currentFeature];
  if (!hexColors) return;
  for (let L = 0; L < {n_layers}; L++) {{
    const canvas = document.querySelector(
      '.minimap-row[data-op="' + op + '"] .minimap-cell[data-layer="' + L + '"] canvas');
    if (canvas) renderMinimap(canvas, op, L, hexColors);
  }}
}}

function showMinimapsForOp(op) {{
  document.querySelectorAll('.minimap-row[data-op]').forEach(row => {{
    row.classList.toggle('visible', row.dataset.op === op);
  }});
  // Wait one frame so layout applies, then render
  requestAnimationFrame(() => renderAllMinimaps(op));
}}

function highlightMinimapLayer(op, layer) {{
  document.querySelectorAll(
    '.minimap-row[data-op="' + op + '"] .minimap-cell').forEach(cell => {{
    cell.classList.toggle('active',
      parseInt(cell.dataset.layer, 10) === layer);
  }});
}}

function getColors(op, feature) {{
  const vals = FEATURE_DATA[op][feature];
  if (vals === undefined) return null;  // feature not available for this op
  if (feature === 'sign_class') {{
    return vals.map(c => SIGN_CLASS_HEX[String(c)]);
  }}
  if (feature === 'predicted_char') {{
    return vals.map(ch => PRED_CHAR_HEX[ch] || '#000');
  }}
  return vals;  // numeric array, used with colorscale
}}

function getMarkerStyle(op, feature) {{
  const colors = getColors(op, feature);
  const meta = FEATURE_META[feature];
  const m = {{
    size: 11,
    color: colors,
    line: {{ width: 0.5, color: 'black' }},
  }};
  if (meta.kind === 'categorical') {{
    m.showscale = false;
  }} else {{
    m.colorscale = meta.scale;
    m.cmin = meta.cmin;
    m.cmax = meta.cmax;
    m.showscale = true;
    m.colorbar = {{
      title: {{ text: meta.label, font: {{ size: 12 }} }},
      len: 0.7, thickness: 14, x: 1.02, y: 0.5,
    }};
  }}
  return m;
}}

function setFeature(feature) {{
  if (!FEATURE_META[feature]) return;
  // Don't switch to a feature that's not available for the current op
  if (FEATURE_DATA[currentOp][feature] === undefined) {{
    console.warn('Feature ' + feature + ' not available for ' + currentOp);
    return;
  }}
  currentFeature = feature;
  document.getElementById('status-feature').textContent = feature;

  // Highlight all matching pills
  document.querySelectorAll('.color-pill').forEach(p => {{
    p.classList.toggle('active', p.dataset.feature === feature);
  }});

  // Rebuild frames for the visible operator (colors stay the same
  // across layers, so we just need to update the color array)
  const op = currentOp;
  const marker = getMarkerStyle(op, feature);
  const plot = plotDiv(op);
  if (!plot) return;

  // Update the current trace.  Since we drive position interpolation
  // manually via RAF (see animateTo), we only need to restyle the
  // marker here — no need to touch Plotly's frame system.
  try {{
    Plotly.restyle(plot, {{ 'marker': [marker] }}, [0]);
  }} catch (e) {{ console.error('restyle failed', e); }}

  // Re-render minimaps with new feature colors
  renderAllMinimaps(currentOp);
}}

// Manual requestAnimationFrame interpolation between layer positions.
// This is more reliable than Plotly.animate for scatter3d, whose
// WebGL transitions can be choppy.  Works identically for 2D scatter.
let _rafId = null;
const _lastTargetLayer = {{ mul: 0, add: 0, sub: 0 }};
const _ANIM_DURATION = 800;  // ms

function _easeInOutCubic(t) {{
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}}

function animateTo(op, layer) {{
  const plot = plotDiv(op);
  if (!plot || !plot.data || !plot.data[0]) return;
  if (_lastTargetLayer[op] === layer && _rafId === null) return;
  _lastTargetLayer[op] = layer;

  if (_rafId !== null) {{
    cancelAnimationFrame(_rafId);
    _rafId = null;
  }}

  const has3d = !!LAYER_DATA[op].u3;
  const startX = Array.from(plot.data[0].x);
  const startY = Array.from(plot.data[0].y);
  const startZ = has3d ? Array.from(plot.data[0].z) : null;
  const endX = LAYER_DATA[op].u1[layer];
  const endY = LAYER_DATA[op].u2[layer];
  const endZ = has3d ? LAYER_DATA[op].u3[layer] : null;

  const t0 = performance.now();

  function step(now) {{
    const t = Math.min(1, (now - t0) / _ANIM_DURATION);
    const e = _easeInOutCubic(t);
    const ix = new Array(startX.length);
    const iy = new Array(startY.length);
    for (let i = 0; i < startX.length; i++) {{
      ix[i] = startX[i] + e * (endX[i] - startX[i]);
      iy[i] = startY[i] + e * (endY[i] - startY[i]);
    }}
    const update = {{ x: [ix], y: [iy] }};
    if (has3d) {{
      const iz = new Array(startZ.length);
      for (let i = 0; i < startZ.length; i++) {{
        iz[i] = startZ[i] + e * (endZ[i] - startZ[i]);
      }}
      update.z = [iz];
    }}
    try {{ Plotly.restyle(plot, update, [0]); }} catch (e) {{}}
    if (t < 1) {{
      _rafId = requestAnimationFrame(step);
    }} else {{
      _rafId = null;
    }}
  }}
  _rafId = requestAnimationFrame(step);
}}

function applyOpToSteps(op) {{
  document.querySelectorAll('.step').forEach(s => {{
    const h2 = s.querySelector('h2');
    const tEl = h2.querySelector('.h2-title');
    if (tEl && tEl.dataset[op + 'Title']) {{
      tEl.textContent = tEl.dataset[op + 'Title'];
    }}
    s.querySelectorAll('[data-op-body]').forEach(div => {{
      div.style.display = (div.dataset.opBody === op) ? '' : 'none';
    }});
  }});
}}

function switchOp(op) {{
  if (op === currentOp) return;
  currentOp = op;
  document.getElementById('status-op').textContent = op;
  document.querySelectorAll('.controls button.op[data-op]').forEach(b =>
    b.classList.toggle('active', b.dataset.op === op));
  applyOpToSteps(op);

  document.querySelectorAll('.viz-fig').forEach(el =>
    el.classList.remove('visible'));
  document.getElementById('fig-' + op).classList.add('visible');

  // If user has not manually picked a feature, use the new op's
  // informative default (otherwise persist their choice).
  const targetFeature = userPickedFeature
    ? currentFeature
    : OP_DEFAULT_FEATURE[op];

  const plot = plotDiv(op);
  if (plot) {{
    requestAnimationFrame(() => {{
      try {{ Plotly.Plots.resize(plot); }} catch (e) {{}}
      setFeature(targetFeature);
      animateTo(op, currentLayer);
      showMinimapsForOp(op);
      highlightMinimapLayer(op, currentLayer);
    }});
  }}
}}

document.querySelectorAll('.controls button.op[data-op]').forEach(b => {{
  b.addEventListener('click', () => switchOp(b.dataset.op));
}});

// Delegate clicks on .color-pill anywhere on the page
document.addEventListener('click', (e) => {{
  const pill = e.target.closest('.color-pill');
  if (pill && pill.dataset.feature) {{
    userPickedFeature = true;
    setFeature(pill.dataset.feature);
  }}
}});

window.addEventListener('load', () => {{
  document.getElementById('status-op').textContent = currentOp;
  document.getElementById('status-layer').textContent = 'L' + currentLayer;
  document.getElementById('status-feature').textContent = currentFeature;
  applyOpToSteps(currentOp);
  // Mark initial feature pill active
  document.querySelectorAll('.color-pill').forEach(p =>
    p.classList.toggle('active', p.dataset.feature === currentFeature));

  // Initial minimap render + click handlers
  showMinimapsForOp(currentOp);
  highlightMinimapLayer(currentOp, currentLayer);
  document.querySelectorAll('.minimap-cell').forEach(cell => {{
    cell.addEventListener('click', () => {{
      const L = parseInt(cell.dataset.layer, 10);
      // Find the matching step section and scroll into view
      const target = document.querySelector(
        '.step[data-layer="' + L + '"]');
      if (target) target.scrollIntoView({{behavior: 'smooth', block: 'start'}});
    }});
  }});
  window.addEventListener('resize', () => renderAllMinimaps(currentOp));

  // Scrollama
  const scroller = scrollama();
  scroller
    .setup({{ step: '.step', offset: 0.55, debug: false }})
    .onStepEnter(({{ index, element }}) => {{
      element.classList.add('is-active');
      const layer = parseInt(element.dataset.layer, 10);
      currentLayer = layer;
      document.getElementById('status-layer').textContent = 'L' + layer;
      animateTo(currentOp, layer);
      highlightMinimapLayer(currentOp, layer);
    }})
    .onStepExit(({{ element }}) => {{
      element.classList.remove('is-active');
    }});
  window.addEventListener('resize', () => scroller.resize());
}});
</script>
</body></html>
"""


STEP_TEMPLATE = """
<section class="step" data-layer="{layer}">
  <span class="layer-tag">LAYER {layer}</span>
  <h2><span class="h2-title" data-mul-title="{mul_title}" data-add-title="{add_title}" data-sub-title="{sub_title}">{default_title}</span></h2>
  <div class="layer-body">
    <div data-op-body="mul">{mul_body}</div>
    <div data-op-body="add" style="display:none">{add_body}</div>
    <div data-op-body="sub" style="display:none">{sub_body}</div>
  </div>
</section>
"""


def build_minimaps_html():
    rows = []
    for op in OPS:
        cells = "".join(
            f'<div class="minimap-cell" data-layer="{L}">'
            f'<canvas></canvas><span class="lbl">L{L}</span></div>'
            for L in range(N_LAYERS)
        )
        rows.append(f'<div class="minimap-row" data-op="{op}">{cells}</div>')
    return "\n".join(rows)


def build_steps_html():
    parts = []
    for L in range(N_LAYERS):
        mul_title, mul_body = LAYER_TEXT["mul"][L]
        add_title, add_body = LAYER_TEXT["add"][L]
        sub_title, sub_body = LAYER_TEXT["sub"][L]
        parts.append(STEP_TEMPLATE.format(
            layer=L,
            default_title=mul_title.replace('"', "'"),
            mul_title=mul_title.replace('"', "'"),
            add_title=add_title.replace('"', "'"),
            sub_title=sub_title.replace('"', "'"),
            mul_body=featurize_text(mul_body),
            add_body=featurize_text(add_body),
            sub_body=featurize_text(sub_body),
        ))
    return "\n".join(parts)


def main():
    print("Loading data and building scroll figures...")
    figs = {}
    feature_data = {}
    layer_data = {}

    def coords_to_jsdict(coords):
        """Serialize (n_layers, n_pairs, N_DIM) to per-axis layer lists."""
        out = {
            "u1": coords[:, :, 0].tolist(),
            "u2": coords[:, :, 1].tolist(),
        }
        if N_DIM >= 3:
            out["u3"] = coords[:, :, 2].tolist()
        return out

    # Process mul first — its L0 becomes the cross-op reference so
    # add and sub start in the same orientation at L0.
    print(f"  mul: computing aligned (N_DIM={N_DIM}) coords...")
    H_mul, a_mul, b_mul = load_stack(FILES["mul"])
    coords_mul = compute_aligned_coords(H_mul, ref=None)
    mul_L0 = coords_mul[0]

    figs["mul"] = build_scroll_figure("mul", coords_mul, a_mul, b_mul)
    feature_data["mul"] = compute_features("mul", a_mul, b_mul)
    layer_data["mul"] = coords_to_jsdict(coords_mul)

    coords_by_op = {"mul": coords_mul}

    for op in ["add", "sub"]:
        print(f"  {op}: computing aligned coords...")
        H_stack, a, b = load_stack(FILES[op])
        coords = compute_aligned_coords(H_stack, ref=mul_L0)
        figs[op] = build_scroll_figure(op, coords, a, b)
        feature_data[op] = compute_features(op, a, b)
        layer_data[op] = coords_to_jsdict(coords)
        coords_by_op[op] = coords

    print("  precomputing minimap colors and axis ranges...")
    hex_colors = compute_all_hex_colors(feature_data)
    minimap_ranges = compute_minimap_ranges(coords_by_op)

    print("  computing joint Laplacian (ops pooled per layer)...")
    H_stacks = {"mul": H_mul}
    for op in ["add", "sub"]:
        H_op, _, _ = load_stack(FILES[op])
        H_stacks[op] = H_op
    joint_coords = compute_joint_coords(H_stacks)
    joint_per_op = split_joint_coords(joint_coords, n_pairs=coords_mul.shape[1])
    joint_layer_data = {op: coords_to_jsdict(joint_per_op[op])
                        for op in ["mul", "add", "sub"]}

    print("  building stacked-3D figure...")
    fig_stacked = build_stacked_3d_figure(joint_per_op, feature_data)

    print("  generating kNN graph image for intro...")
    knn_image_b64 = build_knn_graph_image()
    print("  generating k-sensitivity image for intro...")
    k_image_b64 = build_k_sensitivity_image()
    print("  generating UMAP/PCA/Laplacian comparison image for intro...")
    method_image_b64 = build_method_comparison_image()

    print("  generating OG (1200x630) preview image via matplotlib...")
    og_path = Path("/Users/jdonaldson/Projects/nanogpt-arithmetic-viz/og.png")
    og_path.parent.mkdir(parents=True, exist_ok=True)
    _build_og_image_matplotlib(joint_per_op, feature_data, og_path)
    print(f"  → {og_path}")

    def fig_to_html(fig, div_id):
        return fig.to_html(include_plotlyjs=False, full_html=False,
                           div_id=div_id, config={"displayModeBar": False})

    steps_html = build_steps_html()
    minimaps_html = build_minimaps_html()
    html = PAGE.format(
        steps_html=steps_html,
        minimaps_html=minimaps_html,
        fig_mul_html=fig_to_html(figs["mul"], "fig-mul-inner"),
        fig_add_html=fig_to_html(figs["add"], "fig-add-inner"),
        fig_sub_html=fig_to_html(figs["sub"], "fig-sub-inner"),
        feature_meta_json=json.dumps(FEATURE_META),
        feature_data_json=json.dumps(feature_data),
        layer_data_json=json.dumps(layer_data),
        sign_class_hex_json=json.dumps({str(k): v for k, v in SIGN_CLASS_HEX.items()}),
        pred_char_hex_json=json.dumps(PRED_CHAR_HEX),
        op_default_json=json.dumps(OP_DEFAULT_FEATURE),
        hex_colors_json=json.dumps(hex_colors),
        minimap_ranges_json=json.dumps(minimap_ranges),
        joint_layer_data_json=json.dumps(joint_layer_data),
        fig_stacked_html=fig_to_html(fig_stacked, "fig-stacked3d-inner"),
        method_image_b64=method_image_b64,
        knn_image_b64=knn_image_b64,
        k_image_b64=k_image_b64,
        n_layers=N_LAYERS,
    )
    OUT.write_text(html)
    print(f"Wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    try:
        subprocess.Popen(["open", str(OUT)])
    except Exception:
        pass


if __name__ == "__main__":
    main()
