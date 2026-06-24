"""Scrollytelling narrative for the Laplacian-spectral arithmetic story.

Modeled on `times_table_story.py`: sticky right-hand viz panel, left
column carries the prose split into "steps" that swap the visible
figure as they scroll into view.

Different focus from the original story.py — that one walks through
binding, gutter, commit, rotation, abstraction.  This one walks
through what Laplacian eigenvectors of the kNN graph reveal about
the manifold's *shape* at each layer for `+ - *`:

  - L0: triangle, unordered (min, max) basis
  - L1: operator-conditional axis emerges (sub's u_1 = operand b)
  - L2: shared zero-detection commit (all three pick up zero_flag on u_1)
  - L3: operator-specific answer-dimension commit
  - L4-L5: simplex / answer-class clusters
  - The basis evolution arc
  - Why UMAP/PCA missed this

Uses scrollama (CDN) + Plotly.  No D3.

Usage:
    uv run python experiments/times_table_l0_spectral_story.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import umap
from plotly.subplots import make_subplots
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import kneighbors_graph

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
OUT = ROOT / "l0_spectral_story.html"

FILES = {
    "mul": ROOT / "hidden_states.npz",
    "add": ROOT / "hidden_states_add.npz",
    "sub": ROOT / "hidden_states_sub.npz",
}
N_LAYERS = 6
K_NN = 10
OPS = list(FILES.keys())

CLASS_COLORS = {
    -1: "#1f77b4",
    0: "#7f7f7f",
    1: "#d62728",
    2: "#2ca02c",
}


# ─── Spectral helpers ────────────────────────────────────────────────


def load_stack(path):
    d = np.load(path)
    layers_arr = d["layer"]
    sel0 = layers_arr == 0
    base_a, base_b = d["a"][sel0], d["b"][sel0]
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


def lap_uv(H, k=K_NN):
    A = kneighbors_graph(H, n_neighbors=k, mode="connectivity",
                        include_self=False)
    A = (A + A.T).maximum(A.T)
    L = laplacian(A, normed=True)
    Ld = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
    _, v = np.linalg.eigh(Ld)
    return v[:, 1].copy(), v[:, 2].copy()


def sign_align(u_prev, u_curr):
    if np.dot(u_prev, u_curr) < 0:
        return -u_curr
    return u_curr


def compute_per_layer_uv(H_stack):
    n_pairs = H_stack.shape[1]
    u1 = np.zeros((N_LAYERS, n_pairs))
    u2 = np.zeros((N_LAYERS, n_pairs))
    for L in range(N_LAYERS):
        u1L, u2L = lap_uv(H_stack[L])
        if L > 0:
            u1L = sign_align(u1[L - 1], u1L)
            u2L = sign_align(u2[L - 1], u2L)
        u1[L], u2[L] = u1L, u2L
    return u1, u2


def candidates(a, b):
    return {
        "a": a.astype(float), "b": b.astype(float),
        "min": np.minimum(a, b).astype(float),
        "max": np.maximum(a, b).astype(float),
        "sum": (a + b).astype(float),
        "abs_diff": np.abs(a - b).astype(float),
        "signed_diff": (a - b).astype(float),
        "prod": (a * b).astype(float),
        "zero_flag": ((a == 0) | (b == 0)).astype(float),
    }


def answer_sign_class(op, a, b):
    """Return per-pair sign class: -1, 0, +1.  For mul/add (always >=0)
    we collapse to {0: 'zero answer', 1: 'positive answer'}."""
    if op == "mul":
        return np.where((a * b) == 0, 0, 1)
    if op == "add":
        return np.where((a + b) == 0, 0, 1)
    if op == "sub":
        return np.sign(a - b).astype(int)
    raise ValueError(op)


def r2_uni(y, x):
    x = x - x.mean(); y = y - y.mean()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    beta = (x @ y) / (x @ x)
    return float(1 - ((y - beta * x) ** 2).sum() / (y * y).sum())


def r2_joint(U, Y):
    Uc = U - U.mean(axis=0); Yc = Y - Y.mean(axis=0)
    beta, *_ = np.linalg.lstsq(Uc, Yc, rcond=None)
    return float(1 - ((Yc - Uc @ beta) ** 2).sum() / (Yc ** 2).sum())


# ─── Compute all per-op data once ────────────────────────────────────


def precompute():
    data = {}
    for op, path in FILES.items():
        H_stack, a, b = load_stack(path)
        u1, u2 = compute_per_layer_uv(H_stack)
        data[op] = dict(H_stack=H_stack, a=a, b=b, u1=u1, u2=u2,
                        feats=candidates(a, b),
                        sign_cls=answer_sign_class(op, a, b))
    return data


# ─── Figures ─────────────────────────────────────────────────────────


def _coord_text(a, b, op, sign_cls_val):
    cls_name = {
        ("mul", 0): "zero answer (annihilator)",
        ("mul", 1): "positive answer",
        ("add", 0): "zero answer",
        ("add", 1): "positive answer",
        ("sub", -1): "negative answer (a < b)",
        ("sub", 0): "zero answer (a = b)",
        ("sub", 1): "positive answer (a > b)",
    }
    return cls_name.get((op, int(sign_cls_val)), str(sign_cls_val))


def fig_grid(data, color_feature: str):
    """3 rows × 6 cols of Laplacian (u_1, u_2) scatters.

    color_feature is one of:
        'sign_class'  - categorical, sign of answer
        'max'         - max(a, b)
        'min'         - min(a, b)
        'zero_flag'   - 0/1, is a zero pair?
        'signed_diff' - a - b
    """
    fig = make_subplots(
        rows=3, cols=N_LAYERS,
        subplot_titles=[f"{op} L{L}" for op in OPS for L in range(N_LAYERS)],
        horizontal_spacing=0.02,
        vertical_spacing=0.07,
    )

    for r, op in enumerate(OPS):
        d = data[op]
        a, b = d["a"], d["b"]
        for L in range(N_LAYERS):
            x = d["u1"][L]; y = d["u2"][L]
            hover_text = [
                f"L{L}<br>({ai}, {bi})<br>{_coord_text(ai, bi, op, s)}"
                for ai, bi, s in zip(a, b, d["sign_cls"])
            ]
            if color_feature == "sign_class":
                # split into separate traces per class for clean legend
                for c in sorted(set(d["sign_cls"].tolist())):
                    m = d["sign_cls"] == c
                    fig.add_trace(
                        go.Scatter(
                            x=x[m], y=y[m], mode="markers",
                            marker=dict(size=7, color=CLASS_COLORS[int(c)],
                                        line=dict(width=0.3, color="black")),
                            name=_coord_text(a[m][0], b[m][0], op, c),
                            legendgroup=f"sign_{c}_{op}",
                            showlegend=(L == 0),
                            text=[hover_text[i] for i in np.where(m)[0]],
                            hovertemplate="%{text}<extra></extra>",
                        ),
                        row=r + 1, col=L + 1,
                    )
            else:
                vals = d["feats"][color_feature]
                cmap = "viridis" if color_feature in ("max", "min") else \
                       ("RdBu_r" if color_feature == "signed_diff" else "RdYlBu_r")
                fig.add_trace(
                    go.Scatter(
                        x=x, y=y, mode="markers",
                        marker=dict(size=7, color=vals, colorscale=cmap,
                                    line=dict(width=0.3, color="black"),
                                    showscale=(L == N_LAYERS - 1 and r == 0),
                                    colorbar=dict(
                                        title=color_feature,
                                        x=1.02, y=0.85, len=0.3)
                                    if (L == N_LAYERS - 1 and r == 0) else None),
                        showlegend=False,
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                    ),
                    row=r + 1, col=L + 1,
                )

            fig.update_xaxes(showticklabels=False, row=r + 1, col=L + 1)
            fig.update_yaxes(showticklabels=False, row=r + 1, col=L + 1)

    fig.update_layout(
        height=620, width=None, autosize=True,
        margin=dict(l=30, r=30, t=60, b=30),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="left", x=0.0, font=dict(size=10)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def fig_basis_evolution(data):
    """Per-op joint R² of (u_1,u_2) onto candidate basis pairs across layers."""
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[op for op in OPS],
                        horizontal_spacing=0.06)
    basis_pairs = [("min", "max"), ("a", "b")]
    bp_colors = {"(min, max)": "#1f77b4", "(a, b)": "#ff7f0e"}

    for col, op in enumerate(OPS):
        d = data[op]
        for pa, pb in basis_pairs:
            vals = []
            for L in range(N_LAYERS):
                U = np.stack([d["u1"][L], d["u2"][L]], axis=1)
                Y = np.stack([d["feats"][pa], d["feats"][pb]], axis=1)
                vals.append(r2_joint(U, Y))
            name = f"({pa}, {pb})"
            fig.add_trace(
                go.Scatter(
                    x=list(range(N_LAYERS)), y=vals,
                    mode="lines+markers", name=name,
                    line=dict(color=bp_colors[name], width=2),
                    marker=dict(size=8),
                    legendgroup=name, showlegend=(col == 0),
                    hovertemplate=f"{name} L%{{x}}: R²=%{{y:.3f}}<extra></extra>",
                ),
                row=1, col=col + 1,
            )
        fig.update_xaxes(title="layer", tickvals=list(range(N_LAYERS)),
                         row=1, col=col + 1)
        fig.update_yaxes(range=[0, 1], row=1, col=col + 1)
        if col == 0:
            fig.update_yaxes(title="Joint R² of (u_1, u_2)", row=1, col=col + 1)

    fig.update_layout(
        height=380, autosize=True,
        margin=dict(l=60, r=30, t=60, b=50),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.12, x=0.0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def fig_simplex(data):
    """Sub L5 colored three ways: signed_diff, sign class, magnitude(|a-b|)."""
    d = data["sub"]
    u1, u2 = d["u1"][N_LAYERS - 1], d["u2"][N_LAYERS - 1]
    a, b = d["a"], d["b"]
    signed = a - b
    sign_cls = np.sign(signed).astype(int)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "colored by signed_diff (a − b)",
            "colored by answer sign class",
            "colored by |a − b|",
        ],
        horizontal_spacing=0.06,
    )

    # signed_diff
    fig.add_trace(
        go.Scatter(
            x=u1, y=u2, mode="markers",
            marker=dict(size=10, color=signed, colorscale="RdBu_r",
                        cmin=-9, cmax=9,
                        line=dict(width=0.4, color="black"),
                        colorbar=dict(title="a−b", x=0.30, y=0.5, len=0.7)),
            showlegend=False,
            text=[f"({ai},{bi}) a-b={s}" for ai, bi, s in zip(a, b, signed)],
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1, col=1,
    )

    # sign class
    for c in [-1, 0, 1]:
        m = sign_cls == c
        label = {-1: "a < b", 0: "a = b", 1: "a > b"}[c]
        fig.add_trace(
            go.Scatter(
                x=u1[m], y=u2[m], mode="markers",
                marker=dict(size=10, color=CLASS_COLORS[c],
                            line=dict(width=0.4, color="black")),
                name=f"{label}  (n={int(m.sum())})",
                text=[f"({ai},{bi})" for ai, bi in zip(a[m], b[m])],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1, col=2,
        )

    # |a-b|
    abs_diff = np.abs(signed)
    fig.add_trace(
        go.Scatter(
            x=u1, y=u2, mode="markers",
            marker=dict(size=10, color=abs_diff, colorscale="viridis",
                        cmin=0, cmax=9,
                        line=dict(width=0.4, color="black"),
                        colorbar=dict(title="|a−b|", x=1.02, y=0.5, len=0.7)),
            showlegend=False,
            text=[f"({ai},{bi}) |a-b|={d_}" for ai, bi, d_ in zip(a, b, abs_diff)],
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1, col=3,
    )

    for c in range(1, 4):
        fig.update_xaxes(title="u_1", row=1, col=c)
        fig.update_yaxes(title="u_2", row=1, col=c)

    # Between/within stats annotation
    cents = {}
    spreads = {}
    for c in [-1, 0, 1]:
        m = sign_cls == c
        pts = np.stack([u1[m], u2[m]], axis=1)
        cents[c] = pts.mean(axis=0)
        spreads[c] = float(np.linalg.norm(pts - cents[c], axis=1).mean())
    mean_within = np.mean(list(spreads.values()))
    mean_between = np.mean([np.linalg.norm(cents[s1] - cents[s2])
                            for s1 in [-1, 0, 1] for s2 in [-1, 0, 1]
                            if s1 < s2])
    fig.add_annotation(
        text=(f"<b>Between/within ratio = {mean_between/mean_within:.1f}×</b><br>"
              "10× → three discrete classes, not a filled parameter region."),
        xref="paper", yref="paper", x=0.50, y=-0.18, showarrow=False,
        font=dict(size=12, color="#444"),
    )

    fig.update_layout(
        height=440, autosize=True,
        margin=dict(l=50, r=70, t=60, b=80),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.12, x=0.30),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def fig_method_compare(data):
    """UMAP / PCA / Laplacian on the same mul L0 data — three columns."""
    d = data["mul"]
    H = d["H_stack"][0]
    a, b = d["a"], d["b"]

    Hc = H - H.mean(axis=0)
    U, S, _ = np.linalg.svd(Hc, full_matrices=False)
    pca = U[:, :2] * S[:2]
    um = umap.UMAP(n_components=2, random_state=42).fit_transform(H)
    u1, u2 = d["u1"][0], d["u2"][0]

    zero = (a == 0) | (b == 0)
    diag = (a == b) & ~zero
    offdiag = (a != b) & ~zero & (np.abs(a - b) >= 3)
    mid = ~zero & ~diag & ~offdiag

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "UMAP (n_neighbors=15, min_dist=0.1)",
            "PCA: PC1 × PC2",
            "Laplacian: u_1 × u_2",
        ],
        horizontal_spacing=0.05,
    )

    def add_panel(X, col):
        for m, c, lbl in [
            (zero, "#d62728", "zero pair"),
            (diag, "#2ca02c", "a = b non-zero"),
            (mid, "#7f7f7f", "|a−b| ≤ 2"),
            (offdiag, "#1f77b4", "|a−b| ≥ 3"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=X[m, 0], y=X[m, 1], mode="markers",
                    marker=dict(size=8, color=c,
                                line=dict(width=0.3, color="black")),
                    name=lbl, legendgroup=lbl,
                    showlegend=(col == 1),
                    text=[f"({ai},{bi})" for ai, bi in zip(a[m], b[m])],
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=1, col=col,
            )

    add_panel(um, 1)
    add_panel(pca, 2)
    add_panel(np.stack([u1, u2], axis=1), 3)

    for c in range(1, 4):
        fig.update_xaxes(showticklabels=False, showgrid=True,
                        gridcolor="#f0f0f0", row=1, col=c)
        fig.update_yaxes(showticklabels=False, showgrid=True,
                        gridcolor="#f0f0f0", row=1, col=c)

    fig.update_layout(
        height=440, autosize=True,
        margin=dict(l=30, r=30, t=60, b=30),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.12, x=0.0),
    )
    return fig


# ─── Steps narrative ─────────────────────────────────────────────────


STEPS = [
    dict(
        anchor="triangle",
        viz="grid-max",
        title="L0 — The (min, max) triangle",
        body="""
<p>For each operator we evaluate 100 pairs (a, b ∈ {0..9}) and take
the hidden state at the <code>=</code>-token at every layer.  Build a
kNN graph (k=10) on the 100 L0 states; compute the smallest non-trivial
Laplacian eigenvectors u₁ and u₂.</p>
<p>The shape is a V — two long edges meeting at a vertex.  The vertex
sits where <code>(a, b) = (0, 0)</code>.  One arm runs up to (0, 9):
that's the boundary edge where min = 0 and max varies.  The other arm
runs to (9, 9): the diagonal where min = max.  The interior of the
triangle is between them.</p>
<p>The L0 cluster is literally a triangulated 2D sub-lattice — the
quotient of the 10×10 operand grid by the swap action.  Probes
already told us min/max are 94–96% decodable at L0.  The spectrum
shows min/max <em>is the cluster's geometry</em>, not just decodable
from it.  Confirming corollary: the mean distance between (a, b) and
(b, a) in (u₁, u₂) is 7.6% of the mean distance between random pairs.
Swap-pairs collapse to near-coincidence.</p>""",
        hint="The grid shows mul / add / sub × L0…L5.  Focus on the first "
             "column.  Each operator's L0 cluster has the same triangular "
             "shape — operator dispatch hasn't happened yet."
    ),
    dict(
        anchor="dispatch",
        viz="grid-sign",
        title="L1 — Sub diverges first",
        body="""
<p>Block 1 is the operator dispatcher (causally established by
activation patching in the architecture memory: ≥91% flip rate after
L1).  In the Laplacian readout, mul and add still look L0-like at
L1, but <em>sub's u₁ locks onto operand <code>b</code></em> with
R² = 0.78.  That's the subtractor — the operand whose sign gets
flipped — being isolated as a coordinate.</p>
<p>Why only sub?  Because subtraction is the only one of the three
operators that's anti-commutative.  Mul and add are symmetric in (a, b),
so a symmetric basis (min, max) still works.  Sub needs to know which
operand is which, so its representation begins to peel apart the
ordered pair.</p>""",
        hint="Sub L1: the manifold has rotated relative to L0.  Coloring is "
             "now by answer sign — for mul/add 'positive answer' (red) "
             "dominates, but for sub the negative-answer (blue) cluster "
             "is already separating from the positive-answer cluster.",
    ),
    dict(
        anchor="zero_detect",
        viz="grid-zero",
        title="L2 — Everyone detects zero",
        body="""
<p>By L2, all three operators have u₁ ≈ <code>zero_flag</code> with
R² ≈ 1.0.  The Laplacian's first non-trivial eigenvector — the
Fiedler vector — is the cut that maximally separates zero-pairs from
non-zero pairs.</p>
<p>This is shared substrate, not operator-specific computation yet.
Annihilator/edge detection happens before commitment to an
operator-specific answer dimension.  The cluster topology has
shifted: at L0 it was 2D-LATTICE (a triangulated manifold);
at L2 it's becoming HUB-like — a small zero-pair cluster
splitting off from the main non-zero mass.</p>""",
        hint="The L2 column: zero pairs (red in this coloring) have peeled "
             "into a tight cluster, with non-zero pairs (blue) forming the "
             "larger group.  Same shape for all three operators.",
    ),
    dict(
        anchor="commit",
        viz="grid-sign",
        title="L3 — Operator-specific answer dimension",
        body="""
<p>This is the operator-specific commit point.  Each operator's
leading eigenvector u₁ shifts to its own answer dimension:</p>
<ul>
  <li><b>mul</b>: stays on <code>zero_flag</code> (R² = 1.0).
      The mul commit-class fork is zero-vs-nonzero — its main
      computation-saving shortcut, the annihilator gutter from L3.</li>
  <li><b>add</b>: shifts to <code>sum</code> (R² = 0.71) — the answer
      <em>is</em> the sum.</li>
  <li><b>sub</b>: shifts to <code>signed_diff</code> (R² = 0.76) —
      again, the answer.</li>
</ul>
<p>The Laplacian's leading mode at L3 is literally tracking what the
model has computed.  Probes had to be hand-built to extract these
features; the spectrum finds them unsupervised as the manifold's
principal directions.</p>""",
        hint="Compare L3 across the three rows.  Mul L3 looks 2-cluster "
             "(zero vs non-zero, exactly the gutter).  Add L3 spreads "
             "along a single magnitude axis.  Sub L3 has three discrete "
             "vertices — the simplex we'll inspect next.",
    ),
    dict(
        anchor="simplex",
        viz="simplex",
        title="L4–L5 — Sub's three-vertex simplex",
        body="""
<p>Sub L5's (u₁, u₂) plane has three obvious vertices: a positive-
answer cluster (a > b), a negative-answer cluster (a < b), and a
small zero-answer cluster (a = b).  The vertices are tight; the
interior is sparse.</p>
<p>Numerically: <b>between/within cluster ratio = 10.5×</b>.
That rules out the "continuous filled parameter region" hypothesis
and confirms three discrete classes.  This is the neural-collapse /
equiangular-tight-frame pattern — penultimate-layer classes
converging toward simplex vertices in a low-dimensional projection.
For 3 classes, the ETF is a 2-simplex (triangle) embedded in 2D,
which is exactly what u₁, u₂ show.</p>
<p>Subtler caveat: positive answers 1, 2, …, 9 are all stacked nearly
on top of each other in (u₁, u₂).  Digit identity isn't in u₁ / u₂ —
it must live in u₃, u₄, …, up to the rank-10 subspace the head reads
from.  The Laplacian eigenvectors form a hierarchy: u₁, u₂ pick up the
coarsest class split; finer digit structure is in higher modes.</p>""",
        hint="The right panel (colored by |a−b|) shows that distance "
             "from the diagonal a = b stretches each arm of the V. "
             "The middle panel makes the three answer-sign clusters "
             "explicit.",
    ),
    dict(
        anchor="basis_arc",
        viz="basis-evolution",
        title="The basis-dominance arc, operator by operator",
        body="""
<p>One number per layer per operator: the joint R² of the leading
(u₁, u₂) plane onto candidate basis pairs.  Two lines per
operator — the symmetric basis (min, max) vs. the ordered basis
(a, b).</p>
<ul>
  <li><b>mul</b>: (min, max) wins L0–L3, then collapses at L4–L5.
      Once mul has committed to "say 0 vs. something else," the
      set basis stops being the principal axis.</li>
  <li><b>add</b>: (min, max) wins all six layers — the answer is a
      function of (sum, abs_diff), which spans the same plane as
      (min, max) up to linear basis change.</li>
  <li><b>sub</b>: (min, max) wins at L0, holds through L2, then
      <em>collapses to 0.05 at L3 and L5</em>.  The leading plane
      shifts decisively off the symmetric basis once subtraction's
      signed answer dimension takes over.</li>
</ul>
<p>Note: (a, b) and (sum, signed_diff) span the same plane via linear
basis change; only (min, max) is a non-linear projection.  When
(min, max) wins, the cluster's geometry really is in the unordered
quotient.</p>""",
        hint="The collapse in sub at L3 is the operator-specific commit "
             "happening in real time as a number: 0.65 → 0.05.",
    ),
    dict(
        anchor="why_umap",
        viz="method-compare",
        title="Why UMAP and PCA missed all of this",
        body="""
<p>The same 100 mul L0 states, projected three ways:</p>
<ul>
  <li><b>UMAP</b>: fragments the triangle into three disjoint blobs.
      Zero pairs become one cluster; (9, 9) becomes its own corner;
      the bulk is everything else.  UMAP's <code>min_dist</code> and
      negative-sampling loss <em>actively push neighbors apart</em>
      to make clusters visually distinct — great for "is this
      clustered?" questions, ruinous for "what shape is the cluster?"</li>
  <li><b>PCA</b>: featureless blob.  Anisotropy 0.51 means variance
      is roughly isotropic in the top components; PC1/PC2 doesn't
      align with the manifold's natural axes.</li>
  <li><b>Laplacian</b>: the V is sharp.  u₁, u₂ are the smoothest
      non-trivial functions on the kNN graph — for a connected
      triangulated 2D manifold those are the principal coordinates.</li>
</ul>
<p>Tool-choice rule: UMAP for "is this clustered?"; PCA for "what
linear axes carry the variance?"; <strong>Laplacian eigenvectors for
"what's the cluster's shape / boundary / topology?"</strong>  The
prior nanogpt visualizations (<code>times_table_stacked_umap.html</code>,
<code>times_table_story.html</code>) used UMAP and overlaid all six
layers — the L0 triangle was destroyed twice over.</p>""",
        hint="Same data, different lens.  This is also why the existing "
             "stacked-UMAP visualization in this directory shows L0 as a "
             "rounded blob with no triangle structure visible — UMAP's "
             "fault, not the model's.",
    ),
]


# ─── HTML page template ──────────────────────────────────────────────


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>NanoGPT Laplacian — A Layer-by-Layer Story</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://unpkg.com/scrollama@3.2.0"></script>
<style>
  :root {{ --accent: #2643d4; --warn: #f4a623; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue",
                 Arial, sans-serif;
    color: #222; background: #fafafa; margin: 0;
    line-height: 1.55;
  }}
  header {{
    max-width: 1400px; margin: 0 auto; padding: 44px 32px 12px;
  }}
  header h1 {{ font-size: 30px; margin: 0 0 6px; color: #1a1a1a; }}
  header .subtitle {{
    color: #555; font-size: 16px; margin: 0;
  }}
  .summary {{
    background: #eef2ff; border-left: 4px solid var(--accent);
    padding: 14px 18px; border-radius: 4px; font-size: 15px;
    max-width: 1400px; margin: 16px auto 16px; padding-left: 22px;
  }}
  .scrolly {{
    display: grid; grid-template-columns: 1fr 1.3fr; gap: 32px;
    max-width: 1500px; margin: 0 auto; padding: 0 32px;
  }}
  .story-col {{ padding-top: 4vh; }}
  .step {{
    min-height: 75vh; padding: 32px 0 40vh 0;
    opacity: 0.38; transition: opacity 0.4s ease;
  }}
  .step.is-active {{ opacity: 1; }}
  .step h2 {{
    border-bottom: 1px solid #e3e3e3; padding-bottom: 8px;
    font-size: 22px; color: #1a1a1a; margin-top: 0;
  }}
  .step p {{ font-size: 15px; }}
  .step ul {{ font-size: 14.5px; line-height: 1.6; }}
  .hint {{
    background: #fff8e6; border-left: 3px solid var(--warn);
    padding: 10px 14px; border-radius: 4px; font-size: 14px; color: #555;
    margin-top: 16px;
  }}
  .viz-col {{
    position: sticky; top: 0; height: 100vh;
    display: flex; flex-direction: column; justify-content: center;
    padding: 16px 0;
  }}
  .viz-fig {{
    display: none; width: 100%;
    background: white; border: 1px solid #e6e6e6; border-radius: 6px;
    padding: 8px;
    max-height: 95vh; overflow-y: auto;
  }}
  .viz-fig.visible {{ display: block; }}
  .stepper {{
    position: fixed; right: 18px; top: 50%;
    transform: translateY(-50%);
    display: flex; flex-direction: column; gap: 6px;
    background: rgba(255,255,255,0.85);
    padding: 8px 6px; border-radius: 8px;
    border: 1px solid #ddd; font-size: 11px;
    z-index: 10;
  }}
  .stepper a {{
    color: #777; text-decoration: none; padding: 2px 6px;
    border-radius: 3px;
  }}
  .stepper a.is-active {{ color: var(--accent); background: #eef2ff; }}
  footer {{
    max-width: 1400px; margin: 0 auto;
    padding: 60px 32px 80px; color: #888; font-size: 13px;
  }}
  code {{
    background: #f0f0f4; padding: 1px 5px; border-radius: 3px;
    font-size: 13px;
  }}
  @media (max-width: 920px) {{
    .scrolly {{ grid-template-columns: 1fr; }}
    .viz-col {{ position: sticky; height: 75vh; }}
    .stepper {{ display: none; }}
  }}
</style>
</head><body>

<header>
  <h1>Laplacian Eigenvectors of NanoGPT Arithmetic</h1>
  <p class="subtitle">A layer-by-layer manifold-shape story for
     <code>+ − ×</code> with operands 0–9 in the 1.2M / 6L / 128d
     char-level GPT.</p>
</header>

<div class="summary">
  L0 is a triangular (min, max) sub-lattice — operator-agnostic.
  L1 starts dispatching: sub's u₁ locks onto operand b.  L2 is shared
  zero detection.  L3 is where each operator's answer dimension takes
  over as the leading Laplacian mode — mul stays on zero_flag, add
  shifts to sum, sub shifts to signed_diff.  By L5 the manifold is a
  small ETF-like simplex over answer classes.  UMAP and PCA both
  destroy this; only the graph Laplacian shows the shape.
  <br/><span style="color:#666">Scroll down — the right panel updates
  as you read.</span>
</div>

<nav class="stepper" id="stepper">
  {stepper_links}
</nav>

<div class="scrolly">
  <div class="story-col">
    {steps_html}
  </div>
  <div class="viz-col">
    <div id="grid-max" class="viz-fig visible">{fig_grid_max_html}</div>
    <div id="grid-sign" class="viz-fig">{fig_grid_sign_html}</div>
    <div id="grid-zero" class="viz-fig">{fig_grid_zero_html}</div>
    <div id="simplex" class="viz-fig">{fig_simplex_html}</div>
    <div id="basis-evolution" class="viz-fig">{fig_basis_html}</div>
    <div id="method-compare" class="viz-fig">{fig_method_html}</div>
  </div>
</div>

<footer>
  Figures computed fresh from
  <code>experiments/data/nanogpt_times_table/hidden_states*.npz</code>.
  Companion scripts:
  <code>times_table_l0_spectral.py</code> (single-layer probe of L0),
  <code>times_table_story.py</code> (the earlier binding/commit/rotation
  story).  Memory:
  <code>feedback_manifold_viz_laplacian.md</code> (tool-choice rule).
</footer>

<script>
const STEPS = {steps_data};

function applyStep(idx) {{
  const step = STEPS[idx];
  if (!step) return;
  document.querySelectorAll('.viz-fig').forEach(el =>
    el.classList.remove('visible'));
  const figEl = document.getElementById(step.viz);
  if (figEl) {{
    figEl.classList.add('visible');
    const plot = figEl.querySelector('.js-plotly-plot');
    if (plot) {{
      requestAnimationFrame(() => {{
        try {{ Plotly.Plots.resize(plot); }} catch (e) {{}}
      }});
    }}
  }}
  document.querySelectorAll('.stepper a').forEach((a, i) =>
    a.classList.toggle('is-active', i === idx));
}}

window.addEventListener('load', () => {{
  applyStep(0);
  const scroller = scrollama();
  scroller
    .setup({{ step: '.step', offset: 0.5, debug: false }})
    .onStepEnter(({{ index, element }}) => {{
      element.classList.add('is-active');
      applyStep(index);
    }})
    .onStepExit(({{ element }}) => {{
      element.classList.remove('is-active');
    }});
  window.addEventListener('resize', () => scroller.resize());

  document.querySelectorAll('.stepper a').forEach((a, i) => {{
    a.addEventListener('click', (e) => {{
      e.preventDefault();
      const target = document.querySelector(`.step[data-step="${{i}}"]`);
      if (target) target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }});
  }});
}});
</script>
</body></html>
"""


STEP_TEMPLATE = """
<section class="step" id="{anchor}" data-step="{idx}">
  <h2>{title}</h2>
  {body}
  <div class="hint"><strong>What to look at →</strong> {hint}</div>
</section>
"""


def main():
    print("Loading hidden states and computing Laplacian projections...")
    data = precompute()

    print("Building grid figure (color by max)...")
    fig_grid_max = fig_grid(data, "max")
    print("Building grid figure (color by sign class)...")
    fig_grid_sign = fig_grid(data, "sign_class")
    print("Building grid figure (color by zero_flag)...")
    fig_grid_zero = fig_grid(data, "zero_flag")
    print("Building simplex figure...")
    fig_simplex_ = fig_simplex(data)
    print("Building basis-evolution figure...")
    fig_basis = fig_basis_evolution(data)
    print("Building method comparison (UMAP/PCA/Laplacian)...")
    fig_method = fig_method_compare(data)

    def fig_to_html(fig, div_id):
        return fig.to_html(include_plotlyjs=False, full_html=False,
                           div_id=div_id, config={"displayModeBar": False})

    steps_html = "\n".join(
        STEP_TEMPLATE.format(idx=i, anchor=s["anchor"], title=s["title"],
                             body=s["body"], hint=s["hint"])
        for i, s in enumerate(STEPS)
    )
    stepper_links = "\n".join(
        f'  <a href="#{s["anchor"]}" data-idx="{i}">{s["title"]}</a>'
        for i, s in enumerate(STEPS)
    )
    steps_data = json.dumps([dict(viz=s["viz"]) for s in STEPS])

    html = PAGE.format(
        steps_html=steps_html,
        stepper_links=stepper_links,
        fig_grid_max_html=fig_to_html(fig_grid_max, "grid-max-fig"),
        fig_grid_sign_html=fig_to_html(fig_grid_sign, "grid-sign-fig"),
        fig_grid_zero_html=fig_to_html(fig_grid_zero, "grid-zero-fig"),
        fig_simplex_html=fig_to_html(fig_simplex_, "simplex-fig"),
        fig_basis_html=fig_to_html(fig_basis, "basis-fig"),
        fig_method_html=fig_to_html(fig_method, "method-fig"),
        steps_data=steps_data,
    )
    OUT.write_text(html)
    print(f"Wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    try:
        subprocess.Popen(["open", str(OUT)])
    except Exception:
        pass


if __name__ == "__main__":
    main()
