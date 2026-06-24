"""Animated layer-by-layer Laplacian (u_1, u_2) projection per operator.

Three operators (mul, add, sub), each as its own Plotly animation.
Frames = layers L0..L5, with linear interpolation between consecutive
layer positions.  Each frame has a title + a short interpretation of
what the layer is doing.  Sign-aligned eigenvectors and per-layer
unit-variance normalization keep the morph visually continuous.

User selects which operator to view via buttons at the top of the page;
within an operator, slider + play button drive the animation.

Usage:
    uv run python experiments/times_table_l0_spectral_animation.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import kneighbors_graph

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
OUT = ROOT / "l0_spectral_animation.html"

FILES = {
    "mul": ROOT / "hidden_states.npz",
    "add": ROOT / "hidden_states_add.npz",
    "sub": ROOT / "hidden_states_sub.npz",
}
N_LAYERS = 6
K_NN = 10
OPS = list(FILES.keys())

CLASS_COLORS = {-1: "#1f77b4", 0: "#7f7f7f", 1: "#d62728"}


# ─── Layer narratives ──────────────────────────────────────────────


LAYER_TEXT = {
    "mul": {
        0: ("L0 — Pre-operator (min, max) triangle",
            "Operands encoded as an unordered set; cluster is a "
            "triangulated 2D sub-lattice. Swap pairs (a,b) ↔ (b,a) "
            "collapse to within 8% of the random-pair distance."),
        1: ("L1 — Operator dispatcher fires (block-1 attention)",
            "Patching at this layer flips operator with ≥91% rate. "
            "For mul, u_1 still tracks magnitude (max/sum); no large "
            "geometric change yet — the commit is downstream."),
        2: ("L2 — Shared zero detection (Fiedler cut)",
            "u_1 ≈ zero_flag with R² ≈ 1.0. The Laplacian's leading "
            "non-trivial mode is the cut separating zero pairs from "
            "non-zero pairs. Topology: 2D-LATTICE → HUB."),
        3: ("L3 — Annihilator gutter consolidates",
            "u_1 stays on zero_flag (R² = 1.00). For mul, the "
            "commit-class fork IS zero-vs-nonzero — the gutter from the "
            "architecture memory, found here unsupervised."),
        4: ("L4 — Commit-class MLP writes (typical pairs)",
            "L4 MLP injects +1.57 along correct-digit-direction for "
            "L4-commit pairs; non-zero pairs spread into output sub-blobs. "
            "Zero cluster stays isolated."),
        5: ("L5 — Linear contrastive digit selector",
            "L5 MLP writes +6 on correct digit row, −0.7 on each other. "
            "Effective rank ~10. u_1, u_2 show coarse zero/non-zero "
            "split; digit identity lives in u_3..u_10."),
    },
    "add": {
        0: ("L0 — Pre-operator (min, max) triangle",
            "Same starting geometry as mul — operator-agnostic. "
            "(min, max) joint R² = 0.79, the strongest of any operator."),
        1: ("L1 — Operator dispatcher fires (block-1 attention)",
            "For add, u_1 still tracks magnitude. The combination rule "
            "(raw a+b sum) doesn't need a coordinate flip — addition "
            "preserves the symmetric basis."),
        2: ("L2 — Shared zero detection",
            "u_1 ≈ zero_flag (R² = 0.99). Same Fiedler cut as mul/sub. "
            "Add will not use this — the answer for add is `sum`, "
            "not a zero/non-zero classification."),
        3: ("L3 — Commit to `sum`",
            "u_1 shifts to `sum` (R² = 0.71). The answer IS the sum, "
            "and the leading Laplacian mode now tracks it directly. "
            "Add never needs the zero detection branch."),
        4: ("L4 — Refinement on the magnitude axis",
            "u_1 still on `sum` (R² = 0.71). Pairs spread along a "
            "1D-leaning magnitude axis — answers 0..18 sorted along the "
            "principal coordinate."),
        5: ("L5 — Linear contrastive digit selector",
            "u_1 ≈ `sum`/`prod` composite (~0.6). Add's L5 looks more "
            "continuous than mul's because `sum` doesn't quantize into "
            "discrete commit classes — answers form a 1D ramp."),
    },
    "sub": {
        0: ("L0 — Pre-operator (min, max) triangle",
            "Same starting geometry — sub's L0 has no signed information "
            "at all (R²(signed_diff) = 0.000 on every eigenvector)."),
        1: ("L1 — Sub diverges first: u_1 locks onto operand b",
            "u_1 = b with R² = 0.78. The subtractor is being isolated. "
            "Mul and add never do this — sub is the only "
            "anti-commutative operator and needs to break symmetry early."),
        2: ("L2 — Shared zero detection (overlaid on b-axis)",
            "u_1 shifts toward zero_flag like mul/add, but b's signal "
            "is still strong. The dispatcher's downstream effect is "
            "now visible: u_2 picks up signed_diff."),
        3: ("L3 — Commit to `signed_diff`",
            "u_1 = signed_diff with R² = 0.76. (min, max) joint R² "
            "collapses 0.65 → 0.05. The leading plane has shifted "
            "decisively off the symmetric basis."),
        4: ("L4 — Three answer-sign clusters separate",
            "Refinement of the signed_diff axis. Positive, zero, and "
            "negative answer classes begin tightening into distinct "
            "clusters."),
        5: ("L5 — Three-vertex simplex (neural-collapse pattern)",
            "Between/within cluster ratio = 10.5×. The three answer-sign "
            "classes form a 2-simplex — ETF-like. Within-class digit "
            "identity in u_3+; (u_1, u_2) shows only the class split."),
    },
}


# ─── Spectral helpers ──────────────────────────────────────────────


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


def compute_normalized_uv(H_stack):
    """Per-layer (u_1, u_2), sign-aligned, centered, unit-variance.

    Normalizing keeps animation scale stable; sign-aligning keeps the
    orientation continuous between frames.
    """
    n_pairs = H_stack.shape[1]
    u1 = np.zeros((N_LAYERS, n_pairs))
    u2 = np.zeros((N_LAYERS, n_pairs))
    for L in range(N_LAYERS):
        u1L, u2L = lap_uv(H_stack[L])
        if L > 0:
            u1L = sign_align(u1[L - 1], u1L)
            u2L = sign_align(u2[L - 1], u2L)
        # center + unit variance
        u1L = (u1L - u1L.mean()) / (u1L.std() + 1e-12)
        u2L = (u2L - u2L.mean()) / (u2L.std() + 1e-12)
        u1[L], u2[L] = u1L, u2L
    return u1, u2


def answer_sign_class(op, a, b):
    if op == "mul":
        return np.where((a * b) == 0, 0, 1)
    if op == "add":
        return np.where((a + b) == 0, 0, 1)
    if op == "sub":
        return np.sign(a - b).astype(int)
    raise ValueError(op)


def class_label(op, c):
    return {
        ("mul", 0): "annihilator (answer = 0)",
        ("mul", 1): "positive answer",
        ("add", 0): "answer = 0",
        ("add", 1): "positive answer",
        ("sub", -1): "negative answer (a < b)",
        ("sub", 0): "zero answer (a = b)",
        ("sub", 1): "positive answer (a > b)",
    }.get((op, int(c)), str(c))


# ─── Per-operator animated figure ──────────────────────────────────


def build_animation(op, H_stack, a, b):
    u1, u2 = compute_normalized_uv(H_stack)
    sign_cls = answer_sign_class(op, a, b)
    classes_present = sorted(set(sign_cls.tolist()))

    # Stable hover text per point
    hover = [f"({ai}, {bi})  {class_label(op, sc)}"
             for ai, bi, sc in zip(a, b, sign_cls)]

    # Pre-compute axis range with small margin (~10%) so morph stays inside
    all_x = u1.flatten(); all_y = u2.flatten()
    x_pad = 0.10 * (all_x.max() - all_x.min())
    y_pad = 0.10 * (all_y.max() - all_y.min())
    x_range = [all_x.min() - x_pad, all_x.max() + x_pad]
    y_range = [all_y.min() - y_pad, all_y.max() + y_pad]

    def make_traces(L):
        traces = []
        for c in classes_present:
            m = sign_cls == c
            traces.append(go.Scatter(
                x=u1[L, m], y=u2[L, m], mode="markers",
                marker=dict(size=11, color=CLASS_COLORS[int(c)],
                            line=dict(width=0.5, color="black")),
                name=class_label(op, c),
                legendgroup=f"cls_{c}",
                showlegend=True,
                text=[hover[i] for i in np.where(m)[0]],
                hovertemplate="%{text}<extra></extra>",
            ))
        return traces

    title0, sub0 = LAYER_TEXT[op][0]
    initial = make_traces(0)
    frames = []
    for L in range(N_LAYERS):
        title_L, sub_L = LAYER_TEXT[op][L]
        frames.append(go.Frame(
            data=make_traces(L),
            name=f"L{L}",
            layout=go.Layout(
                title=dict(text=f"<b>{title_L}</b>",
                           x=0.02, xanchor="left",
                           font=dict(size=18)),
                annotations=[dict(
                    text=sub_L,
                    xref="paper", yref="paper",
                    x=0.02, y=-0.16, xanchor="left", yanchor="top",
                    showarrow=False, align="left",
                    font=dict(size=13, color="#444"),
                    width=900,
                )],
            ),
        ))

    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="Layer: ", font=dict(size=13)),
        pad=dict(t=40),
        steps=[
            dict(
                label=f"L{L}", method="animate",
                args=[[f"L{L}"], dict(
                    mode="immediate",
                    frame=dict(duration=900, redraw=True),
                    transition=dict(duration=900, easing="cubic-in-out"),
                )],
            ) for L in range(N_LAYERS)
        ],
    )]

    play_button = dict(
        type="buttons", direction="left", showactive=False,
        x=0.02, y=1.20, xanchor="left", yanchor="top",
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(
                     frame=dict(duration=1500, redraw=True),
                     transition=dict(duration=1200, easing="cubic-in-out"),
                     fromcurrent=True, mode="immediate")]),
            dict(label="◼ Pause", method="animate",
                 args=[[None], dict(
                     frame=dict(duration=0, redraw=False),
                     mode="immediate")]),
        ],
        pad=dict(r=10, t=4),
    )

    fig = go.Figure(
        data=initial,
        frames=frames,
        layout=go.Layout(
            title=dict(text=f"<b>{title0}</b>",
                       x=0.02, xanchor="left", font=dict(size=18)),
            annotations=[dict(
                text=sub0,
                xref="paper", yref="paper",
                x=0.02, y=-0.16, xanchor="left", yanchor="top",
                showarrow=False, align="left",
                font=dict(size=13, color="#444"),
                width=900,
            )],
            xaxis=dict(title="u_1 (normalized)", range=x_range,
                       showgrid=True, gridcolor="#f0f0f0", zeroline=False),
            yaxis=dict(title="u_2 (normalized)", range=y_range,
                       showgrid=True, gridcolor="#f0f0f0", zeroline=False),
            plot_bgcolor="white",
            sliders=sliders,
            updatemenus=[play_button],
            height=620,
            margin=dict(l=70, r=40, t=120, b=180),
            legend=dict(orientation="h", yanchor="bottom", y=1.04,
                        xanchor="left", x=0.20, font=dict(size=11)),
        ),
    )
    return fig


# ─── Page assembly ─────────────────────────────────────────────────


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>NanoGPT Laplacian — Animated</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{ --accent: #2643d4; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue",
                 Arial, sans-serif;
    color: #222; background: #fafafa; margin: 0; line-height: 1.55;
  }}
  header {{
    max-width: 1100px; margin: 0 auto; padding: 36px 32px 12px;
  }}
  header h1 {{ font-size: 26px; margin: 0 0 4px; }}
  header .sub {{ color: #555; font-size: 14px; }}
  .op-switch {{
    max-width: 1100px; margin: 16px auto 8px; padding: 0 32px;
    display: flex; gap: 8px; align-items: center;
  }}
  .op-switch span {{ color: #555; font-size: 13px; }}
  .op-switch button {{
    border: 1px solid #ccc; background: white; padding: 6px 18px;
    border-radius: 4px; cursor: pointer; font-size: 14px;
    font-family: inherit; color: #333;
  }}
  .op-switch button.active {{
    background: var(--accent); color: white; border-color: var(--accent);
  }}
  .fig-container {{
    max-width: 1100px; margin: 8px auto 0; padding: 0 32px 40px;
  }}
  .fig {{ display: none; }}
  .fig.visible {{ display: block; }}
  .note {{
    max-width: 1100px; margin: 0 auto 40px; padding: 14px 32px;
    background: #eef2ff; border-left: 4px solid var(--accent);
    border-radius: 4px; font-size: 13.5px; color: #333;
  }}
  code {{ background: #f0f0f4; padding: 1px 5px; border-radius: 3px;
          font-size: 13px; }}
</style>
</head><body>
<header>
  <h1>Laplacian Eigenvector Evolution — Animated</h1>
  <p class="sub">Per-layer (u_1, u_2) for the <code>=</code>-token at every
     layer L0..L5. Points smoothly interpolate between layers; titles and
     captions update per frame. Press <b>Play</b> or scrub the slider.</p>
</header>

<div class="op-switch">
  <span>Operator:</span>
  <button id="btn-mul" class="active" data-op="mul">multiplication (×)</button>
  <button id="btn-add" data-op="add">addition (+)</button>
  <button id="btn-sub" data-op="sub">subtraction (−)</button>
</div>

<div class="fig-container">
  <div id="fig-mul" class="fig visible">{fig_mul_html}</div>
  <div id="fig-add" class="fig">{fig_add_html}</div>
  <div id="fig-sub" class="fig">{fig_sub_html}</div>
</div>

<div class="note">
  Eigenvectors are sign-aligned between consecutive layers and each
  layer is centered + unit-variance normalized so animation scale stays
  stable. Color encodes <b>answer sign class</b> for each operator —
  fixed per (a, b) pair across all layers, so you can track a point's
  trajectory by its color. The shape of the cluster IS the model's
  state at that layer.
</div>

<script>
function switchOp(op) {{
  document.querySelectorAll('.fig').forEach(el => el.classList.remove('visible'));
  document.getElementById('fig-' + op).classList.add('visible');
  document.querySelectorAll('.op-switch button').forEach(b => {{
    b.classList.toggle('active', b.dataset.op === op);
  }});
  // Force Plotly resize so the newly-visible figure lays out
  const plot = document.querySelector('#fig-' + op + ' .js-plotly-plot');
  if (plot) {{
    requestAnimationFrame(() => {{ try {{ Plotly.Plots.resize(plot); }} catch (e) {{}} }});
  }}
}}
document.querySelectorAll('.op-switch button').forEach(b => {{
  b.addEventListener('click', () => switchOp(b.dataset.op));
}});
</script>
</body></html>
"""


def main():
    print("Loading data and computing animations...")
    figs = {}
    for op, path in FILES.items():
        H_stack, a, b = load_stack(path)
        figs[op] = build_animation(op, H_stack, a, b)
        print(f"  {op}: built")

    htmls = {}
    first = True
    for op, fig in figs.items():
        htmls[op] = fig.to_html(
            include_plotlyjs=False, full_html=False,
            div_id=f"fig-{op}-inner", config={"displayModeBar": False},
        )
        first = False

    html = PAGE.format(
        fig_mul_html=htmls["mul"],
        fig_add_html=htmls["add"],
        fig_sub_html=htmls["sub"],
    )
    OUT.write_text(html)
    print(f"Wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    try:
        subprocess.Popen(["open", str(OUT)])
    except Exception:
        pass


if __name__ == "__main__":
    main()
