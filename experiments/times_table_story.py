"""Scrollytelling narrative for the NanoGPT times-table story.

Sticky right-hand viz panel; left column carries the prose split into
"steps."  As each step scrolls into the viewport, the right panel
updates — either by re-coloring/re-highlighting the stacked UMAP (for
the L0 / L1 / L3 events) or by swapping in a different figure (the
backtrack commit heatmap and the role-abstraction NN curve).

Uses scrollama (CDN) + Plotly.  No D3.

Usage:
    python experiments/times_table_story.py
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import (  # noqa: E402
    GPT, encode, STOI, load, arithmetic_sample, svo_sample,
)
from times_table_stacked_umap import (  # noqa: E402
    PAIRS, build_figure, per_layer_umap,
    precompute_concepts, precompute_highlights,
)

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
OUT = ROOT / "times_table_story.html"
STATES = ROOT / "hidden_states.npz"


# ─── Figures ───────────────────────────────────────────────────────────


def fig_stacked_umap() -> go.Figure:
    """Per-layer stacked 2D UMAP, reusing the existing build_figure but
    stripped of the in-figure narrations so the prose can carry them."""
    data = np.load(STATES)
    H, a_arr, b_arr, p_arr, l_arr = (
        data["H"], data["a"], data["b"], data["product"], data["layer"],
    )
    n_layers = int(l_arr.max()) + 1
    coords = per_layer_umap(H, l_arr, n_layers)
    concept_data = precompute_concepts()
    highlight_data = precompute_highlights()
    fig = build_figure(coords, a_arr, b_arr, p_arr, l_arr,
                       n_layers, concept_data, highlight_data)

    # Drop the right-side per-layer narration annotations (xref="paper").
    # Keep the left-side layer labels ("L0", "L1", …) which have xref="x".
    fig.layout.annotations = tuple(
        a for a in fig.layout.annotations if a.xref != "paper"
    )
    fig.update_layout(
        width=None, autosize=True, height=860,
        margin=dict(l=60, r=30, t=90, b=30),
    )
    return fig


# Binding-zoom — three specific swap pairs across all layers,
# with explicit connector lines so the (a,b)↔(b,a) identity is obvious.


def fig_binding_zoom() -> go.Figure:
    """Line chart of within-pair distance over layers, normalized by
    average state norm.  Compared against a random-pair baseline so the
    "binding gap" between same-set and different-set pairs is visible at
    every layer."""
    data = np.load(STATES)
    H, l_arr = data["H"], data["layer"]
    n_layers = int(l_arr.max()) + 1

    pairs = [
        ((2, 6), (6, 2), "rgb(220, 50, 60)"),
        ((3, 4), (4, 3), "rgb(50, 100, 200)"),
        ((1, 8), (8, 1), "rgb(60, 160, 80)"),
    ]
    layers = list(range(n_layers))

    fig = go.Figure()

    # Random-pair baseline first so it draws under the colored lines.
    rng = np.random.default_rng(0)
    baseline = []
    for l in layers:
        m = l_arr == l
        Hl = H[m]
        norm_avg = float(np.mean(np.linalg.norm(Hl, axis=1)))
        ds = []
        for _ in range(800):
            i = int(rng.integers(0, 100))
            j = int(rng.integers(0, 100))
            if i != j:
                ds.append(
                    float(np.linalg.norm(Hl[i] - Hl[j]) / norm_avg)
                )
        baseline.append(float(np.mean(ds)))
    fig.add_trace(go.Scatter(
        x=layers, y=baseline,
        mode="lines+markers",
        name="random pair (mean)",
        line=dict(color="rgba(110, 110, 110, 0.7)", dash="dash", width=2.5),
        marker=dict(size=8, color="rgba(110, 110, 110, 0.7)"),
        hovertemplate="L%{x}  random≈%{y:.3f}<extra></extra>",
    ))

    # Swap pair distances
    for (a1, b1), (a2, b2), color in pairs:
        idx1 = a1 * 10 + b1
        idx2 = a2 * 10 + b2
        per = []
        for l in layers:
            m = l_arr == l
            Hl = H[m]
            norm_avg = float(np.mean(np.linalg.norm(Hl, axis=1)))
            per.append(
                float(np.linalg.norm(Hl[idx1] - Hl[idx2]) / norm_avg)
            )
        fig.add_trace(go.Scatter(
            x=layers, y=per,
            mode="lines+markers",
            name=f"({a1},{b1}) ↔ ({a2},{b2})",
            line=dict(color=color, width=3),
            marker=dict(size=11),
            hovertemplate=(f"({a1},{b1}) ↔ ({a2},{b2})<br>"
                           "L%{x}  d=%{y:.3f}<extra></extra>"),
        ))

    # Annotations on L0 and L1
    fig.add_annotation(
        x=0, y=0.07, ax=40, ay=-50,
        text="<b>L0</b>  pairs collapsed<br>(swap dist ≪ random)",
        showarrow=True, arrowhead=2, arrowwidth=1.3,
        font=dict(size=11, color="#444"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#aaa", borderwidth=1,
    )
    fig.add_annotation(
        x=1, y=0.29, ax=60, ay=-40,
        text="<b>L1</b>  binding event<br>(swap dist jumps 4–5×)",
        showarrow=True, arrowhead=2, arrowwidth=1.3,
        font=dict(size=11, color="#444"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#aaa", borderwidth=1,
    )
    fig.add_annotation(
        x=4, y=0.42, ax=-30, ay=-60,
        text="<b>L2–L5</b>  binding persists<br>"
             "(swap stays well below random)",
        showarrow=True, arrowhead=2, arrowwidth=1.3,
        font=dict(size=11, color="#444"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#aaa", borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text="Within-pair distance ‖h(a,b) − h(b,a)‖ / ‖h‖<sub>avg</sub>"
                 "  vs layer",
            font=dict(size=14),
        ),
        xaxis=dict(title="Layer", tickmode="linear",
                   gridcolor="#eee", zerolinecolor="#ccc"),
        yaxis=dict(title="Normalized L2 distance",
                   range=[0, None], gridcolor="#eee", zerolinecolor="#ccc"),
        height=560, autosize=True,
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=70, r=30, t=80, b=60),
        legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.03,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ccc", borderwidth=1),
    )
    return fig


def _cv_acc(X, y, n_splits=5, n_seeds=5) -> float:
    accs = []
    for seed in range(n_seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000),
            )
            clf.fit(X[tr], y[tr])
            accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))


def fig_phase_map() -> go.Figure:
    """A summary heatmap: per-layer decodability of 10 features, sorted
    so the input→route→output phase flow shows as a diagonal of brightness
    sweeping from upper-left to lower-right."""
    data = np.load(STATES)
    H, a, b, prod, layer = (
        data["H"], data["a"], data["b"], data["product"], data["layer"],
    )
    n_layers = int(layer.max()) + 1

    first_digit = np.array([int(str(int(p))[0]) for p in prod])
    ones_digit = (prod % 10).astype(int)
    tens_digit = (prod // 10).astype(int)

    features = [
        ("operand a   (input id)",   a,                              "input"),
        ("operand b   (input id)",   b,                              "input"),
        ("min(a, b)   (input set)",  np.minimum(a, b),               "input"),
        ("max(a, b)   (input set)",  np.maximum(a, b),               "input"),
        ("a == b      (special-case)", (a == b).astype(int),         "route"),
        ("zero        (special-case)", (a * b == 0).astype(int),     "route"),
        ("product length (output timing)", (prod >= 10).astype(int), "output"),
        ("tens digit  (output digit)", tens_digit,                   "output"),
        ("first digit (output digit)", first_digit,                  "output"),
        ("ones digit  (output digit)", ones_digit,                   "output"),
    ]

    Z = []
    names = []
    kinds = []
    for name, y, kind in features:
        names.append(name)
        kinds.append(kind)
        Z.append([_cv_acc(H[layer == l], y[layer == l]) for l in range(n_layers)])
    Z = np.array(Z)

    # Row label colors by phase
    color_by_kind = {"input": "#c63a3a", "route": "#888", "output": "#2643d4"}

    fig = go.Figure(go.Heatmap(
        z=Z,
        x=[f"L{l}" for l in range(n_layers)],
        y=names,
        colorscale="YlGnBu",
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in Z],
        texttemplate="%{text}",
        textfont=dict(size=11, color="#222"),
        colorbar=dict(title="decodability", thickness=14, len=0.7),
        hovertemplate="%{y}<br>%{x}  acc=%{z:.3f}<extra></extra>",
    ))

    # Color-tag the row labels by phase (input red / route gray / output blue)
    for i, kind in enumerate(kinds):
        # Plotly doesn't support per-tick colors directly; emulate with
        # annotations on the left axis.
        fig.add_annotation(
            x=-0.07, y=i, xref="paper", yref="y",
            text="●",
            showarrow=False, xanchor="right",
            font=dict(size=12, color=color_by_kind[kind]),
        )

    # Vertical phase guides between L1/L2 (end of binding) and L3/L4
    # (end of compute) — soft transitions, not hard boundaries.
    for x_pos, label in [(1.5, "input → compute"),
                         (3.5, "compute → output")]:
        fig.add_shape(
            type="line", x0=x_pos, x1=x_pos,
            y0=-0.5, y1=len(names) - 0.5,
            line=dict(color="rgba(80,80,80,0.35)", dash="dot", width=1.5),
        )
        fig.add_annotation(
            x=x_pos, y=len(names) - 0.3,
            text=label,
            showarrow=False, xanchor="center", yanchor="bottom",
            font=dict(size=10, color="#666"),
        )

    fig.update_layout(
        title=dict(
            text="Per-layer feature decodability — the phase map.  "
                 "Red rows = input features (decay).  "
                 "Blue rows = output features (rise).  "
                 "Gray rows = routing / special-case features.",
            font=dict(size=13),
        ),
        height=560, autosize=True,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=240, r=30, t=110, b=50),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def fig_input_output_rotation() -> go.Figure:
    """Input-side vs output-side feature decodability across layers.
    Solid lines = features about the operands you saw (input).  Dotted
    lines = features about the characters about to be emitted (output)."""
    data = np.load(STATES)
    H, a, b, prod, layer = (
        data["H"], data["a"], data["b"], data["product"], data["layer"],
    )
    n_layers = int(layer.max()) + 1

    first_digit = np.array([int(str(int(p))[0]) for p in prod])
    ones_digit = (prod % 10).astype(int)

    feats = [
        ("operand a", a, "input", "rgb(220, 60, 70)"),
        ("operand b", b, "input", "rgb(255, 130, 90)"),
        ("first digit of product", first_digit, "output", "rgb(50, 100, 200)"),
        ("ones digit of product", ones_digit, "output", "rgb(80, 180, 200)"),
    ]
    layers = list(range(n_layers))

    fig = go.Figure()
    for name, y, kind, color in feats:
        per = [_cv_acc(H[layer == l], y[layer == l]) for l in layers]
        dash = "solid" if kind == "input" else "dot"
        label = f"{name}  ({kind})"
        fig.add_trace(go.Scatter(
            x=layers, y=per,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=3, dash=dash),
            marker=dict(size=11),
            hovertemplate=f"{label}<br>L%{{x}}  acc=%{{y:.3f}}<extra></extra>",
        ))

    fig.add_annotation(
        x=1, y=0.97, ax=80, ay=-20,
        text="<b>L1</b>  operand_a peaks (0.97)",
        showarrow=True, arrowhead=2, arrowwidth=1.3,
        font=dict(size=11, color="#444"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#aaa", borderwidth=1,
    )
    fig.add_annotation(
        x=3, y=0.55, ax=0, ay=-60,
        text="<b>L3</b>  crossover<br>(input ≈ output)",
        showarrow=True, arrowhead=2, arrowwidth=1.3,
        font=dict(size=11, color="#444"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#aaa", borderwidth=1,
    )
    fig.add_annotation(
        x=5, y=1.00, ax=-40, ay=50,
        text="<b>L5</b>  first_digit at 1.00<br>"
             "(but ones_digit stuck at 0.58 —<br>"
             "deferred to next position)",
        showarrow=True, arrowhead=2, arrowwidth=1.3,
        font=dict(size=11, color="#444"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#aaa", borderwidth=1,
    )
    fig.add_annotation(
        x=5, y=0.27, ax=-50, ay=30,
        text="<b>operand_a destroyed</b><br>"
             "(0.97 → 0.27, below L0 baseline of 0.41)",
        showarrow=True, arrowhead=2, arrowwidth=1.3,
        font=dict(size=11, color="#444"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#aaa", borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text="Input-to-output rotation: operand decodability collapses "
                 "as next-character decodability rises",
            font=dict(size=14),
        ),
        xaxis=dict(title="Layer", tickmode="linear", gridcolor="#eee"),
        yaxis=dict(title="5-fold CV accuracy",
                   range=[0, 1.05], gridcolor="#eee"),
        height=560, autosize=True,
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=70, r=30, t=80, b=60),
        legend=dict(yanchor="middle", y=0.5, xanchor="right", x=0.97,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ccc", borderwidth=1),
    )
    return fig


def fig_cross_operator(model: GPT) -> go.Figure:
    """UMAP of L5 states for all 300 prompts (100 per operator),
    colored by the first character of the result and shaped by operator.
    Shared output circuits show up as colored clusters that mix shapes."""
    import umap as umap_lib

    rows = []
    for op in ("+", "-", "*"):
        for a in range(10):
            for b in range(10):
                r = ((a + b) if op == "+" else
                     (a - b) if op == "-" else (a * b))
                rows.append(dict(prompt=f"{a}{op}{b}=", op=op,
                                 result=r, first_char=str(r)[0]))

    # Capture L5 state at the '=' position for each prompt.
    states = []
    with torch.no_grad():
        for r in rows:
            idx = torch.tensor([encode(r["prompt"])], dtype=torch.long)
            _, _, all_states = model(idx, return_states=True)
            states.append(all_states[-1][0, -1].detach().numpy())
    H = np.stack(states)

    reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.20,
                            random_state=42)
    Y = reducer.fit_transform(H)

    # Map first_char to a position on a discrete colorscale.
    unique_chars = sorted(set(r["first_char"] for r in rows),
                          key=lambda c: (c == "-", c))
    char_to_idx = {c: i for i, c in enumerate(unique_chars)}
    palette = [
        "#8b8b8b",  # '-'
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ][: len(unique_chars)]

    shape_map = {"+": "circle", "-": "diamond", "*": "square"}
    fig = go.Figure()
    for op in ("+", "-", "*"):
        ixs = [i for i, r in enumerate(rows) if r["op"] == op]
        colors = [palette[char_to_idx[rows[i]["first_char"]]] for i in ixs]
        texts = [f"{rows[i]['prompt']}{rows[i]['result']}  "
                 f"(emits '{rows[i]['first_char']}')"
                 for i in ixs]
        fig.add_trace(go.Scatter(
            x=Y[ixs, 0], y=Y[ixs, 1],
            mode="markers",
            marker=dict(size=11, color=colors, symbol=shape_map[op],
                        line=dict(color="white", width=1.2),
                        opacity=0.9),
            text=texts, hoverinfo="text",
            name=f"op = '{op}'",
        ))

    # Color legend strip — small marker traces for each first_char.
    for c, color in zip(unique_chars, palette):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color, symbol="circle"),
            name=f"emit '{c}'",
            legendgroup="firstchar", showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text="L5 hidden states for 300 prompts (100 per operator) — "
                 "shape = operator, color = next character to emit.<br>"
                 "<span style='font-size:12px;color:#666'>"
                 "If output circuits are shared, color forms clusters that "
                 "mix shapes.</span>",
            font=dict(size=13),
        ),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   title="UMAP-1"),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   title="UMAP-2"),
        height=620, autosize=True,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=30, t=110, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ccc", borderwidth=1, font=dict(size=11)),
    )
    return fig


# Backtrack heatmap — when does the answer token become argmax?


@torch.no_grad()
def _probs_for_prompt(model: GPT, prompt: str):
    idx = torch.tensor([encode(prompt)], dtype=torch.long)
    _, _, states = model(idx, return_states=True)
    H = torch.stack(states, dim=0)[:, 0, :, :]
    return torch.softmax(model.head(model.ln_f(H)), dim=-1).numpy()


def fig_commit_heatmap(model: GPT) -> go.Figure:
    cases = [("2 × 3 = 6  (non-zero)", "2*3=", "6"),
             ("0 × 9 = 0  (annihilator)", "0*9=", "0")]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[c[0] for c in cases],
                        horizontal_spacing=0.16)
    for col, (_, prompt, ans) in enumerate(cases, start=1):
        probs = _probs_for_prompt(model, prompt)
        grid = probs[:, :, STOI[ans]]
        labels = list(prompt)
        n_layer = grid.shape[0]
        fig.add_trace(go.Heatmap(
            z=grid, x=labels,
            y=[f"L{l}" for l in range(n_layer)],
            colorscale="Magma", zmin=0, zmax=1.0,
            colorbar=dict(title=f"P('{ans}')",
                          x=0.45 + 0.55 * (col - 1), len=0.7),
            text=[[f"{v:.2f}" for v in row] for row in grid],
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            hovertemplate="L=%{y}  token=%{x}<br>P=%{z:.3f}<extra></extra>",
        ), row=1, col=col)
    fig.update_layout(
        title=dict(
            text="P(answer token) at each (layer, position) — logit lens",
            font=dict(size=14),
        ),
        height=520, autosize=True,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=30, t=100, b=50),
    )
    for col in (1, 2):
        fig.update_xaxes(title_text="position (token)", row=1, col=col)
        fig.update_yaxes(title_text="layer", autorange="reversed",
                         row=1, col=col)
    return fig


# Role-abstraction line chart — fraction of top-K NN with same source char.


def _build_corpus(n: int = 200, win: int = 32) -> list[str]:
    rng = random.Random(1)
    parts = []
    while sum(len(p) for p in parts) < n * win + win:
        parts.append(arithmetic_sample(rng) if rng.random() < 0.7
                     else svo_sample(rng))
    stream = "".join(parts)
    rng2 = random.Random(99)
    return [stream[s:s + win] for s in
            (rng2.randint(0, len(stream) - win - 1) for _ in range(n))]


@torch.no_grad()
def _capture_corpus(model: GPT, windows: list[str]):
    acts_per_layer = [[] for _ in range(model.cfg.n_layer)]
    tok_ids: list[int] = []
    for w in windows:
        idx = torch.tensor([encode(w)], dtype=torch.long)
        _, _, states = model(idx, return_states=True)
        for layer, h in enumerate(states):
            acts_per_layer[layer].append(h[0].numpy())
        tok_ids.extend(STOI[c] for c in w)
    A = np.stack([np.concatenate(a, axis=0) for a in acts_per_layer], axis=0)
    return A, np.array(tok_ids, dtype=np.int32)


def fig_abstraction_curve(model: GPT) -> go.Figure:
    windows = _build_corpus()
    A, tok_ids = _capture_corpus(model, windows)
    n_layer = A.shape[0]
    queries = [("'=' in  2*3=", "2*3=", 3),
               ("'2' in  2*3=", "2*3=", 0)]
    fig = go.Figure()
    K = 10
    for label, prompt, pos in queries:
        idx = torch.tensor([encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            _, _, states = model(idx, return_states=True)
        q_id = STOI[prompt[pos]]
        per_layer = []
        for layer in range(n_layer):
            h_q = states[layer][0, pos].detach().numpy()
            H = A[layer]
            sims = (H @ h_q) / (np.linalg.norm(H, axis=1) *
                                np.linalg.norm(h_q) + 1e-8)
            top = np.argsort(-sims)[:K]
            per_layer.append(float((tok_ids[top] == q_id).mean()))
        fig.add_trace(go.Scatter(
            x=list(range(n_layer)), y=per_layer,
            mode="lines+markers", name=label,
            line=dict(width=3), marker=dict(size=11),
        ))
    fig.update_layout(
        title=dict(
            text=f"Fraction of top-{K} nearest neighbors sharing the "
                 f"query's source character",
            font=dict(size=14),
        ),
        xaxis=dict(title="Layer", tickmode="linear", gridcolor="#eee"),
        yaxis=dict(title=f"frac(top-{K} NN with same char)",
                   range=[0, 1.05], gridcolor="#eee"),
        height=460, autosize=True, hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=20, t=80, b=50),
    )
    return fig


# ─── Steps definition ──────────────────────────────────────────────────


STEPS = [
    dict(
        anchor="set",
        viz="stacked-umap",
        color=3, highlight=0,
        title="L0 — Set encoding",
        body="""
<p>The token-embedding layer already separates the <em>set</em>
{a, b} almost perfectly: linear probes recover <code>min(a, b)</code>
at 98% and <code>max(a, b)</code> at 97%.  But the role assignment
— which operand is <em>a</em>, which is <em>b</em> — sits at just
41%, barely above an unordered-set baseline.  The product itself is
already at R² = 0.95 because Ridge can compose it from min and max.
Nothing here is binding yet.  It's positional encoding plus token
embedding, nothing more.</p>""",
        hint="In the UMAP above: top band (L0), colored by "
             "<code>min(a, b)</code>.  Note that swap pairs (2, 3) and "
             "(3, 2) collapse to the same point at L0.",
    ),
    dict(
        anchor="binding",
        viz="binding-zoom",
        title="L0 → L1 — Position binding",
        body="""
<p>One transformer block is enough to attach each operand to a role.
At L0, <code>h(2, 6)</code> and <code>h(6, 2)</code> are nearly the
same point — within-set variance is only 0.9% of total, the swap
distance is roughly seven times smaller than the distance to a
random pair.  After a single attention pass, within-set variance
jumps to 9.3%.  That's a 10× change in one layer — the largest
information-gain event anywhere in this network.  Operand-a
decodability rises 41% → 97%.</p>
<p>L2+ partly re-mixes role information with other axes, so the
swap pairs drift back closer in UMAP space — but they don't fully
re-collapse, and linear probes still recover order at 50–80% through
L4.  The binding event happens at L1; the later layers
rearrange how that information is stored.</p>""",
        hint="Right panel: three swap pairs and a random-pair baseline. "
             "At L0 every swap distance is ≪ the random baseline — pairs "
             "are collapsed.  At L1 the swap distances jump 4–5× while "
             "random distance only doubles.  Through L2–L5 the random "
             "baseline grows much faster than the swap distances, so the "
             "gap between bound and unbound stays wide — binding persists, "
             "even though L2+ doesn't show it as crisply in UMAP space.",
    ),
    dict(
        anchor="gutter",
        viz="stacked-umap",
        color=6, highlight=1,
        title="L3 — Annihilator gutter",
        body="""
<p>By L3 the network has built a single 1-D direction that cleanly
separates the 19 zero-pairs from the 81 non-zero pairs (Fisher's
discriminant, <code>d′ ≈ 8.5</code>).  Ablating this direction collapses
zero-detection from 94% → 48% (chance), but leaves operand
sub-structure intact.  The gutter is a <em>flag</em>, not a
value — and crucially, zero-detection was already 100% at L0.
The L3 event is not where zero is <em>discovered</em>;
it's where zero is <em>consolidated</em> into a single routing channel
the later layers can read.</p>""",
        hint="In the UMAP: zero-product pairs are highlighted (red ends "
             "of the colorscale).  At L3 they separate sharply along "
             "the layer's primary axis.",
    ),
    dict(
        anchor="commit",
        viz="commit-heatmap",
        title="L3 → L5 — Answer commit timing",
        body="""
<p>Run the logit lens at every (layer, position): apply
<code>head(ln_f(h))</code> and ask <em>P(answer token)</em>.  For
<code>2 × 3 = 6</code>, <code>P('6')</code> at the <code>=</code>
position is 25% at L1, drops to 0.2% at L3 (the routing dip), then
commits to 99.8% at L5.  For <code>0 × 9 = 0</code>, the annihilator
short-cuts everything: <code>P('0')</code> at the <code>=</code>
position is 39% at L3 already and 84% by L4.</p>
<p>Notice also that the <code>*</code> operator's residual stream
lights up at L2–L4 in both cases.  The model uses the operator slot
as a scratchpad, writing the candidate answer there before retrieving
it at the <code>=</code> position.</p>""",
        hint="Left: the non-zero case has to wait until L5 to commit. "
             "Right: the zero case commits two layers earlier (L3). "
             "That's the gutter buying compute back.",
    ),
    dict(
        anchor="rotation",
        viz="io-rotation",
        title="L2 → L5 — Input-to-output rotation",
        body="""
<p>The commit isn't a single event — it's the visible tip of a deeper
rearrangement.  Input-side features (operand identities) <em>decay</em>
through the middle layers while output-side features (the next character
to emit) <em>rise</em>.  At L1, <code>operand_a</code> decodes at 97%
while <code>first_digit_of_product</code> sits at 31% (chance for a
10-way problem).  By L5 they've flipped: <code>operand_a</code> at 27%,
<code>first_digit_of_product</code> at 100%.</p>
<p>Two things to notice.  <strong>The operand is not just diluted, it's
overwritten</strong> — L5 sits <em>below</em> the L0 baseline of 41%,
so the network actively replaces operand encoding with output encoding
in the same residual dimensions.  And <strong>the rotation is
asymmetric</strong>: <code>first_digit_of_product</code> reaches 100%
but <code>ones_digit_of_product</code> tops out at 58%.  The
<code>=</code>-position state only needs to encode <em>what comes
next</em>; the ones digit of a two-digit answer is emitted from the
position after that and is computed there.  Carry / multi-digit support
is implemented as an autoregressive split, not as parallel encoding of
the full answer.</p>""",
        hint="Solid lines = input-side features (decay).  Dotted lines = "
             "output-side features (rise).  The crossover at L3 is where "
             "the representation switches from 'what did I see' to 'what "
             "do I emit.'  The persistent ones-digit gap at L5 is the "
             "carry-handling signature.",
    ),
    dict(
        anchor="shared",
        viz="cross-operator",
        title="L4 – L5 — Shared output circuits across operators",
        body="""
<p>If the late-layer state really is sorted by 'what character do I
emit next,' then prompts with the same next character should cluster
together regardless of which operator produced them.  Run 300 prompts
(100 each across <code>+</code>, <code>−</code>, <code>×</code>) and
capture the L5 state at the <code>=</code> position.  Of every prompt's
top-5 L5 nearest neighbors:</p>
<ul>
<li><strong>99.7%</strong> share the next character (chance: 12.6%)</li>
<li>70.3% share the operator (chance: 33.3%)</li>
</ul>
<p>Within-class cosine similarity by first character is 0.75–0.87;
across-class is ~0.10.  A gap of ~0.7 cosine — nearly orthogonal
between distinct output buckets, tight clusters within them.  The L5
representation is essentially one-hot-by-first-character: 11 channels,
one per emittable character (10 digits plus the minus sign for
negative subtraction results).</p>
<p>So yes — there's a shared "emit '5'" circuit that fires for
<code>2+3=</code>, <code>9−4=</code>, and <code>1×5=</code>.  The model
didn't learn 3 separate operator-specific lookup tables; it learned
operand encoding (early layers, per-operator) and result encoding (late
layers, shared) connected by an arithmetic computation in between.</p>""",
        hint="Each point is one prompt's L5 state, UMAP-reduced.  Shape = "
             "operator (○ +, ◇ −, ▢ ×); color = first character of result. "
             "If output circuits are shared, you should see colored clusters "
             "that mix all three shapes freely.",
    ),
    dict(
        anchor="phasemap",
        viz="phase-map-viz",
        title="Summary — Per-layer phase map",
        body="""
<p>Pulling all of the above into one view: per-layer linear
decodability of 10 features, sorted to make the input → route → output
phase flow visible as a diagonal of brightness.  Red row dots = input
features (operand identity and set membership).  Blue = output features
(what character to emit and when).  Gray = special-case routing
(zero-pair gutter, diagonal pairs).</p>
<p>Read the diagonal from upper-left to lower-right.  L0 has input
identity and set features active.  L1 spikes operand_a / operand_b to
nearly 1 (binding event).  L3 keeps zero and a==b detectors high while
output features start to climb.  L4-L5 has the output digits saturating
while the input rows fade below their L0 baseline.  The fact that the
brightness moves through the table at roughly one column per layer is
the network's pipeline made visible.</p>
<p>Two caveats worth keeping in mind.  The phases are gradients, not
hard boundaries — every layer is doing several things in parallel
because of the residual stream.  And the allocation is
depth-dependent: a 6-layer network has to compress this pipeline to
one phase per layer, while a 50-layer model would spread the same
phases over many more blocks and leave room for filler propagation
layers.</p>""",
        hint="Look for the diagonal: bright cells move from "
             "upper-left (input features at early layers) through the "
             "middle (routing) to lower-right (output features at late "
             "layers).  Each row is a feature; each column is a layer.",
    ),
    dict(
        anchor="abstraction",
        viz="abstraction-curve",
        title="L4 – L5 — Character → role abstraction",
        body="""
<p>Take a query cell (prompt + position + layer) and find its nearest
neighbors in a diverse corpus.  At L0–L3 every neighbor of the
<code>'='</code> in <code>"2*3="</code> is another <code>'='</code>.
At L4 the first cross-character neighbors appear: <code>'1'</code>
tokens at the start of two-digit answers <code>"...=16"</code>.
At L5 the cross-character neighbors include <code>'-'</code> at the
start of negative numbers and the <code>'='</code> in
<code>7*9=63</code>.  Same role, different character.  The activation
has shifted from <em>character</em> to <em>role</em>.</p>""",
        hint="Both queries drop sharply at L4 and L5.  That's when "
             "the late-layer hidden state stops being 'which character' "
             "and starts being 'which structural role.'",
    ),
]


# ─── HTML page template ────────────────────────────────────────────────


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Times-Table: A Layer-by-Layer Story</title>
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
    display: grid; grid-template-columns: 1fr 1.15fr; gap: 32px;
    max-width: 1400px; margin: 0 auto; padding: 0 32px;
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
    .viz-col {{ position: sticky; height: 70vh; }}
    .stepper {{ display: none; }}
  }}
</style>
</head><body>

<header>
  <h1>Times-Table: A Layer-by-Layer Story</h1>
  <p class="subtitle">NanoGPT 1.2M (6L, 128-d, char-vocab) trained on
     <code>+ − ×</code> with operands 0–9.  100% multiplication accuracy.
     What is each layer actually <em>doing</em>?</p>
</header>

<div class="summary">
  The network gets the operand <em>set</em> {{a, b}} for free at L0
  (min/max ≈ 95%) but does not yet know which is <em>a</em> vs <em>b</em>.
  L1 binds positions to roles via a single attention pass.  L3 is routing:
  a 1-D Fisher direction separates the 19 zero-pairs from the rest.  For
  zero pairs the answer commits at L3 already; non-zero pairs only commit
  at L5.  By L5 the <em>character</em> identity of a token has been
  abstracted away into a <em>functional role</em>.
  <br/><span style="color:#666">Scroll down — the right panel updates as
  you read.</span>
</div>

<nav class="stepper" id="stepper">
  {stepper_links}
</nav>

<div class="scrolly">
  <div class="story-col">
    {steps_html}
  </div>
  <div class="viz-col">
    <div id="stacked-umap" class="viz-fig visible">{fig_umap_html}</div>
    <div id="binding-zoom" class="viz-fig">{fig_binding_html}</div>
    <div id="commit-heatmap" class="viz-fig">{fig_commit_html}</div>
    <div id="io-rotation" class="viz-fig">{fig_rotation_html}</div>
    <div id="cross-operator" class="viz-fig">{fig_xop_html}</div>
    <div id="phase-map-viz" class="viz-fig">{fig_phase_html}</div>
    <div id="abstraction-curve" class="viz-fig">{fig_abstract_html}</div>
  </div>
</div>

<footer>
  All figures computed fresh from
  <code>experiments/data/nanogpt_times_table/hidden_states.npz</code> plus,
  for the backtrack heatmap and abstraction NN, a diverse corpus running
  through the model live.  Scripts:
  <code>times_table_information_gain.py</code>,
  <code>times_table_binding_verify.py</code>,
  <code>times_table_backtrack.py</code>,
  <code>times_table_activation_neighbors.py</code>,
  <code>times_table_stacked_umap.py</code>.
</footer>

<script>
const STEPS = {steps_data};

function applyStep(idx) {{
  const step = STEPS[idx];
  if (!step) return;
  // Toggle visible figure
  document.querySelectorAll('.viz-fig').forEach(el =>
    el.classList.remove('visible'));
  const figEl = document.getElementById(step.viz);
  if (figEl) {{
    figEl.classList.add('visible');
    // A Plotly figure inside display:none doesn't lay out correctly
    // until its container is visible.  Force a resize after reveal.
    const plot = figEl.querySelector('.js-plotly-plot');
    if (plot) {{
      requestAnimationFrame(() => {{
        try {{ Plotly.Plots.resize(plot); }} catch (e) {{}}
      }});
    }}
  }}
  // If it's the UMAP, programmatically apply the matching color +
  // highlight by re-using the figure's own button restyle args.
  if (step.viz === 'stacked-umap' &&
      step.color !== null && step.highlight !== null) {{
    const div = document.getElementById('stacked-umap');
    const inner = div.querySelector('.js-plotly-plot') || div;
    try {{
      const menus = inner.layout && inner.layout.updatemenus;
      if (menus) {{
        Plotly.restyle(inner, menus[0].buttons[step.color].args[0]);
        Plotly.restyle(inner, menus[1].buttons[step.highlight].args[0]);
      }}
    }} catch (e) {{
      console.error('restyle failed', e);
    }}
  }}
  // Update side stepper
  document.querySelectorAll('.stepper a').forEach((a, i) =>
    a.classList.toggle('is-active', i === idx));
}}

window.addEventListener('load', () => {{
  // Initial step
  applyStep(0);
  // Set up scrollama
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

  // Make stepper links clickable
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
    print("Building stacked UMAP...")
    fig_umap = fig_stacked_umap()
    print("Building binding zoom...")
    fig_binding = fig_binding_zoom()
    print("Loading model for live-corpus figures...")
    model = load("cpu")
    print("Building commit heatmap...")
    fig_commit = fig_commit_heatmap(model)
    print("Building input-output rotation chart...")
    fig_rotation = fig_input_output_rotation()
    print("Building cross-operator UMAP...")
    fig_xop = fig_cross_operator(model)
    print("Building phase map...")
    fig_phase = fig_phase_map()
    print("Building abstraction curve...")
    fig_abstract = fig_abstraction_curve(model)

    # Render figures.  Plotly JS comes from the page-level <script> tag
    # so include_plotlyjs=False here for all of them.
    fig_umap_html = fig_umap.to_html(
        include_plotlyjs=False, full_html=False,
        div_id="umap-fig", config={"displayModeBar": False},
    )
    fig_binding_html = fig_binding.to_html(
        include_plotlyjs=False, full_html=False,
        div_id="binding-fig", config={"displayModeBar": False},
    )
    fig_commit_html = fig_commit.to_html(
        include_plotlyjs=False, full_html=False,
        div_id="commit-fig", config={"displayModeBar": False},
    )
    fig_rotation_html = fig_rotation.to_html(
        include_plotlyjs=False, full_html=False,
        div_id="rotation-fig", config={"displayModeBar": False},
    )
    fig_xop_html = fig_xop.to_html(
        include_plotlyjs=False, full_html=False,
        div_id="xop-fig", config={"displayModeBar": False},
    )
    fig_phase_html = fig_phase.to_html(
        include_plotlyjs=False, full_html=False,
        div_id="phase-fig", config={"displayModeBar": False},
    )
    fig_abstract_html = fig_abstract.to_html(
        include_plotlyjs=False, full_html=False,
        div_id="abstract-fig", config={"displayModeBar": False},
    )

    steps_html = "\n".join(
        STEP_TEMPLATE.format(
            idx=i, anchor=s["anchor"], title=s["title"],
            body=s["body"], hint=s["hint"],
        )
        for i, s in enumerate(STEPS)
    )
    stepper_links = "\n".join(
        f'  <a href="#{s["anchor"]}" data-idx="{i}">{s["title"]}</a>'
        for i, s in enumerate(STEPS)
    )
    steps_data = json.dumps([
        dict(viz=s["viz"], color=s.get("color"), highlight=s.get("highlight"))
        for s in STEPS
    ])

    html = PAGE.format(
        steps_html=steps_html,
        stepper_links=stepper_links,
        fig_umap_html=fig_umap_html,
        fig_binding_html=fig_binding_html,
        fig_commit_html=fig_commit_html,
        fig_rotation_html=fig_rotation_html,
        fig_xop_html=fig_xop_html,
        fig_phase_html=fig_phase_html,
        fig_abstract_html=fig_abstract_html,
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
