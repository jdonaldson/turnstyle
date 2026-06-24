"""Trace per-layer isolines of properties of (a, b, a*b) through the network.

Loads the (600, 128) =-token states from times_table_trace, recomputes
the joint 2D UMAP, then renders a Plotly 3D plot with a property
dropdown. Selecting a property:
  - recolors the 600 points by that property's value
  - overlays, at each layer Z, the 2D convex hull around each
    property-value cluster

These hulls are the empirical isolines of the property in the
(UMAP-1, UMAP-2) plane at each depth. Watching them deform from
L0..L5 shows when each property becomes geometrically separable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import umap
from scipy.spatial import ConvexHull  # type: ignore

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
STATES = ROOT / "hidden_states.npz"
HTML = ROOT / "times_table_isolines.html"


def leading_digit(p):
    out = np.zeros_like(p)
    nz = p > 0
    out[nz] = (p[nz] // 10 ** np.floor(np.log10(p[nz])).astype(int)).astype(int)
    return out


def build_properties(a, b, p):
    return {
        "a (first operand)": a,
        "b (second operand)": b,
        "a * b (product)": p,
        "a + b (sum)": a + b,
        "min(a,b)": np.minimum(a, b),
        "max(a,b)": np.maximum(a, b),
        "trailing digit of product (p % 10)": p % 10,
        "leading digit of product": leading_digit(p),
        "is product < 10": (p < 10).astype(int),
        "is a==0 or b==0": ((a == 0) | (b == 0)).astype(int),
        "product parity (p % 2)": p % 2,
        "a mod 5": a % 5,
        "b mod 5": b % 5,
        "product mod 5": p % 5,
    }


def hull_lines(xs, ys):
    """Return (hull_x, hull_y) tracing the closed convex hull, or empty if degenerate."""
    if len(xs) < 3:
        return [], []
    pts = np.column_stack([xs, ys])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return [], []
    idx = list(hull.vertices) + [hull.vertices[0]]
    return pts[idx, 0].tolist(), pts[idx, 1].tolist()


def main():
    d = np.load(STATES)
    H = d["H"]
    a = d["a"]
    b = d["b"]
    p = d["product"]
    layer = d["layer"]
    print(f"states: {H.shape}, layers {sorted(set(layer))}")

    print("UMAP fitting (600, 128)...")
    Y = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(H)

    props = build_properties(a, b, p)
    n_layers = int(layer.max()) + 1

    fig = go.Figure()
    # Per property: 2 traces — points (colored by prop value) and hull polylines.
    # Index traces in a flat list, then control visibility per dropdown choice.
    trace_indices_per_prop = {}
    for prop_name, vals in props.items():
        start = len(fig.data)
        n_classes = len(np.unique(vals))
        categorical = n_classes <= 12
        cmin, cmax = (float(vals.min()), float(vals.max())) if not categorical else (0, n_classes - 1)
        colorscale = "Viridis" if not categorical else "Turbo"

        # 1) points
        text = [
            f"{ai}×{bi}={pi}, L{li}<br>{prop_name}={int(v)}"
            for ai, bi, pi, li, v in zip(a, b, p, layer, vals)
        ]
        fig.add_trace(
            go.Scatter3d(
                x=Y[:, 0],
                y=Y[:, 1],
                z=layer,
                mode="markers",
                marker=dict(
                    size=5,
                    color=vals,
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    showscale=True,
                    colorbar=dict(title=prop_name, len=0.7),
                ),
                hovertext=text,
                hoverinfo="text",
                showlegend=False,
                name="points",
                visible=False,
            )
        )

        # 2) per-layer hulls per property value, concatenated via None separators
        hx, hy, hz = [], [], []
        for L in range(n_layers):
            for v in np.unique(vals):
                mask = (layer == L) & (vals == v)
                if mask.sum() < 3:
                    continue
                xs, ys = hull_lines(Y[mask, 0], Y[mask, 1])
                if not xs:
                    continue
                hx.extend(xs + [None])
                hy.extend(ys + [None])
                hz.extend([L] * len(xs) + [None])

        fig.add_trace(
            go.Scatter3d(
                x=hx,
                y=hy,
                z=hz,
                mode="lines",
                line=dict(color="rgba(120,120,120,0.55)", width=2),
                hoverinfo="skip",
                showlegend=False,
                name="hulls",
                visible=False,
            )
        )

        trace_indices_per_prop[prop_name] = (start, start + 1)

    # Default: show the first property
    default_prop = next(iter(props))
    for s, e in [trace_indices_per_prop[default_prop]]:
        fig.data[s].visible = True
        fig.data[e].visible = True

    # Dropdown
    buttons = []
    n_traces = len(fig.data)
    for prop_name in props:
        s, e = trace_indices_per_prop[prop_name]
        vis = [False] * n_traces
        vis[s] = True
        vis[e] = True
        buttons.append(dict(label=prop_name, method="update", args=[{"visible": vis}]))

    fig.update_layout(
        title="Times-table isolines: per-layer convex hulls of property level sets (NanoGPT 1.2M)",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="Layer (=-token)",
            zaxis=dict(tickmode="linear", tick0=0, dtick=1),
        ),
        width=1200,
        height=900,
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.08,
                yanchor="top",
            )
        ],
    )

    fig.write_html(HTML)
    print(f"saved: {HTML}")


if __name__ == "__main__":
    main()
