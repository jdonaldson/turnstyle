"""Stacked per-layer 2D UMAP of times-table activations.

Each layer gets its own UMAP projection (100 points each).
Panels are stacked vertically, L0 at top. For every (a,b) pair a
line connects the point across all layers.

Controls:
  • Color row  — switch color encoding between mathematical concepts
  • Highlight row — dim everything except the selected behavioral group
  • Scroll to zoom, drag to pan

Usage:
    python experiments/times_table_stacked_umap.py
    (requires experiments/data/nanogpt_times_table/hidden_states.npz
     — run times_table_trace.py --no-train first if missing)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
STATES = ROOT / "hidden_states.npz"
HTML = ROOT / "times_table_stacked_umap.html"

Y_SPACING = 14.0
NORM_SCALE = 3.0

PAIRS = [(a, b) for a in range(10) for b in range(10)]  # 100 pairs, stable order

# ---------- color concepts ----------

CONCEPTS: list[tuple] = [
    ("a × b",        lambda a, b: a * b,           0,  81, "Viridis"),
    ("operand a",    lambda a, b: a,                0,   9, "Plasma"),
    ("operand b",    lambda a, b: b,                0,   9, "Plasma"),
    ("min(a,b)",     lambda a, b: min(a, b),        0,   9, "Plasma"),
    ("max(a,b)",     lambda a, b: max(a, b),        0,   9, "Plasma"),
    ("a + b",        lambda a, b: a + b,            0,  18, "Viridis"),
    ("zero product", lambda a, b: int(a * b == 0),  0,   1, "RdBu_r"),
    ("product < 10", lambda a, b: int(a * b < 10),  0,   1, "RdBu_r"),
    ("a == b",       lambda a, b: int(a == b),       0,   1, "RdBu_r"),
]

# ---------- highlight groups ----------

HIGHLIGHTS: list[tuple] = [
    # (label, predicate, tooltip description)
    ("all pairs",
     lambda a, b: True,
     "Show all 100 pairs"),
    ("zero annihilator  a=0 or b=0",
     lambda a, b: a == 0 or b == 0,
     "One operand is 0 → product always 0.  The gutter group."),
    ("identity  a=1 or b=1",
     lambda a, b: a == 1 or b == 1,
     "One operand is 1 → product = the other operand."),
    ("perfect squares  a=b",
     lambda a, b: a == b,
     "Diagonal: same operand twice.  Should cluster as unordered-set encoding collapses a,b."),
    ("commutative group  a×b = 12",
     lambda a, b: a * b == 12,
     "All pairs with product=12: (2,6),(6,2),(3,4),(4,3).  Do they cluster by L5?"),
    ("commutative group  a×b = 6",
     lambda a, b: a * b == 6,
     "All pairs with product=6: (1,6),(6,1),(2,3),(3,2)."),
    ("large products  ≥56",
     lambda a, b: a * b >= 56,
     "Hardest products: 7×8 through 9×9.  Upper-right of the table."),
    ("prime products",
     lambda a, b: a * b in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47},
     "Products that are prime numbers — uniquely factorable."),
]

# ---------- per-layer narrations ----------

LAYER_NARRATIONS = [
    # L0
    ("<b>L0 — Token embedding</b><br>"
     "Raw input. Already encodes operands as an<br>"
     "unordered set: min/max decodable at ~95%.<br>"
     "Zero detection: 100%.  Output-length plan<br>"
     "(product&lt;10): 93%.  Individual operand<br>"
     "positions are not yet encoded."),
    # L1
    ("<b>L1 — Early structure</b><br>"
     "Route-type signal peaks here (100% CV) —<br>"
     "structural/syntactic encoding happens fast.<br>"
     "Operand identities remain stable; the network<br>"
     "has not yet built cross-operand context."),
    # L2
    ("<b>L2 — Context integration</b><br>"
     "First cross-operand interactions.  Operation<br>"
     "type (*vs+vs−) begins to separate clusters.<br>"
     "Min/max encoding persists from L0."),
    # L3
    ("<b>L3 — Annihilator gutter</b><br>"
     "A single 1D direction (d'=8.48) cleanly routes<br>"
     "all zero-product pairs away from the rest.<br>"
     "Zero-pair state norms peak here (+49% over<br>"
     "non-zero pairs).  The gutter is a <i>flag</i>,<br>"
     "not a value — orthogonal to operand encoding."),
    # L4
    ("<b>L4 — Value refinement</b><br>"
     "Product-value clustering strengthens.  The zero<br>"
     "flag remains stable while the network sharpens<br>"
     "its representation of the actual product value<br>"
     "for non-zero pairs."),
    # L5
    ("<b>L5 — Output preparation</b><br>"
     "Zero-detection directions converge across × and<br>"
     "− operations (cosine 0.29→0.81).  A single<br>"
     "'say zero' attractor handles all operations.<br>"
     "Commutative pairs (a,b) and (b,a) reach the<br>"
     "same final position before unembedding."),
]

# ---------- helpers ----------


def scale_color(value: float, cmin: float, cmax: float, cscale: str, alpha: float) -> str:
    t = max(0.0, min(1.0, (value - cmin) / (cmax - cmin + 1e-8)))
    rgb = sample_colorscale(cscale, t)[0]
    r, g, b = [int(c) for c in rgb[4:-1].split(",")]
    return f"rgba({r},{g},{b},{alpha})"


def per_layer_umap(H: np.ndarray, l_arr: np.ndarray, n_layers: int) -> dict[int, np.ndarray]:
    import umap as umap_lib

    coords: dict[int, np.ndarray] = {}
    for layer in range(n_layers):
        mask = l_arr == layer
        H_layer = H[mask]
        print(f"  UMAP layer {layer} ({H_layer.shape[0]} points)...", flush=True)
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=42)
        Y = reducer.fit_transform(H_layer)
        lo, hi = Y.min(0), Y.max(0)
        Y = (Y - lo) / (hi - lo + 1e-8) * (2 * NORM_SCALE) - NORM_SCALE
        coords[layer] = Y
    return coords


def procrustes_align(coords: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Rotate each layer's UMAP to best match L0 (orthogonal Procrustes).

    Each layer is already min-max normalised to the same scale, so we only
    need to correct for arbitrary UMAP rotation/reflection.  After alignment
    the connecting lines between layers reflect genuine representational drift
    rather than UMAP orientation noise.
    """
    ref = coords[0]
    ref_mean = ref.mean(0)
    ref_c = ref - ref_mean

    aligned = {0: ref}
    for layer in range(1, len(coords)):
        Y = coords[layer]
        Y_c = Y - Y.mean(0)
        R, _ = orthogonal_procrustes(Y_c, ref_c)
        aligned[layer] = Y_c @ R + ref_mean
    return aligned


def precompute_concepts() -> list[dict]:
    out = []
    for label, fn, cmin, cmax, cscale in CONCEPTS:
        values = [fn(a, b) for a, b in PAIRS]
        line_colors = [scale_color(v, cmin, cmax, cscale, alpha=0.35) for v in values]
        out.append(dict(label=label, values=values, line_colors=line_colors,
                        cmin=cmin, cmax=cmax, cscale=cscale))
    return out


def precompute_highlights() -> list[dict]:
    out = []
    for label, fn, _ in HIGHLIGHTS:
        flags = [fn(a, b) for a, b in PAIRS]
        out.append(dict(label=label, flags=flags))
    return out


# ---------- figure ----------


def build_figure(coords: dict[int, np.ndarray], a_arr, b_arr, p_arr, l_arr,
                 n_layers: int, concept_data: list[dict],
                 highlight_data: list[dict]) -> go.Figure:
    fig = go.Figure()
    band_w = NORM_SCALE + 1.5
    default_cd = concept_data[0]
    n_pairs = len(PAIRS)

    # Background bands
    for layer in range(n_layers):
        y_off = (n_layers - 1 - layer) * Y_SPACING
        fig.add_shape(
            type="rect",
            x0=-band_w, x1=band_w,
            y0=y_off - NORM_SCALE - 0.8, y1=y_off + NORM_SCALE + 0.8,
            fillcolor="rgba(235,240,255,0.45)" if layer % 2 == 0 else "rgba(255,245,235,0.45)",
            line=dict(color="rgba(180,180,200,0.5)", width=0.5),
            layer="below",
        )

    # ---- Trace block A: line traces (indices 0..n_pairs-1) ----
    for pair_idx, (a, b) in enumerate(PAIRS):
        xs = [coords[layer][pair_idx, 0] for layer in range(n_layers)]
        ys = [coords[layer][pair_idx, 1] + (n_layers - 1 - layer) * Y_SPACING
              for layer in range(n_layers)]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            opacity=1.0,
            line=dict(color=default_cd["line_colors"][pair_idx], width=1.2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ---- Trace block B: dot traces (indices n_pairs..n_pairs+n_layers-1) ----
    for layer in range(n_layers):
        mask = l_arr == layer
        xs = coords[layer][:, 0]
        ys = coords[layer][:, 1] + (n_layers - 1 - layer) * Y_SPACING
        a_vals, b_vals, p_vals = a_arr[mask], b_arr[mask], p_arr[mask]
        text = [f"{a}×{b}={p}  L{layer}" for a, b, p in zip(a_vals, b_vals, p_vals)]
        show_cb = (layer == n_layers - 1)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            opacity=1.0,
            marker=dict(
                size=8,
                opacity=1.0,
                color=default_cd["values"],
                colorscale=default_cd["cscale"],
                cmin=default_cd["cmin"],
                cmax=default_cd["cmax"],
                showscale=show_cb,
                colorbar=dict(title=default_cd["label"], x=1.02) if show_cb else None,
                line=dict(width=0.8, color="white"),
            ),
            hovertext=text,
            hoverinfo="text",
            showlegend=False,
            name=f"L{layer}",
        ))

    # ---- Annotations: layer labels + narrations ----
    for layer in range(n_layers):
        y_off = (n_layers - 1 - layer) * Y_SPACING
        # Left label
        fig.add_annotation(
            x=-(band_w + 0.3), y=y_off,
            text=f"<b>L{layer}</b>",
            showarrow=False, xanchor="right",
            font=dict(size=13, color="#333"),
            xref="x", yref="y",
        )
        # Right narration — fixed paper x, data y
        narration = LAYER_NARRATIONS[layer] if layer < len(LAYER_NARRATIONS) else ""
        fig.add_annotation(
            x=1.03, y=y_off,
            text=narration,
            showarrow=False,
            xanchor="left", yanchor="middle",
            align="left",
            font=dict(size=11, color="#222"),
            xref="paper", yref="y",
            bgcolor="rgba(248,248,255,0.85)",
            bordercolor="rgba(180,180,210,0.6)",
            borderwidth=1,
            borderpad=6,
        )

    # ---- updatemenus: concept color buttons (row 1) ----
    color_buttons = []
    for cd in concept_data:
        all_line_color   = cd["line_colors"] + ["white"] * n_layers
        all_marker_color = cd["values"] + [cd["values"]] * n_layers
        all_cscale   = [cd["cscale"]] * (n_pairs + n_layers)
        all_cmin     = [cd["cmin"]]   * (n_pairs + n_layers)
        all_cmax     = [cd["cmax"]]   * (n_pairs + n_layers)
        all_cb_title = [None] * (n_pairs + n_layers - 1) + [cd["label"]]
        color_buttons.append(dict(
            label=cd["label"],
            method="restyle",
            args=[{
                "line.color":                 all_line_color,
                "marker.color":               all_marker_color,
                "marker.colorscale":          all_cscale,
                "marker.cmin":                all_cmin,
                "marker.cmax":                all_cmax,
                "marker.colorbar.title.text": all_cb_title,
            }],
        ))

    # ---- updatemenus: highlight buttons (row 2) ----
    highlight_buttons = []
    for hd in highlight_data:
        flags = hd["flags"]
        # Line traces: trace-level opacity (1.0 or 0.06)
        line_opacity  = [1.0 if flags[i] else 0.06 for i in range(n_pairs)]
        # Dot traces: per-marker opacity and size arrays
        dot_opacities = [1.0 if flags[i] else 0.07 for i in range(n_pairs)]
        dot_sizes     = [11  if flags[i] else 5    for i in range(n_pairs)]

        all_opacity       = line_opacity + [1.0] * n_layers  # dot trace-level opacity stays 1
        all_marker_opacity = [1.0] * n_pairs + [dot_opacities] * n_layers
        all_marker_size    = [8]   * n_pairs + [dot_sizes]     * n_layers

        highlight_buttons.append(dict(
            label=hd["label"],
            method="restyle",
            args=[{
                "opacity":        all_opacity,
                "marker.opacity": all_marker_opacity,
                "marker.size":    all_marker_size,
            }],
        ))

    total_height = n_layers * Y_SPACING
    fig.update_layout(
        title=dict(
            text="Times-table: per-layer 2D UMAP (Procrustes-aligned to L0) — color by concept, highlight by behavior",
            font=dict(size=14),
        ),
        updatemenus=[
            dict(
                type="buttons", direction="right",
                x=0.0, y=1.13, xanchor="left",
                showactive=True, buttons=color_buttons,
                bgcolor="rgba(235,240,255,0.9)",
                bordercolor="#aaa", font=dict(size=10),
                pad=dict(r=4, t=4),
            ),
            dict(
                type="buttons", direction="right",
                x=0.0, y=1.07, xanchor="left",
                showactive=True, buttons=highlight_buttons,
                bgcolor="rgba(255,245,230,0.9)",
                bordercolor="#aaa", font=dict(size=10),
                pad=dict(r=4, t=4),
            ),
        ],
        annotations=fig.layout.annotations,  # preserve the layer annotations
        xaxis=dict(
            showticklabels=False, showgrid=False, zeroline=False,
            range=[-(band_w + 1.2), band_w + 1.5],
        ),
        yaxis=dict(
            showticklabels=False, showgrid=False, zeroline=False,
            range=[-NORM_SCALE - 1.5, total_height - Y_SPACING + NORM_SCALE + 1.5],
        ),
        width=1480,
        height=max(700, 280 * n_layers),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=420, t=100, b=30),
        dragmode="pan",
        xaxis_fixedrange=False,
        yaxis_fixedrange=False,
    )
    return fig


# ---------- main ----------


def main():
    if not STATES.exists():
        raise FileNotFoundError(
            f"{STATES} not found.\n"
            "Run: python experiments/times_table_trace.py --no-train"
        )

    data = np.load(STATES)
    H, a_arr, b_arr, p_arr, l_arr = (
        data["H"], data["a"], data["b"], data["product"], data["layer"]
    )
    n_layers = int(l_arr.max()) + 1
    print(f"Loaded {H.shape[0]} hidden states  ({n_layers} layers × {len(PAIRS)} pairs)")

    coords = per_layer_umap(H, l_arr, n_layers)
    coords = procrustes_align(coords)
    concept_data   = precompute_concepts()
    highlight_data = precompute_highlights()

    print("Building figure...")
    fig = build_figure(coords, a_arr, b_arr, p_arr, l_arr,
                       n_layers, concept_data, highlight_data)
    fig.write_html(HTML, include_plotlyjs="cdn", config={"scrollZoom": True})
    print(f"Saved: {HTML}")
    import subprocess
    subprocess.Popen(["open", str(HTML)])


if __name__ == "__main__":
    main()
