"""Joint trajectory plot of +, -, * tables through the NanoGPT.

Reuses the existing checkpoint (trained on all three tier-1 ops mixed
with SVO filler). Forwards 300 prompts (100 per op), grabs the
=-token state at every layer, joint UMAP to 2D, renders Plotly 3D
trajectories.

Visual encoding:
  - one polyline per (op, a, b) pair, 6 points L0..L5
  - marker symbol: op   (* circle, + square, - diamond)
  - line dash:     op   (* solid,  + dot,    - dash)
  - color:         result, diverging RdBu_r centered at 0
                   (sub negatives on red side; results==0 are pale;
                   high mul products are deep blue)
  - legend groups: clicking 'mul' / 'add' / 'sub' toggles all 100 lines

Designed to make the late-layer gutter convergence visible: all
result==0 trajectories (19 mul + 1 add + 10 sub) should funnel into a
shared neighborhood by L5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import umap

sys.path.insert(0, str(Path(__file__).parent))
import times_table_trace as ttt  # type: ignore

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
HTML = ROOT / "times_table_trace_all_ops.html"

OP_STYLE = {
    "*": {"name": "multiplication", "symbol": "circle", "dash": "solid"},
    "+": {"name": "addition", "symbol": "square", "dash": "dot"},
    "-": {"name": "subtraction", "symbol": "diamond", "dash": "dash"},
}


@torch.no_grad()
def collect(model, device: str = "cpu"):
    rows = []
    for op in OP_STYLE:
        for a in range(10):
            for b in range(10):
                prompt = f"{a}{op}{b}="
                idx = torch.tensor([ttt.encode(prompt)], dtype=torch.long, device=device)
                _, _, states = model(idx, return_states=True)
                eq_pos = len(prompt) - 1
                result = a + b if op == "+" else a - b if op == "-" else a * b
                for layer, h in enumerate(states):
                    vec = h[0, eq_pos, :].cpu().numpy()
                    rows.append((op, a, b, result, layer, vec))
    op_arr = np.array([r[0] for r in rows])
    a_arr = np.array([r[1] for r in rows], dtype=int)
    b_arr = np.array([r[2] for r in rows], dtype=int)
    r_arr = np.array([r[3] for r in rows], dtype=int)
    l_arr = np.array([r[4] for r in rows], dtype=int)
    H = np.stack([r[5] for r in rows])
    return op_arr, a_arr, b_arr, r_arr, l_arr, H


def main():
    model = ttt.load("cpu")
    print("forwarding 300 prompts × 6 layers...")
    op, a, b, r, layer, H = collect(model)
    print(f"H shape: {H.shape}")

    print("UMAP fitting (1800, 128)...")
    Y = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(H)

    fig = go.Figure()
    first_per_op = {k: True for k in OP_STYLE}
    cmin, cmax = int(r.min()), int(r.max())  # -9, 81
    for op_sym, style in OP_STYLE.items():
        for a_i in range(10):
            for b_i in range(10):
                mask = (op == op_sym) & (a == a_i) & (b == b_i)
                if not mask.any():
                    continue
                order = np.argsort(layer[mask])
                xs = Y[mask][order, 0]
                ys = Y[mask][order, 1]
                zs = layer[mask][order]
                result = int(r[mask][0])
                text = [
                    f"{a_i}{op_sym}{b_i}={result}  L{z}"
                    for z in zs
                ]
                show_scale = (op_sym == "*" and a_i == 0 and b_i == 0)
                show_legend = first_per_op[op_sym]
                first_per_op[op_sym] = False
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines+markers",
                        line=dict(
                            color=[result] * len(zs),
                            colorscale="RdBu_r",
                            cmin=cmin,
                            cmax=cmax,
                            cmid=0,
                            width=2.5,
                            dash=style["dash"],
                        ),
                        marker=dict(
                            size=4,
                            symbol=style["symbol"],
                            color=[result] * len(zs),
                            colorscale="RdBu_r",
                            cmin=cmin,
                            cmax=cmax,
                            cmid=0,
                            showscale=show_scale,
                            colorbar=dict(title="result") if show_scale else None,
                        ),
                        hovertext=text,
                        hoverinfo="text",
                        showlegend=show_legend,
                        legendgroup=op_sym,
                        name=style["name"],
                    )
                )

    fig.update_layout(
        title=(
            "Tier-1 arithmetic trajectories through 6 layers (NanoGPT 1.2M, joint UMAP); "
            "circle=*  square=+  diamond=-  color=result (diverging at 0)"
        ),
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="Layer (=-token)",
            zaxis=dict(tickmode="linear", tick0=0, dtick=1),
        ),
        width=1300,
        height=950,
    )
    fig.write_html(HTML)
    print(f"saved: {HTML}")


if __name__ == "__main__":
    main()
