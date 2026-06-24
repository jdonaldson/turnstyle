"""Decompose times-table trajectory motion through the network.

For each (a,b) pair, the residual stream at the =-token visits 6 states
across L0..L5. The per-pair trajectory length is the total motion in
R^128:

    traj_len(a,b) = sum_{L=0..4} || h_{L+1}(a,b) - h_L(a,b) ||

We compare:
  - trajectory length vs product (do zero-product pairs move more?)
  - per-layer update norm (which block contributes most to the motion?)
  - zero-product vs non-zero distributions

Hypothesis (per current discussion): zero-product pairs should be the
stable trajectories if the model treats them as no-ops. Empirically
they appear to bounce — this quantifies it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
STATES = ROOT / "hidden_states.npz"
HTML = ROOT / "times_table_trajectory_length.html"


def main():
    d = np.load(STATES)
    H = d["H"]  # (600, 128)
    a = d["a"]
    b = d["b"]
    p = d["product"]
    layer = d["layer"]
    n_layers = int(layer.max()) + 1

    # Reshape to (100, 6, 128) indexed by (pair_idx, layer)
    pair_ids = a * 10 + b  # unique 0..99 for each (a,b)
    H_pair = np.zeros((100, n_layers, 128), dtype=H.dtype)
    P_pair = np.zeros(100, dtype=int)
    for i in range(len(H)):
        H_pair[pair_ids[i], layer[i]] = H[i]
        P_pair[pair_ids[i]] = p[i]

    # Per-pair, per-layer update norm: || h(L+1) - h(L) ||  for L in 0..4
    deltas = H_pair[:, 1:, :] - H_pair[:, :-1, :]  # (100, 5, 128)
    update_norms = np.linalg.norm(deltas, axis=2)  # (100, 5)
    traj_len = update_norms.sum(axis=1)  # (100,)

    # Also per-layer state norm (how big the state itself is)
    state_norms = np.linalg.norm(H_pair, axis=2)  # (100, 6)

    is_zero = (P_pair == 0)
    print(f"n zero-product pairs: {is_zero.sum()} / 100")
    print(f"trajectory length — zero    : mean {traj_len[is_zero].mean():.3f}  std {traj_len[is_zero].std():.3f}")
    print(f"trajectory length — non-zero: mean {traj_len[~is_zero].mean():.3f}  std {traj_len[~is_zero].std():.3f}")
    print(f"ratio (zero / non-zero): {traj_len[is_zero].mean() / traj_len[~is_zero].mean():.3f}")

    # Per-layer L→L+1 update norm, zero vs non-zero
    print("\nPer-transition update norm  (mean ± std)")
    print(f"{'transition':14s}  {'zero':>16s}     {'non-zero':>16s}     ratio")
    for t in range(n_layers - 1):
        z_mu, z_sd = update_norms[is_zero, t].mean(), update_norms[is_zero, t].std()
        n_mu, n_sd = update_norms[~is_zero, t].mean(), update_norms[~is_zero, t].std()
        print(f"  L{t}→L{t+1}     :  {z_mu:6.3f} ± {z_sd:5.3f}     {n_mu:6.3f} ± {n_sd:5.3f}     {z_mu/n_mu:.3f}")

    # Per-layer state norm comparison
    print("\nPer-layer state norm  (mean)")
    print(f"{'layer':6s}  {'zero':>8s}  {'non-zero':>10s}  ratio")
    for L in range(n_layers):
        z_mu = state_norms[is_zero, L].mean()
        n_mu = state_norms[~is_zero, L].mean()
        print(f"  L{L}    {z_mu:7.3f}    {n_mu:8.3f}     {z_mu/n_mu:.3f}")

    # Sorted: top-5 longest and shortest trajectories, with (a, b, product)
    order = np.argsort(traj_len)
    print("\nTop 5 SHORTEST trajectories:")
    for i in order[:5]:
        ai, bi = i // 10, i % 10
        print(f"  {ai}×{bi}={P_pair[i]:2d}  traj_len {traj_len[i]:.3f}")
    print("Top 5 LONGEST trajectories:")
    for i in order[-5:]:
        ai, bi = i // 10, i % 10
        print(f"  {ai}×{bi}={P_pair[i]:2d}  traj_len {traj_len[i]:.3f}")

    # ----- viz: 3 subplots -----
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Trajectory length vs product (per pair)",
            "Per-transition update norm (zero vs non-zero)",
            "Per-layer state norm (zero vs non-zero)",
        ),
        column_widths=[0.42, 0.29, 0.29],
    )

    # 1) scatter: traj_len vs product, color by is_zero
    hover = [f"{i//10}×{i%10}={P_pair[i]}" for i in range(100)]
    fig.add_trace(
        go.Scatter(
            x=P_pair[~is_zero],
            y=traj_len[~is_zero],
            mode="markers",
            marker=dict(color="steelblue", size=7, line=dict(width=0)),
            text=[hover[i] for i in range(100) if not is_zero[i]],
            hoverinfo="text+y",
            name="non-zero",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=P_pair[is_zero],
            y=traj_len[is_zero],
            mode="markers",
            marker=dict(color="crimson", size=9, symbol="x"),
            text=[hover[i] for i in range(100) if is_zero[i]],
            hoverinfo="text+y",
            name="zero",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="product (a × b)", row=1, col=1)
    fig.update_yaxes(title_text="trajectory length (sum of layer-update norms)", row=1, col=1)

    # 2) per-transition update norm
    transitions = [f"L{t}→L{t+1}" for t in range(n_layers - 1)]
    zero_means = [update_norms[is_zero, t].mean() for t in range(n_layers - 1)]
    nz_means = [update_norms[~is_zero, t].mean() for t in range(n_layers - 1)]
    zero_std = [update_norms[is_zero, t].std() for t in range(n_layers - 1)]
    nz_std = [update_norms[~is_zero, t].std() for t in range(n_layers - 1)]
    fig.add_trace(
        go.Scatter(
            x=transitions,
            y=zero_means,
            error_y=dict(type="data", array=zero_std),
            mode="lines+markers",
            line=dict(color="crimson"),
            name="zero",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=transitions,
            y=nz_means,
            error_y=dict(type="data", array=nz_std),
            mode="lines+markers",
            line=dict(color="steelblue"),
            name="non-zero",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="|| h(L+1) - h(L) ||", row=1, col=2)

    # 3) per-layer state norm
    layers_lbl = [f"L{L}" for L in range(n_layers)]
    fig.add_trace(
        go.Scatter(
            x=layers_lbl,
            y=[state_norms[is_zero, L].mean() for L in range(n_layers)],
            mode="lines+markers",
            line=dict(color="crimson"),
            name="zero",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=layers_lbl,
            y=[state_norms[~is_zero, L].mean() for L in range(n_layers)],
            mode="lines+markers",
            line=dict(color="steelblue"),
            name="non-zero",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_yaxes(title_text="|| h(L) ||", row=1, col=3)

    fig.update_layout(
        title="Decomposing times-table trajectory motion (zero-product vs non-zero)",
        width=1500,
        height=550,
    )
    fig.write_html(HTML)
    print(f"\nsaved: {HTML}")


if __name__ == "__main__":
    main()
