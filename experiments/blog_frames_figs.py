"""
blog_frames_figs.py — solarized figures for the semantic-frames blog post
("Meaning Has More Than Three Numbers", jjd.io).

Data provenance: F1/F2/F3 read experiments/results/ordering_frames.json (written
by a fresh ordering_frames.py run — no hand-entered matrix cells). The OG card
(F4) is generated art, no data claims.

Outputs -> ~/Projects/jjd.io/posts/images/frames_*.png
Run:  .venv/bin/python experiments/blog_frames_figs.py
"""
from __future__ import annotations
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- solarized ---------------------------------------------------------------
BASE03, BASE02, BASE01, BASE00 = "#002b36", "#073642", "#586e75", "#657b83"
BASE1, BASE2, BASE3 = "#93a1a1", "#eee8d5", "#fdf6e3"
YELLOW, ORANGE, RED, MAGENTA = "#b58900", "#cb4b16", "#dc322f", "#d33682"
VIOLET, BLUE, CYAN, GREEN = "#6c71c4", "#268bd2", "#2aa198", "#859900"

plt.rcParams.update({
    "figure.facecolor": BASE3, "axes.facecolor": BASE3, "savefig.facecolor": BASE3,
    "text.color": BASE02, "axes.edgecolor": BASE01, "axes.labelcolor": BASE02,
    "xtick.color": BASE01, "ytick.color": BASE01, "font.size": 11,
    "axes.grid": True, "grid.color": BASE2, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, "results", "ordering_frames.json")))
OUT = os.path.expanduser("~/Projects/jjd.io/posts/images")
os.makedirs(OUT, exist_ok=True)

# canonical adjective-ordering positions (opinion > size > age > shape > color >
# origin/space > material)
ORDER_POS = {"opinion": 1, "size": 2, "age": 3, "shape": 4, "color": 5,
             "space": 6, "material": 7}
RUNG_COLOR = {"opinion": VIOLET, "size": BLUE, "age": CYAN, "shape": GREEN,
              "color": YELLOW, "space": ORANGE, "material": MAGENTA}


# --- F1: recoverability bars --------------------------------------------------
# space: shown at its frequency-RESIDUALIZED r (the honest number) with a ghost
# outline at the raw fit — the raw 0.89 was word rarity (freq_residual.py).
# color: scalar shown; the hue-ring result (color_ring.py) annotated.
def fig_recoverability():
    rec = RES["recoverability"]
    resid = json.load(open(os.path.join(HERE, "results", "freq_residual.json")))
    ring = json.load(open(os.path.join(HERE, "results", "color_ring.json")))
    space_res = resid["rungs"]["space"]["residualized"]["r"]
    space_raw = resid["rungs"]["space"]["raw"]["r"]
    ring_R = ring["peaks"]["ring_R"]["value"]
    ring_L = ring["peaks"]["ring_R"]["layer"]
    cats = sorted(rec, key=lambda c: ORDER_POS[c])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, c in enumerate(cats):
        r, L = rec[c]["r"], rec[c]["layer"]
        if c == "space":
            ax.bar(i, space_raw, color="none", edgecolor=BASE1, ls="--", lw=1.4,
                   width=0.62, zorder=2)
            ax.bar(i, space_res, color=BASE1, width=0.62, zorder=3)
            ax.text(i, space_raw + 0.02, f"raw {space_raw:.2f}", ha="center",
                    fontsize=8.5, color=BASE01, style="italic")
            ax.text(i, space_res + 0.02, f"{space_res:.2f}", ha="center",
                    fontsize=10.5, fontweight="bold", color=BASE02)
            ax.text(i, 0.52, "fails frequency control", ha="center", va="center",
                    fontsize=8.5, color=RED, fontweight="bold", rotation=90)
            continue
        ax.bar(i, r, color=RUNG_COLOR[c], width=0.62, zorder=3)
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=10.5,
                fontweight="bold", color=BASE02)
        ax.text(i, 0.04, f"L{L}", ha="center", fontsize=9, color=BASE3
                if r > 0.12 else BASE01, fontweight="bold")
        if c == "color":
            ax.annotate(f"scalar read — as a\nhue RING: R={ring_R:.2f} @L{ring_L}",
                        (i - 0.35, r + 0.09), ha="right", fontsize=8.5, color=ORANGE,
                        fontweight="bold")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{c}\n(rung {ORDER_POS[c]})" for c in cats], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("held-out CV r (5-fold, shuffled)")
    ax.axhline(0.8, color=BASE1, lw=1, ls="--", zorder=2)
    ax.text(len(cats) - 0.45, 0.815, "r = 0.8", fontsize=9, color=BASE01)
    ax.set_title("Five rungs recover as scalar frames; two need a closer look\n"
                 "SmolLM2-1.7B · linear read of the hidden state · label = peak layer",
                 fontsize=12, loc="left", color=BASE02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "frames_recoverability.png"), dpi=160)
    plt.close(fig)


# --- F2: orthogonality heatmap -------------------------------------------------
def fig_orthogonality(layer_key="L8"):
    # space dropped (fails the frequency control); stats recomputed over the
    # remaining 6x6 from the same fresh run.
    cm = RES["cos_matrix"][layer_key]
    cats = RES["categories"]
    keep = [i for i, c in enumerate(cats) if c != "space"]
    cats = [cats[i] for i in keep]
    full = np.array(cm["matrix"])[np.ix_(keep, keep)]
    offdiag = full[~np.eye(len(cats), dtype=bool)]
    cm = {"matrix": full.tolist(), "mean_offdiag": float(offdiag.mean()),
          "max_offdiag": float(offdiag.max())}
    order = sorted(range(len(cats)), key=lambda i: ORDER_POS[cats[i]])
    labels = [cats[i] for i in order]
    M = np.array(cm["matrix"])[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color=BASE3 if v < 0.55 else BASE03,
                    fontweight="bold" if i == j else "normal")
    ax.set_title("…and the frames are mutually orthogonal\n"
                 f"|cosine| of frame directions @{layer_key}\n"
                 f"mean off-diag {cm['mean_offdiag']:.2f} · max {cm['max_offdiag']:.2f}",
                 fontsize=11.5, loc="left", color=BASE02)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("|cos|", color=BASE02)
    cb.ax.yaxis.set_tick_params(color=BASE01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "frames_orthogonality.png"), dpi=160)
    plt.close(fig)


# --- F3: rung position vs peak layer -------------------------------------------
def fig_depth():
    # space dropped (frequency, not depth); age flagged — its frequency-clean
    # peak moves L2->L14 (freq_residual.py), so the early cluster is softer
    # than the raw peaks suggest.
    rec = RES["recoverability"]
    cats = [c for c in sorted(rec, key=lambda c: ORDER_POS[c]) if c != "space"]
    xs = [ORDER_POS[c] for c in cats]
    ys = [rec[c]["layer"] for c in cats]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for c, x, y in zip(cats, xs, ys):
        ax.scatter(x, y, s=170, color=RUNG_COLOR[c], zorder=3,
                   edgecolor=BASE02, linewidth=0.8)
        ax.annotate(c, (x, y), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=10.5, color=BASE02, fontweight="bold")
        if c == "age":
            ax.scatter(x, 14, s=90, facecolor="none", edgecolor=RUNG_COLOR[c],
                       ls="--", zorder=2)
            ax.annotate("age, frequency-\ncontrolled: L14", (x, 14),
                        textcoords="offset points", xytext=(38, -4),
                        ha="left", va="center", fontsize=8, color=BASE01,
                        style="italic")
    z = np.polyfit(xs, ys, 1)
    xr = np.linspace(0.6, 7.4, 10)
    ax.plot(xr, np.polyval(z, xr), color=BASE1, lw=1.4, ls="--", zorder=2)
    ax.set_xlabel("position in the canonical adjective order  (opinion → material)")
    ax.set_ylabel("layer of peak recoverability")
    ax.set_xticks(list(range(1, 8)))
    ax.set_title("Suggestive: rung position tracks network depth\n"
                 "early rungs peak in early layers, late rungs deep in the stack "
                 "(n=6, correlational)", fontsize=12, loc="left", color=BASE02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "frames_depth.png"), dpi=160)
    plt.close(fig)


# --- F4: OG card — instrument panel of dials ------------------------------------
def fig_og():
    fig = plt.figure(figsize=(12, 6.28), dpi=100)   # 1200x628
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.patch.set_facecolor(BASE3)
    ASPECT = 12 / 6.28                       # x-units are this much wider than y
    dials = [("opinion", VIOLET, 0.62), ("size", BLUE, 0.80), ("age", CYAN, 0.35),
             ("shape", GREEN, 0.55), ("color", YELLOW, 0.20), ("origin", ORANGE, 0.70),
             ("material", MAGENTA, 0.45), ("number", RED, 0.85), ("time", BASE00, 0.30)]
    rx = 0.042; ry = rx * ASPECT             # circular gauge on a wide canvas
    for i, (name, color, val) in enumerate(dials):
        cx = 0.10 + (i % 3) * 0.155
        cy = 0.76 - (i // 3) * 0.27
        th = np.linspace(np.pi * 7 / 6, -np.pi / 6, 100)   # 210° → -30° gauge arc
        ax.plot(cx + rx * np.cos(th), cy + ry * np.sin(th), color=BASE1, lw=3,
                solid_capstyle="round")
        ang = np.pi * 7 / 6 - val * np.pi * 4 / 3
        ax.plot([cx, cx + 0.82 * rx * np.cos(ang)],
                [cy, cy + 0.82 * ry * np.sin(ang)], color=color, lw=4.5,
                solid_capstyle="round")
        ax.scatter([cx], [cy], s=46, color=color, zorder=5)
        ax.text(cx, cy - ry - 0.045, name, ha="center", fontsize=14.5,
                color=BASE02, fontweight="bold")
    ax.text(0.545, 0.70, "Meaning has more\nthan three numbers", fontsize=33,
            color=BASE02, fontweight="bold", va="center", ha="left")
    ax.text(0.545, 0.42, "a small language model keeps an\n"
            "instrument panel of orthogonal semantic\n"
            "dials — the adjective-ordering rungs,\n"
            "plus number and time", fontsize=16, color=BASE00, va="center", ha="left")
    ax.text(0.545, 0.16, "jjd.io · turnstyle", fontsize=13.5, color=BASE1, ha="left")
    fig.savefig(os.path.join(OUT, "frames_og.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_recoverability()
    fig_orthogonality()
    fig_depth()
    fig_og()
    print("wrote", sorted(f for f in os.listdir(OUT) if f.startswith("frames_")))
