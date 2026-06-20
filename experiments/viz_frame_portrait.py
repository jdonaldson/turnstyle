"""A direct portrait of the semantic frame (no UMAP — a frame is a named signed
coordinate system, so show the coordinates).

Left  — heatmap: mean semantic coordinate per (axis, pole) group × axis. A clean
        signed block-diagonal means each word group loads on its OWN axis with the
        correct sign. Computed per language to show cross-lingual consistency.
Right — a labeled 2-axis scatter: cross-lingual translation sets (hot/caliente/
        chaud/heiß, cold/frío/froid/kalt) annotated, showing translations co-locate.

Fits axes on ENGLISH only (cached cross_language acts, no model run).
Usage:  python experiments/viz_frame_portrait.py [--layer L]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp"))
sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import cross_language as CL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from viz_semantic_frame import fit_coords


def main(layer):
    d = np.load(CL.CACHE, allow_pickle=True)
    acts = d["acts"].astype(np.float32)[:, layer, :]
    lang = d["langs"]; axis = d["axes"]; word = d["words"]
    pole = (d["poles"] == 1).astype(int) * 2 - 1

    coords, names, _, _ = fit_coords(acts, lang, axis, pole)
    # z-score each axis column so the heatmap is comparable across axes
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)

    fig, (axh, axs) = plt.subplots(1, 2, figsize=(15, 7),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ── left: (axis,pole) × axis mean-coordinate heatmap, pooled over ALL langs ──
    rows, rlabels = [], []
    for ax in names:
        for p, tag in ((1, "+"), (-1, "−")):
            m = (axis == ax) & (pole == p)
            rows.append(coords[m].mean(0))
            rlabels.append(f"{ax} {tag}")
    M = np.vstack(rows)
    im = axh.imshow(M, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    axh.set_xticks(range(len(names))); axh.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axh.set_yticks(range(len(rlabels))); axh.set_yticklabels(rlabels, fontsize=7)
    axh.set_title(f"Mean semantic coordinate @L{layer}\n(row = word group, col = axis; "
                  "signed block-diagonal = clean frame, all 4 langs pooled)", fontsize=10)
    axh.set_xlabel("projected onto axis"); axh.set_ylabel("word group (axis, pole)")
    fig.colorbar(im, ax=axh, fraction=0.046, label="coordinate (z)")

    # ── right: a labeled cross-lingual scatter on two axes ──────────────────────
    pop = sorted(names, key=lambda a: -(axis == a).sum())[:2]
    if "temp" in names and "value" in names:
        pop = ["temp", "value"]
    ai, aj = names.index(pop[0]), names.index(pop[1])
    cmap = plt.get_cmap("tab10")
    langs = ["en", "es", "fr", "de"]
    for k, lg in enumerate(langs):
        m = lang == lg
        axs.scatter(coords[m, ai], coords[m, aj], s=18, color=cmap(k), label=lg, alpha=0.7)
    axs.axhline(0, color="gray", lw=0.6); axs.axvline(0, color="gray", lw=0.6)
    axs.set_xlabel(f"{pop[0]} axis  (− low … high +)")
    axs.set_ylabel(f"{pop[1]} axis  (− low … high +)")
    axs.set_title(f"Cross-lingual: {pop[0]} vs {pop[1]} coordinates\n"
                  "(translations land together, English-fit axes)", fontsize=10)
    # annotate the temp translation sets
    sets = {"hot": ["hot", "caliente", "chaud", "heiß"],
            "cold": ["cold", "frío", "froid", "kalt"]}
    seen = set()
    for grp in sets.values():
        for w in grp:
            idx = np.where(word == w)[0]
            if len(idx) and w not in seen:
                seen.add(w)
                i = idx[0]
                axs.annotate(w, (coords[i, ai], coords[i, aj]), fontsize=7,
                             xytext=(3, 3), textcoords="offset points")
    axs.legend(fontsize=8, title="language")

    fig.tight_layout()
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "frame_portrait.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=11)
    args = ap.parse_args()
    main(args.layer)
