"""Visualize the cross-lingual adjective-ordering ladder.

Each language's adjective categories, projected onto the English-fit subjectivity
axis, should descend opinion -> material in the SAME order — the visual proof that
the ordering hierarchy is cross-lingual in the model's geometry.
"""
import os
import sys

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp"))
sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import subjectivity_order as SO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 11
d = np.load(SO.CACHE, allow_pickle=True)
A = d["acts"].astype(np.float32)[:, L, :]
cat = d["cat"]; lang = d["lang"]
cats = sorted(SO.RANK, key=lambda c: SO.RANK[c])

en = lang == "en"
mu = A[en].mean(0); sd = A[en].std(0) + 1e-6
Z = (A - mu) / sd
hi = Z[en & (cat == "opinion")].mean(0)
lo = Z[en & (cat == "material")].mean(0)
dirn = (hi - lo) / (np.linalg.norm(hi - lo) + 1e-12)
proj = Z @ dirn

fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.get_cmap("tab10")
for k, lg in enumerate(["en", "es", "fr", "de"]):
    ys = [proj[(lang == lg) & (cat == c)].mean() for c in cats]
    ax.plot(range(len(cats)), ys, "-o", color=cmap(k), label=lg, lw=2, ms=7,
            alpha=0.85)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([f"{c}\n(rank {SO.RANK[c]})" for c in cats], fontsize=9)
ax.set_ylabel("subjectivity projection  (high = subjective)")
ax.set_xlabel("adjective-ordering category  (canonical: opinion → material)")
ax.set_title("Cross-lingual adjective-ordering ladder @L11\n"
             "English-fit subjectivity axis; every language descends in the same "
             "canonical order\n(opinion/material = fit poles; size…origin held out)",
             fontsize=11)
ax.axvspan(-0.4, 0.4, color="gray", alpha=0.08)
ax.axvspan(len(cats) - 1.4, len(cats) - 0.6, color="gray", alpha=0.08)
ax.legend(title="language", fontsize=9)
ax.grid(True, alpha=0.25)
fig.tight_layout()
out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "subjectivity_ladder.png")
fig.savefig(out, dpi=130)
print(f"saved {out}")
