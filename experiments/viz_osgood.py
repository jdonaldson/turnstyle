"""Visualize the Osgood E-P-A space: are the three factors orthogonal?

Project every pole word onto the English-fit E, P, A axes; scatter the three axis
pairs, colored by each word's TRUE factor. If a factor's words spread along ITS
own axis and sit near zero on the others, the factors are independent (Osgood's
claim) — visible as three perpendicular "spokes".
"""
import os
import sys

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp"))
sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import osgood_epa as OE
from turnstyle.semantic_frame import fit_axis_from_vectors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 13
d = np.load(OE.CACHE, allow_pickle=True)
A = d["acts"].astype(np.float32)[:, L, :]
fac = d["fac"]; lang = d["lang"]; pole = d["pole"]
facs = ["evaluation", "potency", "activity"]

en = lang == "en"
mu = A[en].mean(0); sd = A[en].std(0) + 1e-6
Z = (A - mu) / sd
dirs, cents = {}, {}
for f in facs:
    m = fac == f
    ax = fit_axis_from_vectors(f, "lo", "hi", L, Z[en & m & (pole == 1)], Z[en & m & (pole == -1)])
    dirs[f] = ax.direction; cents[f] = ax.center
proj = {f: Z @ dirs[f] - cents[f] for f in facs}

fig, axx = plt.subplots(1, 3, figsize=(16, 5.3))
pairs = [("evaluation", "potency"), ("evaluation", "activity"), ("potency", "activity")]
cmap = {"evaluation": "C2", "potency": "C3", "activity": "C0"}
for ax, (fa, fb) in zip(axx, pairs):
    for f in facs:
        m = fac == f
        ax.scatter(proj[fa][m], proj[fb][m], s=22, color=cmap[f],
                   label=f, alpha=0.75, edgecolor="none")
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(f"{fa}  (− low … high +)"); ax.set_ylabel(f"{fb}  (− low … high +)")
    cos = abs(float(dirs[fa] @ dirs[fb]))
    ax.set_title(f"{fa[:4].upper()} vs {fb[:4].upper()}   |cos| = {cos:.2f}", fontsize=10)
    ax.legend(fontsize=7)
fig.suptitle("Osgood E-P-A space @L13 — each factor's words spread along ITS axis, "
             "centered on the others (≈ orthogonal = Osgood's independent factors)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "osgood_epa.png")
fig.savefig(out, dpi=130)
print(f"saved {out}")
