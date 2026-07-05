"""
color_ring.py — is color a RING/PLANE the scalar probe was blind to?

Color is the frame family's lone failure: a linear LIGHTNESS scalar over 16
color words recovers at r=0.28 (ordering_frames.py). Hypothesis: shape
mismatch — chromatic color is dominantly a circular HUE dimension plus a
lightness axis, so a 1-D linear fit on a ring scores near zero even if the
ring is cleanly represented. Engels et al. (2405.14860) found circular
day/month features in GPT-2/Mistral via SAEs; this is the supervised-probe
version of the same question, for color, on SmolLM2.

Tests (extended CSS color-word set, template "It is a {w} object.", last-subword):
  1. scalar CV r for L* (lightness), a*, b*   — the plane, channel by channel
  2. HUE ring fit: ridge-predict cos(θ) and sin(θ) (θ = atan2(b*, a*), chromatic
     words only), reconstruct θ̂ = atan2(ŝ, ĉ); metrics = circular correlation
     (mean resultant of angular error) + median |angular error|°
  3. HUE naive-linear baseline: ridge on raw θ as a scalar — pays the
     wraparound penalty iff the representation is circular

Verdicts:
  ring:    circular hue fit ≫ naive linear hue
  plane:   a*/b* recover well even if the explicit ring doesn't
  neither: color really is weakly represented (the r=0.28 stands)

Writes experiments/results/color_ring.json.
Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/color_ring.py
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ordering_frames import collect_last, cv_r
from color_affect_frame import _hex_to_lab

from matplotlib.colors import CSS4_COLORS

# Common single-word color terms (CSS4 hex as ground truth).
COLOR_WORDS = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "grey", "cyan", "magenta", "teal", "navy", "maroon",
    "olive", "lime", "indigo", "violet", "turquoise", "salmon", "coral",
    "crimson", "gold", "silver", "beige", "tan", "khaki", "lavender", "plum",
    "orchid", "ivory", "chocolate", "sienna", "tomato", "aquamarine", "azure",
]
CHROMA_MIN = 15.0   # exclude near-achromatic words from hue fits


def cv_pred(X, y):
    """Cross-validated ridge predictions (same estimator family as cv_r)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, KFold
    est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
    return cross_val_predict(est, X, y, cv=KFold(5, shuffle=True, random_state=0))


def circ_metrics(theta_hat, theta):
    err = np.angle(np.exp(1j * (theta_hat - theta)))
    R = float(abs(np.mean(np.exp(1j * err))))           # 1 = perfect, 0 = chance
    med_deg = float(np.median(np.abs(err)) * 180 / np.pi)
    return R, med_deg


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    words = [w for w in COLOR_WORDS if w in CSS4_COLORS]
    lab = {w: _hex_to_lab(CSS4_COLORS[w].lstrip("#")) for w in words}   # (L*, a*, b*)
    chroma = {w: float(np.hypot(lab[w][1], lab[w][2])) for w in words}
    hue = {w: float(np.arctan2(lab[w][2], lab[w][1])) for w in words}
    chromatic = [w for w in words if chroma[w] >= CHROMA_MIN]
    print(f"{len(words)} color words, {len(chromatic)} chromatic (C*>={CHROMA_MIN})",
          flush=True)

    acts = collect_last(mdl, tok, dev, words)
    n_layers = acts[words[0]].shape[0]

    results = {"model": mid, "n_words": len(words), "n_chromatic": len(chromatic),
               "by_layer": {}}
    best = {"L": (-9, -1), "a": (-9, -1), "b": (-9, -1),
            "ring_R": (-9, -1), "naive_R": (-9, -1)}
    for Lay in range(n_layers):
        X_all = np.array([acts[w][Lay] for w in words])
        X_chr = np.array([acts[w][Lay] for w in chromatic])
        row = {}
        for ch, idx in (("L", 0), ("a", 1), ("b", 2)):
            r = cv_r(X_all, np.array([lab[w][idx] for w in words]))
            row[ch] = round(float(r), 3)
            if r > best[ch][0]: best[ch] = (float(r), Lay)
        th = np.array([hue[w] for w in chromatic])
        # ring: predict cos & sin, reconstruct angle
        c_hat = cv_pred(X_chr, np.cos(th)); s_hat = cv_pred(X_chr, np.sin(th))
        R_ring, med_ring = circ_metrics(np.arctan2(s_hat, c_hat), th)
        # naive: raw angle as a scalar (wraparound penalty iff circular)
        t_hat = cv_pred(X_chr, th)
        R_naive, med_naive = circ_metrics(t_hat, th)
        row.update({"hue_ring_R": round(R_ring, 3), "hue_ring_meddeg": round(med_ring, 1),
                    "hue_naive_R": round(R_naive, 3), "hue_naive_meddeg": round(med_naive, 1)})
        results["by_layer"][Lay] = row
        if R_ring > best["ring_R"][0]: best["ring_R"] = (R_ring, Lay)
        if R_naive > best["naive_R"][0]: best["naive_R"] = (R_naive, Lay)
        if Lay % 4 == 0 or Lay == n_layers - 1:
            print(f"  L{Lay:2d}  L*={row['L']:+.2f} a*={row['a']:+.2f} b*={row['b']:+.2f}"
                  f"  ring R={R_ring:.2f} ({med_ring:.0f}°)  naive R={R_naive:.2f}"
                  f" ({med_naive:.0f}°)", flush=True)

    print("\n=== peaks ===")
    for k, (v, L) in best.items():
        print(f"  {k:8s} {v:+.3f} @L{L}")
    results["peaks"] = {k: {"value": round(v, 4), "layer": L} for k, (v, L) in best.items()}

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "color_ring.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
