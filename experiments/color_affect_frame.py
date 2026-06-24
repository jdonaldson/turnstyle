"""Is there a shared semantic frame for COLOR in SmolLM2, and does it overlap the
affect frame (Positivity/Potency/Activity = EPA) or is it independent?

Method (reuses the EPA external-validation rig):
  1. Ground-truth color geometry: color name -> sRGB -> CIELAB (L*=lightness,
     a*=red-green, b*=blue-yellow) — the perceptual opponent-process axes.
  2. Affect frame: fit ridge directions activations->E,P,A on ACT-lexicon words
     (robust, ~hundreds of words) at a fixed layer, in a shared standardized space.
  3. COLOR recoverable? ridge probe activations->L*,a*,b*, held-out 5-fold CV r
     per channel, layer sweep. (Does the model encode color geometry at all?)
  4. OVERLAP vs INDEPENDENCE, two ways:
     (a) project color words onto the affect axes -> each color's (E,P,A); regress
         Lab ~ (E,P,A): per-channel R^2 = how much of lightness/hue affect explains.
     (b) cosine between each color direction (L,a,b) and each affect direction
         (E,P,A) in the shared standardized activation basis.

Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/color_affect_frame.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "experiments")
from epa_external_validation import collect_acts, load_act  # reuse rig

# ── color name -> sRGB hex (~100 single-word names; canonical/standard hex) ─────
# Excludes dark/light/pale/deep/bright compounds — those leak lightness via the
# literal morpheme and would trivially inflate the L* probe (lexical, not geometry).
COLORS = {
    # primaries / basics
    "red": "ff0000", "orange": "ffa500", "yellow": "ffff00", "green": "008000",
    "blue": "0000ff", "purple": "800080", "pink": "ffc0cb", "brown": "a52a2a",
    "black": "000000", "white": "ffffff", "grey": "808080", "silver": "c0c0c0",
    "gold": "ffd700", "cyan": "00ffff", "magenta": "ff00ff", "lime": "00ff00",
    "maroon": "800000", "olive": "808000", "teal": "008080", "navy": "000080",
    "indigo": "4b0082", "violet": "ee82ee", "turquoise": "40e0d0",
    # reds / pinks
    "crimson": "dc143c", "scarlet": "ff2400", "ruby": "e0115f", "vermilion": "e34234",
    "burgundy": "800020", "wine": "722f37", "rose": "ff007f", "salmon": "fa8072",
    "coral": "ff7f50", "tomato": "ff6347", "cherry": "990000", "rust": "b7410e",
    "fuchsia": "ff00ff", "hotpink": "ff69b4", "raspberry": "e30b5d",
    # oranges / browns / earth
    "amber": "ffbf00", "apricot": "fbceb1", "peach": "ffe5b4", "tangerine": "f28500",
    "chocolate": "d2691e", "sienna": "a0522d", "mahogany": "c04000", "umber": "635147",
    "sepia": "704214", "ochre": "cc7722", "tan": "d2b48c", "beige": "f5f5dc",
    "khaki": "f0e68c", "taupe": "483c32", "fawn": "e5aa70", "auburn": "712f26",
    "copper": "b87333", "bronze": "cd7f32", "brass": "b5a642", "caramel": "c68e17",
    "chestnut": "954535", "hazel": "8e7618",
    # yellows / golds
    "lemon": "fff700", "mustard": "ffdb58", "goldenrod": "daa520", "saffron": "f4c430",
    "cream": "fffdd0", "ivory": "fffff0", "wheat": "f5deb3", "honey": "eb9605",
    # greens
    "emerald": "50c878", "jade": "00a86b", "mint": "98ff98", "sage": "9caf88",
    "moss": "8a9a5b", "forest": "228b22", "chartreuse": "7fff00", "kelly": "4cbb17",
    "pine": "01796f", "fern": "4f7942", "seafoam": "9fe2bf", "pistachio": "93c572",
    # blues / cyans
    "azure": "007fff", "cobalt": "0047ab", "cerulean": "2a52be", "sapphire": "0f52ba",
    "royal": "4169e1", "sky": "87ceeb", "powder": "b0e0e6", "steel": "4682b4",
    "slate": "708090", "denim": "1560bd", "aqua": "00ffff", "periwinkle": "ccccff",
    # purples
    "lavender": "e6e6fa", "lilac": "c8a2c8", "mauve": "e0b0ff", "plum": "8e4585",
    "orchid": "da70d6", "eggplant": "614051", "amethyst": "9966cc", "grape": "6f2da8",
    # neutrals / greys
    "charcoal": "36454f", "ash": "b2beb5", "pewter": "8ba8b7", "ebony": "1b1b1b",
    "graphite": "383428", "smoke": "738276",
}


def _hex_to_lab(h):
    """sRGB hex -> CIELAB (L*, a*, b*) under D65."""
    rgb = np.array([int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = M @ lin
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    L = 116 * f[1] - 16
    a = 500 * (f[0] - f[1])
    b = 200 * (f[1] - f[2])
    return np.array([L, a, b])


def _ridge_dir(Xs, y, alpha=10.0):
    """Ridge weight direction (unit) for y on standardized X (closed form)."""
    w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean()))
    return w / (np.linalg.norm(w) + 1e-9)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    # affect words (subsample ACT for speed) + color words
    act = load_act()
    aw = sorted(act)[:600]
    cw = [c for c in COLORS if c not in ("fuchsia", "aqua")]  # hue dups (magenta/cyan)
    print(f"affect words={len(aw)}  color words={len(cw)}", flush=True)

    print("collecting affect activations...", flush=True)
    a_acts = collect_acts(mdl, tok, dev, aw)
    print("collecting color activations...", flush=True)
    c_acts = collect_acts(mdl, tok, dev, cw)

    EPA = np.array([act[w] for w in aw])                 # (Na, 3)
    LAB = np.array([_hex_to_lab(COLORS[w]) for w in cw])  # (Nc, 3)

    n_layers = a_acts[aw[0]].shape[0]
    aff_names = ["E(valuation)", "P(otency)", "A(ctivity)"]
    col_names = ["L*(light)", "a*(red-grn)", "b*(blu-yel)"]

    # ── (3) is COLOR recoverable? ridge probe acts->Lab, 5-fold CV r, layer sweep ──
    print("\n=== color recoverability (held-out 5-fold CV Pearson r) ===")
    best = {}
    for layer in range(n_layers):
        Xc = np.array([c_acts[w][layer] for w in cw])
        rs = []
        for j in range(3):
            est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
            pred = cross_val_predict(est, Xc, LAB[:, j], cv=5)
            rs.append(float(np.corrcoef(pred, LAB[:, j])[0, 1]))
        for j in range(3):
            if rs[j] > best.get(j, (-9, -9))[0]:
                best[j] = (rs[j], layer)
    for j in range(3):
        print(f"  {col_names[j]:12s} best r={best[j][0]:+.3f} @L{best[j][1]}")

    # ── shared standardized basis at a fixed analysis layer (affect-frame peak) ──
    L0 = 12
    Xa = np.array([a_acts[w][L0] for w in aw])
    Xc = np.array([c_acts[w][L0] for w in cw])
    mu = np.concatenate([Xa, Xc]).mean(0)
    sd = np.concatenate([Xa, Xc]).std(0) + 1e-6
    Xas, Xcs = (Xa - mu) / sd, (Xc - mu) / sd

    aff_dirs = [_ridge_dir(Xas, EPA[:, k]) for k in range(3)]   # E,P,A directions
    col_dirs = [_ridge_dir(Xcs, LAB[:, j]) for j in range(3)]   # L,a,b directions

    # ── (4b) direction cosines: color axis vs affect axis ──
    print(f"\n=== direction cosines (color vs affect) @L{L0} ===")
    print("              " + "  ".join(f"{n:>12s}" for n in aff_names))
    for j in range(3):
        row = [abs(float(col_dirs[j] @ aff_dirs[k])) for k in range(3)]
        print(f"  {col_names[j]:12s} " + "  ".join(f"{v:12.3f}" for v in row))

    # ── (4a) project colors onto affect axes; regress Lab ~ (E,P,A) ──
    proj = np.stack([Xcs @ aff_dirs[k] for k in range(3)], 1)   # (Nc, 3) color affect-coords
    print(f"\n=== color Lab variance explained by affect projection (R^2) @L{L0} ===")
    for j in range(3):
        est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
        pred = cross_val_predict(est, proj, LAB[:, j], cv=5)
        ss_res = np.sum((LAB[:, j] - pred) ** 2)
        ss_tot = np.sum((LAB[:, j] - LAB[:, j].mean()) ** 2)
        print(f"  {col_names[j]:12s} R^2(from E,P,A) = {1 - ss_res/ss_tot:+.3f}")

    # ── where do colors sit on the affect frame? (mean affect coord per color) ──
    print(f"\n=== each color's affect coordinate (z-scored projection) @L{L0} ===")
    pz = (proj - proj.mean(0)) / (proj.std(0) + 1e-9)
    order = np.argsort(-pz[:, 2])     # by Activity
    print(f"  {'color':10s} {'Eval':>6s} {'Pot':>6s} {'Act':>6s}")
    for i in order:
        print(f"  {cw[i]:10s} {pz[i,0]:6.2f} {pz[i,1]:6.2f} {pz[i,2]:6.2f}")


if __name__ == "__main__":
    main()
