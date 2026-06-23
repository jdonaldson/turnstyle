"""The canonical English adjective-ordering hierarchy AS a family of semantic frames.

Order: opinion > size > age > shape > color > origin/SPACE > material (> purpose).
Each rung is built as a scalar ADJECTIVE frame (template "It is a {w} object." — the
hyperbaton template, so this ties to the subjectivity-ordering work). Question: is each
ordering category a recoverable, mutually-orthogonal frame? If so, the ordering hierarchy
is literally an ordering of (near-)orthogonal conceptual frames.

  1. Recoverability: ridge acts -> the category's scalar, shuffled 5-fold CV, last-token.
  2. Orthogonality: cosine matrix of the per-category gradient axes (shared standardization).
purpose (cooking/racing/...) is categorical-functional with no clean scalar — noted, excluded.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")
from color_affect_frame import COLORS, _hex_to_lab

TEMPLATE = "It is a {w} object."

# category -> {adjective: scalar}.  scalars are ordinal (within-category gradient).
FRAMES = {
    "opinion": {"terrible": -3, "horrible": -3, "awful": -3, "bad": -2, "nasty": -2,
                "poor": -1, "mediocre": 0, "decent": 1, "good": 2, "great": 2,
                "lovely": 3, "wonderful": 3, "excellent": 3, "delightful": 3},
    "size": {"tiny": -3, "minuscule": -3, "small": -2, "little": -2, "modest": -1,
             "average": 0, "large": 1, "big": 1, "huge": 2, "enormous": 3,
             "gigantic": 3, "massive": 3},
    "age": {"newborn": -3, "new": -2, "young": -2, "fresh": -2, "recent": -1,
            "modern": -1, "mature": 1, "old": 2, "aged": 2, "elderly": 2,
            "ancient": 3, "antique": 3},
    "shape": {"round": 0, "circular": 0, "spherical": 0, "oval": 1, "curved": 1,
              "square": 2, "rectangular": 2, "boxy": 2, "flat": 2, "long": 3,
              "thin": 3, "narrow": 3, "elongated": 3},
    "space": {"local": 0, "domestic": 1, "native": 1, "regional": 2, "national": 3,
              "foreign": 4, "distant": 5, "remote": 5, "exotic": 5, "faraway": 6,
              "alien": 7, "cosmic": 9},
    "material": {"soft": 0, "woolen": 0, "fluffy": 0, "papery": 1, "leathery": 1,
                 "rubbery": 1, "wooden": 2, "plastic": 2, "glassy": 3, "ceramic": 3,
                 "stony": 4, "concrete": 4, "metallic": 5, "iron": 5, "steely": 5,
                 "golden": 5},
}
# color rung: real color terms, scalar = lightness L*.
_COLOR_TERMS = ["black", "navy", "brown", "maroon", "red", "green", "purple", "blue",
                "teal", "grey", "orange", "gold", "pink", "yellow", "cyan", "white"]
FRAMES["color"] = {c: float(_hex_to_lab(COLORS[c])[0]) for c in _COLOR_TERMS if c in COLORS}
PURPOSE = ["cooking", "racing", "hunting", "sleeping", "sailing", "hiking", "farming"]


def collect_last(model, tok, device, words):
    import torch
    out = {}
    for w in words:
        sent = TEMPLATE.format(w=w)
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        stk = torch.stack(hs, 0)[:, 0, :, :]
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        out[w] = stk[:, idxs[-1], :].float().cpu().numpy()
    return out


def cv_r(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, KFold
    est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
    pred = cross_val_predict(est, X, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(np.corrcoef(pred, y)[0, 1])


def ridge_dir(Xs, y, alpha=10.0):
    w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean()))
    return w / (np.linalg.norm(w) + 1e-9)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    cats = list(FRAMES)
    words = {c: list(FRAMES[c]) for c in cats}
    y = {c: np.array([FRAMES[c][w] for w in words[c]]) for c in cats}
    allw = sorted({w for c in cats for w in words[c]} | set(PURPOSE))
    print("categories:", cats, "  (purpose categorical, excluded from scalars)", flush=True)
    acts = collect_last(mdl, tok, dev, allw)
    n_layers = acts[allw[0]].shape[0]

    print("\n=== recoverability (held-out 5-fold CV r, shuffled, last-token) ===")
    best_layer = {}
    for c in cats:
        best = (-9, -9)
        for layer in range(n_layers):
            X = np.array([acts[w][layer] for w in words[c]])
            r = cv_r(X, y[c])
            if r > best[0]:
                best = (r, layer)
        best_layer[c] = best[1]
        print(f"  {c:9s} n={len(words[c]):2d}  r={best[0]:+.3f} @L{best[1]}")

    for L0 in (8, 12):
        allX = np.concatenate([np.array([acts[w][L0] for w in words[c]]) for c in cats])
        mu, sd = allX.mean(0), allX.std(0) + 1e-6
        dirs = {c: ridge_dir((np.array([acts[w][L0] for w in words[c]]) - mu) / sd, y[c])
                for c in cats}
        print(f"\n=== gradient-axis |cos| matrix @L{L0} (ordering categories) ===")
        print("           " + " ".join(f"{c[:8]:>8s}" for c in cats))
        for a in cats:
            print(f"  {a:9s} " + " ".join(f"{abs(float(dirs[a] @ dirs[b])):8.3f}" for b in cats))
        offdiag = [abs(float(dirs[a] @ dirs[b])) for i, a in enumerate(cats)
                   for b in cats[i + 1:]]
        print(f"  mean off-diagonal |cos| = {np.mean(offdiag):.3f}  max = {np.max(offdiag):.3f}")


if __name__ == "__main__":
    main()
