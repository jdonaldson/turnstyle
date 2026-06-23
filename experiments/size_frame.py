"""A THIRD frame: SIZE (physical magnitude). Tests the "family of orthogonal frames"
hypothesis — if affect and color are distinct orthogonal frames, a size frame should
(a) be recoverable and (b) be orthogonal to BOTH.

  1. Recoverable? ridge probe activations → log10(size in metres), held-out 5-fold CV
     r, layer sweep. (last-token readout — lesson from the color cross-lingual audit.)
  2. Orthogonal? at a shared-standardized layer, cosine(size axis, {Evaluation, color
     L*, color b*}); plus eval·color as the known-orthogonal sanity check.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")
from epa_external_validation import load_act
from color_affect_frame import COLORS, _hex_to_lab

# noun -> log10(characteristic size in metres). Spans ~atom (1e-10) to galaxy (1e21).
SIZE = {
    "atom": -10, "molecule": -9, "virus": -7, "bacterium": -6, "cell": -5,
    "dust": -4, "ant": -2.5, "fly": -2.2, "bee": -2, "mosquito": -2.7,
    "mouse": -1, "sparrow": -0.8, "rat": -0.6, "frog": -1, "cat": -0.3,
    "rabbit": -0.4, "dog": 0, "fox": -0.2, "human": 0.25, "wolf": 0.1,
    "pig": 0.2, "sheep": 0.1, "cow": 0.4, "horse": 0.4, "bear": 0.4,
    "lion": 0.3, "tiger": 0.4, "car": 0.7, "truck": 1, "elephant": 0.6,
    "whale": 1.3, "shark": 0.7, "house": 1, "tree": 1, "bus": 1.1,
    "boat": 1, "building": 1.5, "hill": 2.5, "mountain": 3.5, "island": 4,
    "lake": 3.5, "river": 4, "ocean": 6.5, "planet": 7, "moon": 6.5,
    "sun": 9, "star": 9, "galaxy": 21,
}
TEMPLATE = "It is a {w}."


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
        out[w] = stk[:, idxs[-1], :].float().cpu().numpy()          # last subword
    return out


def cv_r(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, KFold
    est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
    # shuffled folds: SIZE is size-sorted, so contiguous folds would be size bands
    # the model never trains on (systematic extrapolation -> spurious negative r).
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    pred = cross_val_predict(est, X, y, cv=cv)
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

    size_w = list(SIZE)
    SZ = np.array([SIZE[w] for w in size_w])
    act = load_act(); aff_w = sorted(act)[:400]
    E = np.array([act[w][0] for w in aff_w])
    col_w = [c for c in COLORS if c not in ("fuchsia", "aqua")]
    LAB = np.array([_hex_to_lab(COLORS[w]) for w in col_w])

    print(f"size nouns={len(size_w)}  affect={len(aff_w)}  colors={len(col_w)}", flush=True)
    s_acts = collect_last(mdl, tok, dev, size_w)
    a_acts = collect_last(mdl, tok, dev, aff_w)
    c_acts = collect_last(mdl, tok, dev, col_w)
    n_layers = s_acts[size_w[0]].shape[0]

    # (1) recoverability of size, layer sweep
    print("\n=== size recoverability (held-out 5-fold CV r), layer sweep ===")
    best = (-9, -9)
    for layer in range(n_layers):
        X = np.array([s_acts[w][layer] for w in size_w])
        r = cv_r(X, SZ)
        if r > best[0]:
            best = (r, layer)
        if layer % 4 == 0 or r == best[0]:
            print(f"  L{layer:<2d} r={r:+.3f}", flush=True)
    print(f"  BEST size r={best[0]:+.3f} @L{best[1]}")

    # (2) orthogonality at a shared-standardized mid layer
    for L0 in (best[1], 11):
        Xs = np.array([s_acts[w][L0] for w in size_w])
        Xa = np.array([a_acts[w][L0] for w in aff_w])
        Xc = np.array([c_acts[w][L0] for w in col_w])
        allX = np.concatenate([Xs, Xa, Xc])
        mu, sd = allX.mean(0), allX.std(0) + 1e-6
        d_size = ridge_dir((Xs - mu) / sd, SZ)
        d_eval = ridge_dir((Xa - mu) / sd, E)
        d_L = ridge_dir((Xc - mu) / sd, LAB[:, 0])
        d_b = ridge_dir((Xc - mu) / sd, LAB[:, 2])
        cos = lambda u, v: abs(float(u @ v))
        print(f"\n=== axis cosines @L{L0} (|cos|, 0=orthogonal) ===")
        print(f"  size · Evaluation = {cos(d_size, d_eval):.3f}")
        print(f"  size · color-L*   = {cos(d_size, d_L):.3f}")
        print(f"  size · color-b*   = {cos(d_size, d_b):.3f}")
        print(f"  Evaluation · color-L* (sanity) = {cos(d_eval, d_L):.3f}")


if __name__ == "__main__":
    main()
