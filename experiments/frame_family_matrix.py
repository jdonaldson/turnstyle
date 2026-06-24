"""4th frame + the full family matrix. Adds NUMBER (numerical magnitude) to affect /
color / size and computes the complete pairwise axis-cosine matrix.

Sharp question (ATOM — A Theory Of Magnitude): does the model share ONE magnitude axis
across abstract number and physical size, or keep them separate? number·size is the test:
high cosine = shared magnitude system; ~0 = independent frames.

  1. Recoverability (held-out 5-fold CV r, SHUFFLED folds, last-token) for each scalar
     axis: Evaluation, size, number, color-L*, color-b*.
  2. Full pairwise |cos| matrix at shared-standardized layers.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")
from epa_external_validation import load_act
from color_affect_frame import COLORS, _hex_to_lab
from size_frame import SIZE

# number/quantity word -> log10(value)
NUMBER = {
    "one": 0.0, "two": 0.301, "three": 0.477, "four": 0.602, "five": 0.699,
    "six": 0.778, "seven": 0.845, "eight": 0.903, "nine": 0.954, "ten": 1.0,
    "eleven": 1.041, "twelve": 1.079, "fifteen": 1.176, "twenty": 1.301,
    "thirty": 1.477, "fifty": 1.699, "eighty": 1.903, "hundred": 2.0,
    "thousand": 3.0, "million": 6.0, "billion": 9.0, "trillion": 12.0,
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

    act = load_act(); aff_w = sorted(act)[:400]; E = np.array([act[w][0] for w in aff_w])
    col_w = [c for c in COLORS if c not in ("fuchsia", "aqua")]
    LAB = np.array([_hex_to_lab(COLORS[w]) for w in col_w])
    size_w = list(SIZE); SZ = np.array([SIZE[w] for w in size_w])
    num_w = list(NUMBER); NUM = np.array([NUMBER[w] for w in num_w])
    print(f"affect={len(aff_w)} color={len(col_w)} size={len(size_w)} number={len(num_w)}", flush=True)

    A = collect_last(mdl, tok, dev, aff_w)
    C = collect_last(mdl, tok, dev, col_w)
    S = collect_last(mdl, tok, dev, size_w)
    N = collect_last(mdl, tok, dev, num_w)
    n_layers = N[num_w[0]].shape[0]

    # frame spec: name -> (acts_dict, words, target)
    scal = {"Eval": (A, aff_w, E), "size": (S, size_w, SZ), "number": (N, num_w, NUM),
            "color-L*": (C, col_w, LAB[:, 0]), "color-b*": (C, col_w, LAB[:, 2])}

    print("\n=== recoverability (best held-out CV r, shuffled, last-token) ===")
    best_layer = {}
    for name, (acts, words, y) in scal.items():
        best = (-9, -9)
        for layer in range(n_layers):
            X = np.array([acts[w][layer] for w in words])
            r = cv_r(X, y)
            if r > best[0]:
                best = (r, layer)
        best_layer[name] = best[1]
        print(f"  {name:9s} r={best[0]:+.3f} @L{best[1]}")

    names = list(scal)
    for L0 in (4, 8, 11):
        # shared standardization across ALL words at this layer
        allX = np.concatenate([np.array([acts[w][L0] for w in words])
                               for acts, words, _ in scal.values()])
        mu, sd = allX.mean(0), allX.std(0) + 1e-6
        dirs = {}
        for name, (acts, words, y) in scal.items():
            X = (np.array([acts[w][L0] for w in words]) - mu) / sd
            dirs[name] = ridge_dir(X, y)
        print(f"\n=== |cos| matrix @L{L0} (0=orthogonal) ===")
        print("           " + " ".join(f"{n[:8]:>8s}" for n in names))
        for a in names:
            row = " ".join(f"{abs(float(dirs[a] @ dirs[b])):8.3f}" for b in names)
            print(f"  {a:9s} {row}")
        print(f"  >>> ATOM test  number·size = {abs(float(dirs['number'] @ dirs['size'])):.3f}")


if __name__ == "__main__":
    main()
