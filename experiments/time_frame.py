"""A FIFTH frame: TIME (duration). Completes the ATOM quartet (number/size/time/space
minus space). Tests whether duration shares a magnitude axis with abstract number or
physical size — the ATOM trio time·number, time·size, number·size. Independent (~0) =
SmolLM2 keeps each magnitude on its own direction; high = a shared magnitude system.

Duration words use a sense-biasing template ("It lasted a {w}.") so "second"/"minute"
read as durations, not ordinal/tiny. Recoverability: ridge acts -> log10(seconds),
shuffled 5-fold CV, last-token. Orthogonality: cosines vs affect/color/size/number.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")
from epa_external_validation import load_act
from color_affect_frame import COLORS, _hex_to_lab
from size_frame import SIZE
from frame_family_matrix import NUMBER

# duration word -> log10(seconds)
TIME = {
    "millisecond": -3.0, "instant": -1.5, "heartbeat": -0.1, "second": 0.0,
    "moment": 0.5, "minute": 1.78, "hour": 3.56, "day": 4.94, "week": 5.78,
    "fortnight": 6.08, "month": 6.42, "year": 7.50, "decade": 8.50,
    "generation": 8.90, "lifetime": 9.40, "century": 9.50, "millennium": 10.50,
    "era": 12.0, "epoch": 13.0, "eon": 16.5,
}
NEUTRAL = "It is a {w}."
TIME_TMPL = "It lasted a {w}."


def collect_last(model, tok, device, words, template):
    import torch
    out = {}
    for w in words:
        sent = template.format(w=w)
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
    time_w = list(TIME); TM = np.array([TIME[w] for w in time_w])
    print(f"affect={len(aff_w)} color={len(col_w)} size={len(size_w)} "
          f"number={len(num_w)} time={len(time_w)}", flush=True)

    A = collect_last(mdl, tok, dev, aff_w, NEUTRAL)
    C = collect_last(mdl, tok, dev, col_w, NEUTRAL)
    S = collect_last(mdl, tok, dev, size_w, NEUTRAL)
    N = collect_last(mdl, tok, dev, num_w, NEUTRAL)
    T = collect_last(mdl, tok, dev, time_w, TIME_TMPL)
    n_layers = T[time_w[0]].shape[0]

    scal = {"Eval": (A, aff_w, E), "size": (S, size_w, SZ), "number": (N, num_w, NUM),
            "time": (T, time_w, TM), "color-L*": (C, col_w, LAB[:, 0]),
            "color-b*": (C, col_w, LAB[:, 2])}

    print("\n=== time recoverability (held-out 5-fold CV r, shuffled, last-token) ===")
    best = (-9, -9)
    for layer in range(n_layers):
        r = cv_r(np.array([T[w][layer] for w in time_w]), TM)
        if r > best[0]:
            best = (r, layer)
        if layer % 4 == 0 or r == best[0]:
            print(f"  L{layer:<2d} r={r:+.3f}", flush=True)
    print(f"  BEST time r={best[0]:+.3f} @L{best[1]}")

    names = list(scal)
    for L0 in (4, 8, 11):
        allX = np.concatenate([np.array([acts[w][L0] for w in words])
                               for acts, words, _ in scal.values()])
        mu, sd = allX.mean(0), allX.std(0) + 1e-6
        dirs = {n: ridge_dir((np.array([a[w][L0] for w in ws]) - mu) / sd, y)
                for n, (a, ws, y) in scal.items()}
        cos = lambda u, v: abs(float(dirs[u] @ dirs[v]))
        print(f"\n=== |cos| matrix @L{L0} ===")
        print("           " + " ".join(f"{n[:8]:>8s}" for n in names))
        for a in names:
            print(f"  {a:9s} " + " ".join(f"{cos(a, b):8.3f}" for b in names))
        print(f"  >>> ATOM trio  time·number={cos('time','number'):.3f}  "
              f"time·size={cos('time','size'):.3f}  number·size={cos('number','size'):.3f}")


if __name__ == "__main__":
    main()
