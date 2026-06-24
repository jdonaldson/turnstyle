"""Is ATOM (a shared magnitude axis across number/size/time) refuted in BIGGER models too?

frame_family_matrix.py + time_frame.py found number/size/time mutually orthogonal in
SmolLM2-1.7B (all pairwise |cos| < 0.09 → no shared magnitude axis; strong ATOM refuted).
This re-runs the ATOM trio on other models to test whether that's size/family-invariant
(cf. the EPA cross-model replication, which held across SmolLM2/Phi/Qwen).

  python experiments/atom_crossmodel.py Qwen/Qwen2.5-1.5B-Instruct
  python experiments/atom_crossmodel.py microsoft/Phi-4-mini-instruct

Recoverability (shuffled CV r, last-token) + the trio cosines at several layers.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")
from size_frame import SIZE
from frame_family_matrix import NUMBER
from time_frame import TIME

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
    return w / (np.linalg.norm(w) + 1e-12)


def main():
    mid = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"MODEL: {mid}", flush=True)

    frames = {"number": (list(NUMBER), np.array(list(NUMBER.values()), float), NEUTRAL),
              "size": (list(SIZE), np.array(list(SIZE.values()), float), NEUTRAL),
              "time": (list(TIME), np.array(list(TIME.values()), float), TIME_TMPL)}
    acts = {n: collect_last(mdl, tok, dev, ws, tmpl) for n, (ws, y, tmpl) in frames.items()}
    n_layers = acts["number"][frames["number"][0][0]].shape[0]

    print("\n=== recoverability (shuffled CV r, last-token) ===")
    best = {}
    for n, (ws, y, _) in frames.items():
        b = (-9, -9)
        for L in range(n_layers):
            r = cv_r(np.array([acts[n][w][L] for w in ws]), y)
            if r > b[0]:
                b = (r, L)
        best[n] = b
        print(f"  {n:7s} r={b[0]:+.3f} @L{b[1]}")

    print("\n=== ATOM trio |cos| (number/size/time) ===")
    for L0 in (4, 8, max(2, n_layers // 2)):
        allX = np.concatenate([np.array([acts[n][w][L0] for w in frames[n][0]])
                               for n in frames])
        mu, sd = allX.mean(0), allX.std(0) + 1e-6
        d = {n: ridge_dir((np.array([acts[n][w][L0] for w in frames[n][0]]) - mu) / sd,
                          frames[n][1]) for n in frames}
        cos = lambda a, b: abs(float(d[a] @ d[b]))
        print(f"  L{L0:<2d}  number·size={cos('number','size'):.3f}  "
              f"time·number={cos('time','number'):.3f}  time·size={cos('time','size'):.3f}")


if __name__ == "__main__":
    main()
