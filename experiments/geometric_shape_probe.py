"""Does SmolLM2 linearly encode SVG-path GEOMETRY in its hidden states?

The model reads the path string and we probe its hidden state (frame/probe pattern).
The confound: vertex-count N is correlated with string length (more sides -> more L
tokens), so an N-probe might just read length, not geometry. Per the null-design rule
(preserve the confound, test the residual), the SHARP test is trapezoid-vs-kite: both
4-vertex, ~same token length, differ ONLY in geometry. We probe:

  (1) shape class (10-way)        vs majority + token-count-only baselines
  (2) trapezoid vs kite           LENGTH-CONTROLLED clean geometry test
  (3) vertex count N (polygons)   vs token-count-only baseline (expect length-explained)

Readout at the last token of the "...draws a" context (before options), and mean-pooled
over the body. Layer sweep, 5-fold CV.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/geometric_shape_probe.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np

from turnstyle.bbh import load_task

NVERT = {"line": 2, "triangle": 3, "trapezoid": 4, "kite": 4,
         "pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8}


def shape_of(e):
    tgt = e["target"].strip().strip("()")
    m = re.search(rf"\({tgt}\)\s*([^\n]+)", e["input"])
    return m.group(1).strip().lower() if m else None


def body_of(e):
    return e["input"].split("Options:")[0].rstrip()    # path + "... draws a"


def cv_acc(X, y, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y)
    if len(set(y)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()

    ex = load_task("geometric_shapes")
    shapes, ntok, last_by_layer, mean_by_layer = [], [], None, None
    nlayers = None
    for i, e in enumerate(ex):
        sh = shape_of(e)
        if sh is None:
            continue
        body = body_of(e)
        enc = tok(body, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = mdl(**enc, output_hidden_states=True)
        hs = out.hidden_states                      # (L+1) x (1,T,H)
        if nlayers is None:
            nlayers = len(hs)
            last_by_layer = [[] for _ in range(nlayers)]
            mean_by_layer = [[] for _ in range(nlayers)]
        T = enc["input_ids"].shape[1]
        for l in range(nlayers):
            h = hs[l][0].float().cpu().numpy()
            last_by_layer[l].append(h[T - 1])
            mean_by_layer[l].append(h.mean(0))
        shapes.append(sh)
        ntok.append(T)
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(ex)} forward passes", flush=True)

    shapes = np.array(shapes)
    ntok = np.array(ntok).reshape(-1, 1).astype(float)
    Nlabels = np.array([NVERT.get(s, -1) for s in shapes])
    poly = Nlabels > 0
    tk = np.isin(shapes, ["trapezoid", "kite"])

    # cheap baselines
    maj_shape = max((np.mean(shapes == s) for s in set(shapes)))
    maj_tk = max(np.mean(shapes[tk] == "trapezoid"), np.mean(shapes[tk] == "kite"))
    len_shape = cv_acc(ntok, shapes)
    len_N = cv_acc(ntok[poly], Nlabels[poly])
    len_tk = cv_acc(ntok[tk], shapes[tk])
    print(f"\nn={len(shapes)}  shapes={sorted(set(shapes))}")
    print(f"BASELINES  shape: majority={maj_shape:.2f} tokenlen={len_shape:.2f} | "
          f"N(poly): tokenlen={len_N:.2f} | trap/kite: majority={maj_tk:.2f} tokenlen={len_tk:.2f}")

    print(f"\n{'layer':>5} | {'shape':>6} {'shapeMP':>7} | {'N':>5} {'N_MP':>5} | "
          f"{'tk':>5} {'tk_MP':>6}   (MP=mean-pool)")
    best = {"shape": (0, 0.0), "N": (0, 0.0), "tk": (0, 0.0)}
    for l in range(nlayers):
        XL = np.array(last_by_layer[l]); XM = np.array(mean_by_layer[l])
        s_l = cv_acc(XL, shapes);            s_m = cv_acc(XM, shapes)
        n_l = cv_acc(XL[poly], Nlabels[poly]); n_m = cv_acc(XM[poly], Nlabels[poly])
        t_l = cv_acc(XL[tk], shapes[tk]);    t_m = cv_acc(XM[tk], shapes[tk])
        for k, v in (("shape", max(s_l, s_m)), ("N", max(n_l, n_m)), ("tk", max(t_l, t_m))):
            if v > best[k][1]:
                best[k] = (l, v)
        print(f"{l:>5} | {s_l:6.2f} {s_m:7.2f} | {n_l:5.2f} {n_m:5.2f} | {t_l:5.2f} {t_m:6.2f}",
              flush=True)

    print(f"\nBEST  shape: L{best['shape'][0]}={best['shape'][1]:.2f} (maj {maj_shape:.2f}, "
          f"len {len_shape:.2f}) | N: L{best['N'][0]}={best['N'][1]:.2f} (len {len_N:.2f}) | "
          f"trap/kite: L{best['tk'][0]}={best['tk'][1]:.2f} (maj {maj_tk:.2f}, len {len_tk:.2f})")
    print("trap/kite >> majority/len => genuine GEOMETRY encoding (length-controlled).")


if __name__ == "__main__":
    main()
