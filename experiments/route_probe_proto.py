"""Prototype: can a route-classification probe auto-select which recognition probe
to use, instead of the manual dropdown?

(1) 7-way routing accuracy: last-token hidden state at layer L → which probe task.
    Grouped 5-fold CV, layer sweep.
(2) Misroute / coverage gate: train on the 7, score NONE prompts (MC tasks that
    should NOT use a probe). With a confidence threshold τ, report recall on the 7
    vs false-activation on NONE — the conservative gate that decides auto vs abstain.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/route_probe_proto.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from turnstyle.bbh import load_task

PROBE_TASKS = ["snarks", "movie_recommendation", "ruin_names", "disambiguation_qa",
               "salient_translation_error_detection", "temporal_sequences",
               "date_understanding"]
# MC-shaped tasks that should NOT route to a probe (symbolic handles them, or no probe):
NONE_TASKS = ["logical_deduction_three_objects", "penguins_in_a_table",
              "formal_fallacies", "hyperbaton"]
N = 50


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model on {dev}", flush=True)

    def last_hidden(prompt):
        enc = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        return torch.stack(hs, 0)[:, 0, -1, :].float().cpu().numpy()   # [L+1, H]

    def collect(tasks):
        X, y = [], []
        for ti, t in enumerate(tasks):
            for ex in load_task(t)[:N]:
                X.append(last_hidden(ex["input"])); y.append(ti)
            print(f"  collected {t}", flush=True)
        return np.stack(X), np.array(y)                                # [n, L+1, H]

    print("collecting probe-task states…", flush=True)
    X, y = collect(PROBE_TASKS)
    print("collecting NONE states…", flush=True)
    Xn, _ = collect(NONE_TASKS)
    nL = X.shape[1]

    # (1) 7-way routing accuracy, layer sweep
    print("\n=== 7-way routing accuracy (5-fold CV) ===")
    best = (-1, -1)
    for L in range(nL):
        accs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[:, L], y):
            sc = StandardScaler().fit(X[tr, L])
            clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[tr, L]), y[tr])
            accs.append(clf.score(sc.transform(X[te, L]), y[te]))
        acc = float(np.mean(accs))
        if acc > best[0]:
            best = (acc, L)
        if L % 2 == 0 or L == nL - 1:
            print(f"  L{L:>2}: {acc:.3f}")
    acc, bL = best
    print(f"  >>> best 7-way routing = {acc:.3f} @L{bL}  (chance {1/len(PROBE_TASKS):.3f})")

    # per-class recall at best layer (out-of-fold predictions)
    from sklearn.model_selection import cross_val_predict
    sc = StandardScaler().fit(X[:, bL])
    pred = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5),
                             sc.transform(X[:, bL]), y, cv=5)
    print("  per-task routing recall:")
    for ti, t in enumerate(PROBE_TASKS):
        m = y == ti
        print(f"    {t:42s} {(pred[m] == ti).mean():.2f}")

    # (2) coverage gate: fit on all 7, score NONE; threshold sweep
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[:, bL]), y)
    # in-class confidence via out-of-fold proba
    proba_in = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5),
                                 sc.transform(X[:, bL]), y, cv=5, method="predict_proba")
    conf_in = proba_in.max(1)
    correct_in = proba_in.argmax(1) == y
    conf_none = clf.predict_proba(sc.transform(Xn[:, bL])).max(1)
    print("\n=== coverage gate (conservative τ: route only if maxprob ≥ τ) ===")
    print(f"  {'τ':>4} {'route+correct (7)':>18} {'false-activate (NONE)':>22}")
    for tau in (0.5, 0.7, 0.8, 0.9, 0.95):
        recall = float((correct_in & (conf_in >= tau)).mean())     # routed AND right
        false_act = float((conf_none >= tau).mean())               # NONE wrongly routed
        print(f"  {tau:>4.2f} {recall:>18.2f} {false_act:>22.2f}")
    print(f"\n  mean conf: in-class {conf_in.mean():.2f} | NONE {conf_none.mean():.2f}")


if __name__ == "__main__":
    main()
