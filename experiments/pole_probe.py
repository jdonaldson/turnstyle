"""Pole probe for logical_deduction — the stage-2 resolver as an activation probe.

The harness (pole_harness.py) proved frames+solver hit 100% with oracle pole and
61.5% with no pole (structural axes only). This script asks: can a probe on the
adjective token's hidden state supply the pole, recovering the 289 age/price
examples — and does it GENERALIZE (leave-one-word-out, the cross-vocab proxy for
cross-language)?

Pipeline:
  1. collect_activations  — one forward pass per BBH prompt; cache the hidden
     state at every scalar-adjective token (new/old/cheap/expensive forms),
     all layers, with the oracle pole label. Cached to data/ (gitignored).
  2. layer_sweep          — LogReg per layer: random 5-fold CV (in-dist) and
     leave-one-ROOT-out (cross-vocab generalization).
  3. end_to_end           — split examples; train probe on one half; for each
     test example build {root: pole} from its adjective activations; run the
     harness solver with that resolver; report accuracy vs the 100% oracle.

Usage:  python experiments/pole_probe.py [--collect] [--layer L]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_harness as H

CACHE = "experiments/data/pole_probe_acts.npz"
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# content-adjective surface forms; root = strip -er/-est, pole from oracle lexicon
_ADJ_RE = re.compile(r"\b(new(?:er|est)?|old(?:er|est)?|"
                     r"cheap(?:er|est)?|expensive)\b", re.I)
_ROOT_POLE = {"new": H.HIGH, "old": H.LOW, "cheap": H.LOW, "expensive": H.HIGH}


def _root(word: str) -> str:
    w = word.lower()
    if w.endswith("est"):
        w = w[:-3]
    elif w.endswith("er"):
        w = w[:-2]
    return w


def adjective_occurrences(prompt: str):
    """Yield (char_start, char_end, root, pole) for every scalar adjective."""
    for m in _ADJ_RE.finditer(prompt):
        root = _root(m.group(1))
        if root in _ROOT_POLE:
            yield m.start(1), m.end(1), root, _ROOT_POLE[root]


# ── stage 1: collect activations ─────────────────────────────────────────────

def collect_activations():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16).to(device).eval()

    acts, poles, roots, ex_ids, task_ids = [], [], [], [], []
    for ti, t in enumerate(H.TASKS):
        data = json.load(open(f"{H.BBH}/{t}.json"))
        for ei, ex in enumerate(data):
            prompt = ex["input"]
            occ = list(adjective_occurrences(prompt))
            if not occ:
                continue
            enc = tok(prompt, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = mdl(**enc, output_hidden_states=True)
            # (L+1, seq, hidden) at the chosen token, fp16 → fp32 cpu
            hs = torch.stack(out.hidden_states, 0)[:, 0].float().cpu().numpy()
            for cs, ce, root, pole in occ:
                # token whose span ends the content word
                ti_tok = None
                for k, (s, e) in enumerate(offsets):
                    if e > cs and s < ce:
                        ti_tok = k
                if ti_tok is None:
                    continue
                acts.append(hs[:, ti_tok, :].astype(np.float16))
                poles.append(pole)
                roots.append(root)
                ex_ids.append(ei)
                task_ids.append(ti)
            print(f"  [{t[18:].split('_')[0]} {ei}] occ={len(occ)} "
                  f"total={len(acts)}", end="\r", flush=True)
    print()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(
        CACHE,
        acts=np.stack(acts),               # (N, L+1, hidden) fp16
        poles=np.array(poles),
        roots=np.array(roots),
        ex_ids=np.array(ex_ids),
        task_ids=np.array(task_ids),
    )
    print(f"saved {len(acts)} occurrences → {CACHE}")


# ── stage 2: layer sweep (CV + leave-one-root-out) ───────────────────────────

def _fit_eval(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.score(sc.transform(Xte), yte), sc, clf


def layer_sweep():
    from sklearn.model_selection import StratifiedKFold
    d = np.load(CACHE, allow_pickle=True)
    A = d["acts"].astype(np.float32)        # (N, L+1, hidden)
    y = (d["poles"] == H.HIGH).astype(int)
    roots = d["roots"]
    nL = A.shape[1]
    uniq_roots = sorted(set(roots.tolist()))
    print(f"N={len(y)} occurrences  roots={dict(zip(*np.unique(roots, return_counts=True)))}")
    print(f"{'L':>3} {'cv':>7} {'LOroot':>8}")
    best = (0, -1)
    for L in range(nL):
        X = A[:, L, :]
        # random 5-fold CV
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        cv = np.mean([_fit_eval(X[tr], y[tr], X[te], y[te])[0]
                      for tr, te in skf.split(X, y)])
        # leave-one-root-out
        loo = []
        for r in uniq_roots:
            te = roots == r
            tr = ~te
            if len(set(y[tr])) < 2:
                continue
            loo.append(_fit_eval(X[tr], y[tr], X[te], y[te])[0])
        lo = np.mean(loo) if loo else float("nan")
        flag = ""
        if cv > best[0]:
            best = (cv, L)
        print(f"{L:>3} {cv:>7.3f} {lo:>8.3f}{flag}")
    print(f"best CV layer = L{best[1]} ({best[0]:.3f})")


# ── stage 3: end-to-end through the harness ──────────────────────────────────

def end_to_end(layer: int):
    d = np.load(CACHE, allow_pickle=True)
    A = d["acts"].astype(np.float32)[:, layer, :]
    y = (d["poles"] == H.HIGH).astype(int)
    roots = d["roots"]
    ex_ids = d["ex_ids"]
    task_ids = d["task_ids"]

    # split EXAMPLES (not occurrences) into train/test, per task, deterministic
    rng = np.random.RandomState(0)
    is_test = np.zeros(len(y), bool)
    for t in range(3):
        exs = sorted(set(ex_ids[task_ids == t].tolist()))
        test_ex = set(rng.choice(exs, size=len(exs) // 2, replace=False).tolist())
        for i in range(len(y)):
            if task_ids[i] == t and ex_ids[i] in test_ex:
                is_test[i] = True

    sc, clf = _fit_eval(A[~is_test], y[~is_test], A[~is_test], y[~is_test])[1:]
    # per (task, example) predicted {root: pole} from that example's occurrences
    pred = clf.predict(sc.transform(A))
    perex: dict[tuple, dict] = {}
    for i in range(len(y)):
        key = (int(task_ids[i]), int(ex_ids[i]))
        perex.setdefault(key, {})
        r = str(roots[i])
        perex[key].setdefault(r, []).append(int(pred[i]))

    def example_pole_map(key):
        out = {}
        for r, votes in perex[key].items():
            out[r] = H.HIGH if (sum(votes) * 2 >= len(votes)) else H.LOW
        return out

    print(f"\n=== end-to-end @ L{layer} (probe-supplied pole, test half) ===")
    grand_c = grand_n = 0
    for t in range(3):
        data = json.load(open(f"{H.BBH}/{H.TASKS[t]}.json"))
        test_ex = sorted({int(ex_ids[i]) for i in range(len(y))
                          if task_ids[i] == t and is_test[i]})
        c = 0
        for ei in test_ex:
            key = (t, ei)
            pm = example_pole_map(key)

            def resolver(phrase, _pm=pm):
                root, flip = H.normalize_phrase(phrase)
                base = _pm.get(root)
                if base is None:                      # structural-only fallback
                    base = _ROOT_POLE.get(root)
                return None if base is None else base * flip

            ans, _ = H.solve(data[ei]["input"], resolver)
            if ans == data[ei]["target"].strip():
                c += 1
        n = len(test_ex)
        grand_c += c
        grand_n += n
        print(f"  {H.TASKS[t][18:]:16s} {c}/{n} = {c/n:5.1%}  (probe-bearing test exs)")
    print(f"  {'TOTAL':16s} {grand_c}/{grand_n} = {grand_c/grand_n:5.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()
    if args.collect or not os.path.exists(CACHE):
        collect_activations()
    layer_sweep()
    if args.layer is not None:
        end_to_end(args.layer)
