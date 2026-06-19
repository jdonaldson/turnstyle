"""Decisive test: does a BROAD-vocab pole probe generalize to held-out words?

pole_probe.py showed that on BBH's 4-word vocab (new/old/cheap/expensive) the
probe memorizes word identity (leave-one-root-out ~0%). The earlier "81% LOO"
claim was on a ~24-word vocab. So the real question for whether the probe BEATS
the regex (generalizes to unseen words / languages) is:

  Train a pole direction on a broad adjective vocab; does it classify words it
  has NEVER seen — in particular the BBH-4 — by their HIGH/LOW pole?

We collect the adjective-token hidden state for a broad labeled vocab in
comparative + superlative templates, then evaluate:
  (a) leave-one-root-out over the WHOLE broad vocab — the honest generalization rate
  (b) train on (broad minus BBH-4) → test on synthetic BBH-4   (pure cross-vocab)
  (c) train on (broad minus BBH-4) → test on REAL BBH-4 cached acts
      (cross-vocab + cross-context = the actual deployment test)

Usage:  python experiments/pole_generalize.py [--collect]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_harness as H
import pole_probe as PP

CACHE = "experiments/data/pole_generalize_acts.npz"

# root: (comparative, superlative, pole)  — pole = "more of the attribute" = HIGH
VOCAB = {
    "tall": ("taller", "tallest", H.HIGH),
    "short": ("shorter", "shortest", H.LOW),
    "big": ("bigger", "biggest", H.HIGH),
    "small": ("smaller", "smallest", H.LOW),
    "large": ("larger", "largest", H.HIGH),
    "heavy": ("heavier", "heaviest", H.HIGH),
    "light": ("lighter", "lightest", H.LOW),
    "fast": ("faster", "fastest", H.HIGH),
    "slow": ("slower", "slowest", H.LOW),
    "long": ("longer", "longest", H.HIGH),
    "wide": ("wider", "widest", H.HIGH),
    "narrow": ("narrower", "narrowest", H.LOW),
    "hot": ("hotter", "hottest", H.HIGH),
    "cold": ("colder", "coldest", H.LOW),
    "warm": ("warmer", "warmest", H.HIGH),
    "strong": ("stronger", "strongest", H.HIGH),
    "weak": ("weaker", "weakest", H.LOW),
    "bright": ("brighter", "brightest", H.HIGH),
    "dark": ("darker", "darkest", H.LOW),
    "loud": ("louder", "loudest", H.HIGH),
    "quiet": ("quieter", "quietest", H.LOW),
    "rich": ("richer", "richest", H.HIGH),
    "poor": ("poorer", "poorest", H.LOW),
    "good": ("better", "best", H.HIGH),
    "bad": ("worse", "worst", H.LOW),
    "happy": ("happier", "happiest", H.HIGH),
    "sad": ("sadder", "saddest", H.LOW),
    "deep": ("deeper", "deepest", H.HIGH),
    "shallow": ("shallower", "shallowest", H.LOW),
    "young": ("younger", "youngest", H.HIGH),
    # the BBH-4 (held out in tests b/c):
    "new": ("newer", "newest", H.HIGH),
    "old": ("older", "oldest", H.LOW),
    "cheap": ("cheaper", "cheapest", H.LOW),
    "expensive": ("more expensive", "most expensive", H.HIGH),
}
BBH4 = {"new", "old", "cheap", "expensive"}

_NOUNS = ["object", "item", "one", "box", "thing", "car"]
_TEMPLATES = [
    "The first {n} is {comp} than the second {n}.",
    "The red {n} is {comp} than the blue {n}.",
    "The {n} on the left is the {sup}.",
    "Of all of them, this {n} is the {sup}.",
]


def collect():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(PP.MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        PP.MODEL_ID, dtype=torch.float16).to(device).eval()

    acts, poles, roots = [], [], []
    for root, (comp, sup, pole) in VOCAB.items():
        for n in _NOUNS:
            for tpl in _TEMPLATES:
                form = comp if "{comp}" in tpl else sup
                sent = tpl.format(n=n, comp=comp, sup=sup)
                # char span of the LAST word of the adjective form (content word)
                content = form.split()[-1]
                cs = sent.find(content)
                ce = cs + len(content)
                enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
                offs = enc.pop("offset_mapping")[0].tolist()
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = mdl(**enc, output_hidden_states=True)
                hs = torch.stack(out.hidden_states, 0)[:, 0].float().cpu().numpy()
                tk = None
                for k, (s, e) in enumerate(offs):
                    if e > cs and s < ce:
                        tk = k
                if tk is None:
                    continue
                acts.append(hs[:, tk, :].astype(np.float16))
                poles.append(pole)
                roots.append(root)
        print(f"  {root:10s} done ({len(acts)})", end="\r", flush=True)
    print()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, acts=np.stack(acts),
                        poles=np.array(poles), roots=np.array(roots))
    print(f"saved {len(acts)} synthetic occurrences → {CACHE}")


def _fit(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xtr), ytr)
    return sc, clf


def _score(sc, clf, X, y):
    return clf.score(sc.transform(X), y)


def analyze():
    d = np.load(CACHE, allow_pickle=True)
    A = d["acts"].astype(np.float32)
    y = (d["poles"] == H.HIGH).astype(int)
    roots = d["roots"]
    nL = A.shape[1]
    uniq = sorted(set(roots.tolist()))

    # real BBH-4 activations from the other cache
    rb = np.load(PP.CACHE, allow_pickle=True)
    rbA = rb["acts"].astype(np.float32)
    rby = (rb["poles"] == H.HIGH).astype(int)

    print(f"broad vocab: {len(uniq)} roots, N={len(y)} synthetic occ")
    print(f"{'L':>3} {'LOroot':>7} {'b:syn4':>7} {'c:real4':>8}")
    rows = []
    for L in range(nL):
        X = A[:, L, :]
        # (a) leave-one-root-out over whole broad vocab
        loo = []
        for r in uniq:
            te = roots == r
            if len(set(y[~te])) < 2:
                continue
            sc, clf = _fit(X[~te], y[~te])
            loo.append(_score(sc, clf, X[te], y[te]))
        lo = float(np.mean(loo))
        # train on broad minus BBH4
        notbbh = ~np.isin(roots, list(BBH4))
        sc, clf = _fit(X[notbbh], y[notbbh])
        # (b) synthetic BBH4
        synb = np.isin(roots, list(BBH4))
        b = _score(sc, clf, X[synb], y[synb])
        # (c) real BBH4 cached acts at same layer
        c = _score(sc, clf, rbA[:, L, :], rby)
        rows.append((L, lo, b, c))
        print(f"{L:>3} {lo:>7.3f} {b:>7.3f} {c:>8.3f}")
    best = max(rows, key=lambda r: r[1])
    print(f"\nbest LOroot: L{best[0]}  LOroot={best[1]:.3f}  "
          f"syn-BBH4={best[2]:.3f}  real-BBH4={best[3]:.3f}")
    bestc = max(rows, key=lambda r: r[3])
    print(f"best real-BBH4 transfer: L{bestc[0]}  real-BBH4={bestc[3]:.3f}  "
          f"(LOroot={bestc[1]:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect or not os.path.exists(CACHE):
        collect()
    analyze()
