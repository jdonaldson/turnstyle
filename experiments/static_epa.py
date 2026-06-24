"""Do SIMPLE (static) word embeddings capture Osgood's E-P-A — or is it an LLM thing?

Runs the same E-P-A test on classic static fastText vectors (Common Crawl English
300d) as we ran on SmolLM2's contextual activations: fit a bipolar axis per factor
from the English pole words, then measure factor independence (|cos|) and held-out
pole-sign accuracy. Static monolingual embeddings can't be tested cross-lingually
(separate space per language) — which is precisely the LLM's distinctive contribution.

Scans the .vec file for just the pole words (no full load).
Usage:  python experiments/static_epa.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import osgood_epa as OE

VEC = "/Users/jdonaldson/Projects/po-clustering/models/cc.en.300.vec"


def words():
    out = {}
    for f in ("evaluation", "potency", "activity"):
        out[f] = {"hi": OE.EPA[f]["en"]["hi"], "lo": OE.EPA[f]["en"]["lo"]}
    return out


def load_vectors(targets):
    need = set(targets)
    vec = {}
    with open(VEC, encoding="utf-8") as fh:
        next(fh)  # header: count dim
        for line in fh:
            sp = line.rstrip().split(" ")
            w = sp[0]
            if w in need:
                vec[w] = np.asarray(sp[1:], dtype=np.float32)
                need.discard(w)
                if not need:
                    break
    return vec, need


def axis(hi, lo):
    d = hi.mean(0) - lo.mean(0)
    return d / (np.linalg.norm(d) + 1e-12), 0.5 * (hi.mean(0) + lo.mean(0))


def main():
    W = words()
    allw = [w for f in W.values() for w in (f["hi"] + f["lo"])]
    vec, missing = load_vectors(allw)
    if missing:
        print("missing from fastText:", sorted(missing))
    # standardize over the pole words (same as the LLM analysis)
    M = np.vstack([vec[w] for w in allw if w in vec])
    mu, sd = M.mean(0), M.std(0) + 1e-6

    def z(w):
        return (vec[w] - mu) / sd

    dirs, cents = {}, {}
    print("static fastText (cc.en.300) — Osgood E-P-A\n")
    print(f"{'factor':11} {'held-out':>9}")
    for f, poles in W.items():
        hi = np.vstack([z(w) for w in poles["hi"] if w in vec])
        lo = np.vstack([z(w) for w in poles["lo"] if w in vec])
        d, c = axis(hi, lo); dirs[f] = d; cents[f] = c @ d
        # held-out leave-one-word-out sign
        items = [(w, 1) for w in poles["hi"] if w in vec] + \
                [(w, -1) for w in poles["lo"] if w in vec]
        ok = 0
        for w, lab in items:
            keep_hi = np.vstack([z(x) for x, l in items if l == 1 and x != w])
            keep_lo = np.vstack([z(x) for x, l in items if l == -1 and x != w])
            d2, c2 = axis(keep_hi, keep_lo)
            ok += int(((z(w) @ d2 - c2 @ d2) > 0) == (lab > 0))
        print(f"{f:11} {ok/len(items):>9.2f}")
    facs = list(W)
    print("\nfactor independence |cos|:")
    import itertools
    for a, b in itertools.combinations(facs, 2):
        print(f"  {a[:4].title()}-{b[:4].title()}: {abs(float(dirs[a] @ dirs[b])):.2f}")
    print("\n(cross-lingual is N/A for monolingual static embeddings — that's the point)")


if __name__ == "__main__":
    main()
