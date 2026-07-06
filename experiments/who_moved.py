"""
who_moved.py — determine WHICH words a layer operates on, from activations alone.

layer_reorg.py detected reorganization events within curated domains. This drops
the domains: a broad ~800-word vocabulary (top Brysbaert nouns + planted probes:
weekdays, months, numbers, countries/capitals — all unprivileged), and per-word
per-transition MOVEMENT scores computed purely from the geometry:

  churn(i, L)   = 1 - Jaccard(kNN_L(i), kNN_{L+1}(i))          # changed communities
  rearr(i, L)   = 1 - spearman(D_L[i, N], D_{L+1}[i, N])       # community reorganized
                  over N = union of i's top-25 neighbors at either layer
                  (catches the weekday-ring case: neighbors stay, arrangement bends)

Validation is blind: weekdays are 7 words among ~800 — if the method works they
should SURFACE in the top movers at the known ring-assembly band (L26-30), and
numbers should stay near the bottom after L2.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/who_moved.py [model_id]
Writes experiments/results/who_moved.json.
"""
from __future__ import annotations
import csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from time_ring import collect_last

from wordfreq import zipf_frequency

TEMPLATE = "They talked about {w}."
N_VOCAB = 750
K_NN = 10
K_LOCAL = 25

PLANTED = {
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"],
    "month": ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"],
    "number": ["one", "two", "three", "four", "five", "six", "seven", "eight",
               "nine", "ten", "eleven", "twelve"],
    "geo": ["France", "Paris", "Japan", "Tokyo", "Italy", "Rome", "Egypt",
            "Cairo", "Canada", "Ottawa", "Spain", "Madrid"],
}


def load_vocab():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "brysbaert_concreteness.txt")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            w = r["Word"]
            if (r["Dom_Pos"] == "Noun" and w.isalpha() and w.islower()
                    and len(w) >= 3 and float(r["Percent_known"]) > 0.97):
                z = zipf_frequency(w, "en")
                if z >= 4.0:
                    rows.append((w, z))
    rows.sort(key=lambda t: -t[1])
    vocab = [w for w, _ in rows[:N_VOCAB]]
    planted = [w for ws in PLANTED.values() for w in ws]
    return sorted(set(vocab) | set(planted))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def tag(w):
    for t, ws in PLANTED.items():
        if w in ws:
            return t
    return ""


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    words = load_vocab()
    n = len(words)
    print(f"{n} words ({sum(1 for w in words if tag(w))} planted), model {mid}", flush=True)

    acts = collect_last(mdl, tok, dev, words, TEMPLATE)
    n_layers = acts[words[0]].shape[0]
    print("collected; computing geometry...", flush=True)

    # per-layer z-scored distance matrices (float32, n~780 -> ~2.4MB each)
    Ds = []
    for L in range(n_layers):
        X = np.array([acts[w][L] for w in words], dtype=np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        sq = (X ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
        Ds.append(D)

    knn = []
    for L in range(n_layers):
        D = Ds[L].copy(); np.fill_diagonal(D, np.inf)
        knn.append(np.argsort(D, axis=1))          # sorted neighbor indices

    results = {"model": mid, "n_words": n, "transitions": {}}
    med_curve, moves = [], []
    for L in range(n_layers - 1):
        churn = np.zeros(n); rearr = np.zeros(n)
        for i in range(n):
            a, b = set(knn[L][i, :K_NN]), set(knn[L + 1][i, :K_NN])
            churn[i] = 1 - len(a & b) / len(a | b)
            N = list(dict.fromkeys(list(knn[L][i, :K_LOCAL]) +
                                   list(knn[L + 1][i, :K_LOCAL])))
            rearr[i] = 1 - spearman(Ds[L][i, N], Ds[L + 1][i, N])
        move = np.maximum(churn, rearr)
        moves.append(move)
        med_curve.append(round(float(np.median(move)), 3))
        top = np.argsort(move)[-15:][::-1]
        results["transitions"][f"L{L}->{L+1}"] = {
            "median_move": med_curve[-1],
            "top_movers": [{"w": words[i], "tag": tag(words[i]),
                            "churn": round(float(churn[i]), 2),
                            "rearr": round(float(rearr[i]), 2)} for i in top]}

    # report: the interesting transitions (biggest median + the known bands)
    order = np.argsort(med_curve)[::-1]
    show = sorted(set([int(order[0]), int(order[1]), 1, 8, 26, 27, 28, 29, 31]))
    for L in show:
        t = results["transitions"][f"L{L}->{L+1}"]
        names = ", ".join(f"{m['w']}{'[' + m['tag'] + ']' if m['tag'] else ''}"
                          for m in t["top_movers"][:12])
        print(f"\nL{L}->{L+1} (median {t['median_move']:.3f}): {names}", flush=True)

    # planted-set validation: mean movement percentile per tag per transition band
    print("\n=== planted-set mean movement percentile by band ===", flush=True)
    idx_by_tag = {t: [words.index(w) for w in ws] for t, ws in PLANTED.items()}
    bands = {"L1-4": range(1, 4), "L8-12": range(8, 12), "L18-22": range(18, 22),
             "L26-30": range(26, 30), "L31-32": range(31, n_layers - 1)}
    for bname, rng in bands.items():
        parts = []
        for tname, idxs in idx_by_tag.items():
            pctl = []
            for L in rng:
                ranks = np.argsort(np.argsort(moves[L]))
                pctl.append(np.mean([ranks[i] / n for i in idxs]))
            parts.append(f"{tname}={100*np.mean(pctl):.0f}%")
        line = f"  {bname:8s} " + "  ".join(parts)
        results.setdefault("validation", {})[bname] = line
        print(line, flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "who_moved.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
