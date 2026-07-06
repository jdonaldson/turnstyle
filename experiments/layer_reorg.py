"""
layer_reorg.py — hypothesis-free detection of late-layer reorganization events,
then read the movers.

Method:
  1. Per domain wordset, per layer: z-scored pairwise distance matrix D_L.
  2. CHANGE curve: change(L) = 1 - spearman(upper(D_L), upper(D_{L+1})) —
     detects THAT the geometry reorganized, with no hypothesis about the shape.
  3. At each domain's biggest late event (L >= 2/3 depth): rank the MOVERS —
     word pairs whose distance-rank shifted most across the event. Movers are
     word pairs, so the reorganization is readable by construction.
  4. Validation case: the weekday ring is KNOWN to assemble ~L25-30 on Mistral —
     the detector must fire there with wrap-ish pairs converging, or the
     detector is broken.
  5. One designed logic index: country<->capital BINDING (mean within-pair
     distance / mean cross-pair distance, lower = bound) per layer — does
     relational binding tighten late?

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/layer_reorg.py [model_id]
Writes experiments/results/layer_reorg.json.
"""
from __future__ import annotations
import json, os, sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from time_ring import collect_last

DOMAINS = {
    "weekdays": (["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday"], "The event happened on {w}."),
    "months": (["January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"],
               "The event happened in {w}."),
    "numbers": (["one", "two", "three", "four", "five", "six", "seven", "eight",
                 "nine", "ten", "eleven", "twelve"], "The value is {w}."),
    "kinship": (["mother", "father", "sister", "brother", "daughter", "son",
                 "aunt", "uncle", "grandmother", "grandfather", "niece", "nephew"],
                "They visited their {w}."),
    "mixed_nouns": (["hammer", "eagle", "sofa", "river", "doctor", "bottle",
                     "whale", "coin", "teacher", "mountain", "spider", "violin"],
                    "They looked at the {w}."),
    "geo": (["France", "Paris", "Japan", "Tokyo", "Italy", "Rome", "Egypt",
             "Cairo", "Canada", "Ottawa", "Spain", "Madrid"],
            "They wrote about {w}."),
}
CAP_PAIRS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
             ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Spain", "Madrid")]


def dist_matrix(acts, words, L):
    X = np.array([acts[w][L] for w in words])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))


def upper(M):
    n = M.shape[0]
    return M[np.triu_indices(n, 1)]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model {mid} on {dev}", flush=True)

    results = {"model": mid, "domains": {}}
    for name, (words, tmpl) in DOMAINS.items():
        acts = collect_last(mdl, tok, dev, words, tmpl)
        n_layers = acts[words[0]].shape[0]
        Ds = [dist_matrix(acts, words, L) for L in range(n_layers)]
        change = [round(1 - spearman(upper(Ds[L]), upper(Ds[L + 1])), 3)
                  for L in range(n_layers - 1)]
        late_start = 2 * n_layers // 3
        ev = late_start + int(np.argmax(change[late_start:]))   # biggest LATE event
        # movers across the event: rank shift per pair between L=ev and L=ev+1
        pairs = list(combinations(range(len(words)), 2))
        r0 = np.argsort(np.argsort(upper(Ds[ev])))
        r1 = np.argsort(np.argsort(upper(Ds[ev + 1])))
        shift = r1.astype(int) - r0.astype(int)
        order = np.argsort(shift)
        def pname(k): i, j = pairs[k]; return f"{words[i]}~{words[j]}"
        movers_close = [(pname(k), int(shift[k])) for k in order[:5]]
        movers_far = [(pname(k), int(shift[k])) for k in order[-5:][::-1]]
        results["domains"][name] = {
            "change_curve": change,
            "late_event_layer": ev,
            "late_event_change": change[ev],
            "median_change": round(float(np.median(change)), 3),
            "movers_closer": movers_close, "movers_farther": movers_far}
        print(f"\n=== {name} === median change {np.median(change):.3f}, "
              f"late event @L{ev}->{ev+1} (change {change[ev]:.3f})", flush=True)
        top3 = np.argsort(change)[-3:][::-1]
        print(f"  biggest events overall: " +
              ", ".join(f"L{int(t)}->{int(t)+1} ({change[int(t)]:.2f})" for t in top3))
        print(f"  movers closer @late event: {movers_close}")
        print(f"  movers farther @late event: {movers_far}")

        if name == "geo":       # designed logic index: capital binding by layer
            idx = {w: k for k, w in enumerate(words)}
            curve = []
            for L in range(n_layers):
                D = Ds[L]
                within = np.mean([D[idx[a], idx[b]] for a, b in CAP_PAIRS])
                mask = np.ones_like(D, bool); np.fill_diagonal(mask, False)
                for a, b in CAP_PAIRS:
                    mask[idx[a], idx[b]] = mask[idx[b], idx[a]] = False
                cross = D[mask].mean()
                curve.append(round(float(within / cross), 3))
            results["capital_binding"] = curve
            print("  capital binding (within/cross, lower=bound): " +
                  " ".join(f"L{L}:{v:.2f}" for L, v in enumerate(curve) if L % 4 == 0))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "layer_reorg.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
