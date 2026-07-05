"""
warmth_competence.py — Fiske's stereotype-content dimensions vs Osgood's E/P.

Social cognition says people judge others on two universal dimensions: WARMTH
(cold<->kind) and COMPETENCE (inept<->brilliant). Osgood's affect space has
Evaluation and Potency. The horse race: are the social axes the SAME directions
as the affect axes, or a separate social panel? The degree of collinearity IS
the finding (either answer is interesting).

All four fit as person-trait frames in one shared standardized space, same
template "They are a {w} person." for the social + potency lexicons; Evaluation
from the canonical opinion frame (its own template). Audit: zipf on the new
lexicons. Writes experiments/results/warmth_competence.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ordering_frames import cv_r, ridge_dir
from freq_audit_family import collect_last_t, residualize
from turnstyle.frame_library import CANONICAL_FRAMES

from wordfreq import zipf_frequency

T_PERSON = "They are a {w} person."

WARMTH = {"hostile": -3, "cruel": -3, "cold": -2, "unfriendly": -2, "harsh": -2,
          "distant": -1, "aloof": -1, "polite": 1, "pleasant": 1, "friendly": 2,
          "warm": 2, "gentle": 2, "kind": 3, "caring": 3, "affectionate": 3}
# v2 lexicon: v1 had label|zipf +0.49 (rare negatives, common positives) and
# took a residualization haircut (0.933 -> 0.628). Rebalanced: COMMON negatives
# (useless, hopeless, lazy, sloppy) and RARE positives (masterful, adept, deft)
# alongside the originals, targeting label|zipf ~ 0.
COMPETENCE = {"incompetent": -3, "inept": -3, "hopeless": -3, "useless": -2,
              "sloppy": -2, "clumsy": -2, "lazy": -2, "careless": -1,
              "mediocre": -1, "capable": 1, "competent": 2, "skilled": 2,
              "efficient": 2, "adept": 2, "deft": 2, "expert": 3,
              "brilliant": 3, "masterful": 3, "virtuosic": 3}
POTENCY = {"weak": -3, "feeble": -3, "frail": -2, "helpless": -2, "powerless": -2,
           "timid": -1, "assertive": 1, "strong": 2, "powerful": 2, "forceful": 2,
           "mighty": 3, "dominant": 3}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    opinion = CANONICAL_FRAMES["opinion"]
    pools = {
        "warmth": (list(WARMTH), np.array(list(WARMTH.values()), dtype=float), T_PERSON),
        "competence": (list(COMPETENCE), np.array(list(COMPETENCE.values()), dtype=float), T_PERSON),
        "potency": (list(POTENCY), np.array(list(POTENCY.values()), dtype=float), T_PERSON),
        "evaluation": (list(opinion["data"]),
                       np.array(list(opinion["data"].values()), dtype=float),
                       opinion.get("template", "It is a {w} object.")),
    }

    acts, recov = {}, {}
    results = {"model": mid, "recoverability": {}, "label_zipf": {}, "cos": {}}
    for name, (words, y, tmpl) in pools.items():
        acts.update(collect_last_t(mdl, tok, dev, words, tmpl))
        z = np.array([zipf_frequency(w, "en") for w in words])
        lab = float(np.corrcoef(y, z)[0, 1])
        y_res = residualize(y, z)
        n_layers = acts[words[0]].shape[0]
        best_raw = best_res = (-9, -1)
        for L in range(n_layers):
            X = np.array([acts[w][L] for w in words])
            r1, r2 = cv_r(X, y), cv_r(X, y_res)
            if r1 > best_raw[0]: best_raw = (r1, L)
            if r2 > best_res[0]: best_res = (r2, L)
        results["recoverability"][name] = {
            "raw": {"r": round(best_raw[0], 3), "layer": best_raw[1]},
            "zipf_res": {"r": round(best_res[0], 3), "layer": best_res[1]}}
        results["label_zipf"][name] = round(lab, 3)
        print(f"  {name:11s} n={len(words):2d} zipf {lab:+.2f}  "
              f"raw {best_raw[0]:+.3f}@L{best_raw[1]}  res {best_res[0]:+.3f}@L{best_res[1]}",
              flush=True)

    for L0 in (8, 12):
        stack = np.concatenate([np.array([acts[w][L0] for w in pools[n][0]])
                                for n in pools])
        mu, sd = stack.mean(0), stack.std(0) + 1e-6
        dirs = {n: ridge_dir((np.array([acts[w][L0] for w in pools[n][0]]) - mu) / sd,
                             pools[n][1]) for n in pools}
        names = list(pools)
        cos = {f"{a}|{b}": round(abs(float(dirs[a] @ dirs[b])), 3)
               for i, a in enumerate(names) for b in names[i + 1:]}
        results["cos"][f"L{L0}"] = cos
        print(f"  |cos| @L{L0}: " + "  ".join(f"{k}={v:.2f}" for k, v in cos.items()),
              flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "warmth_competence.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
