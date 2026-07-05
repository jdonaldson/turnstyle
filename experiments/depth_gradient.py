"""
depth_gradient.py — the honest version of "rung position tracks network depth":
re-read every frame's peak layer on zipf-RESIDUALIZED labels, and test what the
depth gradient actually tracks.

Two questions:
  1. HONEST PEAKS: raw peaks moved under residualization for age (L2->L14) and
     time (L2->L17) — early peaks can be lexical/frequency flavor. Compute the
     residualized peak for EVERY frame (10 canonical + warmth/potency/weight).
  2. WHAT IS "DEPTH"? Operationalize a frame's abstractness as the mean
     Brysbaert concreteness of its lexicon, and correlate with residualized
     peak depth. Prediction test with a live counterexample built in: animacy's
     words are maximally CONCRETE (rock, dog, woman) yet it peaks at L17. If the
     correlation is weak, the gradient tracks the PROPERTY computed, not the
     WORDS used — a sharper claim than "abstract words peak late".

Writes experiments/results/depth_gradient.json.
Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/depth_gradient.py
"""
from __future__ import annotations
import csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ordering_frames import cv_r
from freq_audit_family import collect_last_t, residualize
from turnstyle.frame_library import CANONICAL_FRAMES
from warmth_competence import WARMTH, POTENCY, T_PERSON
from frame_algebra import WEIGHT, T_OBJ

from wordfreq import zipf_frequency

EXTRA = {
    "warmth": (WARMTH, T_PERSON),
    "potency": (POTENCY, T_PERSON),
    "weight": (WEIGHT, T_OBJ),
}


def load_conc_norms():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "brysbaert_concreteness.txt")
    norms = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            norms[r["Word"]] = float(r["Conc.M"])
    return norms


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    norms = load_conc_norms()

    frames = {}
    for name, spec in CANONICAL_FRAMES.items():
        frames[name] = (spec["data"], spec.get("template", "It is a {w} object."))
    for name, (data, tmpl) in EXTRA.items():
        frames[name] = (data, tmpl)

    results = {"model": mid, "frames": {}}
    print(f"{len(frames)} frames")
    print("frame          n   conc(lex)  raw peak      residualized peak")
    for name, (data, tmpl) in frames.items():
        words = list(data)
        y = np.array([data[w] for w in words], dtype=float)
        z = np.array([zipf_frequency(w, "en") for w in words])
        y_res = residualize(y, z)
        conc = [norms[w] for w in words if w in norms]
        mean_conc = float(np.mean(conc)) if conc else float("nan")
        acts = collect_last_t(mdl, tok, dev, words, tmpl)
        n_layers = acts[words[0]].shape[0]
        best_raw = best_res = (-9, -1)
        for L in range(n_layers):
            X = np.array([acts[w][L] for w in words])
            r1, r2 = cv_r(X, y), cv_r(X, y_res)
            if r1 > best_raw[0]: best_raw = (r1, L)
            if r2 > best_res[0]: best_res = (r2, L)
        results["frames"][name] = {
            "n": len(words), "mean_lexicon_concreteness": round(mean_conc, 2),
            "conc_coverage": f"{len(conc)}/{len(words)}",
            "raw": {"r": round(best_raw[0], 3), "layer": best_raw[1]},
            "residualized": {"r": round(best_res[0], 3), "layer": best_res[1]}}
        print(f"{name:13s} {len(words):3d}   {mean_conc:5.2f}     "
              f"{best_raw[0]:+.3f}@L{best_raw[1]:<3d}  {best_res[0]:+.3f}@L{best_res[1]}",
              flush=True)

    # does lexicon concreteness predict residualized depth?
    ok = [(v["mean_lexicon_concreteness"], v["residualized"]["layer"], k)
          for k, v in results["frames"].items()
          if not np.isnan(v["mean_lexicon_concreteness"])
          and v["residualized"]["r"] > 0.5]          # only frames with real signal
    xs = np.array([t[0] for t in ok]); ys = np.array([t[1] for t in ok])
    pear = float(np.corrcoef(xs, ys)[0, 1])
    rank = float(np.corrcoef(np.argsort(np.argsort(xs)),
                             np.argsort(np.argsort(ys)))[0, 1])
    results["conc_vs_depth"] = {"n": len(ok), "pearson": round(pear, 3),
                                "spearman": round(rank, 3)}
    print(f"\nlexicon-concreteness vs residualized depth (n={len(ok)}): "
          f"pearson {pear:+.3f}, spearman {rank:+.3f}")
    print("(negative = concrete-vocab frames peak early; near-zero = depth tracks "
          "the PROPERTY, not the words)")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "depth_gradient.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
