"""
freq_control.py — the frequency-confound control frame.

Frequency is the universal confound for any "the model encodes X" claim: if a
frame's defining words differ systematically in corpus frequency, a probe could
be reading frequency, not X. Three checks, run over the SAME words/template/
activations as the ordering rungs (ordering_frames.py):

  1. Is log-frequency itself recoverable?  (expected: yes — establishes the
     control direction exists to test against)
  2. Direction orthogonality: |cos| between the zipf direction and every rung
     direction in the shared standardized space. Small = the rungs are not
     frequency in disguise.
  3. Target-level confound: corr(rung scalar labels, zipf of the rung's words).
     Large |r| here means the rung's LABELS are frequency-loaded regardless of
     what the probe reads — the more dangerous version.

Writes experiments/results/freq_control.json.
Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/freq_control.py
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ordering_frames import FRAMES, collect_last, cv_r, ridge_dir

from wordfreq import zipf_frequency


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    cats = list(FRAMES)
    words = {c: list(FRAMES[c]) for c in cats}
    y = {c: np.array([FRAMES[c][w] for w in words[c]]) for c in cats}
    allw = sorted({w for c in cats for w in words[c]})
    zipf = {w: zipf_frequency(w, "en") for w in allw}
    print(f"{len(allw)} words, zipf range {min(zipf.values()):.2f}-{max(zipf.values()):.2f}",
          flush=True)

    acts = collect_last(mdl, tok, dev, allw)
    n_layers = acts[allw[0]].shape[0]
    yf = np.array([zipf[w] for w in allw])

    # 1. recoverability of zipf itself
    print("\n=== zipf recoverability (5-fold CV r) ===", flush=True)
    best = (-9, -9)
    freq_r_by_layer = []
    for L in range(n_layers):
        X = np.array([acts[w][L] for w in allw])
        r = cv_r(X, yf)
        freq_r_by_layer.append(round(float(r), 4))
        if r > best[0]:
            best = (r, L)
    print(f"  frequency  n={len(allw)}  r={best[0]:+.3f} @L{best[1]}")

    # 2. direction orthogonality at the matrix layers used by ordering_frames
    results = {"model": mid, "zipf_recoverability": {"r": round(best[0], 4), "layer": best[1]},
               "zipf_r_by_layer": freq_r_by_layer, "cos_vs_rungs": {}, "label_confound": {}}
    for L0 in (8, 12):
        allX = np.concatenate([np.array([acts[w][L0] for w in words[c]]) for c in cats]
                              + [np.array([acts[w][L0] for w in allw])])
        mu, sd = allX.mean(0), allX.std(0) + 1e-6
        fdir = ridge_dir((np.array([acts[w][L0] for w in allw]) - mu) / sd, yf)
        row = {}
        print(f"\n=== |cos(freq, rung)| @L{L0} ===")
        for c in cats:
            d = ridge_dir((np.array([acts[w][L0] for w in words[c]]) - mu) / sd, y[c])
            row[c] = round(abs(float(fdir @ d)), 4)
            print(f"  {c:9s} {row[c]:.3f}")
        results["cos_vs_rungs"][f"L{L0}"] = row

    # 3. target-level confound: corr(rung labels, zipf) per rung
    print("\n=== label-level confound corr(rung scalar, zipf) ===")
    for c in cats:
        zw = np.array([zipf[w] for w in words[c]])
        r = float(np.corrcoef(y[c], zw)[0, 1])
        results["label_confound"][c] = round(r, 4)
        print(f"  {c:9s} {r:+.3f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "freq_control.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
