"""
freq_residual.py — do the frequency-flagged rungs survive residualization?

freq_control.py flagged space (label↔zipf r = −0.69, dir |cos| ~0.21) and mildly
age (−0.29) / material (+0.33). The decisive test: regress zipf OUT of each
rung's scalar targets and refit. If CV r holds, the frame is real signal wearing
a frequency-correlated wardrobe; if it collapses, the "frame" was frequency.

Also refits the unflagged control rung (opinion) — residualization should be a
no-op there (label corr +0.13), guarding against the test itself destroying
signal.

Writes experiments/results/freq_residual.json.
Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/freq_residual.py
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ordering_frames import FRAMES, collect_last, cv_r

from wordfreq import zipf_frequency

RUNGS = ["space", "age", "material", "opinion"]   # flagged x3 + clean control


def residualize(y, z):
    z1 = np.stack([z, np.ones_like(z)], axis=1)
    beta, *_ = np.linalg.lstsq(z1, y, rcond=None)
    return y - z1 @ beta


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    allw = sorted({w for c in RUNGS for w in FRAMES[c]})
    acts = collect_last(mdl, tok, dev, allw)
    n_layers = acts[allw[0]].shape[0]

    results = {"model": mid, "rungs": {}}
    print("rung        raw r@L      residualized r@L     label|zipf r")
    for c in RUNGS:
        words = list(FRAMES[c])
        y = np.array([FRAMES[c][w] for w in words], dtype=float)
        z = np.array([zipf_frequency(w, "en") for w in words])
        y_res = residualize(y, z)
        lab_r = float(np.corrcoef(y, z)[0, 1])
        best_raw = best_res = (-9, -1)
        for L in range(n_layers):
            X = np.array([acts[w][L] for w in words])
            r_raw, r_res = cv_r(X, y), cv_r(X, y_res)
            if r_raw > best_raw[0]: best_raw = (r_raw, L)
            if r_res > best_res[0]: best_res = (r_res, L)
        results["rungs"][c] = {"raw": {"r": round(best_raw[0], 3), "layer": best_raw[1]},
                               "residualized": {"r": round(best_res[0], 3), "layer": best_res[1]},
                               "label_zipf_r": round(lab_r, 3)}
        print(f"{c:10s}  {best_raw[0]:+.3f}@L{best_raw[1]:<3d}  "
              f"{best_res[0]:+.3f}@L{best_res[1]:<3d}          {lab_r:+.3f}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "freq_residual.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
