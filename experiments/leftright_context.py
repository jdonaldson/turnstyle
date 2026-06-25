"""Is left/right's distractor-dominance just the copular template, or is the lateral sense
genuinely buried? Test CONTEXT-GATING: if left/right are context-disambiguated polysemes,
a SPATIAL template should raise the lateral-partner cosine and lower the distractor cosine,
where the monosemous compass words barely move.

  copular: "It is {w}."   ("it is right" reads as CORRECT; "it is left" as LEFTOVER)
  spatial: "Turn to the {w}."  (forces the lateral reading of left/right)

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/leftright_context.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from turnstyle.frame_library import _collect

LEX = ["left", "right", "correct", "wrong", "true",
       "remaining", "leftover", "departed",
       "north", "south", "east", "west", "up", "down", "thing"]
TMPLS = {"copular": "It is {w}.", "spatial": "Turn to the {w}."}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()

    A = {name: _collect(mdl, tok, dev, LEX, t, pool="last") for name, t in TMPLS.items()}
    nL = A["copular"][LEX[0]].shape[0]
    idx = {w: i for i, w in enumerate(LEX)}

    def cos(acts, L, a, b):
        X = np.array([acts[w][L] for w in LEX])
        X = X - X.mean(0)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        return float(X[idx[a]] @ X[idx[b]])

    # right: lateral(left) vs distractor(correct); left: lateral(right) vs distractor(remaining)
    # compass control: cos(east, west) — should be high & template-insensitive
    print("cos(right,left) / cos(right,correct)  ||  cos(left,right)/cos(left,remaining)"
          "  ||  cos(east,west) control")
    print(f"{'L':>3} | {'R-lat·c':>8} {'R-cor·c':>8} {'R-lat·s':>8} {'R-cor·s':>8}"
          f" | {'L-rem·c':>8} {'L-rem·s':>8} | {'EW·c':>6} {'EW·s':>6}")
    for L in range(nL):
        c, s = A["copular"], A["spatial"]
        rl_c, rc_c = cos(c, L, "right", "left"), cos(c, L, "right", "correct")
        rl_s, rc_s = cos(s, L, "right", "left"), cos(s, L, "right", "correct")
        lr_rem_c = cos(c, L, "left", "remaining")
        lr_rem_s = cos(s, L, "left", "remaining")
        ew_c, ew_s = cos(c, L, "east", "west"), cos(s, L, "east", "west")
        print(f"{L:>3} | {rl_c:>8.2f} {rc_c:>8.2f} {rl_s:>8.2f} {rc_s:>8.2f}"
              f" | {lr_rem_c:>8.2f} {lr_rem_s:>8.2f} | {ew_c:>6.2f} {ew_s:>6.2f}", flush=True)
    print("\nIf spatial context gates the sense: R-lat·s > R-lat·c and R-cor·s < R-cor·c "
          "(lateral up, correctness down), while EW (monosemous) barely moves.")


if __name__ == "__main__":
    main()
