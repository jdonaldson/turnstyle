"""Corrected direction-frame test: does cross-lingual direction transfer MID-STACK?

The first probe tested transfer at each axis's English-CV-optimal layer (NS peaked at
L0 = lexical surface, which can't transfer by construction). The frame-family finding is
that cross-lingual meaning lives MID-STACK (~L11-16). So here we fit the NS/EW direction
at EVERY layer on English compass anchors and measure cross-lingual + synonym transfer
PER LAYER — checking each test word's PRIMARY axis sign (the dominant direction).

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/direction_frame_layers.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np

from turnstyle.frame_library import _collect, _ridge_dir

_TMPL = "Move {w}."
S = 1 / math.sqrt(2)

NS_FIT = {"north": 1, "south": -1, "east": 0, "west": 0,
          "northeast": S, "northwest": S, "southeast": -S, "southwest": -S}
EW_FIT = {"east": 1, "west": -1, "north": 0, "south": 0,
          "northeast": S, "northwest": -S, "southeast": S, "southwest": -S}
FIT_WORDS = list(NS_FIT)

# (word, axis, expected_sign) — test only the PRIMARY axis (dominant direction)
TESTS = {
    "synonyms": [("up", "ns", +1), ("down", "ns", -1), ("left", "ew", -1),
                 ("right", "ew", +1), ("upward", "ns", +1), ("downward", "ns", -1)],
    "es": [("norte", "ns", +1), ("sur", "ns", -1), ("este", "ew", +1), ("oeste", "ew", -1)],
    "fr": [("nord", "ns", +1), ("sud", "ns", -1), ("est", "ew", +1), ("ouest", "ew", -1)],
    "de": [("norden", "ns", +1), ("süden", "ns", -1), ("osten", "ew", +1), ("westen", "ew", -1)],
}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()

    test_words = [w for items in TESTS.values() for (w, _, _) in items]
    fit_acts = _collect(mdl, tok, dev, FIT_WORDS, _TMPL, pool="last")
    test_acts = _collect(mdl, tok, dev, test_words, _TMPL, pool="last")
    n_layers = fit_acts[FIT_WORDS[0]].shape[0]
    ns_y = np.array([NS_FIT[w] for w in FIT_WORDS], float)
    ew_y = np.array([EW_FIT[w] for w in FIT_WORDS], float)

    print(f"{'L':>3} | " + " ".join(f"{g:>9}" for g in TESTS) + " | overallXL")
    for L in range(n_layers):
        Xf = np.array([fit_acts[w][L] for w in FIT_WORDS])
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        Zf = (Xf - mu) / sd
        ns_dir, ew_dir = _ridge_dir(Zf, ns_y), _ridge_dir(Zf, ew_y)
        proj = {}
        for w in test_words:
            z = (test_acts[w][L] - mu) / sd
            proj[w] = {"ns": float(z @ ns_dir), "ew": float(z @ ew_dir)}
        cells, xl_hit, xl_tot = [], 0, 0
        for g, items in TESTS.items():
            hit = sum(1 for (w, ax, sgn) in items
                      if (proj[w][ax] > 0) == (sgn > 0))
            cells.append(f"{hit}/{len(items)}")
            if g != "synonyms":
                xl_hit += hit; xl_tot += len(items)
        print(f"{L:>3} | " + " ".join(f"{c:>9}" for c in cells)
              + f" | {xl_hit}/{xl_tot} xling", flush=True)
    print("\n(primary-axis sign accuracy; cross-lingual = es+fr+de. "
          "Look for a MID-STACK layer where xling jumps.)")


if __name__ == "__main__":
    main()
