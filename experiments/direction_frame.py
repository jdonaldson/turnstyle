"""Does SmolLM2 encode spatial DIRECTION as a recoverable low-D frame?

Direction is 2D, so fit two scalar frames (frame-family idiom, last-token readout):
  NS axis: north=+1 south=-1 (east/west=0, diagonals=±.71)
  EW axis: east=+1  west=-1  (north/south=0, diagonals=±.71)
Fit on English COMPASS words only; then test generalization the principled solver would
need:
  (1) recoverability  — held-out CV r per axis (is direction linearly decodable?)
  (2) orthogonality   — NS vs EW axis cos (are they independent geometric dims?)
  (3) synonym transfer — up/down/left/right (HELD OUT) land in the right quadrant?
  (4) cross-lingual    — es/fr/de direction words pole by sign?

If yes, the keyword direction dict (_ABS_DIR/_FACING in ir.py) has a principled,
synonym/language-robust replacement. Same measure-first step we ran for shapes.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/direction_frame.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np

from turnstyle.frame_library import FrameLibrary

_TMPL = "Move {w}."
S = 1 / math.sqrt(2)

# fit set: English compass (NS = y-component, EW = x-component)
NS_FIT = {"north": 1, "south": -1, "east": 0, "west": 0,
          "northeast": S, "northwest": S, "southeast": -S, "southwest": -S}
EW_FIT = {"east": 1, "west": -1, "north": 0, "south": 0,
          "northeast": S, "northwest": -S, "southeast": S, "southwest": -S}

# held-out tests: (word, expected_ns_sign, expected_ew_sign), 0 = ~neutral
SYNONYMS = [("up", +1, 0), ("down", -1, 0), ("left", 0, -1), ("right", 0, +1),
            ("upward", +1, 0), ("downward", -1, 0)]
CROSSLINGUAL = {
    "es": [("norte", +1, 0), ("sur", -1, 0), ("este", 0, +1), ("oeste", 0, -1)],
    "fr": [("nord", +1, 0), ("sud", -1, 0), ("est", 0, +1), ("ouest", 0, -1)],
    "de": [("norden", +1, 0), ("süden", -1, 0), ("osten", 0, +1), ("westen", 0, -1)],
}


def _sign(x, tol=0.25):
    return 0 if abs(x) < tol else (1 if x > 0 else -1)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()

    lib = FrameLibrary()
    ns = lib.fit_scalar("ns", mdl, tok, dev, NS_FIT, template=_TMPL)
    ew = lib.fit_scalar("ew", mdl, tok, dev, EW_FIT, template=_TMPL)
    print(f"(1) RECOVERABILITY  NS: cv_r={ns.cv_r:.2f} @L{ns.axis.layer} | "
          f"EW: cv_r={ew.cv_r:.2f} @L{ew.axis.layer}")

    names, M = lib.orthogonality(mdl, tok, dev, layer=11)
    i, j = names.index("ns"), names.index("ew")
    print(f"(2) ORTHOGONALITY   |cos(NS, EW)| = {M[i, j]:.3f} @L11  (≈0 ⇒ independent dims)")

    def quadrant_acc(items, label):
        ok = 0
        rows = []
        for w, ens, eew in items:
            c = lib.project_word(w, mdl, tok, dev)
            gs, ge = _sign(c["ns"]), _sign(c["ew"])
            hit = (gs == ens) and (ge == eew)
            ok += hit
            rows.append(f"    {w:10s} ns={c['ns']:+.2f}({gs:+d}/{ens:+d}) "
                        f"ew={c['ew']:+.2f}({ge:+d}/{eew:+d}) {'✓' if hit else '✗'}")
        print(f"{label}: {ok}/{len(items)} quadrant-correct")
        print("\n".join(rows))
        return ok, len(items)

    print("\n(3) SYNONYM TRANSFER (held out — up/down/left/right):")
    quadrant_acc(SYNONYMS, "  synonyms")

    print("\n(4) CROSS-LINGUAL:")
    tot = hit = 0
    for lang, items in CROSSLINGUAL.items():
        o, n = quadrant_acc(items, f"  {lang}")
        hit += o; tot += n
    print(f"\nCROSS-LINGUAL TOTAL: {hit}/{tot}")
    print("\nVerdict: high recoverability + low |cos| + synonym/cross-lingual quadrant "
          "hits ⇒ direction is a real frame; the _ABS_DIR keyword dict has a principled "
          "replacement.")


if __name__ == "__main__":
    main()
