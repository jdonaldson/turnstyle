"""
negated_ordering_demo.py — end-to-end: hedge-detected negation + symbolic
complement over the frame-ordering path.

The consumer loop the operators framework buys:
  1. route the superlative to its frame (FrameLibrary.route, membership-first)
  2. DETECT negation with the hedge probe at the attribute token (no
     keywords — validated on paraphrastic forms the detector never saw)
  3. rank options on the synthesized frame column
  4. apply negation SYMBOLICALLY: complement of the extreme — unique with
     two options, ABSTAIN when ambiguous (commitment = coverage)

Question form is itself a transfer test: the detector was fit on
predicative statements ("It is not warm."), scored here inside
"Which is not the biggest? ..." — different construction entirely.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/negated_ordering_demo.py
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from turnstyle.operators import fit_negation_detector

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# (question, attribute word, options, expected)  expected None = abstain
ITEMS = [
    ("Which is the biggest?", "biggest", ["whale", "ant"], "whale"),
    ("Which is not the biggest?", "biggest", ["whale", "ant"], "ant"),
    ("Which is the oldest?", "oldest", ["pyramid", "smartphone"], "pyramid"),
    ("Which is not the oldest?", "oldest", ["pyramid", "smartphone"],
     "smartphone"),
    ("Which is far from the biggest?", "biggest", ["whale", "ant"], "ant"),
    ("Which is hardly the oldest?", "oldest", ["pyramid", "smartphone"],
     "smartphone"),
    # ambiguous complement: three options negated -> abstain
    ("Which is not the biggest?", "biggest", ["whale", "ant", "mouse"], None),
    # plain three-option control: still answerable
    ("Which is the biggest?", "biggest", ["whale", "ant", "mouse"], "whale"),
]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.frame_library import load_library

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev)
    lib = load_library(mdl)

    print("calibrating negation detector (~380 forwards, mixed constructions)...", flush=True)
    det = fit_negation_detector(mdl, tok, dev, lib)
    print(f"  layer L{det.layer}, {det.meta['n_pairs']} pairs, "
          f"s0 {det.s0:.1f}\n")

    n_ok = 0
    for question, attr, options, expected in ITEMS:
        routed = lib.route(attr)
        if routed is None:
            print(f"  ?? no frame route for '{attr}'")
            continue
        frame, sign = routed
        z = det.score_word(mdl, tok, dev, question, attr)
        negated = det.detect_word(mdl, tok, dev, question, attr)
        ranked = lib.rank(options, frame, mdl, tok, dev,
                          descending=(sign > 0))
        extreme = ranked[0][0]
        if negated is True:
            rest = [o for o in options if o != extreme]
            answer = rest[0] if len(rest) == 1 else None   # unique or abstain
        elif negated is False:
            answer = extreme
        else:
            answer = None                                   # detector abstains
        ok = answer == expected
        n_ok += ok
        print(f"  {'OK ' if ok else 'X  '} z={z:+5.1f} neg={negated!s:5s} "
              f"{frame}:{'+' if sign > 0 else '-'} -> "
              f"{str(answer):11s} (want {expected})   \"{question}\" "
              f"{options}")
    print(f"\n{n_ok}/{len(ITEMS)}")


if __name__ == "__main__":
    main()
