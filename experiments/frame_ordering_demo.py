"""Demo: the frame-as-column solve path answers superlatives over IMPLICIT attributes
(no numeric column) that the SQL path cannot. End-to-end through DispatchTurnstyle
(frames auto-load), so it exercises the real dispatch wiring.
"""
from __future__ import annotations

CASES = [
    ("Which is the biggest?\nOptions:\n(A) ant\n(B) whale\n(C) mouse", "(B)"),
    ("Which is the smallest?\nOptions:\n(A) elephant\n(B) mouse\n(C) horse", "(B)"),
    ("Which is the oldest?\nOptions:\n(A) newborn\n(B) teenager\n(C) grandparent", "(C)"),
    ("Which is the youngest?\nOptions:\n(A) elderly man\n(B) baby\n(C) adult", "(B)"),
    ("Which object is the heaviest?\nOptions:\n(A) feather\n(B) boulder\n(C) leaf", "(B)"),
    # abstain cases (should NOT commit via frame_ordering):
    ("Which is the funniest?\nOptions:\n(A) cat\n(B) dog\n(C) fish", None),
    ("Is the sky blue?\nOptions:\n(A) Yes\n(B) No", None),
]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.dispatch_turnstyle import DispatchTurnstyle
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    dt = DispatchTurnstyle(mdl, tok, dev)
    print(f"frames: {dt.frame_names}\n")

    n_ok = 0
    for prompt, expect in CASES:
        ans = dt.parse(prompt)          # Answer or None
        src = getattr(ans, "source", None)
        committed = ans.text if ans is not None else None
        if expect is None:
            ok = src != "frame_ordering"           # must NOT commit via this path
            verdict = "abstained-correctly" if ok else f"WRONGLY committed {committed}"
        else:
            ok = src == "frame_ordering" and committed == expect
            verdict = f"{committed} via {src}" + ("" if ok else f" (want {expect})")
        n_ok += ok
        q = prompt.split(chr(10))[0]
        print(f"  [{'OK ' if ok else 'XX '}] {q:35s} -> {verdict}")
    print(f"\n{n_ok}/{len(CASES)} correct")


if __name__ == "__main__":
    main()
