"""Why does snarks drop -62.5pp under option reorder (perturbation harness finding)?

The ChoiceProbe scores each option's LAST-TOKEN hidden state at L13 from a single
forward pass over the FULL prompt — so the score is contextualized (attends to the
other option). This probes whether the failure is (a) a pure POSITION prior (probe
favors slot A or B regardless of content) or (b) CONTEXTUAL contamination (content
score shifts when the neighbour changes), by scoring each example in original vs
swapped option order and comparing per-content scores.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/snarks_order_probe.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import re

from turnstyle.bbh import load_task
from turnstyle.dispatch import normalize_option_markers

_OPT = re.compile(r"^\(([A-Z])\)\s*(.*)$", re.MULTILINE)


def swap_two(prompt: str) -> str:
    """Swap the contents of a 2-option block, keep letters (A)(B)."""
    opts = _OPT.findall(prompt)
    if len(opts) != 2:
        return prompt
    head = prompt[: prompt.index("(A)")]
    (_, a), (_, b) = opts
    return f"{head}(A) {b}\n(B) {a}"


def score(probe_cls, artifact, model, tok, dev, prompt):
    from turnstyle.blackboard import Blackboard
    bb = Blackboard(prompt=prompt, context={
        "model": model, "tokenizer": tok, "device": dev})
    probe_cls(artifact).fire(bb)
    ans = bb.terminal_answer()
    if ans is None:
        return None, None
    return ans.payload["answer"], ans.payload["scores"]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.dispatch_turnstyle import DispatchTurnstyle
    from turnstyle.primitives.choice_probe import ChoiceProbe

    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
    dt = DispatchTurnstyle(mdl, tok, dev)
    dt.use_probe("snarks")
    art = dt.ctx.choice_artifact
    print(f"snarks probe: layer={art.layer} finder={getattr(art.finder,'__name__',art.finder)}\n")

    ex = load_task("snarks")[:40]
    # tallies
    pos_pick = {"A": 0, "B": 0}          # which slot wins, original order
    n = orig_ok = swap_ok = both_ok = consistent = marg_ok = 0
    sA = sB = 0.0                          # mean P(sarc) by slot, original
    for e in ex:
        p0 = normalize_option_markers(e["input"])
        p1 = swap_two(p0)
        tgt = e["target"].strip()
        a0, s0 = score(ChoiceProbe, art, mdl, tok, dev, p0)
        a1, s1 = score(ChoiceProbe, art, mdl, tok, dev, p1)
        if s0 is None or s1 is None:
            continue
        n += 1
        pos_pick[a0.strip("()")] += 1
        sA += s0["A"]; sB += s0["B"]
        # content-correctness: in swapped order the target letter flips
        tgt_swapped = "(A)" if tgt == "(B)" else "(B)"
        o_ok = (a0 == tgt); s_ok = (a1 == tgt_swapped)
        orig_ok += o_ok; swap_ok += s_ok; both_ok += (o_ok and s_ok)
        consistent += (a0 == a1)
        # POSITION-MARGINALIZED: each content scored over BOTH slots it occupied.
        # content1 = original A (slot A in p0, slot B in p1); content2 = original B.
        c1 = (s0["A"] + s1["B"]) / 2      # the option originally labelled A
        c2 = (s0["B"] + s1["A"]) / 2      # the option originally labelled B
        marg = "(A)" if c1 >= c2 else "(B)"   # answer in ORIGINAL letter space
        marg_ok += (marg == tgt)
        print(f"tgt={tgt} orig(A={s0['A']:.2f},B={s0['B']:.2f})->{a0} {'✓' if o_ok else '✗'}"
              f"  | swap(A={s1['A']:.2f},B={s1['B']:.2f})->{a1} {'✓' if s_ok else '✗'}"
              f"  | marg(c1={c1:.2f},c2={c2:.2f})->{marg} {'✓' if marg==tgt else '✗'}")

    print(f"\nn={n}")
    print(f"orig acc={orig_ok/n*100:.1f}%  swap acc={swap_ok/n*100:.1f}%  both={both_ok/n*100:.1f}%")
    print(f"MARGINALIZED acc={marg_ok/n*100:.1f}%  (order-invariant by construction)")
    print(f"slot wins (orig): A={pos_pick['A']} B={pos_pick['B']}  "
          f"(pure position prior ⇒ one slot dominates)")
    print(f"mean P(sarc): slotA={sA/n:.3f} slotB={sB/n:.3f}  "
          f"(slot gap ⇒ positional bias in the score itself)")
    print(f"winning slot UNCHANGED after content swap: {consistent}/{n} "
          f"({consistent/n*100:.0f}%)  (high ⇒ probe tracks POSITION not content)")


if __name__ == "__main__":
    main()
