"""Measure the principled object_counting solver on SmolLM2 (limit-40).

Tests whether model yes/no category classification (no hardcoded category sets) is
accurate enough. Reports accuracy + per-example off-by errors + which item
classifications the model got wrong vs a small audit set, so we can see if the
LLM-extraction path is viable or if SmolLM2's world knowledge is too weak.
"""
from __future__ import annotations

import sys

from turnstyle.bbh import load_task
from turnstyle.object_counting import parse_item_list, solve_object_counting, _singular


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    from turnstyle.object_counting import _member_scorer
    scorer = _member_scorer(mdl, tok, dev)   # shared cache across all examples

    ex = load_task("object_counting")[:limit]
    ncorrect = 0
    errors = []
    for i, e in enumerate(ex):
        ans = solve_object_counting(e["input"], mdl, tok, dev, scorer=scorer)
        tgt = e["target"].strip()
        ok = ans == tgt
        ncorrect += ok
        if not ok:
            cat, items = parse_item_list(e["input"])
            cs = _singular(cat)
            members = [(_singular(n), q, scorer(_singular(n), cs)) for q, n in items]
            errors.append((i, ans, tgt, cs, members))
        if i < 5 or not ok:
            print(f"[{i:2d}] {'✓' if ok else '✗'} ans={str(ans):4s} tgt={tgt:4s}", flush=True)

    n = len(ex)
    print(f"\n--- object_counting (model-classified membership) ---")
    print(f"accuracy: {ncorrect}/{n} = {ncorrect/n*100:.1f}%")
    print("\nmisclassification audit (item: model_says_member):")
    for i, ans, tgt, cs, members in errors:
        print(f"[{i}] ans={ans} tgt={tgt} cat={cs}")
        for name, q, mem in members:
            print(f"      {name:18s} x{q} -> {mem}")


if __name__ == "__main__":
    main()
