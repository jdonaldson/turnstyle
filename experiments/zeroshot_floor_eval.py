"""Does the zero-shot content-PMI floor lift the abstaining MC tasks?

Compares floor OFF (abstain -> free generation, ~chance on MC) vs floor ON
(domain-conditional PMI over option content) on tasks that have no calibrated probe.
Same model instance; reports accuracy + how many commit via pmi_floor.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/zeroshot_floor_eval.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from turnstyle.bbh import load_task, answer_matches

TASKS = ["geometric_shapes", "movie_recommendation", "causal_judgement",
         "sports_understanding"]
LIMIT = 20


def run(dt, name, floor: bool):
    dt.ctx.choice_artifact = None
    dt.ctx.zeroshot_floor = floor
    ex = load_task(name)[:LIMIT]
    correct = committed = 0
    for e in ex:
        parsed = dt.parse(e["input"])
        gen, _ = dt.generate(e["input"], max_new_tokens=30)
        ok = answer_matches(gen, e["target"].strip())
        correct += ok
        if parsed is not None and getattr(parsed, "source", "") == "pmi_floor":
            committed += 1
    return correct / len(ex), committed, len(ex)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.dispatch_turnstyle import DispatchTurnstyle
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
    dt = DispatchTurnstyle(mdl, tok, dev)

    print(f"{'task':40s} {'off':>7s} {'on':>7s} {'Δpp':>7s}  pmi_commit")
    for name in TASKS:
        off, _, n = run(dt, name, floor=False)
        on, commit, _ = run(dt, name, floor=True)
        print(f"{name:40s} {off*100:6.1f}% {on*100:6.1f}% {(on-off)*100:+6.1f}  "
              f"{commit}/{n}", flush=True)


if __name__ == "__main__":
    main()
