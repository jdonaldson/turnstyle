"""Measure the pmi_floor confidence distribution on the sub-chance MC tasks, and what
abstaining (floor off -> generation) buys. Derive the abstention threshold from data.

For movie_recommendation + salient_translation (limit N):
  - pmi_floor pick + correctness + a CONFIDENCE = (top1-top2)/(top1-topN) margin ratio
  - generation baseline (floor OFF) = the abstain alternative
Print confidence split by correct/wrong so we can see whether a threshold separates the
(rare) right picks from the (systematic) wrong ones, and whether abstaining helps.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/pmi_floor_diag.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from turnstyle.bbh import load_task, answer_matches, _load_model
from turnstyle.dispatch_turnstyle import DispatchTurnstyle
from turnstyle import dispatch as D

TASKS = ["movie_recommendation", "salient_translation_error_detection"]
N = 40


def confidence(pmis: dict) -> float:
    """(top1 - top2) / (top1 - topN): how cleanly the winner separates from the pack."""
    vals = sorted(pmis.values(), reverse=True)
    if len(vals) < 2:
        return 1.0
    spread = vals[0] - vals[-1]
    return (vals[0] - vals[1]) / spread if spread > 1e-6 else 0.0


def main():
    mdl, tok, device = _load_model("HuggingFaceTB/SmolLM2-1.7B-Instruct", "auto")
    dt = DispatchTurnstyle(mdl, tok, device)
    ctx = dt.ctx

    for task in TASKS:
        exs = load_task(task)[:N]
        print(f"\n=== {task} (N={len(exs)}) ===", flush=True)
        floor_ok = gen_ok = 0
        conf_right, conf_wrong = [], []
        picks = {}
        for i, ex in enumerate(exs):
            tgt = ex["target"].strip()
            # pmi_floor pick
            res = D._score_options_pmi(ex["input"], ctx)
            if res is None:
                continue
            ans, pmis = res
            ok = answer_matches(ans, tgt)
            floor_ok += ok
            picks[ans] = picks.get(ans, 0) + 1
            conf = confidence(pmis)
            (conf_right if ok else conf_wrong).append(conf)
            # generation baseline (the abstain alternative)
            ctx.zeroshot_floor = False
            gen, _ = dt.generate(ex["input"], max_new_tokens=50)
            ctx.zeroshot_floor = True
            gok = answer_matches(gen, tgt)
            gen_ok += gok
            print(f"  [{i+1:2d}/{len(exs)}] floor={ans} {'ok' if ok else 'XX'} conf={conf:.2f} "
                  f"| gen={'ok' if gok else 'XX'} | tgt={tgt}", flush=True)
        n = len(exs)

        def stat(xs):
            return f"n={len(xs)} mean={sum(xs)/len(xs):.2f} max={max(xs):.2f}" if xs else "n=0"
        print(f"\n=== {task} (N={n}) ===")
        print(f"  pmi_floor:        {floor_ok}/{n} = {floor_ok/n*100:.1f}%   picks={picks}")
        print(f"  abstain->gen:     {gen_ok}/{n} = {gen_ok/n*100:.1f}%   (the floor's alternative)")
        print(f"  confidence RIGHT: {stat(conf_right)}")
        print(f"  confidence WRONG: {stat(conf_wrong)}")


if __name__ == "__main__":
    main()
