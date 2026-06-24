"""Judge-free causal test of self-introspection.

For each prompt, hold the injected introspection-reading template fixed and vary
only the region word: REAL (the activation-routed label) vs SHUFFLED (wrong label).
Measure the model's refusal disposition = logP(refusal opener) - logP(compliance
opener) right after the reading. Paired per prompt -> sign test + bootstrap CI.

No generation, no refusal judge -> removes the two confounds (weak judge, mid-stream
meta-commentary) that wrecked introspect_causal.py.

Predictions if the model genuinely conditions on the reading's content:
  HARMFUL: real label = 'refusal', shuf = other  -> disposition(real) > disposition(shuf), delta>0
  BENIGN:  real label = 'factual'/'arith', shuf rotates to 'refusal' -> delta<0 (false refusal raises it)

  .venv/bin/python experiments/introspect_disposition.py --model microsoft/Phi-4-mini-instruct
"""
from __future__ import annotations

import argparse

import numpy as np

from introspect_loop import IntrospectiveModel, _wrong_label
from introspect_causal import HARMFUL, BENIGN


def sign_test(deltas) -> dict:
    from math import comb
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    k = min(pos, neg)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)) if n else 1.0
    return {"pos": pos, "neg": neg, "p": p}


def boot_ci(deltas, iters=10000, seed=0):
    a = np.asarray(deltas)
    n = len(a)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, n, size=(iters, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(a.mean()), float(lo), float(hi)


def run(model_id: str, mode: str = "request"):
    m = IntrospectiveModel(model_id)
    m.build_reference(mode=mode)
    print(f"=== {model_id} mode={mode} (judge-free disposition) ===")

    def condition(prompts, tag, expect_sign):
        deltas, real_labs = [], []
        for p in prompts:
            h = m._ref_state(p)
            res = m.tool.route(h)
            real_lab = res.label
            shuf_lab = _wrong_label(real_lab)
            d_real = m.refusal_disposition(p, real_lab, res.ood_score)
            d_shuf = m.refusal_disposition(p, shuf_lab, res.ood_score)
            deltas.append(d_real - d_shuf)
            real_labs.append(real_lab)
        mean, lo, hi = boot_ci(deltas)
        st = sign_test(deltas)
        frac_ref = sum(1 for x in real_labs if x == "refusal") / len(real_labs)
        print(f"\n[{tag}] n={len(deltas)}  routed_refusal={frac_ref:.0%}  (expect delta {expect_sign}0)")
        print(f"  mean delta(real-shuf) = {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"  sign test: pos={st['pos']} neg={st['neg']}  p={st['p']:.4f}")
        return deltas

    # harmful: real='refusal' raises refusal disposition vs shuffled -> delta>0
    condition(HARMFUL, "HARMFUL", ">")
    # benign: shuffled rotates to 'refusal' -> shuffled higher -> delta<0
    condition(BENIGN, "BENIGN", "<")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--mode", choices=["task", "request"], default="request")
    args = ap.parse_args()
    run(args.model, args.mode)
