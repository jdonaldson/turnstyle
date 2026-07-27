"""Does the certainty FRAME predict the model's own correctness? (measurement -> control)

The frame family has been validated as a MEASUREMENT surface across 14 frames and 6
backbones. It has almost never changed a decision — `dispatch_turnstyle.py` says so
outright, "a measurement surface — not part of" the solve path. This is the first
test of the project's actual thesis: can a verified frame improve a downstream
choice? The choice is ABSTENTION; `dispatch.py` already has a typed
`Abstain(reason=...)` and only lacks a confidence signal to trigger it.

WHY THIS IS SHARP, not hopeful. `MetacognitiveProbe` — trained DIRECTLY on the
model's own success — scored **AUC ~= 0.50 at all 25 layers**. The prior is failure.
The certainty frame differs in one way that might matter: it was never fit on
success, only on external semantic certainty words ("impossible" -3 .. "certain" +3,
L16, cv_r .951, supported 6/6 backbones). Does a direction meaning *certainty in the
world* transfer to *this model is unsure right now*?

DESIGN — three columns, because "beats random" is not the bar that matters:

  base      accuracy with no abstention
  margin    top1-minus-top2 option logprob. THE CHEAP BASELINE: a free byproduct of
            scoring the options we already score. A new signal has to beat THIS, not
            beat random, or it is not worth shipping.
  certainty the frame coordinate at the final prompt token

Correctness comes from per-option scoring (argmax over option logprobs), which is
how the project actually solves MC and is deterministic — generation-with-grading
measured the grader, not the model.

    uv run python experiments/certainty_abstain.py --tasks snarks,ruin_names
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turnstyle.bbh import load_task  # noqa: E402
from turnstyle.frame_library import FrameLibrary  # noqa: E402
from turnstyle.ir import parse_scene  # noqa: E402

RESULTS = Path(__file__).parent / "results"
BBH = Path.home() / "Projects" / "swollm" / "data" / "bbh_cache"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DEFAULT_TASKS = ("snarks,ruin_names,disambiguation_qa,movie_recommendation,"
                 "date_understanding,hyperbaton")


def auc(scores, labels):
    s, l = np.asarray(scores, float), np.asarray(labels, int)
    if l.sum() == 0 or l.sum() == len(l):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = l.sum(), len(l) - l.sum()
    return float((ranks[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def score_options(model, tok, dev, prompt, options, layer):
    """Per-option mean logprob, plus the certainty coordinate at TWO positions.

    Two readouts because the transfer is not obvious. The frame was fit on a
    certainty ADJECTIVE inside "It is {w} that it happened." Reading it on a BBH
    prompt is already a stretch, and the two natural positions ask different
    questions:
      pre   final prompt token — "is the model unsure before committing?"
      post  final token after the chosen option is appended — "is it unsure about
            THIS answer?", which is closer to what abstention needs.
    """
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(dev)
    out = model(**enc, output_hidden_states=True)
    h_pre = out.hidden_states[layer][0, -1].float().cpu().numpy()
    n_prompt = enc["input_ids"].shape[1]

    scores, h_post = {}, {}
    for k, text in options.items():
        full = tok(prompt + " " + text, return_tensors="pt",
                   truncation=True, max_length=1024).to(dev)
        if full["input_ids"].shape[1] <= n_prompt:
            scores[k] = -1e9
            h_post[k] = h_pre
            continue
        o = model(**full, output_hidden_states=True)
        h_post[k] = o.hidden_states[layer][0, -1].float().cpu().numpy()
        logits = o.logits[0, :-1]
        tgt = full["input_ids"][0, 1:]
        lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        scores[k] = float(lp[n_prompt - 1:].mean())
    return scores, h_pre, h_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()

    lib = FrameLibrary.load_npz(glob.glob("src/turnstyle/data/frames/*.npz")[0])
    cert = lib.frames["certainty"]
    L = cert.layer
    print(f"{MODEL}\ncertainty axis L{L} ({cert.axis.low_label}..{cert.axis.high_label}), "
          f"cv_r {cert.cv_r:.3f}\n")

    rows = []
    for task in args.tasks.split(","):
        task = task.strip()
        try:
            exs = load_task(task, cache_dir=BBH)[: args.limit]
        except Exception as e:
            print(f"  {task:26s} SKIP ({type(e).__name__})")
            continue
        n_ok, n_used = 0, 0
        for ex in exs:
            sc = parse_scene(ex["input"])
            if len(sc.options) < 2:
                continue
            scores, h_pre, h_post = score_options(model, tok, dev, ex["input"],
                                                  sc.options, L)
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            pick, top = ranked[0]
            margin = top - ranked[1][1]
            tgt = str(ex["target"]).strip().strip("()").upper()
            correct = int(pick.upper() == tgt)
            n_ok += correct
            n_used += 1
            rows.append({"task": task, "correct": correct, "margin": float(margin),
                         "cert_pre": float(cert.project_norm(h_pre)),
                         "cert_post": float(cert.project_norm(h_post[pick]))})
        if n_used:
            print(f"  {task:26s} n={n_used:3d}  per-option acc {n_ok / n_used:.3f}", flush=True)

    # SCREEN: a task at 0% or 100% has no correctness variance, so it contributes
    # noise and no signal to an AUC. ruin_names is the known case — per-option
    # logprob is documented-broken there (min-perplexity 31%, which is why it needed
    # a probe), and it scored 0/20 in a first pass. Drop such tasks and SAY SO.
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["correct"])
    keep = {t for t, v in by_task.items() if 0.10 <= np.mean(v) <= 0.90}
    dropped = {t: round(float(np.mean(v)), 3) for t, v in by_task.items() if t not in keep}
    if dropped:
        print(f"\n  DROPPED (no correctness variance to predict): {dropped}")
    rows = [r for r in rows if r["task"] in keep]

    if not rows:
        raise SystemExit("no scorable examples")
    ok = np.array([r["correct"] for r in rows])
    mg = np.array([r["margin"] for r in rows])
    cp = np.array([r["cert_pre"] for r in rows])
    ct = np.array([r["cert_post"] for r in rows])
    rng = np.random.default_rng(0)

    print(f"\n{'=' * 78}\nDOES EITHER SIGNAL PREDICT CORRECTNESS?\n{'=' * 78}")
    print(f"  n={len(rows)}  base accuracy {ok.mean():.3f}")
    res = {}
    for nm, sig in (("margin (cheap baseline)", mg), ("certainty pre-answer", cp),
                    ("certainty post-answer", ct)):
        a = auc(sig, ok)
        nulls = np.array([auc(sig, rng.permutation(ok)) for _ in range(400)])
        p = float(np.mean(np.abs(nulls - 0.5) >= abs(a - 0.5)))
        res[nm] = (a, p)
        print(f"  {nm:26s} AUC {a:.3f}   permutation p={p:.3f}")
    print(f"  (MetacognitiveProbe, trained directly on success: AUC ~0.50)")

    print(f"\n{'=' * 78}\nACCURACY vs COVERAGE — certainty must beat MARGIN, not random"
          f"\n{'=' * 78}")
    print(f"  {'coverage':>9s} {'random':>8s} {'margin':>8s} {'certainty':>10s} "
          f"{'cert-margin':>12s}")
    curve = []
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(len(rows) * cov))
        a_r = np.mean([ok[rng.choice(len(ok), k, replace=False)].mean() for _ in range(200)])
        a_m = ok[np.argsort(-mg)[:k]].mean()
        a_c = ok[np.argsort(-(ct if abs(res["certainty post-answer"][0]-0.5) >= abs(res["certainty pre-answer"][0]-0.5) else cp))[:k]].mean()
        curve.append({"coverage": cov, "random": float(a_r), "margin": float(a_m),
                      "certainty": float(a_c)})
        print(f"  {cov:>9.0%} {a_r:>8.3f} {a_m:>8.3f} {a_c:>10.3f} {a_c - a_m:>+12.3f}")

    # judge on whichever readout position transferred better; report both above
    a_c, p_c = max((res["certainty pre-answer"], res["certainty post-answer"]),
                   key=lambda t: abs(t[0] - 0.5))
    best = max(curve, key=lambda c: c["certainty"] - c["margin"])
    lift = best["certainty"] - best["margin"]
    print(f"\n  best certainty-over-margin lift {lift:+.3f} at {best['coverage']:.0%}")
    if p_c >= 0.05:
        print("  => NO TRANSFER. The certainty axis says nothing about this model's own")
        print("     correctness — consistent with the MetacognitiveProbe negative. The")
        print("     frame is a measurement surface and does not become a control surface.")
    elif lift <= 0.0:
        print("  => TRANSFERS BUT REDUNDANT. Real signal, but the free logit margin is at")
        print("     least as good — nothing to ship.")
    else:
        print("  => SHIPPABLE: beats the cheap baseline. Wire into Abstain(reason=...).")

    out = RESULTS / "certainty_abstain.json"
    out.write_text(json.dumps({"model": MODEL, "layer": L, "n": len(rows),
                               "auc": {k: v[0] for k, v in res.items()},
                               "p": {k: v[1] for k, v in res.items()},
                               "curve": curve, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
