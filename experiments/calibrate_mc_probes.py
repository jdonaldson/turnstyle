"""Calibrate + persist per-task MC choice probes into the ModelProfile.

Ports the snarks pattern to the other flat-at-zero-shot MC tasks. fit_choice runs
autoprobe (layer x finder x mode, 5-fold CV, ship rule) and records the shipped
probe in the fingerprint-keyed profile; persist() writes it to the user cache so
DispatchTurnstyle.use_probe(task) activates it in later runs with no re-fitting.

NOTE: the probe trains on the full task data; the harness then evals on the same
examples → in-sample (same regime as swollm's reported in-sample MC numbers). The
autoprobe ship decision itself is 5-fold CV, so ship/no-ship is honest.

    python -m experiments.calibrate_mc_probes
    # extract latent recognition where generation is flat, dropping the trust bar:
    python -m experiments.calibrate_mc_probes movie_recommendation \
        salient_translation_error_detection --ship-abs 0.0

The ship rule has two guards: --ship-lift (validity — the probe must beat the best
cheap baseline by this margin; keep it, else noise ships) and --ship-abs (trust —
minimum absolute CV for standalone deployment). For maximizing aggregate BBH score,
set --ship-abs 0.0: a recognition probe that beats the cheap baseline but lands <60%
still beats the generation fallback (movie 50.6% vs gen 22.3%, salient 42.4% vs 13.8%).
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turnstyle.bbh import load_task
from turnstyle.dispatch_turnstyle import DispatchTurnstyle

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DEFAULT_TASKS = ["ruin_names", "disambiguation_qa", "temporal_sequences"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--ship-abs", type=float, default=0.60,
                    help="trust guard: min absolute CV accuracy (0.0 disables it)")
    ap.add_argument("--ship-lift", type=float, default=0.10,
                    help="validity guard: min lift over best cheap baseline")
    args = ap.parse_args()
    tasks = args.tasks or DEFAULT_TASKS

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
    dt = DispatchTurnstyle(mdl, tok, device)
    print(f"Device: {device}  ship_abs={args.ship_abs} ship_lift={args.ship_lift}  "
          f"existing probe_tasks: {dt.profile_tasks}", flush=True)

    for task in tasks:
        examples = load_task(task)
        print(f"\n=== {task} (n={len(examples)}) — fitting ===", flush=True)
        result = dt.fit_choice(examples, task=task, verbose=True,
                               ship_threshold_abs=args.ship_abs,
                               ship_threshold_lift=args.ship_lift)
        chosen = result.chosen
        print(f"  ship={result.ship} "
              f"{'config=' + str(chosen[:3]) + f' acc={chosen[3]:.3f}' if chosen else ''}",
              flush=True)

    path = dt.persist()
    print(f"\nPersisted profile → {path}")
    print(f"probe_tasks now: {dt.profile_tasks}")


if __name__ == "__main__":
    main()
