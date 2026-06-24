"""Calibrate + persist per-task MC choice probes into the ModelProfile.

Ports the snarks pattern to the other flat-at-zero-shot MC tasks. fit_choice runs
autoprobe (layer x finder x mode, 5-fold CV, ship rule) and records the shipped
probe in the fingerprint-keyed profile; persist() writes it to the user cache so
DispatchTurnstyle.use_probe(task) activates it in later runs with no re-fitting.

NOTE: the probe trains on the full task data; the harness then evals on the same
examples → in-sample (same regime as swollm's reported in-sample MC numbers). The
autoprobe ship decision itself is 5-fold CV, so ship/no-ship is honest.

    python -m experiments.calibrate_mc_probes
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turnstyle.bbh import load_task
from turnstyle.dispatch_turnstyle import DispatchTurnstyle

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASKS = sys.argv[1:] or ["ruin_names", "disambiguation_qa", "temporal_sequences"]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
    dt = DispatchTurnstyle(mdl, tok, device)
    print(f"Device: {device}  existing probe_tasks: {dt.profile_tasks}", flush=True)

    for task in TASKS:
        examples = load_task(task)
        print(f"\n=== {task} (n={len(examples)}) — fitting ===", flush=True)
        result = dt.fit_choice(examples, task=task, verbose=True)
        chosen = result.chosen
        print(f"  ship={result.ship} "
              f"{'config=' + str(chosen[:3]) + f' acc={chosen[3]:.3f}' if chosen else ''}",
              flush=True)

    path = dt.persist()
    print(f"\nPersisted profile → {path}")
    print(f"probe_tasks now: {dt.profile_tasks}")


if __name__ == "__main__":
    main()
