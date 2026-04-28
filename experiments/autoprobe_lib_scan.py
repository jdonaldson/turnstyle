#!/usr/bin/env python3
"""Run the *library* autoprobe across the 4 BBH probe tasks and report which
finder wins. Validates that adding the probe-based per-option finder doesn't
regress any task and ideally finds it preferred.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/autoprobe_lib_scan.py
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

from turnstyle.autoprobe import autoprobe, load_option_boundary_probe, DEFAULT_FINDERS

TASKS = ["snarks", "ruin_names", "disambiguation_qa", "temporal_sequences",
         "salient_translation_error_detection"]


def main():
    artifact = load_option_boundary_probe()
    print(f"Option-boundary probe artifact loaded: {artifact is not None}")
    if artifact is not None:
        print(f"  layer={artifact['layer']}, model_id={artifact['model_id']}")
        print(f"  trained on formats: {artifact['train_formats']}")
    print(f"Default finders available: {list(DEFAULT_FINDERS.keys())}\n")

    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    mdl = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype=torch.float16
    ).to(device).eval()

    rows = []
    for task in TASKS:
        print(f"\n{'='*70}\n{task}\n{'='*70}", flush=True)
        result = autoprobe(
            examples=load_task(task),
            target_fn=lambda ex: ex["target"].strip(),
            model=mdl, tokenizer=tok, device=device,
            verbose=True,
        )
        chosen = result.chosen
        if chosen:
            finder, mode, layer, cv = chosen
            rows.append((task, finder, mode, layer, cv, result.ship))
        print(f"\nTop 5 sweep entries (any finder):")
        for finder, mode, layer, cv in result.sweep[:5]:
            print(f"  {finder:32s} {mode:10s} L{layer:>2}  {cv:>6.1%}")

    print("\n\n" + "=" * 78)
    print(f"{'task':40s}  {'finder':32s}  {'mode':10s}  {'L':>3}  {'CV':>5}  ship")
    print("=" * 78)
    for task, finder, mode, layer, cv, ship in rows:
        print(f"  {task:38s}  {finder:32s}  {mode:10s}  {layer:>3}  {cv:>5.1%}  "
              f"{'SHIP' if ship else 'no-ship'}")


if __name__ == "__main__":
    main()
