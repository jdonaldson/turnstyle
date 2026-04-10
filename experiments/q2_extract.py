#!/usr/bin/env python3
"""Q2 Phase A — extract hidden states for a broader set of BBH tasks.

Extracts L8/L22/L23 hidden states for 10 additional BBH tasks beyond
the 5 already cached in capability_probe_data.npz. Together they span
8 turnstyle categories for the cross-task routing experiment.

Run with swollm venv:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/q2_extract.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# Import from sibling module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from capability_probe import (  # type: ignore
    CACHE_PATH as PHASE0_CACHE,
    MODEL_ID,
    detect_device,
    extract_last_token_hiddens,
    load_task_examples,
)


EXPERIMENT_DIR = Path(__file__).parent
Q2_CACHE_PATH = EXPERIMENT_DIR / "q2_cache.npz"

# New tasks (not in capability_probe_data.npz)
NEW_TASKS = [
    "reasoning_about_colored_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "boolean_expressions",
    "multistep_arithmetic_two",
    "date_understanding",
    "disambiguation_qa",
    "snarks",
    "causal_judgement",
    "movie_recommendation",
]

LAYERS = [8, 22, 23]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tasks", default=None, help="Comma-separated subset")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from swollm.bench.bbh import get_exemplars

    device = detect_device()
    print(f"Device: {device}", flush=True)
    print(f"Model:  {MODEL_ID}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).to(device)
    model.eval()
    n_layers = len(model.model.layers)
    print(f"  model has {n_layers} layers", flush=True)

    # Load or init cache
    if Q2_CACHE_PATH.exists() and not args.overwrite:
        cache = {k: v for k, v in np.load(Q2_CACHE_PATH, allow_pickle=True).items()}
        print(f"Loaded existing Q2 cache: {len(cache)} keys", flush=True)
    else:
        cache = {}
        print("Starting fresh Q2 cache", flush=True)

    task_filter = set(args.tasks.split(",")) if args.tasks else None

    grand_t0 = time.time()
    for task in NEW_TASKS:
        if task_filter and task not in task_filter:
            continue

        # Skip if all layers already cached
        already = all(f"{task}__hidden_l{L}" in cache for L in LAYERS)
        if already and not args.overwrite:
            print(f"\n=== {task} (already cached, skipping) ===", flush=True)
            continue

        print(f"\n=== {task} ===", flush=True)
        examples = load_task_examples(task)
        exemplar_indices = get_exemplars(examples)
        test_indices = [i for i in range(len(examples)) if i not in exemplar_indices]
        n = len(test_indices)
        print(f"  test examples: {n}", flush=True)

        hiddens = {L: np.zeros((n, model.config.hidden_size), dtype=np.float32)
                   for L in LAYERS}

        t0 = time.time()
        for i, idx in enumerate(test_indices):
            text = examples[idx]["input"]
            h = extract_last_token_hiddens(text, model, tokenizer, device, LAYERS)
            for L in LAYERS:
                hiddens[L][i] = h[L]

            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (n - i - 1) / max(rate, 1e-6)
                print(f"  {i+1}/{n}  {task[:20]}  ({elapsed:.0f}s, eta={eta:.0f}s)", flush=True)

        # Store target answers too
        targets = np.array([examples[idx]["target"] for idx in test_indices], dtype=object)
        cache[f"{task}__targets"] = targets
        cache[f"{task}__test_indices"] = np.array(test_indices)
        for L in LAYERS:
            cache[f"{task}__hidden_l{L}"] = hiddens[L]
        np.savez(Q2_CACHE_PATH, **cache)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.0f}s, cache saved ({len(cache)} keys)", flush=True)

    grand_elapsed = time.time() - grand_t0
    print(f"\nTotal: {grand_elapsed:.0f}s ({grand_elapsed/60:.1f} min)", flush=True)
    print(f"Cache: {Q2_CACHE_PATH}", flush=True)


if __name__ == "__main__":
    main()
