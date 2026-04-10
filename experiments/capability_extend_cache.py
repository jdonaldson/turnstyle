#!/usr/bin/env python3
"""Extend the capability_probe cache with L23 hidden states.

SmolLM2 has 24 transformer layers (indices 0-23), so L23 is the deepest
available. The activation-routing doc's "L23-24" was 1-indexed; the
0-indexed equivalents are L22 (already cached) and L23 (added here).

Reuses the helpers from capability_probe.py — same model, same chat-template
prompt, same last-token pooling. Pure forward pass; no generation, no tier
evaluation. Each example is one forward pass with a hook on L23.

Existing cache keys are preserved; only `{task}__hidden_l23` is added
(or overwritten with --overwrite).

Run with the swollm venv:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/capability_extend_cache.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# Import from sibling module — this script lives in experiments/
import sys
sys.path.insert(0, str(Path(__file__).parent))
from capability_probe import (  # type: ignore
    CACHE_PATH,
    MODEL_ID,
    detect_device,
    extract_last_token_hiddens,
    load_task_examples,
)


# Tasks to process — same as TASK_TIERS in capability_probe.py
TASKS = [
    "penguins_in_a_table",
    "tracking_shuffled_objects_three_objects",
    "object_counting",
    "navigate",
    "web_of_lies",
]

NEW_LAYERS = [23]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing L23/L24 entries")
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated subset of tasks")
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

    # Verify L23 and L24 exist on this model
    n_layers = len(model.model.layers)
    print(f"  model has {n_layers} layers", flush=True)
    for L in NEW_LAYERS:
        if L >= n_layers:
            raise ValueError(f"layer {L} out of range (model has {n_layers})")

    # Load cache (must already exist from capability_probe.py phase 0)
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"cache not found: {CACHE_PATH}")
    cache = {k: v for k, v in np.load(CACHE_PATH, allow_pickle=True).items()}
    print(f"Loaded cache: {len(cache)} keys", flush=True)

    task_filter = set(args.tasks.split(",")) if args.tasks else None

    grand_t0 = time.time()
    for task in TASKS:
        if task_filter and task not in task_filter:
            continue

        # Skip if both L23 and L24 already cached and not overwriting
        already = all(f"{task}__hidden_l{L}" in cache for L in NEW_LAYERS)
        if already and not args.overwrite:
            print(f"\n=== {task} (L23/L24 already cached, skipping) ===", flush=True)
            continue

        print(f"\n=== {task} ===", flush=True)
        # Reconstruct test_indices the same way capability_probe.py does
        examples = load_task_examples(task)
        exemplar_indices = get_exemplars(examples)
        test_indices = [i for i in range(len(examples)) if i not in exemplar_indices]
        n = len(test_indices)
        print(f"  test examples: {n}", flush=True)

        hiddens = {L: np.zeros((n, model.config.hidden_size), dtype=np.float32)
                   for L in NEW_LAYERS}

        t0 = time.time()
        for i, idx in enumerate(test_indices):
            text = examples[idx]["input"]
            h = extract_last_token_hiddens(
                text, model, tokenizer, device, NEW_LAYERS)
            for L in NEW_LAYERS:
                hiddens[L][i] = h[L]

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (n - i - 1) / max(rate, 1e-6)
                print(f"  {i+1}/{n} ({elapsed:.0f}s, eta={eta:.0f}s)", flush=True)

        # Store and save
        for L in NEW_LAYERS:
            cache[f"{task}__hidden_l{L}"] = hiddens[L]
        np.savez(CACHE_PATH, **cache)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.0f}s, cache saved → {CACHE_PATH.name}", flush=True)

    grand_elapsed = time.time() - grand_t0
    print(f"\nTotal: {grand_elapsed:.0f}s ({grand_elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
