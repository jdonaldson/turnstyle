#!/usr/bin/env python3
"""Diagnose logical_deduction failures.

Imports ld_solve from qwen_two_stage_experiment, runs it on 250 examples,
categorizes failures, prints examples per category.

Failure categories:
- segment_failed: segmenter returned nothing
- no_objects_or_constraints: extraction produced nothing usable
- no_valid_perm: constraints extracted but CSP has no solution
- no_match: valid perms but option matching failed
- wrong_answer: produced an answer but it's wrong
"""
import json
import os
import sys
from collections import defaultdict

# Import the existing experiment module
sys.path.insert(0, "/tmp")
import qwen_two_stage_experiment as q2  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"
N_EVAL = 250
N_SHOW_PER_CAT = 5


def load_task(name):
    with open(os.path.join(BBH_CACHE, f"{name}.json")) as f:
        return json.load(f)


def main():
    print(f"Device: {DEVICE}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
    model.eval()

    examples = load_task("logical_deduction_three_objects")[:N_EVAL]

    correct = 0
    total = 0
    counters = defaultdict(int)
    failures = defaultdict(list)

    for idx, ex in enumerate(examples):
        text = ex["input"]
        gt = ex["target"].strip()

        answer, diag = q2.ld_solve(model, tokenizer, DEVICE, text)

        total += 1
        if answer == gt:
            correct += 1
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{N_EVAL}] correct={correct}/{total} "
                      f"({100*correct/total:.1f}%)", flush=True)
            continue

        # Categorize
        if answer is None:
            cat = diag.get("error", "unknown") if isinstance(diag, dict) else "unknown"
        else:
            cat = "wrong_answer"

        counters[cat] += 1
        if len(failures[cat]) < N_SHOW_PER_CAT:
            failures[cat].append({
                "idx": idx,
                "text": text,
                "gt": gt,
                "answer": answer,
                "diag": diag,
            })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{N_EVAL}] correct={correct}/{total} "
                  f"({100*correct/total:.1f}%)", flush=True)

    # Summary
    print(f"\n{'='*70}")
    print(f"FINAL: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*70}")
    print("\nFailure categories:")
    for cat, n in sorted(counters.items(), key=lambda x: -x[1]):
        print(f"  {cat:35s} {n:3d}")

    print(f"\n{'='*70}")
    print("FAILURE EXAMPLES")
    print(f"{'='*70}")
    for cat, exs in failures.items():
        print(f"\n--- {cat} ({counters[cat]} total) ---")
        for ex in exs:
            print(f"\n  ex {ex['idx']}: gt={ex['gt']}, answer={ex['answer']}")
            # Split body and options for readability
            body_lines = ex['text'].split('\n')
            body = body_lines[0]
            # Trim the preamble
            body = body.replace(
                "The following paragraphs each describe a set of three objects "
                "arranged in a fixed order. The statements are logically "
                "consistent within each paragraph. ", "")
            print(f"    body: {body[:180]}{'...' if len(body) > 180 else ''}")
            # Options
            opts = [l for l in body_lines if l.startswith('(')]
            for opt in opts:
                print(f"    {opt}")
            d = ex['diag'] if isinstance(ex['diag'], dict) else {}
            if 'objects' in d:
                print(f"    objects: {d['objects']}")
            if 'constraints' in d:
                print(f"    constraints: {d['constraints']}")
            if 'valid' in d:
                print(f"    valid_perms: {d['valid']}")
            if 'extractions' in d:
                print(f"    extractions:")
                for e in d['extractions'][:6]:
                    print(f"      seg='{e['seg']}' -> {e['parsed']}")


if __name__ == "__main__":
    main()
