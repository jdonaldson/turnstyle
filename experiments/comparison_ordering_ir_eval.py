#!/usr/bin/env python3
"""Evaluate the SentenceIRSpec path for comparison_ordering on all 3 logical_deduction tasks.

Forces sentence_ir_solve() with COMPARISON_ORDERING_SPEC — bypasses the regex fast path entirely.
Reports per-task accuracy and failure categorization with per-example diagnostics.

Usage:
    python experiments/comparison_ordering_ir_eval.py [--n-eval N] [--show-failures N]
"""
import argparse
import json
import os
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turnstyle.comparison_ordering import COMPARISON_ORDERING_SPEC
from turnstyle.ir import sentence_ir_solve

BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DEVICE = "mps"

TASKS = [
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
]


def load_task(name: str) -> list[dict]:
    with open(os.path.join(BBH_CACHE, f"{name}.json")) as f:
        return json.load(f)


def categorize_failure(diag: dict) -> str:
    """Categorize a failure from the diagnostic dict."""
    if "error" in diag:
        return diag["error"]
    n_parsed = diag.get("n_parsed", 0)
    n_segments = diag.get("n_segments", 0)
    if n_segments == 0:
        return "no_segments"
    if n_parsed == 0:
        return "no_extractions"

    # Check sentence-level extractions
    extractions = diag.get("sentence_extractions", [])
    failed_parses = sum(1 for e in extractions if not e.get("parsed", False))
    if failed_parses > 0:
        return f"partial_extraction({n_parsed}/{n_segments})"

    if diag.get("answer") is None:
        return "aggregation_failed"

    return "unknown"


def run_task(model, tokenizer, task_name: str, n_eval: int, n_show: int):
    examples = load_task(task_name)[:n_eval]
    correct = 0
    total = 0
    counters = defaultdict(int)
    failures = defaultdict(list)

    for idx, ex in enumerate(examples):
        text = ex["input"]
        gt = ex["target"].strip()
        diag = {}

        answer = sentence_ir_solve(
            model, tokenizer, DEVICE, text,
            COMPARISON_ORDERING_SPEC, diag=diag,
        )

        total += 1
        if answer == gt:
            correct += 1
        else:
            cat = "wrong_answer" if answer is not None else categorize_failure(diag)
            counters[cat] += 1
            if len(failures[cat]) < n_show:
                failures[cat].append({
                    "idx": idx,
                    "text_prefix": text[:200],
                    "gt": gt,
                    "answer": answer,
                    "diag": diag,
                })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(examples)}] {task_name}: "
                  f"correct={correct}/{total} ({100*correct/total:.1f}%)",
                  flush=True)

    pct = 100 * correct / total if total else 0
    print(f"\n  {task_name}: {correct}/{total} ({pct:.1f}%)")

    if counters:
        print("  Failure categories:")
        for cat, n in sorted(counters.items(), key=lambda x: -x[1]):
            print(f"    {cat:40s} {n:3d}")

    if failures:
        print(f"\n  Failure examples:")
        for cat, exs in failures.items():
            print(f"\n  --- {cat} ({counters[cat]} total) ---")
            for ex in exs:
                print(f"\n    ex {ex['idx']}: gt={ex['gt']}, answer={ex['answer']}")
                print(f"    text: {ex['text_prefix']}...")
                d = ex["diag"]

                # Show segments
                segs = d.get("segments", [])
                if segs:
                    print(f"    segments ({len(segs)}):")
                    for s_text, s_type in segs:
                        print(f"      [{s_type}] {s_text[:100]}")

                # Show entity list
                entities = d.get("entities", [])
                if entities:
                    print(f"    entities: {entities}")

                # Show per-sentence extractions
                sentence_exts = d.get("sentence_extractions", [])
                for se in sentence_exts:
                    parsed = "OK" if se.get("parsed") else "FAIL"
                    print(f"      [{se['type']}|{parsed}] {se['sentence'][:80]}")
                    print(f"        response: {se['response'][:120]}")

    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=250,
                        help="Examples per task (default: 250)")
    parser.add_argument("--show-failures", type=int, default=3,
                        help="Number of failure examples to show per category")
    args = parser.parse_args()

    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Examples per task: {args.n_eval}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
    model.eval()

    total_correct = 0
    total_count = 0

    for task_name in TASKS:
        print(f"\n{'='*70}")
        print(f"Task: {task_name}")
        print(f"{'='*70}")
        c, t = run_task(model, tokenizer, task_name, args.n_eval, args.show_failures)
        total_correct += c
        total_count += t

    print(f"\n{'='*70}")
    print(f"OVERALL: {total_correct}/{total_count} "
          f"({100*total_correct/total_count:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
