#!/usr/bin/env python3
"""Three-way test of SmolLM2 chain-of-thought on tracking_shuffled_objects.

Test 1 — CoT accuracy:     does the model's final written answer match the target?
Test 2 — Swap extraction:  does parsing the CoT recover the correct swap sequence?
Test 3 — Intercept+correct: swap sequence from CoT + Python simulator → accuracy?

All three run from a single CoT generation pass per example.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/tracking_cot.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, str(Path(__file__).parent))

from swollm.bench.bbh import load_task
from tracking_deterministic import (
    ALL_ACTORS, ACTION_RE,
    detect_actors, parse_init, parse_action, parse_query, parse_options,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
N_EXAMPLES = 50   # per task variant


# ── model ──────────────────────────────────────────────────────────────────

def load_model():
    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Loading {MODEL_ID} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to(device).eval()
    return mdl, tok, device


# ── CoT generation ─────────────────────────────────────────────────────────

def generate_cot(example: dict, model, tokenizer, device) -> str:
    body = example["input"].split("\nOptions:")[0].strip()
    prompt = body + "\n\nSolve step by step, tracking who holds what after each swap:"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=350, do_sample=False, temperature=1.0)
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── Test 1: CoT final-answer extraction ────────────────────────────────────

def extract_cot_answer(cot: str, query_actor: str, options: dict[str, str]) -> str | None:
    """Find what the model claims the queried actor has at the end."""
    # Search last 6 lines for "[actor] has/is dancing with/is playing X"
    lines = [l.strip() for l in cot.strip().split("\n") if l.strip()]
    for line in reversed(lines[-6:]):
        # "Bob has Frankenstein" / "Alice is dancing with Helga" / "Eve is playing goalkeeper"
        pat = rf"\b{re.escape(query_actor)}\b.{{0,20}}?(?:has|is dancing with|is playing|is holding)\s+(.+?)[\.,]?\s*$"
        m = re.search(pat, line, re.I)
        if m:
            val = m.group(1).strip().rstrip(".,")
            for letter, opt in options.items():
                if val.lower() == opt.lower():
                    return f"({letter})"
                if val.lower() in opt.lower() or opt.lower() in val.lower():
                    return f"({letter})"
    return None


# ── Test 2: swap sequence extraction from CoT ──────────────────────────────

def extract_cot_swaps(cot: str) -> list[tuple[str, str]]:
    """Parse all (actor1, actor2) swap pairs mentioned in the CoT, in order."""
    swaps = []
    for line in cot.split("\n"):
        m = ACTION_RE.search(line)
        if m:
            swaps.append((m.group(1), m.group(2)))
    return swaps


def swaps_match(a: list[tuple[str, str]], b: list[tuple[str, str]]) -> bool:
    """Two swap sequences are equal if same length and same unordered pairs in order."""
    if len(a) != len(b):
        return False
    return all(
        frozenset(p1) == frozenset(p2)
        for p1, p2 in zip(a, b)
    )


# ── Test 3: intercept-and-correct ─────────────────────────────────────────

def simulate(initial: dict[str, str], swaps: list[tuple[str, str]]) -> dict[str, str]:
    state = dict(initial)
    for a1, a2 in swaps:
        if a1 in state and a2 in state:
            state[a1], state[a2] = state[a2], state[a1]
    return state


def intercept_and_correct(
    example: dict, cot: str
) -> tuple[str | None, str | None]:
    """
    Parse swap sequence from CoT, run Python simulator against known initial state.
    Returns (answer_letter, failure_reason).
    """
    text = example["input"]
    lines = [l.strip() for l in text.split(".") if l.strip()]

    actors  = detect_actors(lines[0] if lines else text)
    init_sent = next((l for l in lines if re.search(r"At the start", l, re.I)), None)
    if not actors or not init_sent:
        return None, "extract_fail"

    initial = parse_init(init_sent, actors)
    if len(initial) < len(actors):
        return None, "init_parse_fail"

    cot_swaps = extract_cot_swaps(cot)
    if not cot_swaps:
        return None, "no_swaps_in_cot"

    final  = simulate(initial, cot_swaps)
    queried = parse_query(text)
    if not queried or queried not in final:
        return None, "query_fail"

    answer_val = final[queried]
    opts = parse_options(text)
    for letter, val in opts.items():
        if val.strip().lower() == answer_val.lower():
            return f"({letter})", None
        if val.lower() in answer_val.lower() or answer_val.lower() in val.lower():
            return f"({letter})", None
    return None, f"no_option_match({answer_val!r})"


# ── ground-truth swap extraction ──────────────────────────────────────────

def gt_swaps(example: dict) -> list[tuple[str, str]]:
    text  = example["input"]
    lines = [l.strip() for l in text.split(".") if l.strip()]
    return [pair for line in lines for pair in ([parse_action(line)] if parse_action(line) else [])]


# ── evaluation loop ────────────────────────────────────────────────────────

def evaluate(task: str, model, tokenizer, device) -> None:
    examples = load_task(task)[:N_EXAMPLES]

    t1_correct = t1_fail = 0          # CoT answer accuracy
    t2_match   = t2_mismatch = 0      # swap sequence accuracy
    t3_correct = t3_wrong = t3_fail = 0  # intercept-and-correct

    swap_len_errors: Counter = Counter()  # wrong # of swaps in CoT

    print(f"\n{'─'*70}")
    print(f"{task}  (n={N_EXAMPLES})")
    print(f"{'─'*70}")
    print(f"{'ex':>4}  {'T1':>6}  {'T2':>8}  {'T3':>6}  note", flush=True)

    for i, ex in enumerate(examples):
        target = ex["target"]
        cot    = generate_cot(ex, model, tokenizer, device)

        # Test 1
        query_actor = parse_query(ex["input"])
        opts        = parse_options(ex["input"])
        t1_pred     = extract_cot_answer(cot, query_actor or "", opts)
        t1_ok       = t1_pred == target
        if t1_ok: t1_correct += 1
        else:     t1_fail    += 1

        # Test 2
        gt   = gt_swaps(ex)
        cot_s = extract_cot_swaps(cot)
        t2_ok = swaps_match(gt, cot_s)
        if t2_ok: t2_match    += 1
        else:
            t2_mismatch += 1
            swap_len_errors[f"gt={len(gt)} cot={len(cot_s)}"] += 1

        # Test 3
        t3_pred, t3_why = intercept_and_correct(ex, cot)
        if   t3_pred == target: t3_correct += 1
        elif t3_pred is None:   t3_fail    += 1
        else:                   t3_wrong   += 1

        note = "" if (t1_ok and t2_ok and t3_pred == target) else \
               f"T1={'ok' if t1_ok else t1_pred}  T2={'ok' if t2_ok else f'gt={len(gt)}/cot={len(cot_s)}'}  T3={'ok' if t3_pred==target else (t3_why or t3_pred)}"
        print(f"{i:>4}  {'ok' if t1_ok else '✗':>6}  {'ok' if t2_ok else '✗':>8}  {'ok' if t3_pred==target else '✗':>6}  {note}", flush=True)

    n = N_EXAMPLES
    print(f"\nTest 1 (CoT answer):        {t1_correct}/{n}  ({100*t1_correct/n:.0f}%)")
    print(f"Test 2 (swap extraction):   {t2_match}/{n}  ({100*t2_match/n:.0f}%)")
    print(f"Test 3 (intercept+correct): {t3_correct}/{n}  ({100*t3_correct/n:.0f}%)")
    if swap_len_errors:
        print(f"  swap length mismatches: {dict(swap_len_errors.most_common(5))}")


def main() -> None:
    model, tokenizer, device = load_model()
    for task in [
        "tracking_shuffled_objects_three_objects",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
    ]:
        evaluate(task, model, tokenizer, device)


if __name__ == "__main__":
    main()
