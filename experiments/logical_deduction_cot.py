#!/usr/bin/env python3
"""CoT intercept-and-correct for logical_deduction (all three variants).

Three tests per example from a single CoT generation pass:
  T1 — CoT answer accuracy:   does model's final written answer match target?
  T2 — Ordering extraction:   does parsing the CoT recover the correct ordering?
  T3 — Intercept+correct:     extracted ordering + Python position lookup → accuracy?

Comparison baseline: relational transcription pipeline at 93.2% (Qwen).
This experiment uses SmolLM2-1.7B-Instruct for consistency with tracking_cot.py.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/logical_deduction_cot.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, str(Path(__file__).parent))
from swollm.bench.bbh import load_task

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
N_EXAMPLES = 50


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


# ── problem parsing ────────────────────────────────────────────────────────

def extract_items(text: str) -> list[str]:
    """Extract the named items from the preamble ('there are N X: a, b, and c')."""
    # Matches: "there are/were three X:", "sells/had/includes three X:", etc.
    m = re.search(r"(?:three|four|five|six|seven)\s+\w+:\s*(.+?)\.", text, re.I)
    if not m:
        return []
    raw = m.group(1)
    # Strip articles and split on commas / "and"
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", raw)
    items = []
    for p in parts:
        p = re.sub(r"^(a|an|the)\s+", "", p.strip(), flags=re.I).strip().rstrip(".")
        if p:
            items.append(p)
    return items


def detect_ordering_dimension(text: str) -> tuple[str, str]:
    """Return (low_label, high_label) e.g. ('leftmost', 'rightmost')."""
    body = text.split("Options:")[0].lower()
    pairs = [
        (r"left|right",       "leftmost",    "rightmost"),
        (r"older|newer",      "oldest",      "newest"),
        (r"heavier|lighter",  "heaviest",    "lightest"),
        (r"larger|smaller",   "largest",     "smallest"),
        (r"taller|shorter",   "tallest",     "shortest"),
        (r"faster|slower",    "fastest",     "slowest"),
        (r"cheaper|expensive","cheapest",    "most expensive"),
        (r"earlier|later",    "earliest",    "latest"),
    ]
    for pat, low, high in pairs:
        if re.search(pat, body):
            return low, high
    return "first", "last"


# ── ground truth ordering (constraint solver) ──────────────────────────────

def gt_ordering(text: str, items: list[str]) -> list[str] | None:
    """Find the unique ordering consistent with all constraints (exhaustive search)."""
    body = text.split("Options:")[0]
    n = len(items)
    item_lo = [it.lower() for it in items]
    item_pat = "|".join(re.escape(it) for it in items)
    art = r"(?:(?:a|an|the)\s+)?"
    be  = r"(?:is|are)"

    preds: list[tuple] = []

    def find(name: str) -> int | None:
        lo = name.lower()
        return next((k for k, it in enumerate(item_lo) if it == lo), None)

    def add_before(a_str: str, b_str: str) -> None:
        a, b = find(a_str), find(b_str)
        if a is not None and b is not None and a != b:
            preds.append(('before', a, b))

    def add_at(a_str: str, pos: int) -> None:
        a = find(a_str)
        if a is not None and 1 <= pos <= n:
            preds.append(('at', a, pos))

    adj_h = (r"newer|larger|heavier|taller|faster|more expensive|later|better|higher"
             r"|pricier|more costly")
    adj_l = (r"older|smaller|lighter|shorter|slower|cheaper|earlier|worse|lower"
             r"|less expensive|less costly|less pricey")
    sup_h = (r"rightmost|newest|largest|heaviest|tallest|fastest|most expensive|latest"
             r"|best|highest|most costly|most pricey|most valuable")
    sup_l = (r"leftmost|oldest|smallest|lightest|shortest|slowest|cheapest|earliest"
             r"|worst|lowest|least expensive|least costly|least pricey|least valuable")
    sup_h_words = r"newest|most expensive|largest|heaviest|tallest|fastest|best|highest|most costly|most pricey|most valuable"
    sup_l_words = r"oldest|cheapest|smallest|lightest|shortest|slowest|worst|lowest|least expensive|least costly|least pricey"

    for m in re.finditer(rf"({item_pat})\s+{be}\s+to\s+the\s+right\s+of\s+{art}({item_pat})", body, re.I):
        add_before(m.group(2), m.group(1))
    for m in re.finditer(rf"({item_pat})\s+{be}\s+to\s+the\s+left\s+of\s+{art}({item_pat})", body, re.I):
        add_before(m.group(1), m.group(2))
    for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:{adj_h})\s+than\s+{art}({item_pat})", body, re.I):
        add_before(m.group(2), m.group(1))
    for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:{adj_l})\s+than\s+{art}({item_pat})", body, re.I):
        add_before(m.group(1), m.group(2))
    for m in re.finditer(rf"({item_pat})\s+finished\s+(?:above|ahead of)\s+{art}({item_pat})", body, re.I):
        add_before(m.group(2), m.group(1))
    for m in re.finditer(rf"({item_pat})\s+finished\s+(?:below|behind)\s+{art}({item_pat})", body, re.I):
        add_before(m.group(1), m.group(2))
    for m in re.finditer(rf"({item_pat})\s+{be}\s+the\s+(?:{sup_h})", body, re.I):
        add_at(m.group(1), n)
    for m in re.finditer(rf"({item_pat})\s+{be}\s+the\s+(?:{sup_l})", body, re.I):
        add_at(m.group(1), 1)
    for m in re.finditer(rf"({item_pat})\s+finished\s+(?:first|1st)\b", body, re.I):
        add_at(m.group(1), n)
    for m in re.finditer(rf"({item_pat})\s+finished\s+last\b", body, re.I):
        add_at(m.group(1), 1)

    ORDINALS = [("second", 2), ("third", 3), ("fourth", 4),
                ("fifth", 5), ("sixth", 6), ("seventh", 7)]
    for word, k in ORDINALS:
        for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:the\s+)?{word}\s+from\s+the\s+left", body, re.I):
            add_at(m.group(1), k)
        for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:the\s+)?{word}\s+from\s+the\s+right", body, re.I):
            add_at(m.group(1), n - k + 1)
        for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:the\s+)?{word}-(?:{sup_h_words})", body, re.I):
            add_at(m.group(1), n - k + 1)
        for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:the\s+)?{word}-(?:{sup_l_words})", body, re.I):
            add_at(m.group(1), k)
        for m in re.finditer(rf"({item_pat})\s+finished\s+{word}(?!-to)\b", body, re.I):
            add_at(m.group(1), n - k + 1)

    for word, k in [("second", 2), ("third", 3), ("fourth", 4), ("fifth", 5)]:
        for m in re.finditer(rf"({item_pat})\s+finished\s+{word}-to-last\b", body, re.I):
            add_at(m.group(1), k)

    if not preds:
        return None

    pos_arr = [0] * n

    def check(perm: tuple) -> bool:
        for i, it in enumerate(perm):
            k = find(it)
            if k is not None:
                pos_arr[k] = i
        for kind, a, b in preds:
            if kind == 'before':
                if pos_arr[a] >= pos_arr[b]:
                    return False
            else:
                if pos_arr[a] + 1 != b:
                    return False
        return True

    valid: list[list[str]] = []
    for perm in permutations(items):
        if check(perm):
            valid.append(list(perm))
            if len(valid) > 1:
                break
    return valid[0] if len(valid) == 1 else None


def parse_options(text: str) -> dict[str, str]:
    opts_section = text.split("Options:")[-1] if "Options:" in text else text
    return {
        letter: val.strip()
        for letter, val in re.findall(r"\(([A-Z])\)\s+(.+?)(?=\n\([A-Z]\)|\Z)", opts_section, re.S)
    }


def answer_from_ordering(
    ordering: list[str], text: str, options: dict[str, str]
) -> str | None:
    """Map the given ordering to the correct option letter."""
    n = len(ordering)
    ORDINALS = [("second", 2), ("third", 3), ("fourth", 4),
                ("fifth", 5), ("sixth", 6), ("seventh", 7)]
    sup_h = (r"newest|largest|heaviest|tallest|fastest|most expensive|rightmost|latest"
             r"|best|highest|most costly|most pricey|most valuable")
    sup_l = (r"oldest|smallest|lightest|shortest|slowest|cheapest|leftmost|earliest"
             r"|worst|lowest|least expensive|least costly|least pricey|least valuable")
    sup_h_words = r"newest|most expensive|largest|heaviest|tallest|fastest|best|highest|most costly|most pricey|most valuable"
    sup_l_words = r"oldest|cheapest|smallest|lightest|shortest|slowest|worst|lowest|least expensive|least costly|least pricey"

    for letter, opt in options.items():
        for i, item in enumerate(ordering):
            if item.lower() not in opt.lower():
                continue
            pos = i + 1

            # Extreme superlatives require "the [sup]" to avoid matching "second-newest" etc.
            if re.search(r'\bthe\s+(?:' + sup_h + r')\b', opt, re.I) and pos == n:
                return f"({letter})"
            if re.search(r'\bthe\s+(?:' + sup_l + r')\b', opt, re.I) and pos == 1:
                return f"({letter})"
            if re.search(r"middle|center", opt, re.I) and pos == (n + 1) // 2:
                return f"({letter})"
            if re.search(r"finished first|finished 1st", opt, re.I) and pos == n:
                return f"({letter})"
            if re.search(r"finished last\b", opt, re.I) and pos == 1:
                return f"({letter})"

            for word, k in ORDINALS:
                if re.search(rf"\b{word}\s+from\s+the\s+left", opt, re.I) and pos == k:
                    return f"({letter})"
                if re.search(rf"\b{word}\s+from\s+the\s+right", opt, re.I) and pos == n - k + 1:
                    return f"({letter})"
                if re.search(rf"\b{word}-(?:{sup_h_words})", opt, re.I) and pos == n - k + 1:
                    return f"({letter})"
                if re.search(rf"\b{word}-(?:{sup_l_words})", opt, re.I) and pos == k:
                    return f"({letter})"
                if re.search(rf"\bfinished\s+{word}\b(?!-to-last)", opt, re.I) and pos == n - k + 1:
                    return f"({letter})"

            for word, k in [("second", 2), ("third", 3), ("fourth", 4), ("fifth", 5)]:
                if re.search(rf"\bfinished\s+{word}-to-last\b", opt, re.I) and pos == k:
                    return f"({letter})"
    return None


# ── CoT generation ─────────────────────────────────────────────────────────

def generate_cot(example: dict, model, tokenizer, device) -> str:
    body = example["input"].split("Options:")[0].strip()
    low_label, high_label = detect_ordering_dimension(example["input"])
    items = extract_items(example["input"])
    items_str = ", ".join(items) if items else "the objects"
    prompt = (
        f"{body}\n\n"
        f"Sort {items_str} from {low_label} to {high_label}, "
        f"showing each comparison step:"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=300, do_sample=False, temperature=1.0)
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── CoT ordering extraction ────────────────────────────────────────────────

def _candidates_from_text(text: str, items: list[str]) -> list[str] | None:
    """Extract an ordered list of all items from a fragment of text."""
    item_lower = [it.lower() for it in items]
    # Split on comma, "and", ">", "→", "then"
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+|\s*[>→]\s*|\s+then\s+", text)
    matched = []
    for p in parts:
        p = re.sub(r"^(a|an|the)\s+", "", p.strip(), flags=re.I).strip().rstrip(".,;")
        for i, it in enumerate(item_lower):
            if it in p.lower() or p.lower() in it:
                if items[i] not in matched:
                    matched.append(items[i])
                break
    return matched if len(matched) == len(items) else None


def extract_cot_ordering(cot: str, items: list[str], problem_text: str = "") -> list[str] | None:
    """Parse the sorted ordering from CoT output and verify/fix direction."""
    item_lower = [it.lower() for it in items]

    result = None

    # 1. Scan every colon in CoT for a complete item list after it
    for m in re.finditer(r":\s*(.+?)(?:\.|$)", cot, re.I | re.M):
        cands = _candidates_from_text(m.group(1), items)
        if cands:
            result = cands
            break  # take first complete ordering found

    # 2. Arrow chain: A > B > C or A → B → C
    if result is None:
        for m in re.finditer(r"[\w][\w\s]+(?:\s*[>→]\s*[\w][\w\s]+)+", cot):
            cands = _candidates_from_text(m.group(0), items)
            if cands:
                result = cands
                break

    # 3. Last N lines — scan for line containing all items
    if result is None:
        lines = [l.strip() for l in cot.split("\n") if l.strip()]
        for line in reversed(lines[-10:]):
            positions = {i: line.lower().find(it) for i, it in enumerate(item_lower) if line.lower().find(it) >= 0}
            if len(positions) == len(items):
                ordered = sorted(positions, key=lambda k: positions[k])
                result = [items[i] for i in ordered]
                break

    if result is None:
        return None

    # Verify direction using first comparison constraint from problem
    # If ordering violates it, reverse
    if problem_text:
        item_pat2 = "|".join(re.escape(it) for it in items)
        art2 = r"(?:(?:a|an|the)\s+)?"
        # "A is to the right of [the] B" → A should come after B
        for cm in re.finditer(rf"({item_pat2})\s+is\s+to\s+the\s+right\s+of\s+{art2}({item_pat2})", problem_text, re.I):
            right_item, left_item = cm.group(1), cm.group(2)
            if right_item in result and left_item in result:
                if result.index(right_item) < result.index(left_item):
                    result = list(reversed(result))
                break
        # "A is newer/larger than [the] B" → A should come after B (low→high ordering)
        adj_higher2 = r"newer|larger|heavier|taller|faster|more expensive|later"
        for cm in re.finditer(rf"({item_pat2})\s+is\s+(?:{adj_higher2})\s+than\s+{art2}({item_pat2})", problem_text, re.I):
            higher, lower = cm.group(1), cm.group(2)
            if higher in result and lower in result:
                if result.index(higher) < result.index(lower):
                    result = list(reversed(result))
                break

    return result


def orderings_match(a: list[str] | None, b: list[str] | None) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    return a == b


# ── CoT answer extraction ──────────────────────────────────────────────────

def extract_cot_answer(cot: str, options: dict[str, str]) -> str | None:
    lines = [l.strip() for l in cot.strip().split("\n") if l.strip()]
    for line in reversed(lines[-6:]):
        for letter, val in options.items():
            if val.lower() in line.lower() or line.lower() in val.lower():
                return f"({letter})"
        m = re.search(r"\(([A-Z])\)", line)
        if m:
            letter = m.group(1)
            if letter in options:
                return f"({letter})"
    return None


# ── evaluation ─────────────────────────────────────────────────────────────

def evaluate(task: str, model, tokenizer, device) -> None:
    examples = load_task(task)[:N_EXAMPLES]

    t1_correct = t1_fail = 0
    t2_match = t2_mismatch = 0
    t3_correct = t3_wrong = t3_fail = 0
    order_len_errors: Counter = Counter()

    print(f"\n{'─'*70}")
    print(f"{task}  (n={N_EXAMPLES})")
    print(f"{'─'*70}")
    print(f"{'ex':>4}  {'T1':>6}  {'T2':>8}  {'T3':>6}  note", flush=True)

    for i, ex in enumerate(examples):
        target = ex["target"]
        items = extract_items(ex["input"])
        options = parse_options(ex["input"])

        if not items or len(items) < 2:
            t1_fail += 1; t2_mismatch += 1; t3_fail += 1
            print(f"{i:>4}  {'✗':>6}  {'✗':>8}  {'✗':>6}  no_items", flush=True)
            continue

        cot = generate_cot(ex, model, tokenizer, device)

        # Test 1 — CoT final answer
        t1_pred = extract_cot_answer(cot, options)
        t1_ok = (t1_pred == target)
        if t1_ok: t1_correct += 1
        else:     t1_fail    += 1

        # Test 2 — ordering extraction
        gt_ord  = gt_ordering(ex["input"], items)
        cot_ord = extract_cot_ordering(cot, items, ex["input"])
        t2_ok = orderings_match(gt_ord, cot_ord)
        if t2_ok: t2_match += 1
        else:
            t2_mismatch += 1
            gt_len  = len(gt_ord)  if gt_ord  else 0
            cot_len = len(cot_ord) if cot_ord else 0
            order_len_errors[f"gt={gt_len} cot={cot_len}"] += 1

        # Test 3 — intercept+correct
        if cot_ord is not None:
            t3_pred = answer_from_ordering(cot_ord, ex["input"], options)
        else:
            t3_pred = None
        if   t3_pred == target: t3_correct += 1
        elif t3_pred is None:   t3_fail    += 1
        else:                   t3_wrong   += 1

        note = "" if (t1_ok and t2_ok and t3_pred == target) else \
               f"T1={'ok' if t1_ok else t1_pred}  " \
               f"T2={'ok' if t2_ok else f'gt={len(gt_ord) if gt_ord else 0}/cot={len(cot_ord) if cot_ord else 0}'}  " \
               f"T3={'ok' if t3_pred==target else (t3_pred or 'fail')}"
        print(f"{i:>4}  {'ok' if t1_ok else '✗':>6}  {'ok' if t2_ok else '✗':>8}  {'ok' if t3_pred==target else '✗':>6}  {note}", flush=True)

    n = N_EXAMPLES
    print(f"\nTest 1 (CoT answer):        {t1_correct}/{n}  ({100*t1_correct/n:.0f}%)")
    print(f"Test 2 (order extraction):  {t2_match}/{n}  ({100*t2_match/n:.0f}%)")
    print(f"Test 3 (intercept+correct): {t3_correct}/{n}  ({100*t3_correct/n:.0f}%)")
    if order_len_errors:
        print(f"  order length mismatches: {dict(order_len_errors.most_common(5))}")


def main() -> None:
    model, tokenizer, device = load_model()
    for task in [
        "logical_deduction_three_objects",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
    ]:
        evaluate(task, model, tokenizer, device)


if __name__ == "__main__":
    main()
