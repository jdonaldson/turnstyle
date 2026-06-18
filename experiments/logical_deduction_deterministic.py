#!/usr/bin/env python3
"""Deterministic solver for logical_deduction (all three BBH variants).

Parses ordering constraints from the problem text, then finds the unique
permutation of items consistent with all constraints.

No model needed.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/logical_deduction_deterministic.py
"""

from __future__ import annotations

import re
import sys
from itertools import permutations

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task


# ── item extraction ─────────────────────────────────────────────────────────

def extract_items(text: str) -> list[str]:
    """Extract named items from the preamble ('there are N X: a, b, and c')."""
    # Multi-word category name: "three colored books: ..." or "three golfers: ..."
    m = re.search(r"(?:three|four|five|six|seven)\s+[\w ]+?:\s*(.+?)\.", text, re.I)
    if not m:
        return []
    raw = m.group(1)
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", raw)
    items = []
    for p in parts:
        p = re.sub(r"^(a|an|the)\s+", "", p.strip(), flags=re.I).strip().rstrip(".")
        if p:
            items.append(p)
    return items


# ── constraint solver (exhaustive permutation check) ───────────────────────

def gt_ordering(text: str, items: list[str]) -> list[str] | None:
    """Find the unique ordering consistent with all constraints.

    Uses exhaustive permutation check so positional constraints
    ('second-newest', 'finished last', etc.) are handled uniformly.

    Ordering convention: position 1 = lowest/worst/leftmost,
                         position n = highest/best/rightmost.
    """
    body = text.split("Options:")[0]
    n = len(items)
    item_lo = [it.lower() for it in items]
    item_pat = "|".join(re.escape(it) for it in items)
    art = r"(?:(?:a|an|the)\s+)?"   # optional article
    be  = r"(?:is|are)"             # singular or plural copula

    # Each pred is ('before', a_idx, b_idx) or ('at', a_idx, 1-based-pos)
    preds: list[tuple] = []

    def find(name: str) -> int | None:
        lo = name.lower()
        for k, it in enumerate(item_lo):
            if it == lo:
                return k
        return None

    def add_before(a_str: str, b_str: str) -> None:
        a, b = find(a_str), find(b_str)
        if a is not None and b is not None and a != b:
            preds.append(('before', a, b))

    def add_at(a_str: str, pos: int) -> None:
        a = find(a_str)
        if a is not None and 1 <= pos <= n:
            preds.append(('at', a, pos))

    # ── spatial ──
    for m in re.finditer(rf"({item_pat})\s+{be}\s+to\s+the\s+right\s+of\s+{art}({item_pat})", body, re.I):
        add_before(m.group(2), m.group(1))   # left < right

    for m in re.finditer(rf"({item_pat})\s+{be}\s+to\s+the\s+left\s+of\s+{art}({item_pat})", body, re.I):
        add_before(m.group(1), m.group(2))   # left < right

    # ── comparative adjectives ──
    adj_h = (r"newer|larger|heavier|taller|faster|more expensive|later|better|higher"
             r"|pricier|more costly")
    adj_l = (r"older|smaller|lighter|shorter|slower|cheaper|earlier|worse|lower"
             r"|less expensive|less costly|less pricey")

    for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:{adj_h})\s+than\s+{art}({item_pat})", body, re.I):
        add_before(m.group(2), m.group(1))   # lower before higher

    for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:{adj_l})\s+than\s+{art}({item_pat})", body, re.I):
        add_before(m.group(1), m.group(2))   # lower before higher

    # ── tournament comparatives ──
    for m in re.finditer(rf"({item_pat})\s+finished\s+(?:above|ahead of)\s+{art}({item_pat})", body, re.I):
        add_before(m.group(2), m.group(1))   # B lower, A higher

    for m in re.finditer(rf"({item_pat})\s+finished\s+(?:below|behind)\s+{art}({item_pat})", body, re.I):
        add_before(m.group(1), m.group(2))   # A lower, B higher

    # ── superlatives (extreme positions) ──
    sup_last = (r"rightmost|newest|largest|heaviest|tallest|fastest|most expensive|latest"
                r"|best|highest|most costly|most pricey|most valuable")
    sup_first = (r"leftmost|oldest|smallest|lightest|shortest|slowest|cheapest|earliest"
                 r"|worst|lowest|least expensive|least costly|least pricey|least valuable")

    for m in re.finditer(rf"({item_pat})\s+{be}\s+the\s+(?:{sup_last})", body, re.I):
        add_at(m.group(1), n)

    for m in re.finditer(rf"({item_pat})\s+{be}\s+the\s+(?:{sup_first})", body, re.I):
        add_at(m.group(1), 1)

    # ── tournament extremes ──
    for m in re.finditer(rf"({item_pat})\s+finished\s+(?:first|1st)\b", body, re.I):
        add_at(m.group(1), n)    # first place = best = highest pos

    for m in re.finditer(rf"({item_pat})\s+finished\s+(?:last)\b", body, re.I):
        add_at(m.group(1), 1)    # last place = worst = lowest pos

    # ── ordinal positions (generalized, 2nd through 7th) ──
    ORDINALS = [("second", 2), ("third", 3), ("fourth", 4),
                ("fifth", 5), ("sixth", 6), ("seventh", 7)]
    sup_h_words = (r"newest|most expensive|largest|heaviest|tallest|fastest"
                   r"|best|highest|most costly|most pricey|most valuable")
    sup_l_words = (r"oldest|cheapest|smallest|lightest|shortest|slowest"
                   r"|worst|lowest|least expensive|least costly|least pricey")

    for word, k in ORDINALS:
        # "X from the left" → absolute position k
        for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:the\s+)?{word}\s+from\s+the\s+left", body, re.I):
            add_at(m.group(1), k)
        # "X from the right" → position n-k+1
        for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:the\s+)?{word}\s+from\s+the\s+right", body, re.I):
            add_at(m.group(1), n - k + 1)
        # "X-newest / X-most expensive / etc." → position n-k+1 (Kth from top)
        for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:the\s+)?{word}-(?:{sup_h_words})", body, re.I):
            add_at(m.group(1), n - k + 1)
        # "X-oldest / X-cheapest / etc." → position k (Kth from bottom)
        for m in re.finditer(rf"({item_pat})\s+{be}\s+(?:the\s+)?{word}-(?:{sup_l_words})", body, re.I):
            add_at(m.group(1), k)

    # ── tournament ordinals (finished second/third = 2nd/3rd best) ──
    # Use (?!-to) to avoid matching "finished second-to-last" as "finished second"
    for word, k in ORDINALS:
        for m in re.finditer(rf"({item_pat})\s+finished\s+{word}(?!-to)\b", body, re.I):
            add_at(m.group(1), n - k + 1)   # kth place = kth best = pos n-k+1

    # ── "X-to-last" (second-to-last = pos 2, third-to-last = pos 3, ...) ──
    to_last_map = [("second", 2), ("third", 3), ("fourth", 4), ("fifth", 5)]
    for word, k in to_last_map:
        for m in re.finditer(rf"({item_pat})\s+finished\s+{word}-to-last\b", body, re.I):
            add_at(m.group(1), k)   # kth from last = position k

    if not preds:
        return None

    # ── exhaustive search ──
    pos_arr = [0] * n   # reuse array for each perm

    def check(perm: tuple) -> bool:
        for i, it in enumerate(perm):
            pos_arr[find(it)] = i   # 0-indexed position
        for kind, a, b in preds:
            if kind == 'before':
                if pos_arr[a] >= pos_arr[b]:
                    return False
            else:   # 'at': b is 1-based target position
                if pos_arr[a] + 1 != b:
                    return False
        return True

    valid: list[list[str]] = []
    for perm in permutations(items):
        if check(perm):
            valid.append(list(perm))
            if len(valid) > 1:
                break   # ambiguous — shouldn't happen in BBH

    return valid[0] if len(valid) == 1 else None


# ── option parsing + answer lookup ──────────────────────────────────────────

def parse_options(text: str) -> dict[str, str]:
    opts = text.split("Options:")[-1] if "Options:" in text else text
    return {
        letter: val.strip()
        for letter, val in re.findall(
            r"\(([A-Z])\)\s+(.+?)(?=\n\([A-Z]\)|\Z)", opts, re.S
        )
    }


def answer_from_ordering(ordering: list[str], options: dict[str, str]) -> str | None:
    """Map ordering to option letter via positional description in option text."""
    n = len(ordering)
    ORDINALS = [("second", 2), ("third", 3), ("fourth", 4),
                ("fifth", 5), ("sixth", 6), ("seventh", 7)]
    sup_h = (r"newest|largest|heaviest|tallest|fastest|most expensive|rightmost|latest"
             r"|best|highest|most costly|most pricey|most valuable")
    sup_l = (r"oldest|smallest|lightest|shortest|slowest|cheapest|leftmost|earliest"
             r"|worst|lowest|least expensive|least costly|least pricey|least valuable")
    sup_h_words = (r"newest|most expensive|largest|heaviest|tallest|fastest"
                   r"|best|highest|most costly|most pricey|most valuable")
    sup_l_words = (r"oldest|cheapest|smallest|lightest|shortest|slowest"
                   r"|worst|lowest|least expensive|least costly|least pricey")

    for letter, opt in options.items():
        for i, item in enumerate(ordering):
            if item.lower() not in opt.lower():
                continue
            pos = i + 1   # 1-indexed

            # ── extreme superlatives: require "the [sup]" to avoid matching
            #    inside "second-newest", "third-most expensive", etc. ──
            if re.search(r'\bthe\s+(?:' + sup_h + r')\b', opt, re.I) and pos == n:
                return f"({letter})"
            if re.search(r'\bthe\s+(?:' + sup_l + r')\b', opt, re.I) and pos == 1:
                return f"({letter})"

            # ── middle ──
            if re.search(r"middle|center", opt, re.I) and pos == (n + 1) // 2:
                return f"({letter})"

            # ── tournament extremes ──
            if re.search(r"finished first|finished 1st", opt, re.I) and pos == n:
                return f"({letter})"
            if re.search(r"finished last\b", opt, re.I) and pos == 1:
                return f"({letter})"

            # ── ordinal positions (generalized) ──
            for word, k in ORDINALS:
                if re.search(rf"\b{word}\s+from\s+the\s+left", opt, re.I) and pos == k:
                    return f"({letter})"
                if re.search(rf"\b{word}\s+from\s+the\s+right", opt, re.I) and pos == n - k + 1:
                    return f"({letter})"
                # ordinal-superlative: "second-newest" etc.
                if re.search(rf"\b{word}-(?:{sup_h_words})", opt, re.I) and pos == n - k + 1:
                    return f"({letter})"
                if re.search(rf"\b{word}-(?:{sup_l_words})", opt, re.I) and pos == k:
                    return f"({letter})"
                # tournament ordinals: "finished second" = second best = pos n-k+1
                if re.search(rf"\bfinished\s+{word}\b(?!-to-last)", opt, re.I) and pos == n - k + 1:
                    return f"({letter})"

            # ── X-to-last: "finished second-to-last" = pos 2, etc. ──
            for word, k in [("second", 2), ("third", 3), ("fourth", 4), ("fifth", 5)]:
                if re.search(rf"\bfinished\s+{word}-to-last\b", opt, re.I) and pos == k:
                    return f"({letter})"

    return None


# ── solver ───────────────────────────────────────────────────────────────────

def solve(example: dict) -> str | None:
    text = example["input"]
    items = extract_items(text)
    if not items:
        return None
    ordering = gt_ordering(text, items)
    if ordering is None:
        return None
    options = parse_options(text)
    return answer_from_ordering(ordering, options)


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate(task: str, n: int = 250) -> None:
    examples = load_task(task)[:n]
    correct = wrong = parse_fail = 0

    for ex in examples:
        pred = solve(ex)
        target = ex["target"].strip()
        if pred is None:
            parse_fail += 1
        elif pred == target:
            correct += 1
        else:
            wrong += 1

    total = correct + wrong + parse_fail
    print(
        f"{task:<40}  {correct}/{total}  "
        f"({100*correct/total:.1f}%)  "
        f"wrong={wrong}  parse_fail={parse_fail}"
    )


def main() -> None:
    for task in [
        "logical_deduction_three_objects",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
    ]:
        evaluate(task)


if __name__ == "__main__":
    main()
