#!/usr/bin/env python3
"""Vocabulary routing generalization test.

For each BBH task in the local cache, applies keyword-based routing rules
and reports: correct route, wrong route, or unrouted (falls through).

No model, no probes — pure text statistics.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/vocab_routing.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, str(Path(__file__).parent))
from swollm.bench.bbh import load_task
from segment_homogeneity import SEGMENTERS, sentence_segments

# ── ground-truth task → expected route ────────────────────────────────

TASK_EXPECTED: dict[str, str] = {
    # tasks we've explicitly built routes for
    "navigate":                                "SPATIAL_NAVIGATION",
    "tracking_shuffled_objects_three_objects": "OBJECT_TRACKING",
    "tracking_shuffled_objects_five_objects":  "OBJECT_TRACKING",
    "tracking_shuffled_objects_seven_objects": "OBJECT_TRACKING",
    "logical_deduction_three_objects":         "COMPARISON_ORDERING",
    "logical_deduction_five_objects":          "COMPARISON_ORDERING",
    "logical_deduction_seven_objects":         "COMPARISON_ORDERING",
    "web_of_lies":                             "TRUTH_CHAIN",
    "penguins_in_a_table":                     "TABULAR",
    "object_counting":                         "TABULAR",
    # tasks with clear vocabulary signatures we haven't targeted yet
    "boolean_expressions":                     "BOOLEAN",
    "multistep_arithmetic_two":                "ARITHMETIC",
    "dyck_languages":                          "DYCK",
    "word_sorting":                            "WORD_SORT",
    "geometric_shapes":                        "GEOMETRIC",
    "reasoning_about_colored_objects":         "TABULAR",
    "date_understanding":                      "DATE",
    "formal_fallacies":                        "FORMAL_LOGIC",
}

# ── routing rules ──────────────────────────────────────────────────────
# Each rule: (route_name, test_fn)
# test_fn receives a list of segments (strings) and returns True/False.

def _tok(text: str) -> set[str]:
    """Lowercase token set from text."""
    return set(re.findall(r"[a-zA-Z']+", text.lower()))

def _frac(segs: list[str], words: set[str]) -> float:
    """Fraction of segments containing at least one word from 'words'."""
    if not segs:
        return 0.0
    return sum(1 for s in segs if _tok(s) & words) / len(segs)

def _has(segs: list[str], words: set[str], threshold: float = 0.05) -> bool:
    return _frac(segs, words) >= threshold

def _density(text: str, words: set[str]) -> float:
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in words) / len(toks)

_COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "brown", "grey", "gray", "mauve", "burgundy",
    "teal", "cyan", "fuchsia", "magenta", "beige", "silver",
    "turquoise", "gold",
}

def _color_noun_pairs(segs: list[str]) -> int:
    """Count distinct <color> <noun> bigrams across all segments.

    Detects scenes where multiple objects carry color attributes —
    the actual semantic structure COLORED_OBJECTS reasoning requires.
    Threshold-free: just count distinct pairs.
    """
    pairs: set[tuple[str, str]] = set()
    for s in segs:
        toks = re.findall(r"[a-z]+", s.lower())
        for color, noun in zip(toks, toks[1:]):
            if color in _COLOR_WORDS:
                pairs.add((color, noun))
    return len(pairs)


ROUTING_RULES: list[tuple[str, object]] = [

    # ── Tier 1 structural (no segments needed) ─────────────────────────

    # Dyck: bracket sequences. Exclude "(unit)" patterns like "(cm)","(kg)".
    ("DYCK", lambda segs: any(
        ("[" in s or "]" in s)
        or (s.count("(") + s.count(")") > 2
            and not re.search(r"\d\s*[\+\-\*]\s*\d", s)
            and not re.search(r"\(\w{1,4}\)", s))  # unit labels: (cm), (kg)
        for s in segs
    )),

    ("ARITHMETIC", lambda segs: any(
        re.search(r"\d+\s*[\+\-\*×÷]\s*\d+", s) for s in segs
    )),

    ("WORD_SORT", lambda segs: _has(segs, {"alphabetical", "alphabetically",
                                           "sort", "sorted", "lexicographic"})),

    # ── Structural tabular (CV-based, approximated here by table marker) ─
    # Covers: markdown tables, CSV-ish rows, object_counting lists,
    # and colored-object scenes (enumerated <color> <noun> pairs).
    # All route to the same SQL solver.

    ("TABULAR", lambda segs: (
        any("|" in s for s in segs)               # markdown table
        or any(re.search(r"\w+,\s*\d+,", s) for s in segs)  # CSV-ish rows
        or any(re.search(r"^I have\b", s) for s in segs)    # object_counting
        or _color_noun_pairs(segs) >= 3            # colored object scene
    )),

    # ── Vocabulary routes ──────────────────────────────────────────────

    ("SPATIAL_NAVIGATION", lambda segs: _has(
        segs, {"north", "south", "east", "west",
               "forward", "backward"}, 0.05        # navigate uses relative dirs
    )),

    ("OBJECT_TRACKING", lambda segs: (
        _has(segs, {"gives", "give", "swaps", "swap",
                    "trades", "trade", "gets", "get",
                    "passes", "pass"}, 0.05)
        and _has(segs, {               # proper names present
            "alice", "bob", "carol", "dave", "erin",
            "frank", "grace", "heidi", "ivan", "judy",
            "lila", "mike", "nina", "omar", "pete",
        }, 0.05)
    )),

    ("TRUTH_CHAIN", lambda segs: (
        _has(segs, {"says", "say", "tells", "tell", "lies", "lie",
                    "tells", "told"}, 0.1)
        and _has(segs, {"truth", "lie", "lies", "lied",
                        "yes", "no"}, 0.05)
    )),

    ("COMPARISON_ORDERING", lambda segs: (
        # adjective + "than" comparisons: "the book is older than the cup"
        (_has(segs, {"than"}, 0.1)
         and _has(segs, {
             "older", "newer", "heavier", "lighter", "larger", "smaller",
             "taller", "shorter", "faster", "slower", "cheaper", "pricier",
             "better", "worse", "more", "less",
         }, 0.05))
        # spatial ordering: "to the left of", "directly above"
        or (_has(segs, {"left", "right"}, 0.1)
            and any(re.search(r"\b(left|right)\s+of\b", s) for s in segs))
        # tournament/competition ordering: "finished below/above/first/last"
        or (_has(segs, {"finished", "placed", "ranked"}, 0.05)
            and _has(segs, {"above", "below", "ahead", "behind",
                            "first", "last", "second", "third"}, 0.1))
    )),

    ("BOOLEAN", lambda segs: (
        _has(segs, {"true", "false"}, 0.2)
        and _has(segs, {"and", "or", "not"}, 0.2)
    )),

    ("GEOMETRIC", lambda segs: (
        any(re.search(r"<path\b", s) for s in segs)  # SVG path = geometric shape
        or _has(segs, {"circle", "triangle", "square", "rectangle",
                       "pentagon", "hexagon", "oval", "diamond",
                       "crescent", "heart", "star"}, 0.05)
    )),

    ("DATE", lambda segs: (
        _has(segs, {"january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november",
                    "december", "monday", "tuesday", "wednesday", "thursday",
                    "friday", "saturday", "sunday"}, 0.05)
        or (                                    # date questions use years + date words
            any(re.search(r"\b(19|20)\d{2}\b", s) for s in segs)
            and _has(segs, {"date", "yesterday", "tomorrow", "today",
                            "week", "month", "year", "days"}, 0.05)
        )
    )),

    # FORMAL_LOGIC requires "valid"/"invalid"/"argument" to avoid
    # false-positive on logical_deduction's preamble "logically consistent"
    ("FORMAL_LOGIC", lambda segs: (
        _has(segs, {"valid", "invalid", "argument", "premises"}, 0.05)
    )),
]

ROUTE_NAMES = [r for r, _ in ROUTING_RULES]

# ── per-example routing ────────────────────────────────────────────────

def route_example(example: dict, task: str) -> list[str]:
    """Return list of all routes that fire for this example.

    Always uses sentence_segments so that task-specific segmenters
    (e.g. comma_segments for object_counting) don't strip vocab signals.
    """
    segs = sentence_segments(example["input"])
    fired = [name for name, fn in ROUTING_RULES if fn(segs)]
    return fired


# ── main ───────────────────────────────────────────────────────────────

def main():
    tasks = sorted(TASK_EXPECTED.keys())

    print(f"{'task':<48} {'expected':<22} {'result':<10}  top routes")
    print("─" * 110)

    summary = {"correct": 0, "wrong": 0, "unrouted": 0, "ambiguous": 0}

    for task in tasks:
        try:
            examples = load_task(task)[:100]
        except Exception as e:
            print(f"  [skip] {task}: {e}")
            continue

        expected = TASK_EXPECTED[task]
        route_ct: Counter = Counter()

        for ex in examples:
            fired = route_example(ex, task)
            for r in fired:
                route_ct[r] += 1

        # majority vote across examples
        if not route_ct:
            result = "UNROUTED"
            status = "unrouted"
        else:
            top_route, top_ct = route_ct.most_common(1)[0]
            # a route "fires" for the task if it fires on ≥50% of examples
            dominant = [r for r, c in route_ct.most_common() if c >= 50]
            if not dominant:
                result = "UNROUTED"
                status = "unrouted"
            elif len(dominant) == 1:
                result = dominant[0]
                status = "correct" if result == expected else "wrong"
            else:
                result = f"AMBIGUOUS({','.join(dominant[:2])})"
                status = "ambiguous"

        summary[status] += 1

        marker = "✓" if status == "correct" else ("?" if status == "unrouted" else "✗")
        top = [(r, c) for r, c in route_ct.most_common(3)]
        top_str = "  ".join(f"{r}={c}" for r, c in top)
        print(f"{marker} {task:<46} {expected:<22} {result:<22}  {top_str}")

    print()
    print(f"correct={summary['correct']}  wrong={summary['wrong']}  "
          f"ambiguous={summary['ambiguous']}  unrouted={summary['unrouted']}  "
          f"total={sum(summary.values())}")


if __name__ == "__main__":
    main()
