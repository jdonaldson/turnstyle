#!/usr/bin/env python3
"""Canonical SVO extraction + verb normalization coverage test.

Does a closed-world verb dictionary + simple first-verb regex cover
the relationship structure in BBH tasks? No model needed.

Questions answered:
  1. What fraction of segments get a canonical verb type?
  2. What verbs appear in unmatched segments (candidates to add)?
  3. Is the extracted schema consistent within a task?

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/svo_extraction.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

# reuse segmenters from the homogeneity experiment
sys.path.insert(0, str(Path(__file__).parent))
from segment_homogeneity import SEGMENTERS, sentence_segments

# ── verb normalization dictionary ──────────────────────────────────────

VERB_CANON: dict[str, str] = {
    # TRANSFER: entity moves between holders
    "gives":    "TRANSFER", "give":     "TRANSFER", "gave":     "TRANSFER",
    "swaps":    "TRANSFER", "swap":     "TRANSFER", "swapped":  "TRANSFER",
    "trades":   "TRANSFER", "trade":    "TRANSFER", "traded":   "TRANSFER",
    "passes":   "TRANSFER", "pass":     "TRANSFER", "passed":   "TRANSFER",
    "receives": "TRANSFER", "receive":  "TRANSFER", "received": "TRANSFER",
    "moves":    "TRANSFER", "move":     "TRANSFER", "moved":    "TRANSFER",
    "gets":     "TRANSFER", "get":      "TRANSFER", "got":      "TRANSFER",
    "takes":    "TRANSFER", "take":     "TRANSFER", "took":     "TRANSFER",
    # CLAIM: entity asserts a proposition (or its negation)
    "says":     "CLAIM",    "say":      "CLAIM",    "said":     "CLAIM",
    "tells":    "CLAIM",    "tell":     "CLAIM",    "told":     "CLAIM",
    "lies":     "CLAIM",    "lie":      "CLAIM",    "lied":     "CLAIM",
    "claims":   "CLAIM",    "claim":    "CLAIM",    "claimed":  "CLAIM",
    # COMPARISON: relative ordering / attribute ranking
    "older":    "COMPARISON", "newer":   "COMPARISON",
    "heavier":  "COMPARISON", "lighter": "COMPARISON",
    "larger":   "COMPARISON", "smaller": "COMPARISON",
    "taller":   "COMPARISON", "shorter": "COMPARISON",
    "faster":   "COMPARISON", "slower":  "COMPARISON",
    "better":   "COMPARISON", "worse":   "COMPARISON",
    "cheaper":  "COMPARISON", "pricier": "COMPARISON",
    "left":     "COMPARISON", "right":   "COMPARISON",
    "above":    "COMPARISON", "below":   "COMPARISON",
    "before":   "COMPARISON", "after":   "COMPARISON",
    "first":    "COMPARISON", "last":    "COMPARISON",
    "second":   "COMPARISON", "third":   "COMPARISON",
    "fourth":   "COMPARISON", "fifth":   "COMPARISON",
    # NAVIGATION: displacement / orientation
    "turn":     "NAVIGATION", "face":    "NAVIGATION",
    "walk":     "NAVIGATION", "go":      "NAVIGATION",
    # ENUMERATION: "I have X, Y, Z" — no inter-entity relationship
    "have":     "ENUMERATION", "had":    "ENUMERATION",
    "there":    "ENUMERATION",   # "there are N Xs"
}

TASKS = {
    "web_of_lies":                             "dependency",
    "tracking_shuffled_objects_three_objects": "ordered_ops",
    "logical_deduction_five_objects":          "ordered_ops",
    "navigate":                                "ordered_ops",
    "object_counting":                         "tabular",
    "penguins_in_a_table":                     "tabular",
}


# ── extraction ─────────────────────────────────────────────────────────

def extract_canonical(seg: str) -> dict | None:
    """
    Scan tokens left-to-right; return the first hit in VERB_CANON.
    Subject = tokens before the verb; object = tokens after.
    """
    tokens = re.split(r"\s+", seg.strip().rstrip(".!?"))
    for i, tok in enumerate(tokens):
        v = tok.lower().strip(",.;:\"'")
        if v in VERB_CANON:
            return {
                "subject":    " ".join(tokens[:i]),
                "verb":       v,
                "verb_canon": VERB_CANON[v],
                "object":     " ".join(tokens[i + 1:]),
                "raw":        seg,
            }
    return None


# ── analysis ───────────────────────────────────────────────────────────

def analyse_task(task: str, list_type: str) -> None:
    examples  = load_task(task)[:150]
    segmenter = SEGMENTERS.get(task, sentence_segments)

    n_segs       = 0
    n_hit        = 0
    verb_type_ct = Counter()
    schema_ct    = Counter()   # (verb_canon, subject_pattern, object_pattern)
    missed_toks  = Counter()   # tokens in unmatched segments

    for ex in examples:
        segs = segmenter(ex["input"])
        for seg in segs:
            n_segs += 1
            result = extract_canonical(seg)
            if result:
                n_hit += 1
                verb_type_ct[result["verb_canon"]] += 1
                # coarse schema: first/last word of subject and object
                subj_sig = result["subject"].split()[0].lower() if result["subject"] else "_"
                obj_sig  = result["object"].split()[-1].lower() if result["object"] else "_"
                schema_ct[(result["verb_canon"], subj_sig, obj_sig)] += 1
            else:
                for tok in re.split(r"\s+", seg):
                    clean = tok.lower().strip(",.;:\"'()")
                    if len(clean) > 2:
                        missed_toks[clean] += 1

    coverage = n_hit / n_segs if n_segs else 0.0
    print(f"\n{'─'*60}")
    print(f"{task}  [{list_type}]")
    print(f"  coverage : {coverage:.1%}  ({n_hit}/{n_segs} segments)")
    if verb_type_ct:
        print(f"  verb types: {dict(verb_type_ct.most_common())}")
    # show top schema patterns
    if schema_ct:
        print("  top schemas (verb_canon, subj_start, obj_end):")
        for schema, cnt in schema_ct.most_common(5):
            print(f"    {cnt:4d}×  {schema}")
    # show top unmatched tokens
    if missed_toks and coverage < 1.0:
        # filter out stop words for readability
        STOP = {"the","a","an","is","are","was","were","and","or","not",
                "that","this","it","in","of","to","for","on","at","by",
                "with","from","as","be","been","being","do","does","did",
                "no","yes","true","false","if","then","so","but","all",
                "each","any","one","two","three","four","five","six",
                "seven","eight","nine","ten","they","their","its","has"}
        top = [(t, c) for t, c in missed_toks.most_common(20) if t not in STOP][:10]
        if top:
            print(f"  top unmatched tokens: {top}")


def main() -> None:
    print("=== SVO extraction coverage by task ===")
    for task, list_type in TASKS.items():
        analyse_task(task, list_type)
    print(f"\n{'─'*60}")
    print("\nInterpretation:")
    print("  HIGH coverage + SINGLE verb type  → schema is consistent, dict works")
    print("  HIGH coverage + MIXED verb types  → ambiguity; need disambiguation step")
    print("  LOW  coverage                     → missing verbs; check unmatched tokens")


if __name__ == "__main__":
    main()
