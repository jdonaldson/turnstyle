#!/usr/bin/env python3
"""Deterministic solver for web_of_lies (BBH).

Structure:
  Base facts:    "X tells the truth." / "X lies."
  Derived facts: "X says Y tells the truth." / "X says Y lies."
  Query:         "Does X tell the truth?"  →  Yes / No

Truth propagation:
  If X tells truth and X says Y tells truth  → Y tells truth
  If X tells truth and X says Y lies         → Y lies
  If X lies      and X says Y tells truth    → Y lies
  If X lies      and X says Y lies           → Y tells truth

No model needed — pure regex + propagation.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/wol_deterministic.py
"""

from __future__ import annotations

import re
import sys

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

# ── patterns ───────────────────────────────────────────────────────────────

BASE_RE = re.compile(
    r"^(\w+)\s+(tells\s+the\s+truth|lies)\s*$",
    re.I,
)
SAYS_RE = re.compile(
    r"(\w+)\s+says\s+(\w+)\s+(tells\s+the\s+truth|lies)",
    re.I,
)
QUERY_RE = re.compile(
    r"Does\s+(\w+)\s+tell\s+the\s+truth\?",
    re.I,
)

TRUTH_PHRASES = {"tells the truth"}


def _is_truth(phrase: str) -> bool:
    return phrase.strip().lower() in TRUTH_PHRASES


# ── solver ─────────────────────────────────────────────────────────────────

def solve(example: dict) -> str | None:
    # Strip "Question:" prefix if present
    text = re.sub(r"^Question:\s*", "", example["input"].strip())
    truth: dict[str, bool] = {}

    # Pass 1: base facts — only from standalone sentences (not inside "says" sentences)
    for sent in re.split(r"\.\s+|\.\Z", text):
        sent = sent.strip()
        if re.search(r"\bsays\b", sent, re.I):
            continue
        m = BASE_RE.match(sent)
        if m:
            truth[m.group(1)] = _is_truth(m.group(2))

    # Pass 2: derived — propagate in either direction until stable
    # "X says Y [claim]":
    #   subject known → speaker = (claim matches subject's truth)
    #   speaker known → subject = (claim if speaker tells truth else opposite)
    prev_size = -1
    while len(truth) != prev_size:
        prev_size = len(truth)
        for speaker, subject, claim in SAYS_RE.findall(text):
            asserted = _is_truth(claim)
            if subject in truth and speaker not in truth:
                truth[speaker] = (asserted == truth[subject])
            elif speaker in truth and subject not in truth:
                truth[subject] = asserted if truth[speaker] else not asserted

    # Query
    m = QUERY_RE.search(text)
    if not m:
        return None
    queried = m.group(1)
    if queried not in truth:
        return None
    return "Yes" if truth[queried] else "No"


# ── evaluation ─────────────────────────────────────────────────────────────

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
        f"{task:<24}  {correct}/{total}  "
        f"({100*correct/total:.1f}%)  "
        f"wrong={wrong}  parse_fail={parse_fail}"
    )


def main() -> None:
    evaluate("web_of_lies")


if __name__ == "__main__":
    main()
