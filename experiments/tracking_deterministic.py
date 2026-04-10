#!/usr/bin/env python3
"""Deterministic solver for tracking_shuffled_objects (all three variants).

Replaces Qwen transcription + flexible accumulator + SQL with pure regex.

Observation: every example has exactly two sentence types:
  1. Init   — Actor VERB_PHRASE Item  (comma-separated list)
  2. Action — Actor and Actor (swap|switch|trade) [noun]  (always symmetric)

No LLM needed. Verb vocabulary work confirms: all action verbs in this task
normalise to SWAP (symmetric exchange). Schema is fixed by routing.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/tracking_deterministic.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, str(Path(__file__).parent))
from swollm.bench.bbh import load_task

# ── known actor set ────────────────────────────────────────────────────────
# All three variants use a prefix of this ordered list.
ALL_ACTORS = ["Alice", "Bob", "Claire", "Dave", "Eve", "Fred", "Gertrude"]
ACTOR_PAT  = "|".join(ALL_ACTORS)  # longest-match not needed, all distinct length

# ── init verb phrases ──────────────────────────────────────────────────────
# Each phrase maps to: (regex after actor, item_group_index)
INIT_VERBS = [
    re.compile(r"gets\s+(.+)",          re.I),
    re.compile(r"has\s+a\s+(.+)",       re.I),
    re.compile(r"is\s+dancing\s+with\s+(.+)", re.I),
    re.compile(r"is\s+playing\s+(.+)",  re.I),
]

# ── action verb ────────────────────────────────────────────────────────────
ACTION_RE = re.compile(
    rf"({ACTOR_PAT})\s+and\s+({ACTOR_PAT})\s+(?:swap|switch|trade)",
    re.I,
)

# ── query ─────────────────────────────────────────────────────────────────
QUERY_RE = re.compile(
    rf"At the end of .+?,\s+({ACTOR_PAT})\s+(?:is|has)",
    re.I,
)

# ── options ────────────────────────────────────────────────────────────────
OPTIONS_RE = re.compile(r"\(([A-Z])\)\s+(.+?)(?=\n\([A-Z]\)|\Z)", re.S)


# ── parsers ────────────────────────────────────────────────────────────────

def detect_actors(text: str) -> list[str]:
    """Return ordered actors present in the preamble sentence."""
    return [a for a in ALL_ACTORS if re.search(rf"\b{a}\b", text)]


def parse_init(sent: str, actors: list[str]) -> dict[str, str]:
    """Extract {actor: item} from the init sentence."""
    state: dict[str, str] = {}
    actor_boundary = "|".join(actors)
    # Split on ', Actor' or ', and Actor' boundaries
    chunks = re.split(rf",\s*(?:and\s+)?(?={actor_boundary})", sent, flags=re.I)
    for chunk in chunks:
        chunk = chunk.strip().rstrip(".,")
        for actor in actors:
            # Actor may be preceded by setup text in the first chunk
            m = re.search(rf"\b{actor}\b", chunk, re.I)
            if m:
                rest = chunk[m.end():].strip()
                for pat in INIT_VERBS:
                    vm = pat.match(rest)
                    if vm:
                        state[actor] = vm.group(1).strip().rstrip(".,")
                        break
                break  # one actor per chunk
    return state


def parse_action(sent: str) -> tuple[str, str] | None:
    """Return (actor1, actor2) if this is a swap sentence, else None."""
    m = ACTION_RE.search(sent)
    return (m.group(1), m.group(2)) if m else None


def parse_query(text: str) -> str | None:
    """Return the queried actor name."""
    m = QUERY_RE.search(text)
    return m.group(1) if m else None


def parse_options(text: str) -> dict[str, str]:
    """Return {letter: value} from Options block."""
    opts_section = text.split("Options:")[-1] if "Options:" in text else text
    return {letter: val.strip() for letter, val in OPTIONS_RE.findall(opts_section)}


# ── full solver ────────────────────────────────────────────────────────────

def solve(example: dict) -> str | None:
    """
    Returns the option letter (e.g. '(A)') or None if parsing fails.
    """
    text = example["input"]
    lines = [l.strip() for l in text.split(".") if l.strip()]

    actors = detect_actors(lines[0] if lines else text)
    if not actors:
        return None

    # Locate the init sentence (contains 'At the start')
    init_sent = next((l for l in lines if re.search(r"At the start", l, re.I)), None)
    if not init_sent:
        return None

    state = parse_init(init_sent, actors)
    if len(state) < len(actors):
        return None  # failed to parse all actors

    # Apply actions (every sentence with swap/switch/trade)
    for line in lines:
        pair = parse_action(line)
        if pair:
            a1, a2 = pair
            if a1 in state and a2 in state:
                state[a1], state[a2] = state[a2], state[a1]

    # Parse query
    queried = parse_query(text)
    if not queried or queried not in state:
        return None

    answer_val = state[queried]

    # Match against options
    opts = parse_options(text)
    for letter, val in opts.items():
        if val.lower() == answer_val.lower():
            return f"({letter})"

    return None


# ── evaluation ─────────────────────────────────────────────────────────────

def evaluate(task: str) -> None:
    examples = load_task(task)[:250]
    correct = wrong = parse_fail = 0

    for ex in examples:
        pred = solve(ex)
        if pred is None:
            parse_fail += 1
        elif pred == ex["target"]:
            correct += 1
        else:
            wrong += 1

    total = correct + wrong + parse_fail
    print(
        f"{task:<48}  {correct}/{total}  "
        f"({100*correct/total:.1f}%)  "
        f"wrong={wrong}  parse_fail={parse_fail}"
    )


def main() -> None:
    for task in [
        "tracking_shuffled_objects_three_objects",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
    ]:
        evaluate(task)


if __name__ == "__main__":
    main()
