#!/usr/bin/env python3
"""Deterministic solver for navigate (BBH).

Two instruction formats:
  Format A  "Always face forward. Take N steps [direction]."
            Directions are absolute (forward=N, backward=S, left=W, right=E).
  Format B  "Take N steps. Turn left/right/around."
            Steps are in current facing direction; turns rotate state.

Query: "do you return to the starting point?" → Yes / No

No model needed.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/navigate_deterministic.py
"""

from __future__ import annotations

import re
import sys

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

# ── direction tables ────────────────────────────────────────────────────────
# Absolute (Format A): fixed facing = north
ABS_DIR: dict[str, tuple[int, int]] = {
    "forward":  ( 0,  1),
    "backward": ( 0, -1),
    "left":     (-1,  0),
    "right":    ( 1,  0),
    "north":    ( 0,  1),
    "south":    ( 0, -1),
    "west":     (-1,  0),
    "east":     ( 1,  0),
}

# Relative (Format B): facing index 0=N,1=E,2=S,3=W
FACING = [(0, 1), (1, 0), (0, -1), (-1, 0)]

STEP_ABS_RE = re.compile(
    r"Take\s+(\d+)\s+steps?\s+(forward|backward|left|right|north|south|east|west)",
    re.I,
)
STEP_FWD_RE = re.compile(r"Take\s+(\d+)\s+steps?$", re.I)  # no direction = forward
STEP_REL_RE = re.compile(
    r"Take\s+(\d+)\s+steps?\s+(forward|backward)",  # relative but with dir
    re.I,
)
TURN_RE = re.compile(r"Turn\s+(left|right|around)", re.I)


def solve(example: dict) -> str:
    # Strip question preamble (ends with "?") and options block
    raw = example["input"]
    # Extract instruction body: between first "?" and "\nOptions:" (or end)
    after_q = raw.split("?", 1)[-1]
    body = after_q.split("\nOptions:")[0].strip()

    x, y = 0, 0
    text = raw  # keep for "Always face forward" check

    if "Always face forward" in text:
        # Format A: absolute directions
        for n_str, direction in STEP_ABS_RE.findall(body):
            dx, dy = ABS_DIR[direction.lower()]
            x += int(n_str) * dx
            y += int(n_str) * dy
    else:
        # Format B: relative navigation with turns
        facing = 0  # 0=N
        fwd_map = {"forward": 0, "backward": 2}
        for sent in re.split(r"\.\s*", body):
            sent = sent.strip()
            t = TURN_RE.match(sent)
            if t:
                d = t.group(1).lower()
                if d == "left":    facing = (facing - 1) % 4
                elif d == "right": facing = (facing + 1) % 4
                else:              facing = (facing + 2) % 4
                continue
            # Step with explicit relative direction
            sr = STEP_REL_RE.match(sent)
            if sr:
                n = int(sr.group(1))
                offset = fwd_map.get(sr.group(2).lower(), 0)
                fx, fy = FACING[(facing + offset) % 4]
                x += n * fx; y += n * fy
                continue
            # Step without direction = forward
            sf = STEP_FWD_RE.match(sent)
            if sf:
                n = int(sf.group(1))
                fx, fy = FACING[facing]
                x += n * fx; y += n * fy

    return "Yes" if (x == 0 and y == 0) else "No"


def evaluate(task: str, n: int = 250) -> None:
    examples = load_task(task)[:n]
    correct = wrong = 0
    for ex in examples:
        pred = solve(ex)
        if pred == ex["target"].strip():
            correct += 1
        else:
            wrong += 1
    total = correct + wrong
    print(f"{task:<32}  {correct}/{total}  ({100*correct/total:.1f}%)  wrong={wrong}")


def main() -> None:
    evaluate("navigate")


if __name__ == "__main__":
    main()
