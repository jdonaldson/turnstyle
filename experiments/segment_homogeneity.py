#!/usr/bin/env python3
"""Segment homogeneity experiment.

Does variance in segment length (from simple sentence splitting) separate
list types well enough to use as a routing signal?

Hypothesis:
  tabular    (penguins, object_counting) → low variance  (uniform records)
  ordered_ops (navigate, tracking)       → medium variance
  dependency  (web_of_lies)              → low-medium variance (uniform claims)

If tabular vs non-tabular separates cleanly, that's enough to route SQL vs IR.

No model needed — pure text statistics on raw BBH prompts.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/segment_homogeneity.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

# ── config ────────────────────────────────────────────────────────────

TASKS = {
    "penguins_in_a_table":                  "tabular",
    "object_counting":                      "tabular",
    "tracking_shuffled_objects_three_objects": "ordered_ops",
    "navigate":                             "ordered_ops",
    "web_of_lies":                          "dependency",
}

# ── segmentation ──────────────────────────────────────────────────────

def _strip_options(text: str) -> str:
    """Remove 'Options:' block and trailing boilerplate."""
    for marker in ("Options:", "options:", "\nOptions", "- Yes\n", "- No\n"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def sentence_segments(text: str) -> list[str]:
    """Split on sentence boundaries ('. ' or '.\n') and strip empties."""
    text = _strip_options(text)
    parts = re.split(r"\.\s+|\.\n", text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def comma_segments(text: str) -> list[str]:
    """Split on commas — best for object_counting flat lists."""
    text = _strip_options(text)
    # strip leading "I have a " preamble before the list
    text = re.sub(r"^[^,]+have\s+", "", text)
    parts = text.split(",")
    return [p.strip() for p in parts if len(p.strip()) > 2]


def table_row_segments(text: str) -> list[str]:
    """Split penguins CSV rows.

    The table block lies between 'is a penguin:' and 'For example'.
    The header is the first comma-group; remaining groups are records.
    Records may be space-separated on one line OR newline-separated.
    """
    text = _strip_options(text)
    match = re.search(r"is a penguin:\s+(.*?)(?:For example|$)", text, re.DOTALL)
    if not match:
        return sentence_segments(text)
    table = match.group(1).strip()
    # normalise to one record per line: insert newline before each
    # capitalised name token (Name, digit pattern)
    table = re.sub(r"\s+([A-Z][a-z]+,\s*\d)", r"\n\1", table)
    lines = [l.strip() for l in table.split("\n") if l.strip()]
    # skip header (no digits)
    rows = [l for l in lines if re.search(r"\d", l)]
    return rows if rows else sentence_segments(text)


SEGMENTERS = {
    "penguins_in_a_table":                  table_row_segments,
    "object_counting":                      comma_segments,
    "tracking_shuffled_objects_three_objects": sentence_segments,
    "navigate":                             sentence_segments,
    "web_of_lies":                          sentence_segments,
}


# ── metrics ───────────────────────────────────────────────────────────

def segment_stats(segs: list[str]) -> dict:
    if len(segs) < 2:
        return {"n": len(segs), "mean": 0.0, "std": 0.0, "cv": 0.0}
    lengths = np.array([len(s) for s in segs], dtype=float)
    mean = lengths.mean()
    std = lengths.std()
    cv = std / mean if mean > 0 else 0.0  # coefficient of variation
    return {"n": int(len(segs)), "mean": float(mean), "std": float(std), "cv": float(cv)}


# ── main ──────────────────────────────────────────────────────────────

def main():
    per_task: dict[str, list[dict]] = {}

    for task, list_type in TASKS.items():
        examples = load_task(task)
        segmenter = SEGMENTERS[task]
        stats_list = []
        for ex in examples:
            segs = segmenter(ex["input"])
            stats_list.append(segment_stats(segs))
        per_task[task] = stats_list
        cvs = [s["cv"] for s in stats_list]
        ns  = [s["n"]  for s in stats_list]
        print(
            f"{task:<50} [{list_type:<12}]"
            f"  n_segs={np.mean(ns):.1f}±{np.std(ns):.1f}"
            f"  cv={np.mean(cvs):.3f}±{np.std(cvs):.3f}"
        )

    # ── separability check ───────────────────────────────────────────
    print()
    print("=== CV separability by list type ===")
    by_type: dict[str, list[float]] = {}
    for task, list_type in TASKS.items():
        cvs = [s["cv"] for s in per_task[task]]
        by_type.setdefault(list_type, []).extend(cvs)

    for lt, cvs in sorted(by_type.items()):
        print(f"  {lt:<14}  mean_cv={np.mean(cvs):.3f}  median={np.median(cvs):.3f}  std={np.std(cvs):.3f}")

    # ── tabular vs non-tabular threshold sweep ────────────────────────
    print()
    print("=== tabular vs non-tabular (binary routing) ===")
    tab_cvs    = by_type["tabular"]
    nontab_cvs = by_type["ordered_ops"] + by_type["dependency"]

    best_acc, best_thresh = 0.0, 0.0
    all_cvs = sorted(set(tab_cvs + nontab_cvs))
    for thresh in np.linspace(min(all_cvs), max(all_cvs), 200):
        tab_correct    = sum(cv < thresh for cv in tab_cvs)
        nontab_correct = sum(cv >= thresh for cv in nontab_cvs)
        acc = (tab_correct + nontab_correct) / (len(tab_cvs) + len(nontab_cvs))
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh

    tab_below    = sum(cv < best_thresh for cv in tab_cvs)    / len(tab_cvs)
    nontab_above = sum(cv >= best_thresh for cv in nontab_cvs) / len(nontab_cvs)
    print(f"  best threshold: cv < {best_thresh:.3f}")
    print(f"  tabular    correctly routed: {tab_below:.1%}  ({len(tab_cvs)} examples)")
    print(f"  non-tabular correctly routed: {nontab_above:.1%}  ({len(nontab_cvs)} examples)")
    print(f"  overall accuracy: {best_acc:.1%}")

    # ── ordered_ops vs dependency (harder) ────────────────────────────
    print()
    print("=== ordered_ops vs dependency (harder split) ===")
    ops_cvs = by_type["ordered_ops"]
    dep_cvs = by_type["dependency"]
    best_acc2, best_thresh2 = 0.0, 0.0
    for thresh in np.linspace(min(ops_cvs + dep_cvs), max(ops_cvs + dep_cvs), 200):
        ops_correct = sum(cv >= thresh for cv in ops_cvs)
        dep_correct = sum(cv < thresh  for cv in dep_cvs)
        acc = (ops_correct + dep_correct) / (len(ops_cvs) + len(dep_cvs))
        if acc > best_acc2:
            best_acc2, best_thresh2 = acc, thresh
    print(f"  best threshold: cv={best_thresh2:.3f}")
    print(f"  best accuracy: {best_acc2:.1%}  (baseline: {max(len(ops_cvs), len(dep_cvs)) / (len(ops_cvs) + len(dep_cvs)):.1%})")
    print(f"  (chance = 50% — if this is near chance, ops vs dep is not separable by CV alone)")


def subsegments(seg: str) -> list[str]:
    """Split a segment on secondary delimiters: commas, 'and', 'then'."""
    parts = re.split(r",\s*|\s+and\s+|\s+then\s+", seg)
    return [p.strip() for p in parts if len(p.strip()) > 1]


def recursive_stats(segs: list[str]) -> dict:
    """Compute stats on sub-segment counts across all level-1 segments."""
    sub_counts = [len(subsegments(s)) for s in segs]
    if len(sub_counts) < 2:
        return {"mean_sub": 0.0, "std_sub": 0.0, "cv_sub": 0.0}
    arr = np.array(sub_counts, dtype=float)
    mean = arr.mean()
    std = arr.std()
    cv = std / mean if mean > 0 else 0.0
    return {"mean_sub": float(mean), "std_sub": float(std), "cv_sub": float(cv)}


def main_recursive():
    print("\n=== Recursive segmentation: sub-segment count variance ===\n")
    per_task: dict[str, list[dict]] = {}

    for task, list_type in TASKS.items():
        examples = load_task(task)
        segmenter = SEGMENTERS[task]
        stats_list = []
        for ex in examples:
            segs = segmenter(ex["input"])
            stats_list.append(recursive_stats(segs))
        per_task[task] = stats_list

        cv_subs = [s["cv_sub"] for s in stats_list]
        mean_subs = [s["mean_sub"] for s in stats_list]
        print(
            f"{task:<50} [{list_type:<12}]"
            f"  mean_sub={np.mean(mean_subs):.1f}±{np.std(mean_subs):.1f}"
            f"  cv_sub={np.mean(cv_subs):.3f}±{np.std(cv_subs):.3f}"
        )

    print()
    print("=== cv_sub by list type ===")
    by_type: dict[str, list[float]] = {}
    for task, list_type in TASKS.items():
        cv_subs = [s["cv_sub"] for s in per_task[task]]
        by_type.setdefault(list_type, []).extend(cv_subs)
    for lt, vals in sorted(by_type.items()):
        print(f"  {lt:<14}  mean={np.mean(vals):.3f}  median={np.median(vals):.3f}  std={np.std(vals):.3f}")

    # combined feature: level-1 cv + level-2 cv_sub
    print()
    print("=== Combined (cv + cv_sub) — tabular vs rest ===")

    # rebuild with both metrics
    per_task2: dict[str, list[tuple[float,float]]] = {}
    for task, list_type in TASKS.items():
        examples = load_task(task)
        segmenter = SEGMENTERS[task]
        pairs = []
        for ex in examples:
            segs = segmenter(ex["input"])
            s1 = segment_stats(segs)
            s2 = recursive_stats(segs)
            pairs.append((s1["cv"], s2["cv_sub"]))
        per_task2[task] = pairs

    tab_pairs    = [p for t, lt in TASKS.items() if lt == "tabular"     for p in per_task2[t]]
    nontab_pairs = [p for t, lt in TASKS.items() if lt != "tabular"     for p in per_task2[t]]

    # sweep threshold on cv_sub only (level-2)
    all_cv_sub = [p[1] for p in tab_pairs + nontab_pairs]
    best_acc, best_t = 0.0, 0.0
    for thresh in np.linspace(min(all_cv_sub), max(all_cv_sub), 300):
        c1 = sum(p[1] < thresh for p in tab_pairs)
        c2 = sum(p[1] >= thresh for p in nontab_pairs)
        acc = (c1 + c2) / (len(tab_pairs) + len(nontab_pairs))
        if acc > best_acc:
            best_acc, best_t = acc, thresh

    tab_hit    = sum(p[1] < best_t for p in tab_pairs)    / len(tab_pairs)
    nontab_hit = sum(p[1] >= best_t for p in nontab_pairs) / len(nontab_pairs)
    print(f"  best cv_sub threshold: {best_t:.3f}")
    print(f"  tabular    hit: {tab_hit:.1%}  ({len(tab_pairs)} examples)")
    print(f"  non-tabular hit: {nontab_hit:.1%}  ({len(nontab_pairs)} examples)")
    print(f"  overall: {best_acc:.1%}")


if __name__ == "__main__":
    main()
    main_recursive()
