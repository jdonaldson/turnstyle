#!/usr/bin/env python3
"""autoprobe.scan: automated probe-task playbook over multiple BBH tasks.

For each task:
  1. Detect target format (letter-multiple-choice vs free-text yes/no)
  2. Pick canonical token finders applicable to that shape
  3. One forward pass per example, capture hidden states at all layers
     and all relevant token positions
  4. 5-fold CV layer × finder × mode sweep
  5. Cheap baselines (majority, longest, shortest, longest-or-shortest)
  6. Apply ship/no-ship rule: probe_cv > 0.60 AND lift > +10pp over
     best cheap baseline

Validates the 4 already-shipped probes plus the documented negative,
and probes 3 unattempted flat-baseline tasks.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/autoprobe_scan.py
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re
import sys
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SHIP_ABS_THRESHOLD = 0.60
SHIP_LIFT_THRESHOLD = 0.10

TASKS = [
    # Already-shipped probes (validation)
    "snarks",
    "ruin_names",
    "disambiguation_qa",
    "temporal_sequences",
    # Documented negative (validation)
    "salient_translation_error_detection",
    # Unattempted
    "causal_judgement",
    "movie_recommendation",
    "sports_understanding",
]

_OPTION_RE = re.compile(r"\(([A-Z])\)\s+(.+?)(?=\n\(|\Z)", flags=re.S)


def load_model():
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_ID} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to(device).eval()
    return mdl, tok, device


# ── target-format detection ──────────────────────────────────────────────────

def detect_target_format(examples):
    """Returns ('letter', letters_str) or ('binary', (yes_label, no_label)).

    Lenient: classifies as 'letter' if ≥90% of targets match `(X)` pattern.
    Examples not matching are skipped during collection.
    """
    targets = [ex["target"].strip() for ex in examples]
    letter_pat = re.compile(r"^\(([A-Z])\)$")
    matched = [letter_pat.match(t) for t in targets]
    if sum(1 for m in matched if m) / len(targets) >= 0.90:
        letters = sorted({m.group(1) for m in matched if m})
        return "letter", "".join(letters)
    lower = {t.lower() for t in targets}
    if lower <= {"yes", "no"}:
        return "binary", ("Yes", "No")
    return "unknown", None


def detect_options_in_text(text):
    """For letter-format tasks: return list of (letter, opt_text) pairs."""
    return [(m.group(1), m.group(2).strip())
            for m in _OPTION_RE.finditer(text)]


# ── token finders ────────────────────────────────────────────────────────────

def find_last_token_of_prompt(text, tokenizer):
    """Returns [(token_idx, '_LAST_')]."""
    encoded = tokenizer(text, return_offsets_mapping=True,
                        add_special_tokens=True)
    return [(len(encoded["input_ids"]) - 1, "_LAST_")], encoded


def find_per_option_last_token(text, tokenizer, letters):
    """Returns [(token_idx, letter), …] or (None, None) if any miss."""
    encoded = tokenizer(text, return_offsets_mapping=True,
                        add_special_tokens=True)
    offsets = encoded["offset_mapping"]
    positions = []
    found_letters = set()
    for m in _OPTION_RE.finditer(text):
        letter = m.group(1)
        if letter not in letters:
            continue
        opt_text = m.group(2).rstrip()
        opt_end_char = m.start(2) + len(opt_text)
        last_idx = None
        for tok_idx, (s, e) in enumerate(offsets):
            if s < opt_end_char and e >= opt_end_char and s != e:
                last_idx = tok_idx
        if last_idx is None:
            return None, None
        positions.append((last_idx, letter))
        found_letters.add(letter)
    if found_letters != set(letters):
        return None, None
    return positions, encoded


# ── data collection ──────────────────────────────────────────────────────────

def collect_records(examples, model, tokenizer, device, target_fmt, target_meta):
    """One forward pass per example. Capture hidden states at all relevant
    positions across all layers."""
    n_layers = model.config.num_hidden_layers + 1
    records = []
    skipped = 0

    for i, ex in enumerate(examples):
        text = ex["input"]
        target = ex["target"].strip()

        if target_fmt == "letter":
            m_t = re.match(r"\(([A-Z])\)", target)
            if not m_t:
                skipped += 1
                continue
            target_letter = m_t.group(1)
            letters = target_meta
            popt_positions, popt_enc = find_per_option_last_token(
                text, tokenizer, letters)
            last_positions, last_enc = find_last_token_of_prompt(text, tokenizer)
            if popt_positions is None:
                # fall back to single mode only
                positions = last_positions
                encoded = last_enc
            else:
                # combine: per-option positions plus the prompt-end
                positions = popt_positions + [(last_positions[0][0], "_LAST_")]
                encoded = popt_enc
        else:  # binary or unknown
            target_letter = None
            positions, encoded = find_last_token_of_prompt(text, tokenizer)

        ids = {
            "input_ids": torch.tensor([encoded["input_ids"]]).to(device),
            "attention_mask": torch.tensor([encoded["attention_mask"]]).to(device),
        }
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hidden = out.hidden_states

        per_layer = {}
        for layer_idx in range(n_layers):
            per_layer[layer_idx] = {
                tag: hidden[layer_idx][0, pos].float().cpu().numpy()
                for pos, tag in positions
            }

        # Record metadata
        opts = detect_options_in_text(text) if target_fmt == "letter" else []

        records.append({
            "target": target,
            "target_letter": target_letter,
            "per_layer": per_layer,
            "opts": opts,
            "n_layers": n_layers,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(examples)}] records={len(records)}",
                  flush=True)

    if skipped:
        print(f"  skipped {skipped}", flush=True)
    return records


# ── cheap baselines ──────────────────────────────────────────────────────────

def cheap_baselines(records, target_fmt, target_meta):
    n = len(records)
    baselines = {}
    if target_fmt == "letter":
        letter_dist = Counter(r["target_letter"] for r in records)
        baselines["majority"] = max(letter_dist.values()) / n

        # Longest / shortest option as predictor
        def pick_by(key, reducer):
            correct = 0
            for r in records:
                if not r["opts"]:
                    continue
                lens = {l: len(t) for l, t in r["opts"]}
                pred = reducer(lens, key=lens.get)
                if pred == r["target_letter"]:
                    correct += 1
            return correct / n

        baselines["longest"] = pick_by(None, max)
        baselines["shortest"] = pick_by(None, min)

    else:  # binary
        yes_label, no_label = target_meta
        target_dist = Counter(r["target"].lower() for r in records)
        baselines["majority"] = max(target_dist.values()) / n

    return baselines


# ── probe sweeps ─────────────────────────────────────────────────────────────

def cv_single_mode(records, layer_idx, key, n_splits=5, seed=42):
    """Single-vector-per-example multinomial classifier."""
    if key == "_LAST_":
        # Some records may not have _LAST_ if popt finder failed; skip those
        usable = [r for r in records if "_LAST_" in r["per_layer"][layer_idx]]
    else:
        usable = [r for r in records if key in r["per_layer"][layer_idx]]
    if len(usable) < 50:
        return None

    if usable[0]["target_letter"] is not None:
        y = np.array([ord(r["target_letter"]) - ord("A") for r in usable])
    else:
        y = np.array([1 if r["target"].lower() == "yes" else 0 for r in usable])

    X = np.array([r["per_layer"][layer_idx][key] for r in usable])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append((clf.predict(sc.transform(X[te])) == y[te]).mean())
    return float(np.mean(accs))


def cv_per_option(records, layer_idx, letters, n_splits=5, seed=42):
    """Per-option binary "is correct" classifier; argmax at eval."""
    usable = []
    for r in records:
        if all(l in r["per_layer"][layer_idx] for l in letters):
            usable.append(r)
    if len(usable) < 50:
        return None

    n_ex = len(usable)
    y_class = np.array([ord(r["target_letter"]) - ord("A") for r in usable])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr_idx, te_idx in skf.split(np.arange(n_ex), y_class):
        X_tr, y_tr = [], []
        for i in tr_idx:
            for letter in letters:
                X_tr.append(usable[i]["per_layer"][layer_idx][letter])
                y_tr.append(1 if letter == usable[i]["target_letter"] else 0)
        X_tr = np.array(X_tr); y_tr = np.array(y_tr)
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
        clf.fit(sc.transform(X_tr), y_tr)

        correct = 0
        for i in te_idx:
            scores = {}
            for letter in letters:
                h = usable[i]["per_layer"][layer_idx][letter]
                scores[letter] = clf.predict_proba(sc.transform(h[None]))[0, 1]
            pred = max(scores, key=scores.get)
            if pred == usable[i]["target_letter"]:
                correct += 1
        accs.append(correct / len(te_idx))
    return float(np.mean(accs))


# ── orchestration ────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task: str
    n: int
    target_fmt: str
    target_meta: object
    baselines: dict
    sweep: list  # list of (finder, mode, layer, cv) tuples
    chosen: tuple  # (finder, mode, layer, cv) or None
    ship: bool
    reason: str


def autoprobe(task_name, model, tokenizer, device):
    print(f"\n{'='*70}\n[autoprobe {task_name}]\n{'='*70}", flush=True)
    examples = load_task(task_name)
    target_fmt, target_meta = detect_target_format(examples)
    print(f"  N={len(examples)}, target_fmt={target_fmt}, options={target_meta}",
          flush=True)

    if target_fmt == "unknown":
        return TaskResult(task_name, len(examples), target_fmt, target_meta,
                          {}, [], None, False, "unknown target format")

    print("  collecting hidden states…", flush=True)
    t0 = time.time()
    records = collect_records(examples, model, tokenizer, device,
                              target_fmt, target_meta)
    print(f"  collected {len(records)}/{len(examples)} in {time.time()-t0:.1f}s",
          flush=True)

    baselines = cheap_baselines(records, target_fmt, target_meta)
    print(f"  cheap baselines: " + ", ".join(
        f"{k}={v:.1%}" for k, v in baselines.items()), flush=True)

    n_layers = records[0]["n_layers"]
    sweep = []

    # Single-mode at "_LAST_" (if available) or at letter "A" if popt stored too
    last_key = "_LAST_" if "_LAST_" in records[0]["per_layer"][0] else None
    if last_key:
        for layer in range(n_layers):
            cv = cv_single_mode(records, layer, last_key)
            if cv is not None:
                sweep.append(("last_token_of_prompt", "single", layer, cv))

    # Per-option at letters (if letter format)
    if target_fmt == "letter":
        letters = target_meta
        for layer in range(n_layers):
            cv = cv_per_option(records, layer, letters)
            if cv is not None:
                sweep.append(("per_option_last_token", "per_option", layer, cv))

    # Sort by CV
    sweep.sort(key=lambda r: -r[3])
    print("  sweep top 5:")
    for finder, mode, layer, cv in sweep[:5]:
        print(f"    {finder:25s} {mode:10s} L{layer:>2}  {cv:>6.1%}",
              flush=True)

    # Decision
    if not sweep:
        return TaskResult(task_name, len(examples), target_fmt, target_meta,
                          baselines, sweep, None, False, "no probe runnable")
    chosen = sweep[0]
    best_baseline = max(baselines.values()) if baselines else 0
    abs_ok = chosen[3] >= SHIP_ABS_THRESHOLD
    lift = chosen[3] - best_baseline
    lift_ok = lift >= SHIP_LIFT_THRESHOLD
    ship = abs_ok and lift_ok

    print(f"  best: {chosen[0]} {chosen[1]} L{chosen[2]} = {chosen[3]:.1%}")
    print(f"  best cheap baseline: {best_baseline:.1%}, lift: {lift*100:+.1f}pp")
    if ship:
        reason = f"SHIP: cv {chosen[3]:.1%} ≥ {SHIP_ABS_THRESHOLD:.0%} and lift {lift*100:+.1f}pp ≥ {SHIP_LIFT_THRESHOLD*100:+.1f}pp"
    else:
        why = []
        if not abs_ok:
            why.append(f"absolute {chosen[3]:.1%} < {SHIP_ABS_THRESHOLD:.0%}")
        if not lift_ok:
            why.append(f"lift {lift*100:+.1f}pp < {SHIP_LIFT_THRESHOLD*100:+.1f}pp")
        reason = "NO SHIP: " + " and ".join(why)
    print(f"  → {reason}")

    return TaskResult(task_name, len(examples), target_fmt, target_meta,
                      baselines, sweep, chosen, ship, reason)


def main():
    model, tokenizer, device = load_model()
    results = []
    for task in TASKS:
        try:
            r = autoprobe(task, model, tokenizer, device)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {task}: {e}", flush=True)
            results.append(None)

    # Summary table
    print("\n\n" + "="*78)
    print(f"{'task':40s}  {'baseline':>9s}  {'best CV':>9s}  ship?")
    print("="*78)
    for r in results:
        if r is None:
            continue
        bb = max(r.baselines.values()) if r.baselines else 0
        cv = r.chosen[3] if r.chosen else 0
        print(f"  {r.task:38s}  {bb:>8.1%}  {cv:>8.1%}  "
              f"{'SHIP' if r.ship else 'no-ship'}")


if __name__ == "__main__":
    main()
