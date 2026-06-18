#!/usr/bin/env python3
"""Diagnostic: does the temporal option-token probe learn constraint-start matching
or genuine slot occupancy?

Hypothesis: the probe might be exploiting the fact that wrong options have their
start times appearing as constraint STARTS in the body, while the correct option's
start time does not (or appears only as an end time).

Test: split examples by whether the correct option's start time IS or IS NOT a
constraint start in the body. If probe accuracy drops on the overlap group, it's
structural matching. If accuracy holds, it's doing semantic constraint reasoning.

Probes at L13 and L14 (best layers from temporal_option_probe.py).

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/temporal_probe_diagnostic.py
"""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import re
import numpy as np
import torch
from collections import defaultdict
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from common import load_model
from swollm.bench.bbh import load_task
from turnstyle.sql import extract_options

# ── model ─────────────────────────────────────────────────────────────────────

print('Loading model…', flush=True)
tok, mdl, device = load_model()

PROBE_LAYERS = [13, 14]

# ── helpers ───────────────────────────────────────────────────────────────────

_TIME_RE = re.compile(r'\b(\d{1,2}(?:am|pm))\b')
_CONSTRAINT_START_RE = re.compile(r'\bfrom (\d{1,2}(?:am|pm)) to')


def option_start_time(opt_value: str) -> str | None:
    m = _TIME_RE.search(opt_value)
    return m.group(1) if m else None


def constraint_starts(body: str) -> set[str]:
    """All times that appear as the START of a 'from X to Y' constraint."""
    return set(_CONSTRAINT_START_RE.findall(body))


def split_body_options(text: str) -> tuple[str, str]:
    """Split BBH input into body (before Options:) and options section."""
    idx = text.find('Options:')
    if idx == -1:
        return text, ''
    return text[:idx], text[idx:]


def get_all_hidden(text: str) -> tuple[list[int], list[np.ndarray]]:
    inputs = tok(text, return_tensors='pt').to(device)
    token_ids = inputs['input_ids'][0].tolist()
    with torch.no_grad():
        out = mdl(**inputs, output_hidden_states=True)
    return token_ids, [h[0].float().cpu().numpy() for h in out.hidden_states]


def find_last_occurrence(token_ids: list[int], surface: str) -> int | None:
    for prefix in (f' {surface}', surface):
        sub = tok.encode(prefix, add_special_tokens=False)
        if not sub:
            continue
        target = sub[-1]
        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] == target:
                if len(sub) == 1:
                    return i
                start = i - len(sub) + 1
                if start >= 0 and token_ids[start:i + 1] == sub:
                    return i
    return None


# ── data collection ───────────────────────────────────────────────────────────

print('Loading temporal_sequences…', flush=True)
examples = load_task('temporal_sequences')
N_EX = len(examples)

# Per-item storage
X_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
binary_labels: list[int] = []
example_ids:   list[int] = []

# Per-example metadata
ex_start_overlap: list[bool] = []   # True = correct start IS a constraint start

skipped = 0
n_overlap = 0
n_no_overlap = 0

print(f'Collecting hidden states ({N_EX} examples, layers {PROBE_LAYERS})…', flush=True)

for ex_idx, ex in enumerate(examples):
    opts = extract_options(ex['input'])
    if not opts or len(opts) != 4:
        skipped += 1
        continue

    correct_letter = ex['target'].strip('()')
    correct_val = opts.get(correct_letter)
    if not correct_val:
        skipped += 1
        continue

    correct_start = option_start_time(correct_val)
    if not correct_start:
        skipped += 1
        continue

    body, _ = split_body_options(ex['input'])
    c_starts = constraint_starts(body)
    is_overlap = correct_start in c_starts

    text = tok.apply_chat_template(
        [{'role': 'user', 'content': ex['input']}],
        tokenize=False, add_generation_prompt=True)

    token_ids, layers = get_all_hidden(text)

    positions: dict[str, int] = {}
    for letter, opt_val in opts.items():
        st = option_start_time(opt_val)
        if st is None:
            break
        pos = find_last_occurrence(token_ids, st)
        if pos is None:
            break
        positions[letter] = pos
    else:
        for letter, pos in positions.items():
            for L in PROBE_LAYERS:
                X_by_layer[L].append(layers[L][pos])
            binary_labels.append(1 if letter == correct_letter else 0)
            example_ids.append(ex_idx)
        ex_start_overlap.append(is_overlap)
        if is_overlap:
            n_overlap += 1
        else:
            n_no_overlap += 1
        if (ex_idx + 1) % 50 == 0:
            print(f'  {ex_idx + 1}/{N_EX}', flush=True)
        continue

    skipped += 1

binary_labels  = np.array(binary_labels)
example_ids    = np.array(example_ids)
ex_start_overlap = np.array(ex_start_overlap)   # one per example, in order
n_examples = len(ex_start_overlap)

print(f'\n{n_examples} examples collected  (skipped={skipped})')
print(f'  start_overlap   (correct start IS a constraint start): {n_overlap}')
print(f'  no_start_overlap (correct start NOT a constraint start): {n_no_overlap}')

# ── show some examples of each group ─────────────────────────────────────────

print('\n--- Sample start_overlap cases (correct start appears as constraint start) ---')
shown = 0
for ex_idx, ex in enumerate(examples):
    if shown >= 4:
        break
    # check if this example was collected and is overlap
    arr_idx = np.where(np.unique(example_ids) == ex_idx)[0]
    if len(arr_idx) == 0:
        continue
    pos_in_meta = arr_idx[0]
    if not ex_start_overlap[pos_in_meta]:
        continue
    opts = extract_options(ex['input'])
    correct_letter = ex['target'].strip('()')
    correct_val = opts[correct_letter]
    correct_start = option_start_time(correct_val)
    body, _ = split_body_options(ex['input'])
    c_starts = constraint_starts(body)
    print(f'  [{ex_idx}] correct=({correct_letter}) "{correct_val}"  start={correct_start}')
    print(f'         constraint_starts={sorted(c_starts)}')
    shown += 1

print('\n--- Sample no_start_overlap cases ---')
shown = 0
for ex_idx, ex in enumerate(examples):
    if shown >= 4:
        break
    arr_idx = np.where(np.unique(example_ids) == ex_idx)[0]
    if len(arr_idx) == 0:
        continue
    pos_in_meta = arr_idx[0]
    if ex_start_overlap[pos_in_meta]:
        continue
    opts = extract_options(ex['input'])
    correct_letter = ex['target'].strip('()')
    correct_val = opts[correct_letter]
    correct_start = option_start_time(correct_val)
    body, _ = split_body_options(ex['input'])
    c_starts = constraint_starts(body)
    print(f'  [{ex_idx}] correct=({correct_letter}) "{correct_val}"  start={correct_start}')
    print(f'         constraint_starts={sorted(c_starts)}')
    shown += 1

# ── train probe on full data, evaluate split ─────────────────────────────────
# Train on all data, report accuracy by group.
# For proper eval, use leave-group-out: train on all examples, test on held-out.
# Use 5-fold but report accuracy stratified by overlap group within test fold.

from sklearn.model_selection import StratifiedKFold

unique_ex = np.unique(example_ids)
# stratify by overlap label
ex_overlap_label = ex_start_overlap.astype(int)   # 0/1 per example

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('\n' + '='*60)
print('Per-group answer accuracy by layer (5-fold CV)')
print('='*60)
print(f'{"Layer":>6}  {"overall":>8}  {"overlap":>8}  {"no-overlap":>11}  {"n_overlap":>10}  {"n_no_overlap":>13}')
print('-'*70)

for L in PROBE_LAYERS:
    X = np.array(X_by_layer[L])

    fold_overall, fold_overlap, fold_no_overlap = [], [], []
    fold_n_ov, fold_n_no = [], []

    for tr_ex_idx, te_ex_idx in kf.split(unique_ex, ex_overlap_label):
        tr_ex = unique_ex[tr_ex_idx]
        te_ex = unique_ex[te_ex_idx]

        tr_mask = np.isin(example_ids, tr_ex)
        te_mask = np.isin(example_ids, te_ex)

        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_mask])
        X_te = sc.transform(X[te_mask])
        y_tr = binary_labels[tr_mask]

        clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42,
                            class_weight='balanced', alpha=1e-4, n_jobs=1, tol=1e-4)
        clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_te)[:, 1]
        te_ex_ids = example_ids[te_mask]
        te_labels = binary_labels[te_mask]

        # Map te_ex indices to overlap flag
        te_ex_unique = unique_ex[te_ex_idx]
        te_overlap = ex_start_overlap[te_ex_idx]   # one per test example

        overall_c = overlap_c = no_overlap_c = 0
        overlap_t = no_overlap_t = 0

        for i, ex_id in enumerate(te_ex_unique):
            item_mask = te_ex_ids == ex_id
            ex_proba = proba[item_mask]
            ex_labels = te_labels[item_mask]
            if ex_labels.sum() != 1:
                continue
            correct = int(ex_labels[np.argmax(ex_proba)] == 1)
            overall_c += correct
            if te_overlap[i]:
                overlap_c += correct
                overlap_t += 1
            else:
                no_overlap_c += correct
                no_overlap_t += 1

        n_te = len(te_ex_unique)
        fold_overall.append(overall_c / n_te)
        fold_overlap.append(overlap_c / overlap_t if overlap_t else float('nan'))
        fold_no_overlap.append(no_overlap_c / no_overlap_t if no_overlap_t else float('nan'))
        fold_n_ov.append(overlap_t)
        fold_n_no.append(no_overlap_t)

    ov = float(np.nanmean(fold_overlap))
    no = float(np.nanmean(fold_no_overlap))
    print(f'  L{L:>2}   {float(np.mean(fold_overall)):.3f}    {ov:.3f}      {no:.3f}         '
          f'{int(np.mean(fold_n_ov)):>5}         {int(np.mean(fold_n_no)):>5}')

print()
print('Interpretation:')
print('  If overlap << no_overlap: probe exploits constraint-start matching (structural).')
print('  If overlap ≈ no_overlap:  probe does genuine slot-occupancy reasoning (semantic).')
