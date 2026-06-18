#!/usr/bin/env python3
"""Option-token probe for temporal_sequences.

For each example, extract hidden states at the start-time token of each option
(e.g. "6pm" in "(A) 6pm to 9pm") at all 25 layers. Train a binary probe:
  free=1  (correct option)
  occupied=0  (wrong options, 3 per example)

Answer accuracy: for each test example pick the option with the highest
free-probability from the probe. This mirrors the entity-ordering probe that
achieved 87.3% on logical_deduction.

Sweeps all layers and reports both binary accuracy and answer-level accuracy.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/temporal_option_probe.py
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import load_model
from swollm.bench.bbh import load_task
from turnstyle.sql import extract_options

# ── model ─────────────────────────────────────────────────────────────────────

print('Loading model…', flush=True)
tok, mdl, device = load_model()
N_LAYERS = mdl.config.num_hidden_layers + 1


# ── helpers ───────────────────────────────────────────────────────────────────

def get_all_hidden(text: str) -> tuple[list[int], list[np.ndarray]]:
    inputs = tok(text, return_tensors='pt').to(device)
    token_ids = inputs['input_ids'][0].tolist()
    with torch.no_grad():
        out = mdl(**inputs, output_hidden_states=True)
    layers = [h[0].float().cpu().numpy() for h in out.hidden_states]
    return token_ids, layers


def find_last_occurrence(token_ids: list[int], surface: str) -> int | None:
    """Last position of surface's last BPE subtoken in token_ids."""
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


_TIME_RE = re.compile(r'(\d{1,2}(?:am|pm))')

def option_start_time(opt_value: str) -> str | None:
    """Extract the start time from an option string like '6pm to 9pm'."""
    m = _TIME_RE.search(opt_value)
    return m.group(1) if m else None


# ── data collection ───────────────────────────────────────────────────────────

print('Loading temporal_sequences…', flush=True)
examples = load_task('temporal_sequences')
N_EX = len(examples)

# layer → list of hidden vecs (one per option token, 4 per example)
X_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
binary_labels: list[int] = []   # 1=free, 0=occupied
example_ids:   list[int] = []   # which example this option belongs to
option_ids:    list[str] = []   # letter A/B/C/D

skipped = 0

print(f'Collecting option-token hidden states ({N_EX} examples)…', flush=True)

for ex_idx, ex in enumerate(examples):
    text = tok.apply_chat_template(
        [{'role': 'user', 'content': ex['input']}],
        tokenize=False, add_generation_prompt=True)

    opts = extract_options(ex['input'])   # {'A': '6pm to 9pm', ...}
    if not opts or len(opts) != 4:
        skipped += 1
        continue

    correct_letter = ex['target'].strip('()')

    token_ids, layers = get_all_hidden(text)

    # Find a start-time position for each option
    positions: dict[str, int] = {}
    for letter, opt_val in opts.items():
        start_t = option_start_time(opt_val)
        if start_t is None:
            break
        pos = find_last_occurrence(token_ids, start_t)
        if pos is None:
            break
        positions[letter] = pos
    else:
        # All 4 options found — collect hidden states
        for letter, pos in positions.items():
            for L in range(N_LAYERS):
                X_by_layer[L].append(layers[L][pos])
            binary_labels.append(1 if letter == correct_letter else 0)
            example_ids.append(ex_idx)
            option_ids.append(letter)
        if (ex_idx + 1) % 50 == 0:
            print(f'  {ex_idx + 1}/{N_EX}', flush=True)
        continue

    skipped += 1

binary_labels = np.array(binary_labels)
example_ids   = np.array(example_ids)
n_items       = len(binary_labels)
n_examples    = n_items // 4
print(f'  {n_examples} examples, {n_items} option-token items  (skipped={skipped})\n')


# ── probe sweep ───────────────────────────────────────────────────────────────

def answer_accuracy(
    probe, scaler,
    X_te: np.ndarray,
    ex_ids_te: np.ndarray,
    labels_te: np.ndarray,
) -> float:
    """Convert binary free-probability to per-example answer pick."""
    X_te_s = scaler.transform(X_te)
    proba = probe.predict_proba(X_te_s)[:, 1]   # P(free)

    correct = 0
    total = 0
    for ex_id in np.unique(ex_ids_te):
        mask = ex_ids_te == ex_id
        ex_proba = proba[mask]
        ex_labels = labels_te[mask]
        if ex_labels.sum() != 1:
            continue   # shouldn't happen
        pred_idx = np.argmax(ex_proba)
        if ex_labels[pred_idx] == 1:
            correct += 1
        total += 1
    return correct / total if total else 0.0


# 5-fold CV split at the EXAMPLE level (not item level)
unique_ex = np.unique(example_ids)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Stratify by correct-option letter for the unique examples
ex_correct_letter = []
for ex_id in unique_ex:
    mask = example_ids == ex_id
    ex_correct_letter.append(binary_labels[mask].argmax())   # index of the 1
ex_correct_letter = np.array(ex_correct_letter)

print('Layer sweep (5-fold CV, binary SGD + answer accuracy):')
print(f'{"Layer":>6}  {"bin-acc":>8}  {"ans-acc":>8}  {"vs-chance":>9}')
print('-' * 42)

best_layer, best_ans = -1, 0.0
for L in range(N_LAYERS):
    X = np.array(X_by_layer[L])
    fold_bin, fold_ans = [], []

    for tr_ex_idx, te_ex_idx in kf.split(unique_ex, ex_correct_letter):
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

        fold_bin.append(clf.score(X_te, binary_labels[te_mask]))
        fold_ans.append(answer_accuracy(clf, sc, X[te_mask],
                                        example_ids[te_mask],
                                        binary_labels[te_mask]))

    bin_acc = float(np.mean(fold_bin))
    ans_acc = float(np.mean(fold_ans))
    if ans_acc > best_ans:
        best_ans, best_layer = ans_acc, L
    print(f'  L{L:>2}   {bin_acc:.3f}    {ans_acc:.3f}    {ans_acc - 0.25:+.3f}')

print(f'\nBest layer: L{best_layer}  answer-acc={best_ans:.3f}  (last-token best was 0.664)')
if best_ans > 0.80:
    print('=> Strong — option-token probe significantly beats last-token.')
elif best_ans > 0.664:
    print('=> Better than last-token probe — token position matters.')
else:
    print('=> No improvement over last-token — token position does not help.')
