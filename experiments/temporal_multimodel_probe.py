#!/usr/bin/env python3
"""Multi-model sweep of the temporal option-token probe.

For each model, probes hidden states at option start-time token positions
across all layers. Reports best layer and answer accuracy per model.

Hypothesis: the structural signal (free-slot start ≠ constraint start) should
appear across architectures, but at different depths depending on model capacity.

Models (all locally cached):
  - SmolLM2-360M-Instruct    (small, same family as 1.7B)
  - Qwen2.5-1.5B-Instruct    (different architecture, similar size)
  - Phi-4-mini-instruct       (larger, Phi family — deeper stable features)
  - bitnet-b1.58-2B-4T-bf16  (requires bfloat16)

SmolLM2-1.7B result for reference: best L14, answer-acc=0.948

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/temporal_multimodel_probe.py
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from swollm.bench.bbh import load_task
from turnstyle.sql import extract_options

# ── models to sweep ───────────────────────────────────────────────────────────

MODELS = [
    ("HuggingFaceTB/SmolLM2-360M-Instruct",   torch.float16),
    ("Qwen/Qwen2.5-1.5B-Instruct",             torch.float16),
    ("microsoft/Phi-4-mini-instruct",           torch.float16),
    ("microsoft/bitnet-b1.58-2B-4T-bf16",      torch.bfloat16),
]

# ── helpers ───────────────────────────────────────────────────────────────────

_TIME_RE            = re.compile(r'\b(\d{1,2}(?:am|pm))\b')
_CONSTRAINT_START_RE = re.compile(r'\bfrom (\d{1,2}(?:am|pm)) to')


def option_start_time(opt_value: str) -> str | None:
    m = _TIME_RE.search(opt_value)
    return m.group(1) if m else None


def get_all_hidden(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    token_ids = inputs['input_ids'][0].tolist()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    layers = [h[0].float().cpu().numpy() for h in out.hidden_states]
    return token_ids, layers


def find_last_occurrence(token_ids, surface, tokenizer):
    for prefix in (f' {surface}', surface):
        sub = tokenizer.encode(prefix, add_special_tokens=False)
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


def answer_accuracy(clf, sc, X_te, ex_ids_te, labels_te):
    X_te_s = sc.transform(X_te)
    proba = clf.predict_proba(X_te_s)[:, 1]
    correct = total = 0
    for ex_id in np.unique(ex_ids_te):
        mask = ex_ids_te == ex_id
        if labels_te[mask].sum() != 1:
            continue
        if labels_te[mask][np.argmax(proba[mask])] == 1:
            correct += 1
        total += 1
    return correct / total if total else 0.0


# ── dataset (shared across models) ───────────────────────────────────────────

print('Loading temporal_sequences…', flush=True)
examples = load_task('temporal_sequences')

# ── per-model sweep ───────────────────────────────────────────────────────────

def detect_device():
    if torch.cuda.is_available():  return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

DEVICE = detect_device()

summary = []   # (model_short, n_layers, best_layer, best_acc)

for model_id, dtype in MODELS:
    short = model_id.split('/')[-1]
    print(f'\n{"="*70}')
    print(f'Model: {short}  (dtype={dtype})')
    print('='*70)

    print('  Loading…', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype).to(DEVICE).eval()
    N_LAYERS = model.config.num_hidden_layers + 1

    print(f'  {N_LAYERS} layers, hidden={model.config.hidden_size}', flush=True)

    # Collect option-token hidden states
    X_by_layer = defaultdict(list)
    binary_labels = []
    example_ids   = []
    skipped = 0

    for ex_idx, ex in enumerate(examples):
        opts = extract_options(ex['input'])
        if not opts or len(opts) != 4:
            skipped += 1
            continue

        correct_letter = ex['target'].strip('()')

        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': ex['input']}],
            tokenize=False, add_generation_prompt=True)

        token_ids, layers = get_all_hidden(text, model, tokenizer, DEVICE)

        positions = {}
        for letter, opt_val in opts.items():
            st = option_start_time(opt_val)
            if st is None:
                break
            pos = find_last_occurrence(token_ids, st, tokenizer)
            if pos is None:
                break
            positions[letter] = pos
        else:
            for letter, pos in positions.items():
                for L in range(N_LAYERS):
                    X_by_layer[L].append(layers[L][pos])
                binary_labels.append(1 if letter == correct_letter else 0)
                example_ids.append(ex_idx)
            if (ex_idx + 1) % 50 == 0:
                print(f'    {ex_idx + 1}/{len(examples)}', flush=True)
            continue
        skipped += 1

    binary_labels = np.array(binary_labels)
    example_ids   = np.array(example_ids)
    n_examples = len(np.unique(example_ids))
    print(f'  {n_examples} examples  (skipped={skipped})')

    # 5-fold CV probe sweep
    unique_ex = np.unique(example_ids)
    # stratify by which option letter is correct
    ex_correct = []
    for eid in unique_ex:
        mask = example_ids == eid
        ex_correct.append(int(binary_labels[mask].argmax()))
    ex_correct = np.array(ex_correct)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f'\n  Layer sweep (answer accuracy, 5-fold CV):')
    print(f'  {"L":>4}  {"ans-acc":>8}  {"vs-chance":>10}')
    print(f'  {"-"*28}')

    best_layer, best_acc = -1, 0.0
    layer_accs = []

    for L in range(N_LAYERS):
        X = np.array(X_by_layer[L])
        fold_ans = []
        for tr_idx, te_idx in kf.split(unique_ex, ex_correct):
            tr_ex = unique_ex[tr_idx]
            te_ex = unique_ex[te_idx]
            tr_mask = np.isin(example_ids, tr_ex)
            te_mask = np.isin(example_ids, te_ex)
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr_mask])
            clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42,
                                class_weight='balanced', alpha=1e-4,
                                n_jobs=1, tol=1e-4)
            clf.fit(X_tr, binary_labels[tr_mask])
            fold_ans.append(answer_accuracy(clf, sc, X[te_mask],
                                            example_ids[te_mask],
                                            binary_labels[te_mask]))
        acc = float(np.mean(fold_ans))
        layer_accs.append(acc)
        if acc > best_acc:
            best_acc, best_layer = acc, L
        marker = ' ◄' if acc == best_acc else ''
        print(f'  L{L:>2}   {acc:.3f}    {acc-0.25:+.3f}{marker}')

    summary.append((short, N_LAYERS, best_layer, best_acc))
    print(f'\n  Best: L{best_layer}  acc={best_acc:.3f}')

    # Free memory before next model
    del model
    if DEVICE == 'mps':
        torch.mps.empty_cache()
    elif DEVICE == 'cuda':
        torch.cuda.empty_cache()

# ── cross-model summary ───────────────────────────────────────────────────────

print('\n\n' + '='*70)
print('CROSS-MODEL SUMMARY  (reference: SmolLM2-1.7B → L14, acc=0.948)')
print('='*70)
print(f'  {"Model":<40}  {"Layers":>6}  {"Best L":>7}  {"Acc":>6}  {"L/N":>6}')
print(f'  {"-"*68}')
for short, n_layers, best_layer, best_acc in summary:
    frac = best_layer / (n_layers - 1)
    print(f'  {short:<40}  {n_layers:>6}  {best_layer:>6}    {best_acc:.3f}   {frac:.2f}')

print()
print('L/N = best layer as fraction of total depth.')
print('If L/N is consistent across models, the feature lives at the same')
print('relative depth regardless of architecture.')
