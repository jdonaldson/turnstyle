#!/usr/bin/env python3
"""Temporal encoding probe — two experiments on temporal_sequences.

Experiment 1 — Time-token RSA:
  Controlled prompts ("The time is 7am.") for all 22 hours (1am-12pm, 1pm-11pm).
  Extracts hidden states at the time-token position for all 24 layers, then
  computes RSA: do pairwise cosine distances correlate with |hour_a - hour_b|?
  If high: model encodes time ordinally (time has a linear/log representation).

Experiment 2 — Answer decodability:
  Uses temporal_sequences examples. Extracts last-token hidden states at all
  24 layers. SGD classifier 5-fold CV at each layer → predicts correct option
  letter (A-D). If any layer hits >40% (chance=25%), there's exploitable signal.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/temporal_encoding_probe.py
"""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import re
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common import load_model
from swollm.bench.bbh import load_task

# ── model ─────────────────────────────────────────────────────────────────────

print('Loading model…', flush=True)
tok, mdl, device = load_model()
N_LAYERS = mdl.config.num_hidden_layers + 1  # embeddings + 24 transformer layers


# ── shared: hidden state extraction ──────────────────────────────────────────

def get_all_hidden(text: str) -> tuple[list[int], list[np.ndarray]]:
    """Forward pass → (token_ids, [layer_hidden: (seq, D)])."""
    inputs = tok(text, return_tensors='pt').to(device)
    token_ids = inputs['input_ids'][0].tolist()
    with torch.no_grad():
        out = mdl(**inputs, output_hidden_states=True)
    layers = [h[0].float().cpu().numpy() for h in out.hidden_states]
    return token_ids, layers


def find_subtoken_pos(token_ids: list[int], surface: str) -> int | None:
    """Return position of last subtoken of `surface` in token_ids (BPE-safe)."""
    # Try with leading space (most tokens appear mid-sentence)
    for prefix in (f' {surface}', surface):
        sub = tok.encode(prefix, add_special_tokens=False)
        if not sub:
            continue
        # Search right-to-left for the last-subtoken ID
        target = sub[-1]
        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] == target:
                if len(sub) == 1:
                    return i
                start = i - len(sub) + 1
                if start >= 0 and token_ids[start:i + 1] == sub:
                    return i
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Time-token RSA
# ═══════════════════════════════════════════════════════════════════════════════

def to_hour24(h: int, meridiem: str) -> int:
    """Convert 12-hour + meridiem to 24-hour int."""
    if meridiem == 'am':
        return 0 if h == 12 else h
    else:
        return 12 if h == 12 else h + 12


# All 22 distinct hours (skip 12am midnight — unlikely in dataset)
HOURS_22 = []
for h in range(1, 13):
    HOURS_22.append((h, 'am', to_hour24(h, 'am')))
for h in range(1, 11):
    HOURS_22.append((h, 'pm', to_hour24(h, 'pm')))
# Results in hours 1–12 (am) + 13–22 (1pm–10pm) → 22 points

PROMPT_TEMPLATE = "The current time is {surface}."


print('\n' + '='*70)
print('EXPERIMENT 1 — Time-token RSA')
print('='*70)
print(f'Probing {len(HOURS_22)} time tokens across {N_LAYERS} layers…', flush=True)

hour24_vals: list[int] = []
# layer → list of hidden vecs (one per hour)
rsa_vecs: dict[int, list[np.ndarray]] = defaultdict(list)

for h, mer, h24 in HOURS_22:
    surface = f'{h}{mer}'
    prompt = PROMPT_TEMPLATE.format(surface=surface)
    token_ids, layers = get_all_hidden(prompt)
    pos = find_subtoken_pos(token_ids, surface)
    if pos is None:
        print(f'  WARNING: could not find token for {surface!r}', flush=True)
        continue
    for L in range(N_LAYERS):
        rsa_vecs[L].append(layers[L][pos])
    hour24_vals.append(h24)
    print(f'  {surface:>5} (h24={h24:2d})  tok_pos={pos}', flush=True)

# Build predictor: pairwise |hour_a - hour_b|
n = len(hour24_vals)
pred_linear = []
pred_log    = []
for i in range(n):
    for j in range(i + 1, n):
        diff = abs(hour24_vals[i] - hour24_vals[j])
        pred_linear.append(float(diff))
        pred_log.append(float(np.log(diff + 1)))
pred_linear = np.array(pred_linear)
pred_log    = np.array(pred_log)

print(f'\nRSA results (Spearman r: neural cosine-dist vs time-gap):')
print(f'{"Layer":>6}  {"linear-r":>9}  {"log-r":>7}  {"better":>7}')
print('-' * 40)

best_layer, best_r = -1, -1.0
for L in range(N_LAYERS):
    vecs = np.array(rsa_vecs[L])  # (n_hours, D)
    # Pairwise cosine distances
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs_n = vecs / norms
    cos_sim = vecs_n @ vecs_n.T
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(1.0 - cos_sim[i, j]))
    dists = np.array(dists)

    r_lin, _ = spearmanr(dists, pred_linear)
    r_log, _ = spearmanr(dists, pred_log)
    better = 'log' if r_log > r_lin else 'linear'
    best_r_here = max(r_lin, r_log)
    if best_r_here > best_r:
        best_r, best_layer = best_r_here, L
    print(f'  L{L:>2}   {r_lin:+.4f}    {r_log:+.4f}   {better}')

print(f'\nBest RSA layer: L{best_layer}  r={best_r:.4f}')
print('(r>0.3 = meaningful ordinal encoding; r>0.6 = strong)')


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Answer decodability probe
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('EXPERIMENT 2 — Answer decodability (last-token probe)')
print('='*70)

examples = load_task('temporal_sequences')
# Use all examples (exemplars + test — we're probing representation, not solver perf)
N_EX = len(examples)
print(f'Collecting last-token hidden states for {N_EX} examples…', flush=True)

X_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
labels: list[str] = []

for idx, ex in enumerate(examples):
    text = tok.apply_chat_template(
        [{'role': 'user', 'content': ex['input']}],
        tokenize=False, add_generation_prompt=True)
    _, layers = get_all_hidden(text)
    for L in range(N_LAYERS):
        X_by_layer[L].append(layers[L][-1])   # last token
    # Target: letter only, e.g. '(A)' -> 'A'
    label = ex['target'].strip('()')
    labels.append(label)
    if (idx + 1) % 50 == 0:
        print(f'  {idx+1}/{N_EX}', flush=True)

le = LabelEncoder()
y = le.fit_transform(labels)
n_classes = len(le.classes_)
chance = 1 / n_classes
print(f'\nClasses: {list(le.classes_)}  chance={chance:.2f}  N={len(y)}')

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f'\nAnswer decodability by layer (5-fold CV, SGD log-loss):')
print(f'{"Layer":>6}  {"CV acc":>7}  {"vs chance":>9}')
print('-' * 32)

best_probe_layer, best_probe_acc = -1, 0.0
for L in range(N_LAYERS):
    X = np.array(X_by_layer[L])
    fold_accs = []
    for tr_idx, te_idx in kf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        clf = SGDClassifier(loss='log_loss', max_iter=1000,
                            random_state=42, alpha=1e-4, n_jobs=1, tol=1e-4)
        clf.fit(X_tr, y[tr_idx])
        fold_accs.append(clf.score(X_te, y[te_idx]))
    acc = float(np.mean(fold_accs))
    delta = acc - chance
    if acc > best_probe_acc:
        best_probe_acc, best_probe_layer = acc, L
    print(f'  L{L:>2}   {acc:.3f}    {delta:+.3f}')

print(f'\nBest probe layer: L{best_probe_layer}  acc={best_probe_acc:.3f}  chance={chance:.2f}')
if best_probe_acc < chance + 0.05:
    print('=> No meaningful answer signal in hidden states.')
    print('   Regex/symbolic solver is the right path.')
elif best_probe_acc < chance + 0.20:
    print('=> Weak signal — probe may help marginally but regex is cleaner.')
else:
    print('=> Strong answer signal — probe-based solver is viable.')
