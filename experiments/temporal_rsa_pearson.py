#!/usr/bin/env python3
"""Weber's Law RSA re-run using Pearson r (not Spearman).

Spearman r is rank-based, so r(neural, linear) ≡ r(neural, log) for any
monotone predictor — the comparison was meaningless. Pearson r is sensitive
to actual magnitudes, so it CAN distinguish linear from log scale.

Also fits explicit OLS regression (R²) for both predictors to confirm.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/temporal_rsa_pearson.py
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

from common import load_model

print('Loading model…', flush=True)
tok, mdl, device = load_model()
N_LAYERS = mdl.config.num_hidden_layers + 1


def to_hour24(h: int, meridiem: str) -> int:
    if meridiem == 'am':
        return 0 if h == 12 else h
    else:
        return 12 if h == 12 else h + 12


HOURS_22 = []
for h in range(1, 13):
    HOURS_22.append((h, 'am', to_hour24(h, 'am')))
for h in range(1, 11):
    HOURS_22.append((h, 'pm', to_hour24(h, 'pm')))

PROMPT_TEMPLATE = "The current time is {surface}."


def find_last_occurrence(token_ids, surface):
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
                if start >= 0 and token_ids[start:i+1] == sub:
                    return i
    return None


print(f'Collecting hidden states for {len(HOURS_22)} time tokens…', flush=True)

hour24_vals = []
vecs_by_layer: dict[int, list[np.ndarray]] = {L: [] for L in range(N_LAYERS)}

for h, mer, h24 in HOURS_22:
    surface = f'{h}{mer}'
    prompt = PROMPT_TEMPLATE.format(surface=surface)
    inputs = tok(prompt, return_tensors='pt').to(device)
    token_ids = inputs['input_ids'][0].tolist()
    pos = find_last_occurrence(token_ids, surface)
    if pos is None:
        print(f'  WARNING: no token for {surface}', flush=True)
        continue
    with torch.no_grad():
        out = mdl(**inputs, output_hidden_states=True)
    for L in range(N_LAYERS):
        vecs_by_layer[L].append(out.hidden_states[L][0, pos].float().cpu().numpy())
    hour24_vals.append(h24)
    print(f'  {surface:>5} h24={h24:2d}', flush=True)

n = len(hour24_vals)

# Build pairwise predictors
pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
diffs = np.array([abs(hour24_vals[i] - hour24_vals[j]) for i, j in pairs])
pred_linear = diffs.astype(float)
pred_log    = np.log(diffs + 1)

# Confirm Spearman equivalence
spr_lin = spearmanr(pred_linear, pred_log).statistic
print(f'\nSpearman r(linear, log) = {spr_lin:.6f}  '
      f'{"(identical ranks — Spearman cannot distinguish them)" if abs(spr_lin) > 0.9999 else ""}')

print(f'\n{"Layer":>6}  {"Pear-lin":>9}  {"Pear-log":>9}  '
      f'{"R²-lin":>7}  {"R²-log":>7}  {"winner":>8}')
print('-' * 58)

results = []
for L in range(N_LAYERS):
    vecs = np.array(vecs_by_layer[L])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vn = vecs / norms
    cos_sim = vn @ vn.T
    neural = np.array([1.0 - cos_sim[i, j] for i, j in pairs])

    r_lin, _ = pearsonr(neural, pred_linear)
    r_log, _ = pearsonr(neural, pred_log)

    # OLS R²
    def r2(x, y):
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        ss_res = ((y - reg.predict(x.reshape(-1, 1)))**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    r2_lin = r2(pred_linear, neural)
    r2_log = r2(pred_log,    neural)

    winner = 'log' if r_log > r_lin else 'linear'
    marker = ' ◄' if abs(r_log - r_lin) > 0.01 else '  ~'
    results.append((L, r_lin, r_log, r2_lin, r2_log, winner))
    print(f'  L{L:>2}   {r_lin:+.4f}    {r_log:+.4f}    {r2_lin:.4f}    {r2_log:.4f}   {winner}{marker}')

print()
log_wins    = sum(1 for _, rl, rlog, *_ in results if rlog > rl + 0.01)
linear_wins = sum(1 for _, rl, rlog, *_ in results if rl  > rlog + 0.01)
ties        = N_LAYERS - log_wins - linear_wins

print(f'Log wins:    {log_wins}/{N_LAYERS} layers')
print(f'Linear wins: {linear_wins}/{N_LAYERS} layers')
print(f'Ties (<0.01 diff): {ties}/{N_LAYERS} layers')

best = max(results, key=lambda x: max(x[1], x[2]))
print(f'\nBest layer overall: L{best[0]}  '
      f'Pearson lin={best[1]:+.4f}  log={best[2]:+.4f}  '
      f'R²-lin={best[3]:.4f}  R²-log={best[4]:.4f}')
print()
print('Interpretation:')
print('  log > linear  → Weber encoding (compressed, equal ratios feel equal)')
print('  linear > log  → linear clock encoding (equal intervals feel equal)')
print('  ~tie           → both fit equally (Pearson cannot distinguish at this layer)')
