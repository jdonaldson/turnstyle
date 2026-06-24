#!/usr/bin/env python3
"""Weber's Law sweep across semantic domains.

Tests whether SmolLM2's representational geometry is log-compressed (Weber/Fechner)
or linear — and whether this varies by domain.

Method: fit power-law exponent α to pairwise RSA:
    neural_cosine_distance ≈ |magnitude_i − magnitude_j|^α

  α ≪ 1  →  strong Weber / log compression
  α ≈ 1  →  linear encoding
  α > 1  →  super-linear (large gaps over-represented)

Critical control: bare integers — if all domains match this, it is generic
numeric encoding. Domain-specific deviation is the interesting signal.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/weber_sweep.py
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from common import load_model

print('Loading model…', flush=True)
tok, mdl, device = load_model()
N_LAYERS = mdl.config.num_hidden_layers + 1


# ── helpers ──────────────────────────────────────────────────────────────────

def find_last_occurrence(token_ids: list[int], surface: str) -> int | None:
    """Find position of last subtoken of surface string in token_ids."""
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


def fit_alpha(diffs: np.ndarray, neural: np.ndarray) -> tuple[float, float]:
    """Power-law α via log-log OLS: log(neural) ~ α·log(diff).
    Returns (α, R²_of_log_log_fit)."""
    mask = (diffs > 0) & (neural > 1e-8)
    if mask.sum() < 5:
        return float('nan'), float('nan')
    log_d = np.log(diffs[mask])
    log_n = np.log(neural[mask])
    reg = LinearRegression().fit(log_d.reshape(-1, 1), log_n)
    alpha = float(reg.coef_[0])
    ss_res = ((log_n - reg.predict(log_d.reshape(-1, 1))) ** 2).sum()
    ss_tot = ((log_n - log_n.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return alpha, r2


def scale_label(alpha: float) -> str:
    if np.isnan(alpha):
        return '?'
    if alpha < 0.8:
        return 'Weber'
    if alpha > 1.2:
        return 'super'
    return 'linear'


# ── domain definitions ────────────────────────────────────────────────────────
# Each domain: name, prompt template, list of (surface_string, magnitude).
# surface_string is what appears in the prompt AND what we search for as a token.
# For prices, surface is the numeric part (not "$") since "$" is a separate token.

DOMAINS = [
    dict(
        name='integers',
        template='The value is {surface}.',
        stimuli=[(str(n), float(n)) for n in
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100]],
    ),
    dict(
        name='durations',
        template='The task took {surface} hours to complete.',
        stimuli=[(str(n), float(n)) for n in
                 [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]],
    ),
    dict(
        name='distances',
        template='The city is {surface} miles away.',
        stimuli=[(str(n), float(n)) for n in
                 [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]],
    ),
    dict(
        name='prices',
        # Surface is numeric part; "$" tokenizes separately, we want the number token.
        template='The item costs ${surface}.',
        stimuli=[(str(n), float(n)) for n in
                 [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]],
    ),
    dict(
        name='temperatures',
        template='The temperature outside is {surface} degrees Fahrenheit.',
        stimuli=[(str(n), float(n)) for n in
                 [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]],
    ),
    dict(
        name='years',
        template='The year is {surface}.',
        stimuli=[(str(n), float(n)) for n in
                 [1900, 1920, 1940, 1960, 1970, 1980, 1990, 2000, 2005,
                  2010, 2015, 2020, 2024]],
    ),
]


# ── main sweep ────────────────────────────────────────────────────────────────

domain_results = []

for domain in DOMAINS:
    name     = domain['name']
    template = domain['template']
    stimuli  = domain['stimuli']

    print(f'\n{"=" * 62}')
    print(f'Domain: {name}  [{len(stimuli)} stimuli]')
    print('=' * 62)

    mags: list[float] = []
    vecs_by_layer: dict[int, list[np.ndarray]] = {L: [] for L in range(N_LAYERS)}
    skipped = 0

    for surface, magnitude in stimuli:
        prompt = template.format(surface=surface)
        inputs = tok(prompt, return_tensors='pt').to(device)
        token_ids = inputs['input_ids'][0].tolist()
        pos = find_last_occurrence(token_ids, surface)
        if pos is None:
            print(f'  WARNING: token not found for {surface!r} — skipping', flush=True)
            skipped += 1
            continue
        with torch.no_grad():
            out = mdl(**inputs, output_hidden_states=True)
        for L in range(N_LAYERS):
            vecs_by_layer[L].append(out.hidden_states[L][0, pos].float().cpu().numpy())
        mags.append(magnitude)
        print(f'  {surface:>6}  mag={magnitude:8.1f}  pos={pos}', flush=True)

    n = len(mags)
    if skipped:
        print(f'  ({skipped} stimuli skipped)')
    if n < 6:
        print(f'  Too few stimuli ({n}), skipping domain.')
        continue

    # Pairwise magnitude differences (constant across layers)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    diffs = np.array([abs(mags[i] - mags[j]) for i, j in pairs])

    # Layer sweep
    print(f'\n  Layer sweep (power-law α, log-log R²):')
    print(f'  {"L":>3}   {"α":>6}   {"R²":>5}   scale')
    print(f'  {"-" * 32}')

    layer_data: list[tuple] = []
    for L in range(N_LAYERS):
        vecs  = np.array(vecs_by_layer[L])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        vn    = vecs / norms
        cos   = vn @ vn.T
        neural = np.array([1.0 - cos[i, j] for i, j in pairs])

        alpha, r2 = fit_alpha(diffs, neural)
        label = scale_label(alpha)
        marker = ' ◄' if not np.isnan(alpha) and abs(alpha - 1.0) > 0.2 else ''
        alpha_str = f'{alpha:+.3f}' if not np.isnan(alpha) else '   nan'
        r2_str    = f'{r2:.3f}'     if not np.isnan(r2)    else '  nan'
        print(f'  L{L:>2}   {alpha_str}   {r2_str}   {label}{marker}', flush=True)
        layer_data.append((L, alpha, r2, label, neural))

    # Best layer by log-log R²
    valid = [(L, a, r2, lbl, neu) for L, a, r2, lbl, neu in layer_data
             if not np.isnan(r2)]
    if not valid:
        continue
    bL, ba, br2, blbl, bneu = max(valid, key=lambda x: x[2])
    r_lin = pearsonr(bneu, diffs.astype(float)).statistic
    r_log = pearsonr(bneu, np.log(diffs + 1)).statistic
    print(f'\n  Best: L{bL}  α={ba:+.3f}  R²={br2:.3f}  ({blbl})')
    print(f'        Pearson lin={r_lin:+.3f}  log={r_log:+.3f}')

    domain_results.append(dict(
        name=name, n=n, best_layer=bL, alpha=ba, r2=br2,
        scale=blbl, r_lin=r_lin, r_log=r_log,
    ))


# ── cross-domain summary ──────────────────────────────────────────────────────

print(f'\n\n{"=" * 72}')
print('CROSS-DOMAIN SUMMARY')
print('=' * 72)
print(f'  {"Domain":<16}  {"n":>3}  {"BestL":>5}  {"α":>7}  '
      f'{"R²":>5}  {"Plin":>6}  {"Plog":>6}  Scale')
print(f'  {"-" * 65}')
for r in domain_results:
    print(f'  {r["name"]:<16}  {r["n"]:>3}  {r["best_layer"]:>5}  '
          f'{r["alpha"]:>+7.3f}  {r["r2"]:>5.3f}  '
          f'{r["r_lin"]:>+6.3f}  {r["r_log"]:>+6.3f}  {r["scale"]}')

print()
print('α interpretation:')
print('  α < 0.8  → Weber / log compression  (equal ratios feel equal)')
print('  α ≈ 1.0  → linear                  (equal intervals feel equal)')
print('  α > 1.2  → super-linear            (large gaps over-represented)')
print()
print('Domain-specific deviation from the integers (control) baseline')
print('is the signal — matching control = generic numeric encoding.')
