"""Route-type probe sweep — SGD classifier on last-token hidden states.

Finding: L1 classifies 5 route types with 100% 5-fold CV and 92-100% LOO.
Signal at L1 = structural/syntactic, not semantic. Replaces keyword routing.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import torch
from collections import defaultdict
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common import load_model
from swollm.bench.bbh import load_task

N_EX = 100

ROUTE_TASKS = {
    'COMP':  ['logical_deduction_three_objects',
              'logical_deduction_five_objects',
              'logical_deduction_seven_objects'],
    'TRACK': ['tracking_shuffled_objects_three_objects',
              'tracking_shuffled_objects_five_objects',
              'tracking_shuffled_objects_seven_objects'],
    'TRUTH': ['web_of_lies'],
    'TABLE': ['penguins_in_a_table', 'object_counting',
              'reasoning_about_colored_objects'],
    'NAV':   ['navigate'],
}


def fit_predict(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42,
                        alpha=1e-4, n_jobs=1, tol=1e-4)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)


print('Loading model…', flush=True)
tok, mdl, device = load_model()
N_LAYERS = mdl.config.num_hidden_layers + 1

print('Collecting hidden states…', flush=True)
X_by_layer = defaultdict(list)
labels, groups = [], []

for route, tasks in ROUTE_TASKS.items():
    for task in tasks:
        examples = load_task(task)[:N_EX]
        for ex in examples:
            text = tok.apply_chat_template(
                [{'role': 'user', 'content': ex['input']}],
                tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors='pt').to(device)
            with torch.no_grad():
                out = mdl(**inputs, output_hidden_states=True)
            for L in range(N_LAYERS):
                h = out.hidden_states[L][0, -1].float().cpu().numpy()
                X_by_layer[L].append(h)
            labels.append(route)
            groups.append(task)
        print(f'  {task}', flush=True)

le = LabelEncoder()
y = le.fit_transform(labels)
groups_arr = np.array(groups)
n_classes = len(le.classes_)
print(f'\nClasses: {list(le.classes_)}  chance={1/n_classes:.2f}  N={len(y)}\n', flush=True)

# ── 5-fold CV ─────────────────────────────────────────────────────────
print('5-fold CV sweep…', flush=True)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accs = []
for L in range(N_LAYERS):
    X = np.array(X_by_layer[L])
    correct = 0
    for tr, te in kf.split(X, y):
        preds = fit_predict(X[tr], y[tr], X[te])
        correct += (preds == y[te]).sum()
    cv_accs.append(correct / len(y))
    if L % 4 == 0 or L == N_LAYERS - 1:
        print(f'  L{L:>2}  {cv_accs[-1]:.3f}', flush=True)

peak_L = int(np.argmax(cv_accs))
print('\nFull curve:')
for L, a in enumerate(cv_accs):
    bar = '█' * int(a * 40)
    mk = ' ← peak' if L == peak_L else ''
    print(f'  L{L:>2}  {a:.3f}  {bar}{mk}')

# ── leave-one-task-out ────────────────────────────────────────────────
print(f'\nLeave-one-task-out at L{peak_L}:', flush=True)
X_peak = np.array(X_by_layer[peak_L])
loto_c, loto_n = defaultdict(int), defaultdict(int)
for tr, te in LeaveOneGroupOut().split(X_peak, y, groups_arr):
    preds = fit_predict(X_peak[tr], y[tr], X_peak[te])
    for pred, true, grp in zip(preds, y[te], groups_arr[te]):
        loto_n[grp] += 1
        if pred == true: loto_c[grp] += 1

for route, tasks in ROUTE_TASKS.items():
    for task in tasks:
        n, c = loto_n[task], loto_c[task]
        pct = 100*c/n if n else 0
        bar = '█' * int(pct/5)
        print(f'  {task:<52}  {route:<6}  {c}/{n} ({pct:.0f}%)  {bar}')

oc = sum(loto_c.values()); on = sum(loto_n.values())
print(f'\n  Overall LOO: {oc}/{on} ({100*oc/on:.1f}%)  chance {100/n_classes:.1f}%')
