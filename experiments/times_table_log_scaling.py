"""Does log scaling reveal hidden structure?

Three angles:
  1. Log-probability margin per layer — is the L4→L5 jump more dramatic
     in log space than in linear space?
  2. Commit layer vs log(product) — does difficulty scale with log magnitude?
  3. Decodability of log(a), log(b), log(a*b) vs their raw versions — does
     the network internally use log-scale features (i.e., implement
     multiplication as log-add)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import STOI, load  # noqa: E402

STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
          / "hidden_states.npz")


def cv_r2(X, y):
    r2s = []
    for seed in range(3):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            m.fit(X[tr], y[tr])
            r2s.append(m.score(X[te], y[te]))
    return float(np.mean(r2s))


def main():
    model = load("cpu")
    data = np.load(STATES)
    a = data["a"]; b = data["b"]; prod = data["product"]; l_arr = data["layer"]
    H_np = data["H"]
    H = torch.from_numpy(H_np).float()
    n_layers = int(l_arr.max()) + 1

    # ─── (1) Log-probability margin ───
    with torch.no_grad():
        logits = model.head(model.ln_f(H))
        probs = torch.softmax(logits, dim=-1).numpy()
    true_ids = np.array([STOI[str(int(p))[0]] for p in prod])
    p_correct = probs[np.arange(len(probs)), true_ids]
    p_copy = probs.copy()
    p_copy[np.arange(len(probs)), true_ids] = 0
    p_top_wrong = p_copy.max(axis=1)

    log_p_correct = np.log(p_correct + 1e-10)
    log_p_top_wrong = np.log(p_top_wrong + 1e-10)
    log_margin = log_p_correct - log_p_top_wrong

    print("Linear vs log probability per layer:")
    print(f"{'L':>3}  {'P(corr)':>8}  {'logP(corr)':>10}  "
          f"{'P(top wrong)':>12}  {'logP(top wr)':>12}  "
          f"{'lin margin':>10}  {'log margin':>10}")
    for l in range(n_layers):
        m = l_arr == l
        print(f"  L{l}  {p_correct[m].mean():8.3f}  "
              f"{log_p_correct[m].mean():10.3f}  "
              f"{p_top_wrong[m].mean():12.3f}  "
              f"{log_p_top_wrong[m].mean():12.3f}  "
              f"{(p_correct[m] - p_top_wrong[m]).mean():+10.3f}  "
              f"{log_margin[m].mean():+10.3f}")

    # ─── (2) Commit layer vs log(product) ───
    correct = probs.argmax(axis=1) == true_ids
    commit = np.full(100, -1, dtype=int)
    for a_v in range(10):
        for b_v in range(10):
            per_layer = []
            for l in range(n_layers):
                m = (l_arr == l) & (a == a_v) & (b == b_v)
                per_layer.append(bool(correct[m].item()))
            for l in range(n_layers):
                if all(per_layer[l:]):
                    commit[a_v * 10 + b_v] = l
                    break

    pair_prod = np.array([(a_v * b_v) for a_v in range(10)
                          for b_v in range(10)])
    pair_log_prod = np.log(pair_prod.astype(float) + 1)

    print()
    print("Commit layer vs log(product+1):")
    print(f"{'group':6}  n  {'mean log(p)':>12}  {'median':>8}  range")
    for l in range(n_layers):
        in_group = commit == l
        if in_group.sum() == 0:
            continue
        lps = pair_log_prod[in_group]
        print(f"  L{l}    {int(in_group.sum()):2d}  "
              f"{lps.mean():12.2f}  {np.median(lps):8.2f}  "
              f"[{lps.min():.2f}, {lps.max():.2f}]")

    # Correlation
    valid = commit >= 0
    r = float(np.corrcoef(commit[valid], pair_log_prod[valid])[0, 1])
    print(f"\nPearson correlation(commit_layer, log(product+1)): r = {r:.3f}")

    # ─── (3) Decodability of log vs raw features ───
    print()
    print("Per-layer R² for log vs raw targets:")
    targets = {
        "a * b (raw)":      prod.astype(float),
        "log(a * b + 1)":   np.log(prod.astype(float) + 1),
        "a + b":            (a + b).astype(float),
        "log(a + 1)":       np.log(a.astype(float) + 1),
        "log(b + 1)":       np.log(b.astype(float) + 1),
        "log(a + 1) + log(b + 1)":
            np.log(a.astype(float) + 1) + np.log(b.astype(float) + 1),
    }
    print(f"{'feature':28}  " +
          "  ".join(f"L{l}".center(6) for l in range(n_layers)))
    for name, y in targets.items():
        per = [cv_r2(H_np[l_arr == l], y[l_arr == l])
               for l in range(n_layers)]
        cells = "  ".join(f"{v:+.3f}" for v in per)
        print(f"{name:28}  {cells}")


if __name__ == "__main__":
    main()
