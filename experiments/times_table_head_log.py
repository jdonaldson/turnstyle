"""Does the head's digit-token readout have log-keyed structure?

Prior: in `times_table_digit_embeddings.py` we showed that the digit rows
'0'..'9' of W_E (= W_head, since weights tied) are LINEAR-spaced in PCA-3
arc length (r=0.998 vs r=0.956 for log). So the unembed matrix's digit
axis is linear.

But the L1/L2 internal representation prefers log for multiplication.
Somewhere between L2 and L5 the model must convert log magnitude → linear
digit token. This script asks: what does that conversion look like at L5?

Tests:
  (1) Fit a Ridge probe H_L5 → log(product+1). Get direction d_log in
      residual space. Apply head: digit_logits = head[digit_ids] @ d_log.
      → How does each digit token's logit respond to a unit of log magnitude?

  (2) Same with raw product target, get d_raw. Compare.

  (3) For each (a, b) pair, project h_L5 onto d_log → log_score. Plot
      head_digit_d's logit vs this log_score, partitioned by first_digit(a*b).
      → Does each digit's logit peak at the right log_score band?

  (4) Sanity: take the average h_L5 within each {a*b first-digit class}
      (10 classes), compute digit_logits via head @ ln_f(mean_h). Show
      the diagonal of the confusion matrix — should be near 1 if the
      model commits cleanly.

  (5) Direct readout check: head_digit @ d_log shows the slope of each
      digit-token's logit as log magnitude changes. If the model uses
      log-magnitude as the universal scalar, the slope-vs-digit-value
      curve should look like the inverse of "first digit of N", which is
      a discontinuous function — interesting if any structure emerges.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import STOI, ITOS, load  # noqa: E402

STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
          / "hidden_states.npz")


def fit_dir(H, y, alpha=1.0):
    s = StandardScaler().fit(H)
    m = Ridge(alpha=alpha, fit_intercept=True).fit(s.transform(H), y)
    return m.coef_.copy(), s, float(m.intercept_)


@torch.no_grad()
def main():
    model = load("cpu")
    data = np.load(STATES)
    a = data["a"]; b = data["b"]; prod = data["product"]; l_arr = data["layer"]
    H_all = data["H"]

    digit_ids = [STOI[str(d)] for d in range(10)]
    W = model.head.weight.detach().cpu().numpy()      # (vocab, 128)
    W_d = W[digit_ids]                                 # (10, 128)

    # ── (1) Direction in residual space for log(product+1) at L5 ──
    L = 5
    m = l_arr == L
    H_L5 = H_all[m]
    a_L5 = a[m]; b_L5 = b[m]; prod_L5 = prod[m]
    y_log = np.log(prod_L5.astype(float) + 1)
    y_raw = prod_L5.astype(float)

    d_log, s_log, _ = fit_dir(H_L5, y_log)
    d_raw, s_raw, _ = fit_dir(H_L5, y_raw)

    # The Ridge coef lives in standardized space. To get the residual-space
    # direction that controls log magnitude in the actual unscaled hidden
    # state, divide by per-feature std: residual-space direction = coef / scale.
    d_log_res = d_log / s_log.scale_
    d_raw_res = d_raw / s_raw.scale_

    # head[digit_ids] @ d_log_res gives the change in digit-token logit per
    # unit increase in d_log_res-projected residual stream.
    digit_logit_per_log = W_d @ d_log_res
    digit_logit_per_raw = W_d @ d_raw_res
    print("Digit-token logit slope per unit of `log(product+1)` direction "
          "(d_log_res) at L5:")
    print(f"  {'digit':>5}  {'logit/unit Δlog':>16}  {'logit/unit Δraw':>16}")
    for d in range(10):
        print(f"  {d:>5}  {digit_logit_per_log[d]:+16.3f}  "
              f"{digit_logit_per_raw[d]:+16.3f}")

    # ── (2) Look at correlation between digit-logit-slope and digit value /
    # log(digit+1) ──
    log_vals = np.log(np.arange(10) + 1)
    lin_vals = np.arange(10, dtype=float)
    r_lin_log_slope = float(np.corrcoef(digit_logit_per_log, lin_vals)[0, 1])
    r_log_log_slope = float(np.corrcoef(digit_logit_per_log, log_vals)[0, 1])
    r_lin_raw_slope = float(np.corrcoef(digit_logit_per_raw, lin_vals)[0, 1])
    r_log_raw_slope = float(np.corrcoef(digit_logit_per_raw, log_vals)[0, 1])
    print()
    print("Correlation of digit-slope with digit value:")
    print(f"  slope-from-d_log   vs digit (linear): r = {r_lin_log_slope:+.3f}")
    print(f"  slope-from-d_log   vs log(digit+1):    r = {r_log_log_slope:+.3f}")
    print(f"  slope-from-d_raw   vs digit (linear): r = {r_lin_raw_slope:+.3f}")
    print(f"  slope-from-d_raw   vs log(digit+1):    r = {r_log_raw_slope:+.3f}")

    # ── (3) Verify L5 hits the right digit per pair (sanity) ──
    H = torch.from_numpy(H_L5).float()
    logits_L5 = model.head(model.ln_f(H)).detach().cpu().numpy()
    digit_logits_L5 = logits_L5[:, digit_ids]  # (100, 10)
    true_first_digit = np.array([int(str(int(p))[0]) for p in prod_L5])
    pred_first_digit = digit_logits_L5.argmax(axis=1)
    acc = float((pred_first_digit == true_first_digit).mean())
    print(f"\nL5 first-digit accuracy (sanity): {acc:.1%}")

    # ── (4) Mean h_L5 per first-digit class, project through head ──
    print()
    print("Mean digit-logit profile per first-digit class:")
    print(f"  {'class':>5}  {'n':>3}  " +
          "  ".join(f"d={d}".center(7) for d in range(10)) +
          "  argmax")
    for first in range(10):
        m = true_first_digit == first
        n = int(m.sum())
        if n == 0:
            continue
        mean_h = torch.from_numpy(H_L5[m].mean(axis=0)).float()[None, :]
        digit_logit = model.head(model.ln_f(mean_h)).detach().cpu().numpy()
        dl = digit_logit[0, digit_ids]
        cells = "  ".join(f"{v:+.2f}".center(7) for v in dl)
        am = int(dl.argmax())
        print(f"  {first:>5}  {n:>3}  {cells}  d={am}")

    # ── (5) For each (a, b), project h_L5 onto d_log_res → log_score.
    # Plot relationship with digit logits, see if curves are linear or
    # piecewise. ──
    proj_log = (H_L5 @ d_log_res) - (s_log.mean_ / s_log.scale_) @ d_log
    # Actually simpler:
    proj_log = ((H_L5 - s_log.mean_) / s_log.scale_) @ d_log
    print()
    print("L5 d_log projection vs log(product+1) Pearson check:")
    r = float(np.corrcoef(proj_log, y_log)[0, 1])
    print(f"  r(proj, log(p+1)) = {r:.3f}  (Ridge in-sample fit quality)")

    # Group by first digit, average projection
    print()
    print("Mean d_log projection by first-digit class:")
    print(f"  {'class':>5}  {'n':>3}  {'mean proj':>10}  "
          f"{'mean log(p+1)':>13}")
    for first in range(10):
        m = true_first_digit == first
        if m.sum() == 0:
            continue
        print(f"  {first:>5}  {int(m.sum()):>3}  "
              f"{proj_log[m].mean():+10.3f}  {y_log[m].mean():+13.3f}")

    # ── (6) Compare d_log direction to centroid-of-digit-rows ──
    centroid_digits = W_d.mean(axis=0)
    print()
    print("Geometry of d_log_res vs digit-row geometry:")
    print(f"  cos(d_log_res, centroid_digit_row): "
          f"{float(np.dot(d_log_res, centroid_digits) / (np.linalg.norm(d_log_res) * np.linalg.norm(centroid_digits))):+.3f}")
    # Direction along which digit rows vary the most: PC1 of (W_d - centroid)
    W_d_centered = W_d - centroid_digits
    U, S, Vt = np.linalg.svd(W_d_centered, full_matrices=False)
    pc1 = Vt[0]
    print(f"  cos(d_log_res, PC1_digit_rows):     "
          f"{float(np.dot(d_log_res, pc1) / (np.linalg.norm(d_log_res) * np.linalg.norm(pc1))):+.3f}")
    proj_pc1 = W_d_centered @ pc1   # how digits 0..9 project on PC1
    r_pc1_lin = float(np.corrcoef(proj_pc1, lin_vals)[0, 1])
    r_pc1_log = float(np.corrcoef(proj_pc1, log_vals)[0, 1])
    print(f"  PC1 of digit rows vs digit (linear):   r = {r_pc1_lin:+.3f}")
    print(f"  PC1 of digit rows vs log(digit+1):    r = {r_pc1_log:+.3f}")


if __name__ == "__main__":
    main()
