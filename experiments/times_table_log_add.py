"""Is multiplication implemented as log-feature addition?

If the network truly computes log(a*b) = log(a) + log(b) in feature space,
then at the layer where log(a*b+1) is most decodable, the regression
direction d_ab for log(a*b+1) should approximately equal the SUM of the
regression directions d_a and d_b for log(a+1) and log(b+1).

Tests at each layer:
  1. cos(d_ab, d_a + d_b)    — should be high if log-add hypothesis true
  2. cos(d_ab, d_a - d_b)    — control: sign-flipped sum (should be lower)
  3. cos(d_ab, d_a), (d_b)   — control: just one operand
  4. R² of (h·d_a + h·d_b) vs log(a*b+1)
     — if the log-product representation is essentially the sum of two
       log-operand projections, this single 1D feature should reach the
       full-hidden-state R² ceiling.
  5. R² of (h·d_a) and (h·d_b) separately, for comparison.

Also fit a 2-feature regression [h·d_a, h·d_b] → log(a*b+1) and inspect
the learned weights: if they're ≈ [1, 1] (or any constant equal pair),
the model is symmetric-summing in log space.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import load  # noqa: E402

STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
          / "hidden_states.npz")


def cos(u, v):
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def fit_dir(H, y, alpha=1.0):
    """Fit a Ridge probe, return (coef, scaler).

    Coefficients live in standardized feature space; project new H via
    `(H - scaler.mean_) / scaler.scale_ @ coef`.
    """
    s = StandardScaler().fit(H)
    Hs = s.transform(H)
    m = Ridge(alpha=alpha, fit_intercept=True).fit(Hs, y)
    return m.coef_.copy(), s, float(m.intercept_)


def project(H, coef, scaler):
    Hs = (H - scaler.mean_) / scaler.scale_
    return Hs @ coef


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def cv_r2_full(H, y, alpha=1.0, n_splits=5, seeds=3):
    r2s = []
    for seed in range(seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(H):
            s = StandardScaler().fit(H[tr])
            m = Ridge(alpha=alpha, fit_intercept=True).fit(s.transform(H[tr]),
                                                            y[tr])
            r2s.append(m.score(s.transform(H[te]), y[te]))
    return float(np.mean(r2s))


def cv_r2_features(F, y, alpha=1.0, n_splits=5, seeds=3):
    """5-fold CV R² when fitting a Ridge on 1-D or 2-D feature(s) F."""
    if F.ndim == 1:
        F = F[:, None]
    r2s = []
    for seed in range(seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(F):
            m = Ridge(alpha=alpha, fit_intercept=True).fit(F[tr], y[tr])
            r2s.append(m.score(F[te], y[te]))
    return float(np.mean(r2s))


def main():
    _ = load("cpu")  # checkpoint exists; we don't need the model itself here
    data = np.load(STATES)
    a = data["a"]; b = data["b"]; prod = data["product"]; l_arr = data["layer"]
    H_all = data["H"]
    n_layers = int(l_arr.max()) + 1

    print("Sanity: log(a+1) + log(b+1) vs log(a*b+1) over 100 pairs")
    a0 = a[l_arr == 0]; b0 = b[l_arr == 0]; p0 = prod[l_arr == 0]
    y_la0 = np.log(a0.astype(float) + 1)
    y_lb0 = np.log(b0.astype(float) + 1)
    y_lab0 = np.log(p0.astype(float) + 1)
    print(f"  Pearson r:        {np.corrcoef(y_la0 + y_lb0, y_lab0)[0,1]:.4f}")
    print(f"  Note: equal when a, b > 0; differ when either is 0.")
    print()

    print(f"{'L':>3}  {'cos(d_ab, d_a+d_b)':>20}  {'cos(d_a, d_b)':>14}  "
          f"{'cos(d_ab,d_a-d_b)':>18}  {'cos(d_ab,d_a)':>14}  "
          f"{'cos(d_ab,d_b)':>14}")
    print("-" * 100)
    saved = {}
    for l in range(n_layers):
        m = l_arr == l
        H = H_all[m]
        y_la = np.log(a[m].astype(float) + 1)
        y_lb = np.log(b[m].astype(float) + 1)
        y_lab = np.log(prod[m].astype(float) + 1)
        d_a,  s_a,  _ = fit_dir(H, y_la)
        d_b,  s_b,  _ = fit_dir(H, y_lb)
        d_ab, s_ab, _ = fit_dir(H, y_lab)
        d_sum = d_a + d_b
        print(f"  L{l}  "
              f"{cos(d_ab, d_sum):20.3f}  "
              f"{cos(d_a, d_b):14.3f}  "
              f"{cos(d_ab, d_a - d_b):18.3f}  "
              f"{cos(d_ab, d_a):14.3f}  "
              f"{cos(d_ab, d_b):14.3f}")
        saved[l] = (H, d_a, s_a, d_b, s_b, d_ab, s_ab, y_la, y_lb, y_lab)

    print()
    print("1-D feature R² ceiling check: predict log(a*b+1) from a single")
    print("projection h·d (5-fold CV, 3 seeds).")
    print(f"{'L':>3}  {'full H':>8}  {'h·d_a':>8}  {'h·d_b':>8}  "
          f"{'h·(d_a+d_b)':>13}  {'[h·d_a, h·d_b]':>16}  {'h·d_ab':>8}")
    print("-" * 80)
    for l in range(n_layers):
        H, d_a, s_a, d_b, s_b, d_ab, s_ab, _y_la, _y_lb, y_lab = saved[l]
        f_a = project(H, d_a, s_a)
        f_b = project(H, d_b, s_b)
        f_ab = project(H, d_ab, s_ab)
        F2 = np.stack([f_a, f_b], axis=1)
        r2_full   = cv_r2_full(H, y_lab)
        r2_fa     = cv_r2_features(f_a,  y_lab)
        r2_fb     = cv_r2_features(f_b,  y_lab)
        r2_fsum   = cv_r2_features(f_a + f_b, y_lab)
        r2_f2     = cv_r2_features(F2, y_lab)
        r2_fab    = cv_r2_features(f_ab, y_lab)
        print(f"  L{l}  {r2_full:+8.3f}  {r2_fa:+8.3f}  {r2_fb:+8.3f}  "
              f"{r2_fsum:+13.3f}  {r2_f2:+16.3f}  {r2_fab:+8.3f}")

    print()
    print("[h·d_a, h·d_b] → log(a*b+1) learned coefficients (5-fold avg, no CV):")
    print(f"{'L':>3}  {'w_a':>8}  {'w_b':>8}  {'bias':>8}")
    for l in range(n_layers):
        H, d_a, s_a, d_b, s_b, _, _, _, _, y_lab = saved[l]
        f_a = project(H, d_a, s_a)
        f_b = project(H, d_b, s_b)
        F = np.stack([f_a, f_b], axis=1)
        m = Ridge(alpha=1.0, fit_intercept=True).fit(F, y_lab)
        print(f"  L{l}  {m.coef_[0]:+8.3f}  {m.coef_[1]:+8.3f}  "
              f"{m.intercept_:+8.3f}")


if __name__ == "__main__":
    main()
