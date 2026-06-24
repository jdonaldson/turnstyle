"""Addition control for the log-multiplication hypothesis.

Mul result (times_table_log_scaling.py): log(a*b+1) decodes 1-3pp above raw
a*b at every layer; peaks at L2 (+0.987 vs +0.969).

This script forwards all 100 `a+b=` prompts through the same checkpoint,
captures the `=`-token hidden state at every layer, and re-runs the same
log-vs-raw Ridge-R² comparison. Three outcomes:

  (A) raw `a+b` decodes >= its log at every layer
      → log-bias is multiplication-specific. The model implements `×` in
        log space and `+` linearly. Strong evidence for log-add architecture.

  (B) log(a+b+1) decodes 1-3pp above raw `a+b` everywhere
      → log-bias is generic. Either CE-loss naturally produces log-aligned
        intermediate features, or it's a regression-difficulty artifact
        from log compressing the target range.

  (C) Mixed: log wins at some layers, raw at others
      → ambiguous; deeper testing needed.

Also: replicate the [h·d_a, h·d_b] → target test from times_table_log_add.py
on the addition prompts. If the addition target's regression learns
asymmetric or non-additive weights, the log-add symmetric-pair signature
is product-specific.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import encode, load  # noqa: E402

ADD_STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
              / "hidden_states_add.npz")
MUL_STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
              / "hidden_states.npz")


@torch.no_grad()
def collect_add_states(model, device: str = "cpu") -> dict:
    rows = []
    for a in range(10):
        for b in range(10):
            prompt = f"{a}+{b}="
            idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            _, _, states = model(idx, return_states=True)
            eq_pos = len(prompt) - 1
            for layer, h in enumerate(states):
                vec = h[0, eq_pos, :].cpu().numpy()
                rows.append((a, b, a + b, layer, vec))

    a_arr = np.array([r[0] for r in rows], dtype=np.int32)
    b_arr = np.array([r[1] for r in rows], dtype=np.int32)
    s_arr = np.array([r[2] for r in rows], dtype=np.int32)
    l_arr = np.array([r[3] for r in rows], dtype=np.int32)
    H = np.stack([r[4] for r in rows])
    np.savez(ADD_STATES, a=a_arr, b=b_arr, sum=s_arr, layer=l_arr, H=H)
    return {"a": a_arr, "b": b_arr, "sum": s_arr, "layer": l_arr, "H": H}


def cv_r2(X, y, alpha=1.0, n_splits=5, seeds=3):
    r2s = []
    for seed in range(seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            s = StandardScaler().fit(X[tr])
            m = Ridge(alpha=alpha, fit_intercept=True).fit(s.transform(X[tr]),
                                                           y[tr])
            r2s.append(m.score(s.transform(X[te]), y[te]))
    return float(np.mean(r2s))


def cv_r2_feature(F, y, alpha=1.0, n_splits=5, seeds=3):
    if F.ndim == 1:
        F = F[:, None]
    r2s = []
    for seed in range(seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(F):
            m = Ridge(alpha=alpha, fit_intercept=True).fit(F[tr], y[tr])
            r2s.append(m.score(F[te], y[te]))
    return float(np.mean(r2s))


def fit_dir(H, y, alpha=1.0):
    s = StandardScaler().fit(H)
    m = Ridge(alpha=alpha, fit_intercept=True).fit(s.transform(H), y)
    return m.coef_.copy(), s


def project(H, coef, scaler):
    return (H - scaler.mean_) / scaler.scale_ @ coef


def cos(u, v):
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    return float(np.dot(u, v) / (nu * nv)) if nu and nv else float("nan")


def report_block(label, data, target_key, model_target_fn):
    """Run log-vs-raw Ridge R² comparison on either + or * data."""
    a = data["a"]; b = data["b"]; tgt = data[target_key]; l_arr = data["layer"]
    H_all = data["H"]
    n_layers = int(l_arr.max()) + 1

    print(f"\n=== {label} ===")
    print(f"Per-layer R² for log vs raw targets:")
    targets = {
        f"{model_target_fn} (raw)":       tgt.astype(float),
        f"log({model_target_fn} + 1)":    np.log(tgt.astype(float) + 1),
        "a + b":                          (a + b).astype(float),
        "log(a + 1)":                     np.log(a.astype(float) + 1),
        "log(b + 1)":                     np.log(b.astype(float) + 1),
        "log(a + 1) + log(b + 1)":
            np.log(a.astype(float) + 1) + np.log(b.astype(float) + 1),
    }
    header = "  ".join(f"L{l}".center(7) for l in range(n_layers))
    print(f"  {'feature':30}  {header}")
    for name, y in targets.items():
        per = []
        for l in range(n_layers):
            m = l_arr == l
            per.append(cv_r2(H_all[m], y[m]))
        cells = "  ".join(f"{v:+.3f}".center(7) for v in per)
        print(f"  {name:30}  {cells}")

    # log-add direction test on the target
    print(f"\n  Log-add direction test on `{model_target_fn}`:")
    print(f"  {'L':>3}  {'cos(d_t, d_a+d_b)':>20}  "
          f"{'h·(d_a+d_b) R²':>16}  {'[h·d_a, h·d_b] w_a':>20}  "
          f"{'w_b':>8}")
    for l in range(n_layers):
        m = l_arr == l
        H = H_all[m]
        y_la = np.log(a[m].astype(float) + 1)
        y_lb = np.log(b[m].astype(float) + 1)
        y_log_target = np.log(tgt[m].astype(float) + 1)
        d_a, s_a = fit_dir(H, y_la)
        d_b, s_b = fit_dir(H, y_lb)
        d_t, _ = fit_dir(H, y_log_target)
        f_a = project(H, d_a, s_a)
        f_b = project(H, d_b, s_b)
        r2_sum = cv_r2_feature(f_a + f_b, y_log_target)
        F = np.stack([f_a, f_b], axis=1)
        m_ridge = Ridge(alpha=1.0, fit_intercept=True).fit(F, y_log_target)
        print(f"  L{l}  "
              f"{cos(d_t, d_a + d_b):20.3f}  "
              f"{r2_sum:+16.3f}  "
              f"{m_ridge.coef_[0]:+20.3f}  "
              f"{m_ridge.coef_[1]:+8.3f}")


def main():
    model = load("cpu")
    if not ADD_STATES.exists():
        print(f"Collecting addition states → {ADD_STATES}")
        add = collect_add_states(model)
    else:
        print(f"Loading cached addition states from {ADD_STATES}")
        d = np.load(ADD_STATES)
        add = {k: d[k] for k in d.files}
    mul = np.load(MUL_STATES)
    mul = {k: mul[k] for k in mul.files}

    # Quick sanity: confirm trained model still does + correctly.
    print("\nSanity: per-pair logit-lens accuracy at L5 for `+`:")
    H = torch.from_numpy(add["H"]).float()
    l_arr = add["layer"]
    eq_l5 = l_arr == 5
    with torch.no_grad():
        logits = model.head(model.ln_f(H[eq_l5]))
        probs = torch.softmax(logits, dim=-1).numpy()
    from times_table_trace import STOI
    s = add["sum"][eq_l5]
    true_ids = np.array([STOI[str(int(v))[0]] for v in s])
    acc = (probs.argmax(axis=1) == true_ids).mean()
    print(f"  L5 first-digit accuracy on a+b: {acc:.1%}")

    report_block("MULTIPLICATION (a*b)", mul, "product", "a*b")
    report_block("ADDITION (a+b)", add, "sum", "a+b")


if __name__ == "__main__":
    main()
