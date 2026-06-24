"""Three-operator analysis: subtraction control + dispatch localization.

Builds on `times_table_addition_control.py`:
  - mul prompts (cached): a*b=
  - add prompts (cached): a+b=
  - sub prompts (this run): a-b=

Two questions:

  (1) Does subtraction pick the linear branch (like +) or the log branch
      (like ×)?  Run the same log-vs-raw R² table on the sub states. For
      sub the target is signed (a-b), so we compare:
         raw (a-b)           — linear branch
         |a-b|               — magnitude only
         sign(a-b)·log(|a-b|+1)  — signed log
         log(|a-b|+1)            — magnitude log
      Prediction: if subtraction is "linear branch like +", raw (a-b)
      should beat sign·log everywhere.

  (2) Where does operator dispatch happen?  Two views:
        (a) Operator decodability: fit 3-way logistic regression on the
            `=`-token hidden state H_L → {*, +, -}.  At what layer can the
            operator be recovered with high accuracy?
        (b) Operator divergence: for each (a, b), compute cos similarity
            between H_op1[(a,b)] and H_op2[(a,b)] at each layer L.  If
            block 1 is the dispatcher, pairs should be very similar at L0
            (just digits + operator token) and diverge at L1 / L2.

  Bonus (c): is operator decodable BEFORE the `=` token?  Run a forward
      pass and check the operator token's own L0/L1/... state — sanity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import encode, load  # noqa: E402

DATA = Path(__file__).parent / "data" / "nanogpt_times_table"
SUB_STATES = DATA / "hidden_states_sub.npz"
ADD_STATES = DATA / "hidden_states_add.npz"
MUL_STATES = DATA / "hidden_states.npz"


@torch.no_grad()
def collect_states(model, op: str, device: str = "cpu"):
    rows = []
    for a in range(10):
        for b in range(10):
            prompt = f"{a}{op}{b}="
            idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            _, _, states = model(idx, return_states=True)
            eq_pos = len(prompt) - 1
            if op == "+":
                target = a + b
            elif op == "-":
                target = a - b
            else:
                target = a * b
            for layer, h in enumerate(states):
                vec = h[0, eq_pos, :].cpu().numpy()
                rows.append((a, b, target, layer, vec))
    a_arr = np.array([r[0] for r in rows], dtype=np.int32)
    b_arr = np.array([r[1] for r in rows], dtype=np.int32)
    t_arr = np.array([r[2] for r in rows], dtype=np.int32)
    l_arr = np.array([r[3] for r in rows], dtype=np.int32)
    H = np.stack([r[4] for r in rows])
    return {"a": a_arr, "b": b_arr, "target": t_arr, "layer": l_arr, "H": H}


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


def cv_acc(X, y, n_splits=5, seeds=3):
    accs = []
    for seed in range(seeds):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(X, y):
            s = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=2000, C=1.0).fit(s.transform(X[tr]),
                                                              y[tr])
            accs.append(m.score(s.transform(X[te]), y[te]))
    return float(np.mean(accs))


def report_log_vs_raw_sub(sub):
    a = sub["a"]; b = sub["b"]; tgt = sub["target"]; l_arr = sub["layer"]
    H_all = sub["H"]
    n_layers = int(l_arr.max()) + 1
    print("\n=== SUBTRACTION (a-b) — log vs raw decodability ===")
    targets = {
        "a-b (raw, signed)":           tgt.astype(float),
        "|a-b|":                       np.abs(tgt).astype(float),
        "sign(a-b)·log(|a-b|+1)":
            np.sign(tgt) * np.log(np.abs(tgt).astype(float) + 1),
        "log(|a-b|+1)":                np.log(np.abs(tgt).astype(float) + 1),
        "a + b":                       (a + b).astype(float),
        "log(a + 1)":                  np.log(a.astype(float) + 1),
        "log(b + 1)":                  np.log(b.astype(float) + 1),
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


def operator_decodability(mul, add, sub):
    """3-way logistic regression on H[L] → {0=*, 1=+, 2=-}.

    Stack 100 mul + 100 add + 100 sub per layer.
    """
    n_layers = int(mul["layer"].max()) + 1
    print("\n=== Operator decodability from =-token hidden state ===")
    print(f"  {'L':>3}  {'CV acc (3-way, chance=33.3%)':>30}")
    for l in range(n_layers):
        Hm = mul["H"][mul["layer"] == l]
        Ha = add["H"][add["layer"] == l]
        Hs = sub["H"][sub["layer"] == l]
        X = np.concatenate([Hm, Ha, Hs], axis=0)
        y = np.concatenate([np.zeros(len(Hm)),
                            np.ones(len(Ha)),
                            np.full(len(Hs), 2)])
        acc = cv_acc(X, y)
        print(f"  L{l}  {acc:30.3f}")


def cross_op_divergence(mul, add, sub):
    """Per (a,b) pair, cosine sim between H_op1 and H_op2 at each layer.

    If block 1 is the dispatcher, sim should drop sharply L0 → L1.
    """
    n_layers = int(mul["layer"].max()) + 1
    print("\n=== Hidden-state divergence across operators (paired by a,b) ===")
    print(f"  {'L':>3}  {'cos(mul, add)':>15}  {'cos(mul, sub)':>15}  "
          f"{'cos(add, sub)':>15}")
    for l in range(n_layers):
        Hm = mul["H"][mul["layer"] == l]
        Ha = add["H"][add["layer"] == l]
        Hs = sub["H"][sub["layer"] == l]
        # rows are in matching (a, b) order across all three (10×10 sweep)
        def avg_cos(X, Y):
            xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
            return float(np.mean(np.sum(xn * yn, axis=1)))
        print(f"  L{l}  {avg_cos(Hm, Ha):15.3f}  {avg_cos(Hm, Hs):15.3f}  "
              f"{avg_cos(Ha, Hs):15.3f}")


def operator_token_decodability(model, device="cpu"):
    """Sanity: forward "{a}{op}{b}=" and read out the OPERATOR token's
    own hidden state at every layer.  Fit 3-way LR.

    This tells us: is operator identity localized to the operator's
    position throughout, or does it 'spread' to the = position?
    """
    print("\n=== Operator decodability from operator-token hidden state ===")
    ops = ['*', '+', '-']
    rows_per_layer = {l: [] for l in range(6)}
    labels = []
    for op_id, op in enumerate(ops):
        for a in range(10):
            for b in range(10):
                prompt = f"{a}{op}{b}="
                idx = torch.tensor([encode(prompt)], dtype=torch.long,
                                   device=device)
                with torch.no_grad():
                    _, _, states = model(idx, return_states=True)
                op_pos = 1  # position of the operator token
                for layer, h in enumerate(states):
                    rows_per_layer[layer].append(
                        h[0, op_pos, :].cpu().numpy()
                    )
                labels.append(op_id)
    labels = np.array(labels)
    print(f"  {'L':>3}  {'CV acc (op-token, chance=33.3%)':>32}")
    for l in range(6):
        X = np.stack(rows_per_layer[l])
        acc = cv_acc(X, labels)
        print(f"  L{l}  {acc:32.3f}")


def main():
    model = load("cpu")
    if not SUB_STATES.exists():
        print(f"Collecting subtraction states → {SUB_STATES}")
        sub = collect_states(model, "-")
        np.savez(SUB_STATES, **{k: v if k != "target" else v
                                for k, v in sub.items()})
        # Re-label for consistency with mul/add files
        np.savez(SUB_STATES,
                 a=sub["a"], b=sub["b"], diff=sub["target"],
                 layer=sub["layer"], H=sub["H"])
    d = np.load(SUB_STATES)
    sub = {"a": d["a"], "b": d["b"], "target": d["diff"],
           "layer": d["layer"], "H": d["H"]}

    dm = np.load(MUL_STATES)
    mul = {"a": dm["a"], "b": dm["b"], "target": dm["product"],
           "layer": dm["layer"], "H": dm["H"]}
    da = np.load(ADD_STATES)
    add = {"a": da["a"], "b": da["b"], "target": da["sum"],
           "layer": da["layer"], "H": da["H"]}

    # Sanity: subtraction accuracy at L5
    from times_table_trace import STOI, ITOS
    H = torch.from_numpy(sub["H"]).float()
    eq_l5 = sub["layer"] == 5
    with torch.no_grad():
        logits = model.head(model.ln_f(H[eq_l5]))
        pred = logits.argmax(dim=-1).numpy()
    pred_chars = np.array([ITOS[int(p)] for p in pred])
    diffs = sub["target"][eq_l5]
    # First char: '-' if negative, else first digit
    true_chars = np.array([('-' if d < 0 else str(int(d))[0]) for d in diffs])
    print(f"Sanity: L5 first-char accuracy on a-b: "
          f"{(pred_chars == true_chars).mean():.1%}")

    report_log_vs_raw_sub(sub)
    operator_decodability(mul, add, sub)
    cross_op_divergence(mul, add, sub)
    operator_token_decodability(model)


if __name__ == "__main__":
    main()
