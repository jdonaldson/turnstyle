"""Does the model's marginal first-digit distribution match Benford's Law?

Benford: P(d) = log10(1 + 1/d) for d ∈ 1..9.
         ≈ {1: .301, 2: .176, 3: .125, 4: .097, 5: .079,
            6: .067, 7: .058, 8: .051, 9: .046}.

Compares three distributions per operator, per layer:
  (a) DATA: empirical distribution of leading digits in correct outputs.
  (b) MODEL: averaged logit-lens softmax over the 100 prompts at the
            =-token. P_layer(d) = mean_pairs softmax(head(ln_f(h_L)))[d].
  (c) BENFORD: log10(1 + 1/d) reference.

For `*`, the data is roughly log-uniform → expect data ≈ Benford ≈ model
at L5. For `+`, products span 0..18 narrowly → expect skew away from
Benford. For `-`, first char can be '-' so the digit-only distribution
is conditioned.

Also tracks per-layer KL(model‖Benford) on digits 1..9 (re-normalized).
If early layers carry a Benford-like prior that L5 commits away from,
KL should be U-shaped over layers.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import STOI, load  # noqa: E402

DATA = Path(__file__).parent / "data" / "nanogpt_times_table"


def kl(p, q, eps=1e-12):
    p = np.array(p) + eps
    q = np.array(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def main():
    model = load("cpu")
    digit_ids = [STOI[str(d)] for d in range(10)]
    neg_id = STOI["-"]

    benford_19 = np.array([math.log10(1 + 1 / d) for d in range(1, 10)])

    OPS = [("*", "mul", "hidden_states.npz", "product"),
           ("+", "add", "hidden_states_add.npz", "sum"),
           ("-", "sub", "hidden_states_sub.npz", "diff")]

    for op, name, fname, target_key in OPS:
        d = np.load(DATA / fname)
        H = torch.from_numpy(d["H"]).float()
        l_arr = d["layer"]
        tgt = d[target_key]
        # Data: empirical leading-digit (and '-' for sub) distribution
        first_chars = []
        for v in tgt[l_arr == 0]:
            v = int(v)
            first_chars.append("-" if v < 0 else str(v)[0])
        from collections import Counter
        ctr = Counter(first_chars)
        data_probs = np.array(
            [ctr.get(str(i), 0) for i in range(10)] + [ctr.get("-", 0)],
            dtype=float
        )
        data_probs /= data_probs.sum()

        # Data, restricted to digits 1..9 only (for Benford KL)
        data_19 = data_probs[1:10]
        if data_19.sum() > 0:
            data_19 = data_19 / data_19.sum()

        print(f"\n=== {op} ({name}) ===")
        print(f"  Empirical data P over chars '0'..'9','-':")
        cells = "  ".join(f"{c}:{p:.2f}" for c, p in zip(
            list("0123456789-"), data_probs))
        print(f"    {cells}")
        if data_19.sum() > 0:
            print(f"  Empirical KL(data_1..9 ‖ Benford_1..9): "
                  f"{kl(data_19, benford_19):.4f}")
        print()

        # Model marginal per layer (logit-lens)
        n_layers = int(l_arr.max()) + 1
        print(f"  Model marginal P (logit-lens averaged over 100 prompts):")
        header = ("    L  " +
                  "  ".join(f"P({c})".center(6) for c in
                            list("0123456789") + ["-"]) +
                  "  KL(M_1..9‖Benford)")
        print(header)
        for l in range(n_layers):
            m = l_arr == l
            with torch.no_grad():
                logits = model.head(model.ln_f(H[m]))
                probs = torch.softmax(logits, dim=-1).numpy()
            marg = probs.mean(axis=0)
            digit_marg = marg[digit_ids]
            neg_marg = marg[neg_id]
            row = list(digit_marg) + [neg_marg]
            # Renormalize on 1..9 for Benford KL
            model_19 = digit_marg[1:10]
            if model_19.sum() > 0:
                model_19 = model_19 / model_19.sum()
                k = kl(model_19, benford_19)
            else:
                k = float("nan")
            cells = "  ".join(f"{p:.3f}".center(6) for p in row)
            print(f"    L{l}  {cells}  {k:.4f}")

    print(f"\nBenford reference P(d) for d=1..9:")
    print(f"  " + "  ".join(f"{d}:{p:.3f}" for d, p in zip(range(1, 10),
                                                            benford_19)))


if __name__ == "__main__":
    main()
