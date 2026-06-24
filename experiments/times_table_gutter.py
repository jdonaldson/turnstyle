"""Is the annihilator direction a shared gutter across math operations?

The mul-annihilator direction (mean(a×0 ∪ 0×b) - mean(rest)) was a clean
1D feature at L3 that separates zero-output mul from non-zero-output mul
with d'=8.48.

Hypothesis: this is a *shared* "output-equals-zero" gutter, also used by
subtraction's self-subtraction cases (a-a=0). Test:

  1) Compute the analogous direction for subtraction: mean(a-a) - mean(rest).
  2) cos(mul-gutter, sub-gutter) at every layer — high cosine ⇒ shared.
  3) Universal gutter: mean(all output=0 cases across ops) -
     mean(all output!=0 cases). Compare to per-op gutters.
  4) Cross-op portability: does the mul gutter separate sub-zero from
     sub-nonzero? Does the sub gutter separate mul-zero from mul-nonzero?
  5) Op direction: is op identity (mul vs add vs sub) encoded in a
     direction orthogonal to the gutter, as the hypothesis predicts
     ("gutter weaves around other useful computational structure")?

Uses the existing checkpoint (no retraining).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import times_table_trace as ttt  # type: ignore  # the original module

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
CKPT = ROOT / "checkpoint.pt"


@torch.no_grad()
def collect_all_ops(model, device: str = "cpu"):
    """Forward 300 prompts (10x10 for each of +, -, *), return states + metadata."""
    rows = []  # (op, a, b, result, layer, vec)
    for op in ("+", "-", "*"):
        for a in range(10):
            for b in range(10):
                prompt = f"{a}{op}{b}="
                idx = torch.tensor([ttt.encode(prompt)], dtype=torch.long, device=device)
                _, _, states = model(idx, return_states=True)
                eq_pos = len(prompt) - 1
                if op == "+":
                    result = a + b
                elif op == "-":
                    result = a - b
                else:
                    result = a * b
                for layer, h in enumerate(states):
                    vec = h[0, eq_pos, :].cpu().numpy()
                    rows.append((op, a, b, result, layer, vec))
    op_arr = np.array([r[0] for r in rows])
    a_arr = np.array([r[1] for r in rows], dtype=int)
    b_arr = np.array([r[2] for r in rows], dtype=int)
    r_arr = np.array([r[3] for r in rows], dtype=int)
    l_arr = np.array([r[4] for r in rows], dtype=int)
    H = np.stack([r[5] for r in rows])
    return op_arr, a_arr, b_arr, r_arr, l_arr, H


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(u, v):
    return float(unit(u) @ unit(v))


def direction(states: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return states[mask].mean(0) - states[~mask].mean(0)


def dprime(proj: np.ndarray, mask: np.ndarray) -> float:
    pz, pn = proj[mask], proj[~mask]
    denom = np.sqrt(0.5 * (pz.var() + pn.var()) + 1e-12)
    return float((pz.mean() - pn.mean()) / denom)


def main():
    model = ttt.load("cpu")
    print("forwarding 300 prompts across 6 layers...")
    op, a, b, r, layer, H = collect_all_ops(model)
    print(f"H shape: {H.shape}")
    n_layers = int(layer.max()) + 1

    is_mul = op == "*"
    is_sub = op == "-"
    is_add = op == "+"
    zero_mul = is_mul & (r == 0)  # 19 * 6 = 114 rows
    zero_sub = is_sub & (r == 0)  # 10 * 6 = 60 rows (a==b)
    zero_add = is_add & (r == 0)  # 1 * 6 = 6 rows (0+0)
    zero_any = (r == 0)  # 30 unique cases × 6 = 180 rows
    print(
        f"\ncounts (unique pairs per op):  "
        f"mul-zero {is_mul.sum() // n_layers}/{zero_mul.sum() // n_layers}   "
        f"sub-zero {is_sub.sum() // n_layers}/{zero_sub.sum() // n_layers}   "
        f"add-zero {is_add.sum() // n_layers}/{zero_add.sum() // n_layers}"
    )

    # --- Per-layer: directions and cosines ---
    print("\n" + "=" * 76)
    print("Per-layer gutter directions  (annihilator = result==0 within op)")
    print("=" * 76)
    print(
        f"{'layer':6s}  {'d_mul':>7s}  {'d_sub':>7s}  "
        f"{'cos(mul,sub)':>13s}  {'cos(mul,uni)':>13s}  {'cos(sub,uni)':>13s}"
    )

    for L in range(n_layers):
        layer_mask = layer == L
        H_L = H[layer_mask]
        zm = zero_mul[layer_mask]
        zs = zero_sub[layer_mask]
        za = zero_any[layer_mask]

        # within-mul gutter (zero-result vs other mul)
        Hm = H_L[op[layer_mask] == "*"]
        zm_in_mul = zero_mul[layer_mask][op[layer_mask] == "*"]
        d_mul = direction(Hm, zm_in_mul)
        proj_mul = Hm @ unit(d_mul)
        dp_mul = dprime(proj_mul, zm_in_mul)

        # within-sub gutter
        Hs = H_L[op[layer_mask] == "-"]
        zs_in_sub = zero_sub[layer_mask][op[layer_mask] == "-"]
        d_sub = direction(Hs, zs_in_sub)
        proj_sub = Hs @ unit(d_sub)
        dp_sub = dprime(proj_sub, zs_in_sub)

        # universal gutter (any-op zero result vs nonzero)
        d_uni = direction(H_L, za)

        print(
            f"  L{L}    "
            f"{dp_mul:7.2f}  {dp_sub:7.2f}  "
            f"{cos(d_mul, d_sub):+13.3f}  "
            f"{cos(d_mul, d_uni):+13.3f}  "
            f"{cos(d_sub, d_uni):+13.3f}"
        )

    # Focus on L3 (peak bulge for mul) for the deeper tests
    L = 3
    print(f"\n{'=' * 76}\nDeep dive at L{L} (peak mul bulge)\n{'=' * 76}")
    layer_mask = layer == L
    H_L = H[layer_mask]
    op_L = op[layer_mask]
    r_L = r[layer_mask]
    zm_L = zero_mul[layer_mask]
    zs_L = zero_sub[layer_mask]
    za_L = zero_any[layer_mask]

    Hm = H_L[op_L == "*"]
    Hs = H_L[op_L == "-"]
    Ha = H_L[op_L == "+"]
    zm_in_mul = zm_L[op_L == "*"]
    zs_in_sub = zs_L[op_L == "-"]

    d_mul = unit(direction(Hm, zm_in_mul))
    d_sub = unit(direction(Hs, zs_in_sub))
    d_uni = unit(direction(H_L, za_L))

    # cross-portability: does d_mul separate sub-zero in subtraction?
    proj_sub_on_mul_dir = Hs @ d_mul
    dp_sub_on_mul = dprime(proj_sub_on_mul_dir, zs_in_sub)
    proj_mul_on_sub_dir = Hm @ d_sub
    dp_mul_on_sub = dprime(proj_mul_on_sub_dir, zm_in_mul)
    print(f"  mul-gutter d' on subtraction (zero vs nonzero) : {dp_sub_on_mul:6.2f}")
    print(f"  sub-gutter d' on multiplication (zero vs nonzero): {dp_mul_on_sub:6.2f}")
    print(f"  universal d' on all 300 (zero vs nonzero)       : {dprime(H_L @ d_uni, za_L):6.2f}")

    # op direction: encode op identity
    # mean of each op at this layer
    mu_mul = Hm.mean(0)
    mu_sub = Hs.mean(0)
    mu_add = Ha.mean(0)
    mu_all = H_L.mean(0)
    op_dirs = {"mul-vs-all": mu_mul - mu_all, "sub-vs-all": mu_sub - mu_all, "add-vs-all": mu_add - mu_all}

    print("\n  Gutter ⊥ op directions? cos(d_uni, op-direction):")
    for name, v in op_dirs.items():
        print(f"    cos(d_uni, {name:14s}) = {cos(d_uni, v):+.3f}")

    print("\n  Cross-op gutter cosines at L3:")
    print(f"    cos(d_mul, d_sub) = {cos(d_mul, d_sub):+.3f}")
    print(f"    cos(d_mul, d_uni) = {cos(d_mul, d_uni):+.3f}")
    print(f"    cos(d_sub, d_uni) = {cos(d_sub, d_uni):+.3f}")

    # Ablation portability: ablate d_uni from ALL states; does zero-detection collapse?
    print("\n  Ablation test (project out d_uni from L3 states):")
    H_abl = H_L - (H_L @ d_uni)[:, None] * d_uni[None, :]
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.model_selection import cross_val_score  # type: ignore

    for name, H_use in [("intact", H_L), ("d_uni ablated", H_abl)]:
        sc = cross_val_score(LogisticRegression(max_iter=2000), H_use, za_L.astype(int), cv=5).mean()
        print(f"    zero/non-zero detection ({name:14s}): {sc:.1%}")


if __name__ == "__main__":
    main()
