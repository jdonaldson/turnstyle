"""Stratified per-block decomposition of NanoGPT's late layers.

For each of the 100 mul pairs, capture at each block L=0..5:
  Δh_attn[L]   = attn sublayer output at =-position (h += attn(ln1(h)))
  Δh_mlp[L]    = mlp  sublayer output at =-position
  h_L[L]       = total residual after block L

Derived per-pair, per-block:
  ‖Δh_attn‖, ‖Δh_mlp‖
  Δh · digit_axis           (PC1 of head digit rows, the head's digit ruler)
  Δh · correct_digit_dir    (head[correct_first_digit_id] − head_centroid)
  ΔP(correct digit)          via logit-lens before/after the block
  attn weights at =-position (where each head attends)

Then stratify by commit class — at which block does the pair first become
correct-and-stays-correct. 19 pairs commit at L3, 49 at L4, 26 at L5
(plus 6 oddballs at L2).

This distinguishes four hypotheses for what each block does:
  (1) Commit-specific:  block_L writes big only for its commit-class pairs.
  (2) Universal sharpening: every block writes small amounts toward correct
                            for every pair; no class-specific concentration.
  (3) Refinement-specific: block_L writes big for pairs NOT yet committed.
  (4) Confound-suppressing: writes against the dominant wrong-digit
                            candidate, independent of pair class.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import STOI, encode, load  # noqa: E402

DATA = Path(__file__).parent / "data" / "nanogpt_times_table"


def capture_decomposition(model, device="cpu"):
    """Forward all 100 mul prompts, capture per-block sublayer outputs.

    Returns dict with arrays of shape (100, n_layers, hidden) for attn_out,
    mlp_out, h_after; arrays (100, n_layers, n_heads, T) for attn weights;
    int arrays for (a, b, product).
    """
    n_layers = model.cfg.n_layer
    n_head = model.cfg.n_head
    hidden = model.cfg.n_embd

    # Storage per pair
    attn_outs = []
    mlp_outs = []
    h_afters = []
    attn_weights_list = []
    a_arr, b_arr, prod_arr = [], [], []

    # Hooks
    attn_buf: Dict[int, torch.Tensor] = {}
    mlp_buf: Dict[int, torch.Tensor] = {}
    h_buf: Dict[int, torch.Tensor] = {}
    weights_buf: Dict[int, torch.Tensor] = {}

    def make_attn_hook(L):
        def hook(module, inputs, output):
            attn_buf[L] = output.detach().clone()
        return hook

    def make_mlp_hook(L):
        def hook(module, inputs, output):
            mlp_buf[L] = output.detach().clone()
        return hook

    def make_block_hook(L):
        def hook(module, inputs, output):
            h_buf[L] = output.detach().clone()
        return hook

    # Monkey-patch attention to capture weights (NanoGPT uses
    # F.scaled_dot_product_attention which doesn't expose weights).
    original_forwards = {}

    def make_attn_forward(orig, L):
        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            head = C // self.n_head
            q = q.view(B, T, self.n_head, head).transpose(1, 2)
            k = k.view(B, T, self.n_head, head).transpose(1, 2)
            v = v.view(B, T, self.n_head, head).transpose(1, 2)
            # Manual attention to expose weights
            scores = q @ k.transpose(-2, -1) / math.sqrt(head)
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            weights_buf[L] = attn.detach().clone()  # (B, n_head, T, T)
            y = attn @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(y)
        return forward

    handles = []
    for L, blk in enumerate(model.blocks):
        handles.append(blk.attn.register_forward_hook(make_attn_hook(L)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(L)))
        handles.append(blk.register_forward_hook(make_block_hook(L)))
        # Override attention forward
        original_forwards[L] = blk.attn.forward
        blk.attn.forward = make_attn_forward(
            blk.attn.forward, L
        ).__get__(blk.attn)

    model.eval()
    try:
        with torch.no_grad():
            for a in range(10):
                for b in range(10):
                    prompt = f"{a}*{b}="
                    ids = torch.tensor([encode(prompt)], dtype=torch.long,
                                       device=device)
                    eq_pos = len(prompt) - 1
                    attn_buf.clear(); mlp_buf.clear()
                    h_buf.clear(); weights_buf.clear()
                    _ = model(ids)

                    per_layer_attn = []
                    per_layer_mlp = []
                    per_layer_h = []
                    per_layer_weights = []
                    for L in range(n_layers):
                        per_layer_attn.append(
                            attn_buf[L][0, eq_pos, :].cpu().numpy()
                        )
                        per_layer_mlp.append(
                            mlp_buf[L][0, eq_pos, :].cpu().numpy()
                        )
                        per_layer_h.append(
                            h_buf[L][0, eq_pos, :].cpu().numpy()
                        )
                        # weights_buf[L]: (B=1, n_head, T, T); attention from
                        # eq_pos query to all key positions
                        per_layer_weights.append(
                            weights_buf[L][0, :, eq_pos, :].cpu().numpy()
                        )
                    attn_outs.append(np.stack(per_layer_attn))   # (n_L, H)
                    mlp_outs.append(np.stack(per_layer_mlp))
                    h_afters.append(np.stack(per_layer_h))
                    attn_weights_list.append(
                        np.stack(per_layer_weights)
                    )  # (n_L, n_head, T)
                    a_arr.append(a); b_arr.append(b); prod_arr.append(a * b)
    finally:
        for h in handles:
            h.remove()
        for L, orig in original_forwards.items():
            model.blocks[L].attn.forward = orig

    return {
        "attn":     np.stack(attn_outs),         # (100, n_L, H)
        "mlp":      np.stack(mlp_outs),          # (100, n_L, H)
        "h":        np.stack(h_afters),          # (100, n_L, H)
        "weights":  np.stack(attn_weights_list), # (100, n_L, n_head, T=4)
        "a":        np.array(a_arr, dtype=int),
        "b":        np.array(b_arr, dtype=int),
        "product":  np.array(prod_arr, dtype=int),
    }


def commit_layer_per_pair(model, h, prod):
    """Apply logit lens at every layer's =-token; return first L where pair
    is argmax-correct AND stays correct through L5."""
    n_pairs, n_L, H = h.shape
    H_flat = torch.from_numpy(h.reshape(-1, H)).float()
    with torch.no_grad():
        logits = model.head(model.ln_f(H_flat)).numpy()
    pred = logits.argmax(axis=1).reshape(n_pairs, n_L)
    true_ids = np.array([STOI[str(int(p))[0]] for p in prod])
    correct = pred == true_ids[:, None]  # (n_pairs, n_L)
    commit = np.full(n_pairs, -1, dtype=int)
    for i in range(n_pairs):
        for L in range(n_L):
            if correct[i, L:].all():
                commit[i] = L
                break
    return commit, correct, logits.reshape(n_pairs, n_L, -1)


def main():
    model = load("cpu")
    n_layers = model.cfg.n_layer
    H = model.cfg.n_embd
    digit_ids = [STOI[str(d)] for d in range(10)]
    W_head = model.head.weight.detach().cpu().numpy()  # (vocab, H)
    W_digits = W_head[digit_ids]  # (10, H)
    digit_centroid = W_digits.mean(axis=0)
    W_digits_c = W_digits - digit_centroid
    U, S, Vt = np.linalg.svd(W_digits_c, full_matrices=False)
    digit_axis = Vt[0]  # PC1 of digit rows in residual space
    digit_axis = digit_axis / np.linalg.norm(digit_axis)

    # Get d_log direction at L5 from cached states (same as head_log script)
    states_npz = np.load(DATA / "hidden_states.npz")
    L5_mask = states_npz["layer"] == 5
    H_L5 = states_npz["H"][L5_mask]
    prod_L5 = states_npz["product"][L5_mask]
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    s_log = StandardScaler().fit(H_L5)
    rl = Ridge(alpha=1.0).fit(
        s_log.transform(H_L5), np.log(prod_L5.astype(float) + 1)
    )
    d_log_res = rl.coef_ / s_log.scale_
    d_log_res = d_log_res / np.linalg.norm(d_log_res)

    print(f"digit_axis dim={digit_axis.shape}, d_log_res dim={d_log_res.shape}")
    print(f"cos(digit_axis, d_log_res) = "
          f"{float(np.dot(digit_axis, d_log_res)):+.3f}")

    # Capture decomposition
    print("\nCapturing per-block decomposition (100 mul prompts) ...")
    dec = capture_decomposition(model, device="cpu")
    print(f"  attn  shape {dec['attn'].shape}")
    print(f"  mlp   shape {dec['mlp'].shape}")
    print(f"  h     shape {dec['h'].shape}")
    print(f"  weights shape {dec['weights'].shape}")

    # Commit classes
    commit, correct, logits_lens = commit_layer_per_pair(
        model, dec["h"], dec["product"]
    )
    from collections import Counter
    print(f"\nCommit-class distribution: "
          f"{dict(sorted(Counter(commit.tolist()).items()))}")

    # Per-pair correct-digit direction (row of head for correct first digit
    # minus centroid)
    true_first_digit = np.array([int(str(int(p))[0])
                                  for p in dec["product"]])
    correct_digit_rows = W_digits[true_first_digit] - digit_centroid
    correct_digit_dir = correct_digit_rows / (
        np.linalg.norm(correct_digit_rows, axis=1, keepdims=True) + 1e-9
    )  # (100, H)

    # Per-pair, per-block: sublayer norms + projections
    attn = dec["attn"]; mlp = dec["mlp"]
    delta_total = attn + mlp  # by construction in residual stream

    attn_norm = np.linalg.norm(attn, axis=-1)   # (100, n_L)
    mlp_norm = np.linalg.norm(mlp, axis=-1)
    total_norm = np.linalg.norm(delta_total, axis=-1)

    # Projections
    proj_attn_digit = (attn @ digit_axis)        # (100, n_L)
    proj_mlp_digit  = (mlp  @ digit_axis)
    proj_attn_correct = np.einsum("plh,ph->pl", attn, correct_digit_dir)
    proj_mlp_correct  = np.einsum("plh,ph->pl", mlp,  correct_digit_dir)
    proj_attn_log = (attn @ d_log_res)
    proj_mlp_log  = (mlp  @ d_log_res)

    # ΔP(correct digit): apply head(ln_f(h_L)) per layer, compute softmax,
    # take P(correct first digit), then take per-layer differences.
    P_correct = np.zeros((100, n_layers))
    H_t = torch.from_numpy(dec["h"].reshape(-1, H)).float()
    with torch.no_grad():
        logits_l = model.head(model.ln_f(H_t)).numpy().reshape(
            100, n_layers, -1
        )
    probs_l = np.exp(logits_l - logits_l.max(axis=-1, keepdims=True))
    probs_l = probs_l / probs_l.sum(axis=-1, keepdims=True)
    for i in range(100):
        P_correct[i] = probs_l[i, :, [STOI[str(int(dec['product'][i]))[0]]]]
    # ΔP per block: post − pre. For L=0 the "pre" is the embedding output.
    # We'll approximate "pre" for L0 by subtracting Δh_total from h_after.
    # Embedding probs (with ln_f + head on h_pre_L0):
    h_pre = dec["h"] - delta_total  # h before block L
    H_pre_t = torch.from_numpy(h_pre.reshape(-1, H)).float()
    with torch.no_grad():
        logits_pre = model.head(model.ln_f(H_pre_t)).numpy().reshape(
            100, n_layers, -1
        )
    probs_pre = np.exp(logits_pre - logits_pre.max(axis=-1, keepdims=True))
    probs_pre = probs_pre / probs_pre.sum(axis=-1, keepdims=True)
    P_correct_pre = np.zeros((100, n_layers))
    for i in range(100):
        P_correct_pre[i] = probs_pre[
            i, :, [STOI[str(int(dec['product'][i]))[0]]]
        ]
    dP = P_correct - P_correct_pre  # (100, n_L) — gain of each block

    # ── Print stratified tables ──
    classes = {
        "L2_commit (n=6)":   commit == 2,
        "L3_commit (n=19)":  commit == 3,
        "L4_commit (n=49)":  commit == 4,
        "L5_commit (n=26)":  commit == 5,
    }

    def fmt(arr, mask):
        return "  ".join(f"{arr[mask].mean():+7.3f}"
                          for _ in range(1))

    print()
    print("=" * 100)
    print("Per-block sublayer write norms (averaged within commit class)")
    print("=" * 100)
    print(f"{'class':>22}  ",
          "  ".join(f"L{L}".center(13) for L in range(n_layers)))
    print("                       attn_norm | mlp_norm at each layer:")
    for name, mask in classes.items():
        cells = []
        for L in range(n_layers):
            cells.append(f"{attn_norm[mask, L].mean():.2f}|"
                          f"{mlp_norm[mask, L].mean():.2f}")
        print(f"  {name:>20}  " + "  ".join(c.center(13) for c in cells))

    print()
    print("=" * 100)
    print("Projection onto correct-digit direction (commit class × block)")
    print("attn_proj | mlp_proj  (both per Δh_sublayer · correct_digit_dir)")
    print("=" * 100)
    print(f"{'class':>22}  ",
          "  ".join(f"L{L}".center(15) for L in range(n_layers)))
    for name, mask in classes.items():
        cells = []
        for L in range(n_layers):
            a_p = proj_attn_correct[mask, L].mean()
            m_p = proj_mlp_correct[mask, L].mean()
            cells.append(f"{a_p:+.2f}|{m_p:+.2f}")
        print(f"  {name:>20}  " + "  ".join(c.center(15) for c in cells))

    print()
    print("=" * 100)
    print("ΔP(correct first digit) attributed to each block")
    print("=" * 100)
    print(f"{'class':>22}  ",
          "  ".join(f"L{L}".center(8) for L in range(n_layers)))
    for name, mask in classes.items():
        cells = []
        for L in range(n_layers):
            cells.append(f"{dP[mask, L].mean():+.3f}")
        print(f"  {name:>20}  " + "  ".join(c.center(8) for c in cells))

    print()
    print("=" * 100)
    print("Attention pattern at =-token (averaged over heads, then pairs)")
    print(f"Prompt positions: 0='a', 1='*', 2='b', 3='='")
    print("=" * 100)
    weights = dec["weights"].mean(axis=2)  # (100, n_L, 4) — head-averaged
    print(f"{'class':>22}  ",
          "  ".join(f"L{L}".center(28) for L in range(n_layers)))
    print("                       weights[a, *, b, =] per layer")
    for name, mask in classes.items():
        cells = []
        for L in range(n_layers):
            w = weights[mask, L].mean(axis=0)  # (4,)
            cells.append(f"[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}]")
        print(f"  {name:>20}  " + "  ".join(c.center(28) for c in cells))


if __name__ == "__main__":
    main()
