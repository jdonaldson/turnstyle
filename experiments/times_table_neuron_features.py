"""Probe per-neuron token relationships in the NanoGPT times-table model.

Two complementary signals per (layer, neuron):

  INPUT side  — Pearson correlation of activation with one-hot indicators
                of the character at positions (t, t-1) across windows
                sampled from a *diverse* corpus.  Diversity (varying
                operator/SVO contexts) is what breaks the
                position↔character degeneracy you get from pure 'a*b='
                prompts.

  OUTPUT side — head.weight[:, j] tells us which characters that neuron
                boosts when it fires.  Report the top positive boost
                (highest logit gain) and the top negative suppression.

A neuron is a clean "character pass-through" when its top input
correlation char == its top output-boost char.  A "context detector"
fires on char X but writes toward Y (i.e., implements "after seeing X,
predict Y").  The richer cases are written out below.

Usage:
    python experiments/times_table_neuron_features.py [--top-k N] [--min-r 0.3]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import (  # noqa: E402
    GPT, encode, ITOS, STOI, load,
    arithmetic_sample, svo_sample,
)


WINDOW = 32   # matches model.cfg.block_size
N_WINDOWS = 250


def build_diverse_windows(seed: int = 1) -> list[str]:
    """Sample windows from a mixed arithmetic + SVO corpus.

    Each window is WINDOW chars sliced from a long text, so characters
    appear at many positions and contexts.
    """
    rng = random.Random(seed)
    parts: list[str] = []
    while sum(len(p) for p in parts) < N_WINDOWS * WINDOW + WINDOW:
        if rng.random() < 0.7:
            parts.append(arithmetic_sample(rng))
        else:
            parts.append(svo_sample(rng))
    stream = "".join(parts)

    windows: list[str] = []
    rng2 = random.Random(seed + 99)
    for _ in range(N_WINDOWS):
        start = rng2.randint(0, len(stream) - WINDOW - 1)
        windows.append(stream[start : start + WINDOW])
    return windows


@torch.no_grad()
def capture(model: GPT, windows: list[str], device: str):
    """Returns A of shape (n_layer, n_cells, dim) and char_ids of shape (n_cells,).
    n_cells = sum of window lengths (= N_WINDOWS * WINDOW)."""
    acts_per_layer = [[] for _ in range(model.cfg.n_layer)]
    cur_ids: list[int] = []
    prev_ids: list[int] = []  # for (t-1) correlations

    for w in windows:
        idx = torch.tensor([encode(w)], dtype=torch.long, device=device)
        _, _, states = model(idx, return_states=True)
        for layer, h in enumerate(states):
            acts_per_layer[layer].append(h[0].cpu().numpy())  # (T, dim)
        for t in range(len(w)):
            cur_ids.append(STOI[w[t]])
            prev_ids.append(STOI[w[t - 1]] if t > 0 else -1)

    A = np.stack([np.concatenate(a, axis=0) for a in acts_per_layer], axis=0)
    return A, np.array(cur_ids, dtype=np.int32), np.array(prev_ids, dtype=np.int32)


def correlate_with_onehot(values_per_cell: np.ndarray, char_ids: np.ndarray,
                          n_vocab: int) -> np.ndarray:
    """For each (neuron, char), Pearson r between neuron activation and the
    indicator 'this cell's char == c'.

    values_per_cell: (n_cells, dim)
    char_ids:        (n_cells,)  with -1 meaning "no char" (skip)
    Returns r matrix shape (n_vocab, dim)."""
    valid = char_ids >= 0
    V = values_per_cell[valid]
    cids = char_ids[valid]

    Vz = (V - V.mean(0, keepdims=True)) / (V.std(0, keepdims=True) + 1e-8)
    R = np.zeros((n_vocab, V.shape[1]), dtype=np.float32)
    for c in range(n_vocab):
        ind = (cids == c).astype(np.float32)
        p = float(ind.mean())
        if p == 0.0 or p == 1.0:
            continue
        ind_z = (ind - p) / np.sqrt(p * (1 - p))
        R[c] = (ind_z @ Vz) / V.shape[0]
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-r", type=float, default=0.30)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("Loading model...")
    model = load(args.device)
    n_vocab = model.cfg.vocab_size
    n_layer = model.cfg.n_layer
    dim = model.cfg.n_embd

    print(f"Sampling {N_WINDOWS} diverse windows of {WINDOW} chars each...")
    windows = build_diverse_windows()
    print(f"Sample window: {windows[0]!r}")
    print(f"Char distribution: {np.bincount([STOI[c] for w in windows for c in w], minlength=n_vocab)}")

    A, cur_ids, prev_ids = capture(model, windows, args.device)
    print(f"Activation tensor: {A.shape}  (n_layer, n_cells, dim)")

    W_U = model.head.weight.detach().cpu().numpy()  # (vocab, dim)

    # Whitespace chars are excluded from the *input* correlation (they dominate
    # raw frequency).  We still allow them in output-boost reporting.
    interp_chars = set("0123456789+-*=\n ")
    suppress_ids = {STOI[c] for c in ("\n", " ") if c in STOI}

    digit_ids    = {STOI[c] for c in "0123456789" if c in STOI}
    op_ids       = {STOI[c] for c in "+-*=" if c in STOI}
    letter_ids   = {i for c, i in STOI.items() if c.isalpha()}

    def fmt_char(c: str) -> str:
        return c.replace("\n", "\\n").replace(" ", "·")

    def category_corrs(values: np.ndarray, char_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Pearson r of activation with category membership at current position."""
        valid = char_ids >= 0
        V = values[valid]
        cids = char_ids[valid]
        Vz = (V - V.mean(0, keepdims=True)) / (V.std(0, keepdims=True) + 1e-8)
        out = {}
        for name, ids in [("digit", digit_ids), ("op", op_ids), ("letter", letter_ids)]:
            ind = np.isin(cids, list(ids)).astype(np.float32)
            p = float(ind.mean())
            if p == 0.0 or p == 1.0:
                out[name] = np.zeros(V.shape[1], dtype=np.float32)
                continue
            ind_z = (ind - p) / np.sqrt(p * (1 - p))
            out[name] = (ind_z @ Vz) / V.shape[0]
        return out

    print(f"\nReporting top-{args.top_k} cleanest neurons per layer "
          f"(|input r| >= {args.min_r})\n")

    for layer in range(n_layer):
        R_cur = correlate_with_onehot(A[layer], cur_ids, n_vocab)
        R_prev = correlate_with_onehot(A[layer], prev_ids, n_vocab)
        # Zero out whitespace-char rows so they don't dominate
        for sid in suppress_ids:
            R_cur[sid] = 0.0
            R_prev[sid] = 0.0
        cat_cur = category_corrs(A[layer], cur_ids)

        # Best feature (offset, char) per neuron
        abs_cur = np.abs(R_cur)
        abs_prev = np.abs(R_prev)
        best_per_neuron_cur = abs_cur.argmax(axis=0)
        best_per_neuron_prev = abs_prev.argmax(axis=0)

        # Pick the larger of (current-char, previous-char) per neuron
        cur_best_r = R_cur[best_per_neuron_cur, np.arange(dim)]
        prev_best_r = R_prev[best_per_neuron_prev, np.arange(dim)]

        use_cur = np.abs(cur_best_r) >= np.abs(prev_best_r)
        best_r = np.where(use_cur, cur_best_r, prev_best_r)
        best_offset = np.where(use_cur, 0, -1)
        best_char_id = np.where(use_cur, best_per_neuron_cur, best_per_neuron_prev)

        order = np.argsort(-np.abs(best_r))
        print(f"━━━ Layer {layer} ━━━")
        printed = 0
        for j in order:
            r = best_r[j]
            if abs(r) < args.min_r:
                break
            if printed >= args.top_k:
                break
            offset_lbl = "t" if best_offset[j] == 0 else "t-1"
            c_in = ITOS[int(best_char_id[j])]

            # Output side: top boost (positive) and top suppression (negative)
            wu = W_U[:, j]
            order_boost = np.argsort(-wu)
            top_boost = [(ITOS[int(i)], float(wu[i]))
                         for i in order_boost[:3]]
            order_supp = np.argsort(wu)
            top_supp = [(ITOS[int(i)], float(wu[i]))
                        for i in order_supp[:2]]

            # Restrict reporting to interpretable chars for human reading
            boost_str = " ".join(
                f"{fmt_char(c)!r}:{v:+.2f}"
                for c, v in top_boost if c in interp_chars
            ) or " ".join(f"{fmt_char(c)!r}:{v:+.2f}" for c, v in top_boost)
            supp_str = " ".join(
                f"{fmt_char(c)!r}:{v:+.2f}"
                for c, v in top_supp if c in interp_chars
            ) or " ".join(f"{fmt_char(c)!r}:{v:+.2f}" for c, v in top_supp)

            # Pass-through detector: best input char also among top output boost?
            top_boost_chars = {c for c, _ in top_boost}
            marker = ""
            if c_in in top_boost_chars and offset_lbl == "t":
                marker = "   <-- token pass-through"
            elif c_in in top_boost_chars and offset_lbl == "t-1":
                marker = "   <-- copy-from-prev"

            # Categorical correlations (only for current-position)
            cat_str = " ".join(
                f"{name}:{cat_cur[name][j]:+.2f}"
                for name in ("digit", "op", "letter")
                if abs(cat_cur[name][j]) >= 0.15
            )
            cat_str = f"  cat[{cat_str}]" if cat_str else ""
            print(f"  N{j:3d}  r={r:+.3f} char {fmt_char(c_in)!r}@{offset_lbl}"
                  f"  boost[{boost_str}]  supp[{supp_str}]{cat_str}{marker}")
            printed += 1

    # How many neurons per layer have strong correlations?
    print("\nSummary  (neurons with |best input r| >= 0.5):")
    for layer in range(n_layer):
        R_cur = correlate_with_onehot(A[layer], cur_ids, n_vocab)
        R_prev = correlate_with_onehot(A[layer], prev_ids, n_vocab)
        best = np.maximum(np.abs(R_cur).max(0), np.abs(R_prev).max(0))
        clean = int((best >= 0.5).sum())
        print(f"  L{layer}: {clean}/{dim}")


if __name__ == "__main__":
    main()
