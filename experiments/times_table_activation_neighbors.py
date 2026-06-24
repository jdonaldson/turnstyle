"""Trace an activation backwards: find other contexts that produce similar
hidden states, and report their source tokens.

For a chosen (query prompt, query position, layer), compute the hidden
state h_q at that layer.  Search a diverse corpus of (prompt, position)
cells and return the top-K most cosine-similar h_c, with the source
character at each.  Filter view to "different source token" highlights
the cases where the activation is encoding something other than the
identity of the current character — context, role, or structural slot.

Usage:
    python experiments/times_table_activation_neighbors.py "2*3=" 3      # the '=' in 2*3=
    python experiments/times_table_activation_neighbors.py "2*3=" 0      # the '2'
    python experiments/times_table_activation_neighbors.py "0*9=" 3 --top-k 12
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

WINDOW = 32
N_WINDOWS = 250


def build_corpus(seed: int = 1) -> list[str]:
    rng = random.Random(seed)
    parts: list[str] = []
    while sum(len(p) for p in parts) < N_WINDOWS * WINDOW + WINDOW:
        if rng.random() < 0.7:
            parts.append(arithmetic_sample(rng))
        else:
            parts.append(svo_sample(rng))
    stream = "".join(parts)
    rng2 = random.Random(seed + 99)
    return [stream[s : s + WINDOW]
            for s in (rng2.randint(0, len(stream) - WINDOW - 1)
                      for _ in range(N_WINDOWS))]


@torch.no_grad()
def capture(model: GPT, texts: list[str], device: str):
    """Returns A of shape (n_layer, total_cells, dim) plus token-id and
    (window_idx, position) metadata aligned with the n_cells axis."""
    acts_per_layer = [[] for _ in range(model.cfg.n_layer)]
    tok_ids: list[int] = []
    contexts: list[str] = []  # the +/-3 char window around each cell
    for w_i, text in enumerate(texts):
        idx = torch.tensor([encode(text)], dtype=torch.long, device=device)
        _, _, states = model(idx, return_states=True)
        for layer, h in enumerate(states):
            acts_per_layer[layer].append(h[0].cpu().numpy())
        for t in range(len(text)):
            tok_ids.append(STOI[text[t]])
            lo, hi = max(0, t - 3), min(len(text), t + 4)
            ctx = text[lo:t] + "[" + text[t] + "]" + text[t + 1 : hi]
            contexts.append(ctx)
    A = np.stack([np.concatenate(a, axis=0) for a in acts_per_layer], axis=0)
    return A, np.array(tok_ids, dtype=np.int32), contexts


def cosine_topk(h: np.ndarray, H: np.ndarray, k: int):
    """Top-k cosine-similar rows of H to h. Returns (indices, similarities)."""
    h_n = h / (np.linalg.norm(h) + 1e-8)
    H_n = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    sims = H_n @ h_n
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def fmt_ctx(s: str) -> str:
    return s.replace("\n", "\\n").replace(" ", "·")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", help="query prompt, e.g. '2*3='")
    ap.add_argument("position", type=int, help="position in the prompt (0-indexed)")
    ap.add_argument("--layers", default="all",
                    help="comma-separated layer indices, or 'all'")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.position >= len(args.prompt):
        raise ValueError(f"position {args.position} out of range for {args.prompt!r}")
    query_char = args.prompt[args.position]
    query_id = STOI[query_char]

    print("Loading model...")
    model = load(args.device)
    n_layer = model.cfg.n_layer

    print(f"Building corpus ({N_WINDOWS} windows of {WINDOW} chars each)...")
    corpus = build_corpus()
    A, tok_ids, contexts = capture(model, corpus, args.device)
    print(f"Corpus: {A.shape[1]} cells, {n_layer} layers, dim={A.shape[2]}")

    # Capture query
    q_idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=args.device)
    with torch.no_grad():
        _, _, q_states = model(q_idx, return_states=True)
    q_acts = [h[0, args.position].cpu().numpy() for h in q_states]  # one per layer

    if args.layers == "all":
        layers = list(range(n_layer))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    print(f"\nQuery: prompt {args.prompt!r}  position {args.position}  "
          f"char {fmt_ctx(query_char)!r}\n")

    for layer in layers:
        h_q = q_acts[layer]
        H = A[layer]
        idx, sims = cosine_topk(h_q, H, k=args.top_k * 3)  # over-fetch then split

        # Split into same-source-token vs different-source-token
        same, diff = [], []
        for i, s in zip(idx, sims):
            entry = (i, s, tok_ids[i], contexts[i])
            if tok_ids[i] == query_id:
                same.append(entry)
            else:
                diff.append(entry)
            if len(same) >= args.top_k and len(diff) >= args.top_k:
                break

        print(f"━━━ Layer {layer} ━━━")
        print(f"  Same-token neighbors (source char {fmt_ctx(query_char)!r}):")
        for _, s, _, ctx in same[:args.top_k]:
            print(f"    cos={s:.3f}  {fmt_ctx(ctx)}")
        print(f"  Different-token neighbors:")
        if not diff:
            print(f"    (none in top-{args.top_k * 3})")
        else:
            for _, s, tid, ctx in diff[:args.top_k]:
                src = fmt_ctx(ITOS[int(tid)])
                print(f"    cos={s:.3f}  src={src!r:>5}  {fmt_ctx(ctx)}")
        print()


if __name__ == "__main__":
    main()
