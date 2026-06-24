"""Per-position logit lens for the NanoGPT times-table model.

Given a prompt (e.g. "2*3=") and a target character (e.g. "6"), forward
the model and capture hidden states at every (layer, position).  Apply
head(ln_f(h)) to each cell to get a vocab distribution.  Report:

  - P(target | layer, position) — when does the answer signal appear?
  - top-K candidate tokens at each cell — what other tokens were strong?

Usage:
    python experiments/times_table_backtrack.py 2*3= 6
    python experiments/times_table_backtrack.py 7*8= 56 --top-k 4
    python experiments/times_table_backtrack.py 0*9= 0   # annihilator case
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# reuse the model code from the trace experiment
sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import GPT, encode, ITOS, STOI, load  # noqa: E402


@torch.no_grad()
def capture_all(model: GPT, prompt: str, device: str):
    """Return (hidden_states, ln_f_states, logits, probs) of shape
    (n_layer, T, vocab) where layer L is the residual AFTER block L."""
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    _, _, states = model(idx, return_states=True)  # list of (1, T, C)

    H = torch.stack(states, dim=0)[:, 0, :, :]            # (n_layer, T, C)
    Hn = model.ln_f(H)                                    # (n_layer, T, C)
    logits = model.head(Hn)                               # (n_layer, T, V)
    probs = torch.softmax(logits, dim=-1)                 # (n_layer, T, V)
    return H.cpu(), Hn.cpu(), logits.cpu(), probs.cpu()


def heatmap_p_target(probs, target_id: int, prompt: str):
    """Plot text heatmap of P(target) over (layer, position)."""
    n_layer, T, _ = probs.shape
    grid = probs[:, :, target_id].numpy()  # (n_layer, T)

    print(f"\nP('{ITOS[target_id]}' | layer, position):")
    print(f"  {'position':>8}  " + "  ".join(f"  p{p}  " for p in range(T)))
    print(f"  {'  token':>8}  " + "  ".join(f"  {prompt[p]:>3} " for p in range(T)))
    print(f"  {'':>8}  " + "  ".join(["-------"] * T))
    for l in range(n_layer):
        cells = []
        for p in range(T):
            v = grid[l, p]
            bar = "█" * int(v * 8)
            cells.append(f"{v:5.3f}{bar:<2}")
        print(f"  {'L'+str(l):>8}  " + "  ".join(cells))


def top_k_cells(probs, prompt: str, k: int = 3, target_id: int | None = None):
    """For each (layer, position) print top-k tokens + their probability.
    If target_id is given, prepend a marker when the target is in top-k."""
    n_layer, T, _ = probs.shape
    print(f"\nTop-{k} tokens per (layer, position) — '*' marks the answer if in top-{k}:")
    for l in range(n_layer):
        print(f"  L{l}")
        for p in range(T):
            vals, ids = torch.topk(probs[l, p], k=k)
            entries = []
            for v, i in zip(vals.tolist(), ids.tolist()):
                tok = ITOS[i]
                tok = tok.replace("\n", "\\n").replace(" ", "·")
                mark = "*" if (target_id is not None and i == target_id) else " "
                entries.append(f"{mark}{tok!r}:{v:.3f}")
            print(f"    p{p} ({prompt[p]!r}): " + "  ".join(entries))


def first_layer_target_dominant(probs, target_id: int, prompt: str):
    """Compact summary: at each position, which is the earliest layer where
    target is the argmax?"""
    n_layer, T, _ = probs.shape
    print(f"\nEarliest layer where '{ITOS[target_id]}' becomes argmax (per position):")
    for p in range(T):
        wins = [l for l in range(n_layer) if int(probs[l, p].argmax()) == target_id]
        if wins:
            print(f"  p{p} ({prompt[p]!r}): L{min(wins)}  P={probs[min(wins), p, target_id].item():.3f}")
        else:
            print(f"  p{p} ({prompt[p]!r}): (never)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", help="prompt string, e.g. '2*3='")
    ap.add_argument("target", help="target character (the answer the model should produce)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if len(args.target) > 1 and args.target.isdigit():
        # Multi-digit answers — backtrack against the FIRST digit
        target_char = args.target[0]
        print(f"[note] target {args.target!r} is multi-char; backtracking on first digit "
              f"{target_char!r}")
    else:
        target_char = args.target

    if target_char not in STOI:
        raise ValueError(f"target char {target_char!r} not in vocab")
    target_id = STOI[target_char]

    print(f"Loading model from checkpoint...")
    model = load(args.device)

    print(f"Prompt: {args.prompt!r}  ->  next-token target {target_char!r} (id {target_id})")
    H, _, _, probs = capture_all(model, args.prompt, args.device)
    print(f"hidden shape: {tuple(H.shape)}  (n_layer, T, C)")

    # Sanity: what does the model actually predict at position T-1, last layer?
    last = probs[-1, -1]
    top3 = torch.topk(last, k=3)
    pred = "  ".join(f"{ITOS[i]!r}:{v:.3f}" for v, i in zip(top3.values.tolist(), top3.indices.tolist()))
    print(f"\nModel's actual next-token prediction (L{probs.shape[0]-1}, last position): {pred}")

    heatmap_p_target(probs, target_id, args.prompt)
    first_layer_target_dominant(probs, target_id, args.prompt)
    top_k_cells(probs, args.prompt, k=args.top_k, target_id=target_id)


if __name__ == "__main__":
    main()
