"""Activation patching: localize the operator-dispatch block.

For each (a, b), forward three prompts (a*b=, a+b=, a-b=) and save the
`=`-token state at every layer. Then, for each ordered (donor_op,
recipient_op) pair and each layer L:

  - Run the recipient prompt's forward pass.
  - After block L runs, OVERWRITE the `=`-token residual with the
    corresponding donor's L-th `=`-token state.
  - Continue blocks L+1..L_n_layer-1, then ln_f, then head[:, -1].
  - argmax the next-character distribution.
  - Flip success := pred-char matches the DONOR's expected first char.

If patching after block L flips the prediction → that block has already
done enough operator-specific work that the downstream blocks can't undo
the swap from the recipient's other tokens.

Reads the prior result: cos(mul, add) drops most sharply L0→L1 (0.80 →
0.60). Prediction: patching after block 1 should already give high
flip rates; patching after block 0 should give moderate flips; later
blocks should plateau or saturate.

Baselines: each (op, a, b) prompt's unpatched first-char accuracy.
Filter: skip (a, b) where donor and recipient happen to share the same
first character (no-op; would inflate flip rate spuriously).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import ITOS, encode, load  # noqa: E402


OPS = ["*", "+", "-"]


def first_char(op: str, a: int, b: int) -> str:
    if op == "*":
        c = a * b
    elif op == "+":
        c = a + b
    else:
        c = a - b
    return "-" if c < 0 else str(c)[0]


@torch.no_grad()
def collect_eq_states(model, prompt: str, device: str = "cpu"):
    ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    _, _, states = model(ids, return_states=True)
    eq_pos = len(prompt) - 1
    return [s[0, eq_pos, :].detach().clone() for s in states], eq_pos


@torch.no_grad()
def patched_forward(model, recipient_ids, donor_eq_state, patch_layer,
                    eq_pos):
    """Forward recipient_ids but replace eq_pos state after block patch_layer.

    Returns logits over vocab for the LAST token position.
    """
    B, T = recipient_ids.shape
    pos = torch.arange(T, device=recipient_ids.device)
    x = model.wte(recipient_ids) + model.wpe(pos)
    for l, blk in enumerate(model.blocks):
        x = blk(x)
        if l == patch_layer:
            x = x.clone()
            x[:, eq_pos, :] = donor_eq_state
    x = model.ln_f(x)
    return model.head(x[:, [-1], :])


@torch.no_grad()
def baseline_pred(model, prompt: str, device: str = "cpu") -> str:
    ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    logits, _, _ = model(ids)
    pred_id = int(logits[0, 0].argmax().item())
    return ITOS[pred_id]


def main():
    model = load("cpu")
    n_layers = model.cfg.n_layer

    # ── unpatched baselines ──
    print("Unpatched baseline first-char accuracy:")
    for op in OPS:
        correct = 0
        for a in range(10):
            for b in range(10):
                prompt = f"{a}{op}{b}="
                pred = baseline_pred(model, prompt)
                if pred == first_char(op, a, b):
                    correct += 1
        print(f"  {op}:  {correct}/100 = {correct/100:.1%}")

    # ── collect =-token states for all 300 prompts ──
    print("\nCollecting states for 300 prompts ...")
    eq_states = {}  # (op, a, b) -> list of states (1 per block)
    for op in OPS:
        for a in range(10):
            for b in range(10):
                prompt = f"{a}{op}{b}="
                states, _ = collect_eq_states(model, prompt)
                eq_states[(op, a, b)] = states

    # ── patch sweep ──
    print(f"\nPatch sweep ({n_layers} layers × 6 directional pairs):")
    print(f"  Flip success = pred-char matches DONOR's expected first char.")
    print(f"  Skipped pairs where donor and recipient share first char.\n")

    header = "  ".join(f"L{l}".center(7) for l in range(n_layers))
    print(f"  {'direction':>20}  {'n_eligible':>10}  {header}  "
          f"{'no-patch flip':>14}")
    print("  " + "-" * (24 + 13 + 9 * n_layers + 15))

    # For "no-patch flip" baseline: how often does the UNPATCHED recipient
    # prediction already match the donor's expected first char? Should be 0
    # for the cases we keep (where the first chars differ).

    for donor_op in OPS:
        for recipient_op in OPS:
            if donor_op == recipient_op:
                continue
            eligible = []  # (a, b) where donor_fc != recipient_fc
            for a in range(10):
                for b in range(10):
                    if first_char(donor_op, a, b) != first_char(recipient_op,
                                                                a, b):
                        eligible.append((a, b))
            n_eligible = len(eligible)

            # Count unpatched accidental flips
            no_patch_flips = 0
            for a, b in eligible:
                prompt = f"{a}{recipient_op}{b}="
                pred = baseline_pred(model, prompt)
                if pred == first_char(donor_op, a, b):
                    no_patch_flips += 1
            no_patch_rate = no_patch_flips / max(n_eligible, 1)

            flip_rates = []
            for layer in range(n_layers):
                flips = 0
                for a, b in eligible:
                    recipient_prompt = f"{a}{recipient_op}{b}="
                    recipient_ids = torch.tensor(
                        [encode(recipient_prompt)], dtype=torch.long
                    )
                    eq_pos = len(recipient_prompt) - 1
                    donor_state = eq_states[(donor_op, a, b)][layer]
                    logits = patched_forward(
                        model, recipient_ids, donor_state, layer, eq_pos
                    )
                    pred_id = int(logits[0, 0].argmax().item())
                    if ITOS[pred_id] == first_char(donor_op, a, b):
                        flips += 1
                flip_rates.append(flips / max(n_eligible, 1))

            cells = "  ".join(f"{r:.2f}".center(7) for r in flip_rates)
            print(f"  {donor_op}→{recipient_op}                  "
                  f"  {n_eligible:10d}  {cells}  "
                  f"{no_patch_rate:14.2f}")

    # ── digest: compare biggest single-block jumps ──
    print()
    print("Where is the biggest per-layer flip-rate increment?")
    print(f"  (averaged across the 6 directional pairs)")


if __name__ == "__main__":
    main()
