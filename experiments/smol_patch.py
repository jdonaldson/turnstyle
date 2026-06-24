"""Activation patching on SmolLM2-1.7B: where (if anywhere) is the operator
dispatcher?

Same setup as NanoGPT `times_table_patch.py`:
  - Capture =-token (last position) hidden state at every layer for mul and
    add prompts.
  - For each (donor, recipient) op pair and each layer L, replace the
    recipient's =-token state after layer L with the donor's, continue the
    forward pass, and check whether the predicted next token now matches
    the donor's expected first character.

Layers indexed 0..23 = LlamaDecoderLayer indices. Hidden states from
`output_hidden_states=True` are (n_layers+1) elements; index L+1 = output
of layer L. To patch "after layer L" we hook `model.model.layers[L]` to
overwrite the last-position output.

Format: `\\n5*3=` (5 tokens). Baseline 100% on mul and add.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def first_char_expected(op: str, a: int, b: int) -> str:
    if op == "*":
        c = a * b
    elif op == "+":
        c = a + b
    else:
        c = a - b
    return "-" if c < 0 else str(c)[0]


def make_patch_hook(donor_state, position_idx):
    """Return a forward_hook that overwrites output[:, position_idx, :]
    with `donor_state` (shape (B, D))."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[:, position_idx, :] = donor_state.to(hs.dtype).to(hs.device)
            return (hs,) + output[1:]
        else:
            hs = output.clone()
            hs[:, position_idx, :] = donor_state.to(hs.dtype).to(hs.device)
            return hs
    return hook


@torch.no_grad()
def batched_forward(model, ids, output_hidden_states=False, hook=None,
                    hook_layer=None):
    if hook is not None:
        handle = model.model.layers[hook_layer].register_forward_hook(hook)
    try:
        out = model(ids, use_cache=False,
                    output_hidden_states=output_hidden_states)
    finally:
        if hook is not None:
            handle.remove()
    return out


def main():
    print(f"Device: {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, local_files_only=True
    ).to(DEVICE).eval()
    n_layers = model.config.num_hidden_layers
    print(f"  layers={n_layers}, hidden={model.config.hidden_size}")

    # Build prompts (100 each for mul and add)
    pairs = [(a, b) for a in range(10) for b in range(10)]
    mul_prompts = [f"\n{a}*{b}=" for a, b in pairs]
    add_prompts = [f"\n{a}+{b}=" for a, b in pairs]
    # All prompts tokenize to the same 5 tokens; last position = 4.
    pos = 4

    mul_ids = torch.tensor(
        [tok.encode(p, add_special_tokens=False) for p in mul_prompts],
        dtype=torch.long, device=DEVICE
    )
    add_ids = torch.tensor(
        [tok.encode(p, add_special_tokens=False) for p in add_prompts],
        dtype=torch.long, device=DEVICE
    )
    assert mul_ids.shape == add_ids.shape == (100, 5), \
        f"unexpected shapes {mul_ids.shape}, {add_ids.shape}"

    # ── Capture donor states + baseline predictions ──
    print("\nCapturing donor hidden states (mul, add) ...")
    with torch.no_grad():
        mul_out = model(mul_ids, use_cache=False, output_hidden_states=True)
        add_out = model(add_ids, use_cache=False, output_hidden_states=True)
    # hidden_states is (n_layers+1) of (B, T, D). Index L+1 = output of layer L.
    mul_states = torch.stack(
        [h[:, pos, :].detach() for h in mul_out.hidden_states], dim=0
    )  # (n_layers+1, 100, 2048)
    add_states = torch.stack(
        [h[:, pos, :].detach() for h in add_out.hidden_states], dim=0
    )
    print(f"  mul_states {tuple(mul_states.shape)}, "
          f"add_states {tuple(add_states.shape)}")

    # Baseline first-char accuracy
    mul_logits = mul_out.logits[:, -1, :]
    add_logits = add_out.logits[:, -1, :]
    mul_pred_ids = mul_logits.argmax(dim=-1).cpu().tolist()
    add_pred_ids = add_logits.argmax(dim=-1).cpu().tolist()
    mul_preds = [tok.decode([i]).lstrip() for i in mul_pred_ids]
    add_preds = [tok.decode([i]).lstrip() for i in add_pred_ids]

    mul_acc = sum(int(p.startswith(first_char_expected("*", a, b)))
                  for p, (a, b) in zip(mul_preds, pairs))
    add_acc = sum(int(p.startswith(first_char_expected("+", a, b)))
                  for p, (a, b) in zip(add_preds, pairs))
    print(f"  baseline mul: {mul_acc}/100, add: {add_acc}/100")

    # Eligible pairs: donor and recipient first-chars differ
    def eligible_mask(donor_op, recipient_op):
        return np.array([
            first_char_expected(donor_op, a, b) !=
            first_char_expected(recipient_op, a, b)
            for a, b in pairs
        ])

    # ── Patch sweep ──
    print("\nPatch sweep: replace =-token state after layer L from donor "
          "→ recipient, check if recipient's prediction matches donor's "
          "expected first char.\n")

    directions = [("*", "+", mul_states, add_ids, mul_acc),
                  ("+", "*", add_states, mul_ids, add_acc)]
    flip_rates = {}

    for donor_op, recipient_op, donor_states, recipient_ids, _ in directions:
        elig = eligible_mask(donor_op, recipient_op)
        n_elig = int(elig.sum())
        donor_chars = [first_char_expected(donor_op, a, b)
                       for a, b in pairs]
        print(f"--- {donor_op} → {recipient_op}   "
              f"({n_elig} eligible pairs) ---")
        print(f"  {'state':>20}  {'flip rate':>10}  "
              f"{'flips/n_elig':>14}")
        # Embedding patch (L = -1, output of embed = hidden_states[0])
        # Use hook on embed_tokens layer? Simpler: don't patch embeddings;
        # report only block outputs L0..L_{n-1}.
        for L in range(n_layers):
            # donor_states[L+1] is output of layer L (batch, hidden).
            donor_state_batch = donor_states[L + 1]  # (100, 2048)
            hook = make_patch_hook(donor_state_batch, pos)
            out = batched_forward(model, recipient_ids, hook=hook,
                                  hook_layer=L)
            preds = out.logits[:, -1, :].argmax(dim=-1).cpu().tolist()
            pred_chars = [tok.decode([i]).lstrip()[:1] for i in preds]
            flips = sum(int(pred_chars[i] == donor_chars[i])
                        for i in range(100) if elig[i])
            rate = flips / max(n_elig, 1)
            flip_rates[(donor_op, recipient_op, L)] = rate
            if L < 6 or L % 4 == 0 or L >= n_layers - 3:
                print(f"  {'L'+str(L):>20}  {rate:>10.3f}  "
                      f"{flips:>5d}/{n_elig}")

    # ── Find biggest increment block ──
    print()
    for donor_op, recipient_op in [("*", "+"), ("+", "*")]:
        rates = [flip_rates[(donor_op, recipient_op, L)]
                 for L in range(n_layers)]
        increments = [(L, rates[L] - (rates[L - 1] if L > 0 else 0))
                      for L in range(n_layers)]
        increments.sort(key=lambda x: -x[1])
        print(f"{donor_op}→{recipient_op}  biggest single-layer flip-rate "
              f"increments:")
        for L, inc in increments[:5]:
            base = rates[L - 1] if L > 0 else 0
            print(f"  L{L}: {base:.3f} → {rates[L]:.3f}  (Δ {inc:+.3f})")
        print()


if __name__ == "__main__":
    main()
