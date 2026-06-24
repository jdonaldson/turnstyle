"""SmolLM2-360M-Instruct activation patching: third data point for the
dispatcher-depth question.

NanoGPT (6 layers):  dispatcher at L1   (17% depth)
SmolLM2-1.7B (24L):  dispatcher at L15  (62.5% depth)
SmolLM2-360M (32L):  ???

Tries baseline format compatibility first; falls back to few-shot if raw
`\\n5*3=` doesn't hit 100%.

Same setup as `smol_patch.py`: =-token (last position) patching, mul ↔ add.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
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
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[:, position_idx, :] = donor_state.to(hs.dtype).to(hs.device)
            return (hs,) + output[1:]
        hs = output.clone()
        hs[:, position_idx, :] = donor_state.to(hs.dtype).to(hs.device)
        return hs
    return hook


@torch.no_grad()
def baseline_for_format(model, tok, fmt_fn, device):
    """Return (mul_acc, add_acc, mul_ids, add_ids, last_pos)."""
    pairs = [(a, b) for a in range(10) for b in range(10)]
    mul_prompts = [fmt_fn("*", a, b) for a, b in pairs]
    add_prompts = [fmt_fn("+", a, b) for a, b in pairs]
    # Verify all share the same token-length
    mul_token_lens = [len(tok.encode(p, add_special_tokens=False))
                      for p in mul_prompts]
    add_token_lens = [len(tok.encode(p, add_special_tokens=False))
                      for p in add_prompts]
    if (len(set(mul_token_lens)) != 1 or
        len(set(add_token_lens)) != 1 or
        mul_token_lens[0] != add_token_lens[0]):
        return None, None, None, None, None, "ragged token lengths"
    L_tok = mul_token_lens[0]
    mul_ids = torch.tensor(
        [tok.encode(p, add_special_tokens=False) for p in mul_prompts],
        dtype=torch.long, device=device
    )
    add_ids = torch.tensor(
        [tok.encode(p, add_special_tokens=False) for p in add_prompts],
        dtype=torch.long, device=device
    )
    last_pos = L_tok - 1

    mul_out = model(mul_ids, use_cache=False)
    add_out = model(add_ids, use_cache=False)
    mul_pred = mul_out.logits[:, -1, :].argmax(dim=-1).cpu().tolist()
    add_pred = add_out.logits[:, -1, :].argmax(dim=-1).cpu().tolist()
    mul_chars = [tok.decode([i]).lstrip()[:1] for i in mul_pred]
    add_chars = [tok.decode([i]).lstrip()[:1] for i in add_pred]
    mul_acc = sum(int(c == first_char_expected("*", a, b))
                  for c, (a, b) in zip(mul_chars, pairs))
    add_acc = sum(int(c == first_char_expected("+", a, b))
                  for c, (a, b) in zip(add_chars, pairs))
    return mul_acc, add_acc, mul_ids, add_ids, last_pos, None


def main():
    print(f"Device: {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, local_files_only=True
    ).to(DEVICE).eval()
    n_layers = model.config.num_hidden_layers
    print(f"  layers={n_layers}, hidden={model.config.hidden_size}, "
          f"vocab={model.config.vocab_size}")

    # Try several formats; pick the one with highest mul+add baseline
    fmts = {
        "raw":     (lambda op, a, b: f"{a}{op}{b}="),
        "newline": (lambda op, a, b: f"\n{a}{op}{b}="),
        "fewshot": (lambda op, a, b:
                    f"{ {'*':'1*2=2', '+':'1+2=3', '-':'3-1=2'}[op] }\n"
                    f"{ {'*':'3*4=12', '+':'3+4=7', '-':'7-4=3'}[op] }\n"
                    f"{a}{op}{b}="),
    }
    best = None
    for name, fn in fmts.items():
        with torch.no_grad():
            res = baseline_for_format(model, tok, fn, DEVICE)
        mul_acc, add_acc, mul_ids, add_ids, last_pos, err = res
        if err is not None:
            print(f"  format={name}: SKIPPED ({err})")
            continue
        print(f"  format={name}: mul={mul_acc}/100, add={add_acc}/100, "
              f"prompt_len={last_pos+1} tokens, last_pos={last_pos}")
        if best is None or (mul_acc + add_acc) > (best[0] + best[1]):
            best = (mul_acc, add_acc, mul_ids, add_ids, last_pos, name)

    mul_acc, add_acc, mul_ids, add_ids, last_pos, fmt_name = best
    print(f"\nUsing format={fmt_name!r}  (mul={mul_acc}, add={add_acc})")

    if min(mul_acc, add_acc) < 90:
        print("Baseline accuracy < 90%. Patching results would be noisy. "
              "Proceeding anyway but interpret with caution.")

    # Capture donor states
    with torch.no_grad():
        mul_out = model(mul_ids, use_cache=False, output_hidden_states=True)
        add_out = model(add_ids, use_cache=False, output_hidden_states=True)
    mul_states = torch.stack(
        [h[:, last_pos, :].detach() for h in mul_out.hidden_states], dim=0
    )
    add_states = torch.stack(
        [h[:, last_pos, :].detach() for h in add_out.hidden_states], dim=0
    )

    # Eligible pairs per direction
    pairs = [(a, b) for a in range(10) for b in range(10)]

    def elig(donor_op, recipient_op):
        return np.array([
            first_char_expected(donor_op, a, b) !=
            first_char_expected(recipient_op, a, b)
            for a, b in pairs
        ])

    directions = [("*", "+", mul_states, add_ids),
                  ("+", "*", add_states, mul_ids)]
    flip_rates = {}

    for donor_op, recipient_op, donor_states, recipient_ids in directions:
        e = elig(donor_op, recipient_op)
        n_e = int(e.sum())
        donor_chars = [first_char_expected(donor_op, a, b) for a, b in pairs]
        print(f"\n--- {donor_op}→{recipient_op}  ({n_e} eligible) ---")
        for L in range(n_layers):
            donor_state_batch = donor_states[L + 1]
            hook = make_patch_hook(donor_state_batch, last_pos)
            handle = model.model.layers[L].register_forward_hook(hook)
            try:
                out = model(recipient_ids, use_cache=False)
            finally:
                handle.remove()
            preds = out.logits[:, -1, :].argmax(dim=-1).cpu().tolist()
            pred_chars = [tok.decode([i]).lstrip()[:1] for i in preds]
            flips = sum(int(pred_chars[i] == donor_chars[i])
                        for i in range(100) if e[i])
            rate = flips / max(n_e, 1)
            flip_rates[(donor_op, recipient_op, L)] = rate

    # Full table
    print()
    print(f"{'L':>3}  {'L/n_layers':>10}  "
          f"{'*→+ flip':>10}  {'+→* flip':>10}")
    for L in range(n_layers):
        print(f"  L{L:<2d}  {L/n_layers:>10.3f}  "
              f"{flip_rates[('*','+',L)]:>10.3f}  "
              f"{flip_rates[('+','*',L)]:>10.3f}")

    # Biggest jumps
    print()
    for donor_op, recipient_op in [("*", "+"), ("+", "*")]:
        rates = [flip_rates[(donor_op, recipient_op, L)]
                 for L in range(n_layers)]
        increments = [(L, rates[L] - (rates[L - 1] if L > 0 else 0))
                      for L in range(n_layers)]
        increments.sort(key=lambda x: -x[1])
        print(f"{donor_op}→{recipient_op}  biggest jumps:")
        for L, inc in increments[:5]:
            base = rates[L - 1] if L > 0 else 0
            print(f"  L{L}  (depth {L/n_layers:.2f}): "
                  f"{base:.3f} → {rates[L]:.3f}  (Δ {inc:+.3f})")
        print()


if __name__ == "__main__":
    main()
