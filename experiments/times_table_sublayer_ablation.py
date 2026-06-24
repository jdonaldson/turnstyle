"""Single-sublayer ablation: zero out attn OR mlp at one block at a time,
measure first-digit accuracy on the 100 mul prompts.

Prediction (gradient-flow view):
  L0 attn:  big drop (operand gathering)
  L1 attn:  ~chance (operator dispatcher)
  L2 attn:  big drop (operand combination)
  L3-L5 attn: small drop, attention is doing minor fine-tuning at best.

  L0-L5 mlp: progressively larger drops, with L5 mlp catastrophic
             (carries the contrastive selector).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import STOI, ITOS, encode, load  # noqa: E402


@torch.no_grad()
def measure_accuracy(model, device="cpu"):
    correct = 0
    for a in range(10):
        for b in range(10):
            prompt = f"{a}*{b}="
            ids = torch.tensor([encode(prompt)], dtype=torch.long,
                               device=device)
            logits, _, _ = model(ids)
            pred = int(logits[0, 0].argmax().item())
            expected = str(a * b)[0]
            if ITOS[pred] == expected:
                correct += 1
    return correct


def make_zero_hook():
    def hook(module, inputs, output):
        return torch.zeros_like(output)
    return hook


def main():
    model = load("cpu")
    n_layers = model.cfg.n_layer

    baseline = measure_accuracy(model)
    print(f"Baseline (no ablation): {baseline}/100")
    print()

    print(f"{'sublayer':>12}  " +
          "  ".join(f"L{L}".center(7) for L in range(n_layers)))
    # Attention ablation per layer
    attn_acc = []
    for L in range(n_layers):
        h = model.blocks[L].attn.register_forward_hook(make_zero_hook())
        try:
            acc = measure_accuracy(model)
        finally:
            h.remove()
        attn_acc.append(acc)
    print(f"  {'attn':>10}  " +
          "  ".join(f"{a:>3}/100".center(7) for a in attn_acc))

    # MLP ablation per layer
    mlp_acc = []
    for L in range(n_layers):
        h = model.blocks[L].mlp.register_forward_hook(make_zero_hook())
        try:
            acc = measure_accuracy(model)
        finally:
            h.remove()
        mlp_acc.append(acc)
    print(f"  {'mlp':>10}  " +
          "  ".join(f"{a:>3}/100".center(7) for a in mlp_acc))

    print()
    print("Drop from baseline (negative = ablation hurts):")
    print(f"  {'attn':>10}  " +
          "  ".join(f"{a-baseline:+4d}".center(7) for a in attn_acc))
    print(f"  {'mlp':>10}  " +
          "  ".join(f"{a-baseline:+4d}".center(7) for a in mlp_acc))

    # Also: ablate ALL attention sublayers at once
    print()
    print("Cumulative ablations:")
    # Ablate attn at multiple late layers
    for late_set in [[3], [4], [5], [3, 4], [3, 4, 5]]:
        handles = []
        for L in late_set:
            handles.append(
                model.blocks[L].attn.register_forward_hook(make_zero_hook())
            )
        try:
            acc = measure_accuracy(model)
        finally:
            for h in handles:
                h.remove()
        labels = ",".join(f"L{L}" for L in late_set)
        print(f"  attn ablated at {labels:<12}  {acc}/100  "
              f"(drop {acc - baseline:+d})")


if __name__ == "__main__":
    main()
