"""OptionDetector — wraps the option-boundary structural probe artifact.

Detects multiple-choice option boundaries via the L1 per-token binary
classifier shipped at `data/structural_probes/option_boundary.npz`. Emits
one `option` fact per detected option, with the marker-token index, the
last-content-token index (used by downstream choice probes), and the
A/B/C/... label assigned in document order.

The probe is format-agnostic — was trained on a 7-format mix and tested
held-out on `(i)` Roman numerals at F1=1.000. So the same primitive handles
`(A)`, `A.`, `1.`, `Choice A:`, `[A]`, lowercase variants, etc., without
regex.

The forward pass and tokenizer encoding are cached on
`state.context["hidden_states"]` and `state.context["encoded"]` so the next
primitive that needs them (e.g., a ChoiceProbe at the option last-tokens)
doesn't re-run inference.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from turnstyle.autoprobe import load_option_boundary_probe
from turnstyle.blackboard import Blackboard, Has, Not, Primitive


class OptionDetector(Primitive):
    """Per-prompt option-boundary detection. Fires once when no `option`
    facts exist yet; emits one `option` fact per detected option, or zero
    facts if the prompt has no options."""

    def __init__(self, artifact: Optional[dict] = None, priority: int = 5):
        super().__init__(
            name="option_detector",
            selector=Not(Has("option")),
            priority=priority,
        )
        loaded = artifact if artifact is not None else load_option_boundary_probe()
        if loaded is None:
            raise RuntimeError(
                "option_boundary.npz artifact not found at "
                "turnstyle/data/structural_probes/option_boundary.npz — "
                "train it via experiments/train_option_boundary_probe.py"
            )
        self.artifact: dict = loaded

    def fire(self, state: Blackboard) -> None:
        model = state.context.get("model")
        tokenizer = state.context.get("tokenizer")
        device = state.context.get("device")
        if model is None or tokenizer is None or device is None:
            return

        encoded = state.context.get("encoded")
        if encoded is None:
            encoded = tokenizer(
                state.prompt,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            state.context["encoded"] = encoded
        assert encoded is not None

        hidden = state.context.get("hidden_states")
        if hidden is None:
            ids = {
                "input_ids": torch.tensor([encoded["input_ids"]]).to(device),
                "attention_mask": torch.tensor([encoded["attention_mask"]]).to(device),
            }
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            hidden = out.hidden_states
            state.context["hidden_states"] = hidden
        assert hidden is not None

        layer = self.artifact["layer"]
        mean = self.artifact["mean"]
        std = self.artifact["std"]
        weights = self.artifact["weights"]
        bias = self.artifact["bias"]

        h = hidden[layer][0].float().cpu().numpy()
        z = (h - mean) / (std + 1e-9)
        scores = 1.0 / (1.0 + np.exp(-(z @ weights + bias)))

        positives = np.where(scores > 0.5)[0]
        if len(positives) < 2:
            return

        # Group adjacent marker tokens (e.g. "(", "A", ")" → one marker).
        groups = [[int(positives[0])]]
        for p in positives[1:]:
            if p - groups[-1][-1] <= 3:
                groups[-1].append(int(p))
            else:
                groups.append([int(p)])
        starts = [g[-1] for g in groups]

        # Last content token per option = walk back from next-marker-minus-one
        # (or end-of-sequence) skipping whitespace tokens.
        offsets = encoded["offset_mapping"]
        n_tok = len(scores)
        for i, start in enumerate(starts):
            end = starts[i + 1] - 1 if i + 1 < len(starts) else n_tok - 1
            last_idx = end
            while last_idx > start:
                s, e = offsets[last_idx]
                if s != e and state.prompt[s:e].strip():
                    break
                last_idx -= 1
            label = chr(ord("A") + i)
            state.emit(
                kind="option",
                payload={
                    "label": label,
                    "marker_token_idx": start,
                    "last_content_token_idx": last_idx,
                    "marker_score": float(scores[start]),
                },
                source=self.name,
            )
