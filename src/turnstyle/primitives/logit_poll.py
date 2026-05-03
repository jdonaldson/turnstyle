"""LogitPoll — calibrated next-token probability over option letters.

For each option letter (A, B, C, ...), measure the model's next-token logit
for that letter after the prompt, subtract a neutral-prompt prior to remove
positional bias, and pick the highest score. The fallback for tasks where
no probe ships and no symbolic solver applies — knowledge tasks where the
model has weak-but-nonzero opinions worth extracting.

Calibration follows turnstyle.sql.SQLTurnstyle._logit_poll exactly: prior is
the model's logit distribution after a neutral "Please select one option."
prompt, encoded with the chat template. Cached on `state.context["logit_prior"]`
so repeated calls (same model) don't pay the prior cost.

Priority defaults to -10 — LogitPoll is the last-resort fallback. Higher-
priority primitives (probes, deterministic solvers) fire first; LogitPoll
only triggers when nothing else has emitted an answer.
"""
from __future__ import annotations

from typing import Optional

import torch

from turnstyle.blackboard import Blackboard, Has, Primitive


_PRIOR_LETTERS = "ABCDEFGHIJKLMNOPQR"


def _compute_logit_prior(model, tokenizer, device) -> dict[str, float]:
    """Logit prior over single-letter tokens from a neutral prompt."""
    neutral = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Please select one option."}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(neutral, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    prior: dict[str, float] = {}
    for letter in _PRIOR_LETTERS:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if token_ids:
            prior[letter] = float(logits[token_ids[0]].item())
    return prior


class LogitPoll(Primitive):
    """Prior-corrected option-letter logit poll. Selector: ≥2 option facts."""

    def __init__(self, name: str = "logit_poll", priority: int = -10):
        super().__init__(
            name=name,
            selector=Has("option", min_count=2),
            priority=priority,
        )

    def fire(self, state: Blackboard) -> None:
        model = state.context.get("model")
        tokenizer = state.context.get("tokenizer")
        device = state.context.get("device")
        if model is None or tokenizer is None or device is None:
            return

        prior: Optional[dict[str, float]] = state.context.get("logit_prior")
        if prior is None:
            prior = _compute_logit_prior(model, tokenizer, device)
            state.context["logit_prior"] = prior

        # Score each detected option's letter.
        opts = state.has("option")
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": state.prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1]

        scores: dict[str, float] = {}
        for opt in opts:
            letter = opt.payload["label"]
            token_ids = tokenizer.encode(letter, add_special_tokens=False)
            if not token_ids:
                continue
            raw = float(logits[token_ids[0]].item())
            scores[letter] = raw - prior.get(letter, 0.0)

        if not scores:
            return

        best = max(scores, key=lambda k: scores[k])
        state.emit(
            kind="answer",
            payload={
                "mode": "poll",
                "answer": f"({best})",
                "scores": scores,
            },
            source=self.name,
            parent_indices=[o.timestamp for o in opts],
            confidence=0.5,  # poll is a weak fallback — don't claim high confidence
        )
