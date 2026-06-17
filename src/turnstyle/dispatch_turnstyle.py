"""Production wiring for the Task ADT: DispatchTurnstyle.

The Task ADT (turnstyle.dispatch) was an unwired island — it produced Answers
that nothing consumed. This is its first real consumer: a Turnstyle subclass that
routes+solves via dispatch.run() and grounds the model's generation in the result
through the existing SequenceLogitsProcessor.

generate(prompt):
  - dispatch.run() routes the prompt to a variant and solves it → Answer
  - non-abstain  → bias the model to emit Answer.text (immediate), grounded output
  - abstain      → plain generation (the base Turnstyle fallback path)

So the deterministic variants (arithmetic, dyck, web_of_lies, …) ground the
output exactly; multiple-choice routes through the probe when an artifact is
supplied; everything else falls through to the model unbiased.
"""
from __future__ import annotations

from turnstyle.core import SequenceLogitsProcessor, Turnstyle
from turnstyle.dispatch import Ctx, run as dispatch_run


class DispatchTurnstyle(Turnstyle):
    """Consumes the Task ADT and grounds generation in its Answer."""

    probe_label = "dispatch"

    def __init__(self, model, tokenizer, device, bias_strength: float = 15.0,
                 choice_artifact=None, legacy_registry=None):
        super().__init__(model, tokenizer, device, bias_strength)
        self.ctx = Ctx(model=model, tokenizer=tokenizer, device=device,
                       choice_artifact=choice_artifact,
                       legacy_registry=legacy_registry)

    def parse(self, prompt: str):
        """Route+solve via the ADT. Returns the Answer, or None on abstain so the
        base class falls back to plain generation."""
        ans = dispatch_run(prompt, self.ctx)
        return ans if ans.source != "abstain" else None

    def make_processor(self, parsed, max_new_tokens: int):
        """`parsed` is a dispatch.Answer; bias generation toward its text."""
        answer_ids = self.tokenizer.encode(parsed.text, add_special_tokens=False)
        return SequenceLogitsProcessor(
            self.tokenizer,
            answer_ids,
            expression=parsed.source,
            answer_str=parsed.text,
            bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens,
            immediate=True,
        )


__all__ = ["DispatchTurnstyle"]
