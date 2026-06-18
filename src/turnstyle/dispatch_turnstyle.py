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

Probes come from a per-model ModelProfile (turnstyle.profile), fingerprint-loaded
on init: calibrate once (`fit_choice` + `persist`), then `use_probe(task)` activates
a saved probe in later sessions with no re-fitting. The profile's component map is
the multi-MC-task registry.
"""
from __future__ import annotations

from turnstyle.core import SequenceLogitsProcessor, Turnstyle
from turnstyle.dispatch import Answer, Ctx, run as dispatch_run


class DispatchTurnstyle(Turnstyle):
    """Consumes the Task ADT and grounds generation in its Answer."""

    probe_label = "dispatch"

    def __init__(self, model, tokenizer, device, bias_strength: float = 15.0,
                 choice_artifact=None, legacy_registry=None,
                 profile=None, load_profile_on_init: bool = True):
        super().__init__(model, tokenizer, device, bias_strength)
        if profile is None and load_profile_on_init:
            from turnstyle.profile import load_profile
            profile = load_profile(model)        # fingerprint match, or None
        self.profile = profile
        self.ctx = Ctx(model=model, tokenizer=tokenizer, device=device,
                       choice_artifact=choice_artifact,
                       legacy_registry=legacy_registry)

    @property
    def profile_tasks(self) -> list[str]:
        """Tasks with a calibrated probe in the loaded profile (the MC registry)."""
        return sorted(self.profile.components) if self.profile else []

    def use_probe(self, task: str) -> bool:
        """Activate a calibrated probe from the loaded profile for the MC path —
        no re-fitting. Returns True if the profile ships a probe for `task`."""
        if self.profile is None:
            return False
        art = self.profile.get_probe(task)
        if art is None:
            return False
        self.ctx.choice_artifact = art
        return True

    def fit_choice(self, examples, target_fn=lambda ex: ex["target"].strip(),
                   task: str | None = None, verbose: bool = False):
        """Fit a per-option ChoiceProbe artifact for an MC task (via autoprobe) and
        attach it, so MultipleChoice prompts route through the probe instead of
        abstaining. If `task` is given, the shipped probe is also recorded in the
        profile (call `persist()` to save it). Returns the AutoprobeResult."""
        from turnstyle.autoprobe import autoprobe
        result = autoprobe(examples=examples, target_fn=target_fn,
                           model=self.model, tokenizer=self.tokenizer,
                           device=self.device, verbose=verbose)
        if result.ship and result.fitted is not None and result.chosen is not None:
            self.ctx.choice_artifact = result.fitted
            if task is not None:
                if self.profile is None:
                    from turnstyle.profile import ModelProfile, model_fingerprint
                    self.profile = ModelProfile(
                        fingerprint=model_fingerprint(self.model),
                        model_id=getattr(self.model.config, "_name_or_path", "") or "unknown")
                self.profile.set_probe(task, result.fitted, result.chosen[0],
                                       accuracy=result.chosen[3])
        return result

    def persist(self):
        """Write the calibrated profile to the user cache (fingerprint-addressed).
        Explicit by design — fit_choice does not write to disk on its own."""
        from turnstyle.profile import save_profile
        if self.profile is None:
            raise ValueError("no profile to persist — fit a probe with a task first")
        return save_profile(self.profile)

    def parse(self, prompt: str):
        """Route+solve via the ADT. Returns the Answer, or None on abstain so the
        base class falls back to plain generation."""
        ans = dispatch_run(prompt, self.ctx)
        return ans if isinstance(ans, Answer) else None

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
