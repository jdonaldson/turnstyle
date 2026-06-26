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

A per-model FrameLibrary (turnstyle.frame_library) also auto-loads by fingerprint:
`self.frames` / `frame_coordinates(word)` project any word onto the calibrated semantic
frames (affect/size/age/.../number/time). It's a measurement surface — not part of
solve() — and lives in its own .npz store, separate from the profile JSON.
"""
from __future__ import annotations

from turnstyle.core import SequenceLogitsProcessor, Turnstyle
from turnstyle.dispatch import Answer, Ctx, run as dispatch_run


def _augment_option_orderings(examples, k: int = 6, seed: int = 0):
    """Expand MC examples into k option orderings each (identity + random perms),
    remapping the target letter, so a single-mode selection probe learns order-
    invariant selection features instead of a letter-position prior. Examples that
    don't parse as a canonical option block pass through unchanged."""
    import random
    import re
    from turnstyle.dispatch import _split_canonical_options, normalize_option_markers
    LET = re.compile(r"\(([A-Z])\)")
    rng = random.Random(seed)
    out = []
    for ex in examples:
        parsed = _split_canonical_options(normalize_option_markers(ex["input"]))
        m = LET.search(ex.get("target", "") or "")
        if parsed is None or m is None:
            out.append(ex)
            continue
        head, contents = parsed
        n = len(contents)
        correct = ord(m.group(1)) - ord("A")
        if not (0 <= correct < n):
            out.append(ex)
            continue
        perms, seen, tries = [list(range(n))], {tuple(range(n))}, 0
        while len(perms) < k and tries < 50:
            p = list(range(n)); rng.shuffle(p)
            if tuple(p) not in seen:
                perms.append(p); seen.add(tuple(p))
            tries += 1
        for perm in perms:
            new_contents = [contents[perm[s]] for s in range(n)]
            slot = perm.index(correct)
            new_input = head + "\n".join(f"({chr(ord('A') + s)}) {c}"
                                         for s, c in enumerate(new_contents))
            out.append({"input": new_input, "target": f"({chr(ord('A') + slot)})"})
    return out


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
                       legacy_registry=legacy_registry,
                       pole_cache={})
        # activate the model-level polarity probe (Ordering) + subjectivity axis
        # (Hyperbaton) from the profile, if calibrated
        if profile is not None:
            self.ctx.polarity_probe = profile.get_polarity()
            self.ctx.subjectivity_axis = profile.get_subjectivity()
        # Calibrated semantic frames (affect/size/age/.../number/time) — a separate
        # fingerprint-addressed .npz store (kept OUT of the profile JSON). This is a
        # measurement surface on the instance, not wired into solve(); use
        # frame_coordinates() / self.frames. Auto-loads the bundled library by fingerprint.
        self.frames = None
        if load_profile_on_init:
            from turnstyle.frame_library import load_library
            self.frames = load_library(model)
        self.ctx.frame_library = self.frames      # enables the FrameOrdering solve path

    @property
    def profile_tasks(self) -> list[str]:
        """Tasks with a calibrated probe in the loaded profile (the MC registry)."""
        return sorted(self.profile.components) if self.profile else []

    @property
    def frame_names(self) -> list[str]:
        """Names of the auto-loaded semantic frames, or [] if none for this model."""
        return self.frames.names if self.frames else []

    def frame_coordinates(self, word: str):
        """Project a word onto every calibrated semantic frame (affect/size/age/shape/
        space/material/number/time) → {name: signed coordinate}, or None if no
        FrameLibrary is bundled for this model's fingerprint."""
        if self.frames is None:
            return None
        return self.frames.project_word(word, self.model, self.tokenizer, self.device)

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
                   task: str | None = None, verbose: bool = False,
                   ship_threshold_abs: float = 0.60,
                   ship_threshold_lift: float = 0.10):
        """Fit a per-option ChoiceProbe artifact for an MC task (via autoprobe) and
        attach it, so MultipleChoice prompts route through the probe instead of
        abstaining. If `task` is given, the shipped probe is also recorded in the
        profile (call `persist()` to save it). Returns the AutoprobeResult.

        The two ship thresholds gate whether the probe is recorded. They split into
        a *validity* guard (`ship_threshold_lift` — the probe must beat the best cheap
        baseline by this margin, i.e. it's real signal not noise) and a *trust* guard
        (`ship_threshold_abs` — minimum absolute CV accuracy for standalone use). When
        the goal is maximizing aggregate BBH score rather than deploying a trustworthy
        standalone solver, set `ship_threshold_abs=0.0` to keep only the validity guard
        — a recognition probe that beats the cheap baseline but lands <60% (e.g.
        movie_recommendation 50.6%, salient 42.4%) still beats the generation fallback."""
        from turnstyle.autoprobe import autoprobe, DEFAULT_FINDERS
        # The MC dispatch path consumes a per_option ChoiceProbe (with order-marginalized
        # scoring), so restrict the sweep to per_option finders — a single-mode letter
        # classifier would win on some tasks (e.g. movie last_token_of_prompt) but can't
        # be loaded by ChoiceProbe and bypasses the order-robustness guarantees.
        per_option_finders = {n: f for n, f in DEFAULT_FINDERS.items()
                              if n.startswith("per_option")}
        result = autoprobe(examples=examples, target_fn=target_fn,
                           model=self.model, tokenizer=self.tokenizer,
                           device=self.device, verbose=verbose,
                           finders=per_option_finders,
                           ship_threshold_abs=ship_threshold_abs,
                           ship_threshold_lift=ship_threshold_lift)
        return self._record_probe_result(result, task)

    def fit_selection(self, examples, target_fn=lambda ex: ex["target"].strip(),
                      task: str | None = None, verbose: bool = False,
                      ship_threshold_abs: float = 0.0,
                      ship_threshold_lift: float = 0.10,
                      augment_orderings: int = 6):
        """Fit a SINGLE-mode SELECTION probe — a letter classifier read off the prompt's
        final token (the model's own answer intent, the `mc_selection_two_stage`
        "interrupt argmax and solve" readout). For MC tasks where per-option scoring
        underperforms but the model still recognizes the answer late (e.g.
        movie_recommendation: single 50.6% vs per_option no-ship vs generation 22.3%).
        Dispatch routes single-mode artifacts through order-marginalized selection
        scoring. Defaults to ship_threshold_abs=0.0 (these recognizers are often <60%
        but still beat the generation fallback).

        A single-mode probe reads the whole option arrangement, so it carries a letter-
        position prior that inference-time cyclic marginalization can't fully cancel
        (movie reorder Δ -15pp). `augment_orderings` > 1 trains on that many option
        orderings per example (identity + random perms, target remapped), so the probe
        learns order-INVARIANT selection features at the source. NOTE: with augmentation
        the autoprobe CV is leaky (orderings of one example span folds) — judge order-
        robustness by the perturbation harness reorder Δ, not the reported CV."""
        from turnstyle.autoprobe import autoprobe, DEFAULT_FINDERS
        single_finders = {n: f for n, f in DEFAULT_FINDERS.items()
                          if not n.startswith("per_option")}
        train_examples = (_augment_option_orderings(examples, augment_orderings)
                          if augment_orderings > 1 else examples)
        if verbose and augment_orderings > 1:
            print(f"[fit_selection] order-augmented {len(examples)} -> "
                  f"{len(train_examples)} training rows ({augment_orderings}x)", flush=True)
        result = autoprobe(examples=train_examples, target_fn=target_fn,
                           model=self.model, tokenizer=self.tokenizer,
                           device=self.device, verbose=verbose,
                           finders=single_finders,
                           ship_threshold_abs=ship_threshold_abs,
                           ship_threshold_lift=ship_threshold_lift)
        return self._record_probe_result(result, task)

    def _record_probe_result(self, result, task: str | None):
        """Attach a shipped probe to ctx and (if task given) the profile; on no-ship,
        evict any stale probe for that task so a re-calibration can't leave one behind."""
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
        elif task is not None and self.profile is not None:
            self.profile.remove_probe(task)
        return result

    def calibrate_polarity(self, verbose: bool = False):
        """Calibrate the model-level adjective-polarity primitive (detect_polarity),
        activate it for the Ordering path, and record it in the profile. The
        Ordering solver then resolves scalar-adjective poles via the probe
        (cross-lingual) instead of the regex lexicon. Call `persist()` to save."""
        from turnstyle.polarity import detect_polarity
        probe = detect_polarity(self.model, self.tokenizer, self.device)
        self.ctx.polarity_probe = probe
        self.ctx.pole_cache = {}
        if self.profile is None:
            from turnstyle.profile import ModelProfile, model_fingerprint
            self.profile = ModelProfile(
                fingerprint=model_fingerprint(self.model),
                model_id=getattr(self.model.config, "_name_or_path", "") or "unknown")
        self.profile.set_polarity(probe)
        if verbose:
            cap = probe.capability
            print(f"[_polarity] {'SHIP' if cap and cap.ship else 'no-ship'} "
                  f"@L{probe.layer} loo_axis={cap.loo_axis:.3f}")
        return probe

    def calibrate_subjectivity(self, layer: int | None = None, accuracy=None,
                               verbose: bool = False):
        """Fit the model-level subjectivity BipolarAxis (opinion vs material),
        activate it for the Hyperbaton path, and record it in the profile. Call
        `persist()` to save. Built on the same machinery as the polarity primitive."""
        from turnstyle.hyperbaton import fit_subjectivity_axis, DEFAULT_LAYER
        axis = fit_subjectivity_axis(self.model, self.tokenizer, self.device,
                                     layer or DEFAULT_LAYER)
        self.ctx.subjectivity_axis = axis
        if self.profile is None:
            from turnstyle.profile import ModelProfile, model_fingerprint
            self.profile = ModelProfile(
                fingerprint=model_fingerprint(self.model),
                model_id=getattr(self.model.config, "_name_or_path", "") or "unknown")
        self.profile.set_subjectivity(axis, accuracy=accuracy)
        if verbose:
            print(f"[_subjectivity] axis @L{axis.layer}"
                  + (f" acc={accuracy:.3f}" if accuracy is not None else ""))
        return axis

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

    def generate(self, prompt: str, max_new_tokens: int = 50):
        """Canonicalize option markers before generation so the model sees the same
        "(A)" form the parser commits on (grounding to a letter stays coherent) and
        the abstain path gets a clean format. Idempotent on already-canonical input."""
        from turnstyle.dispatch import normalize_option_markers
        return super().generate(normalize_option_markers(prompt), max_new_tokens)

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
