"""Typed dispatch: a `Task` sum type + a total `solve()`, replacing the
stringly-typed `Fact`/`Selector` routing in `blackboard.py`.

Why this exists: `blackboard.py` simulates sum-type dispatch with `Fact(kind:
str, payload: dict)` and a `Has/And/Or` boolean algebra over those string tags.
That's an untyped, non-exhaustive stand-in for an algebraic data type. Here the
shape of a prompt is a real sum type (`Task`), and routing is `match` —
exhaustiveness checked statically by pyright via `typing.assert_never` (keep
pyright in CI; Python won't enforce it at runtime the way rustc would).

Migration strategy (strangler-fig): `parse()` classifies a prompt into a typed
variant for the shapes we model; `solve()` routes each to its solver. Anything
not yet modelled becomes `FreeForm`, which delegates to the legacy blackboard.
As variants move into this ADT the blackboard shrinks; when it's empty, delete
it. The model-touching leaves still call the existing primitives — only the
routing layer changes.

This enum is one `s/@dataclass/enum/` away from the Rust version.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, assert_never

from turnstyle.arithmetic import parse_arithmetic, parse_expression


# ── Sub-variants discovered from the interpretability work (2026-06-16) ────────

class Gather(Enum):
    """Which option-gathering circuit the model uses (clustered from the
    selection-head attention signatures)."""
    BROAD_UNIFORM = "broad_uniform"
    MARKER_LOCKED = "marker_locked"
    SCORE_TOKEN = "score_token"
    POSITIONAL = "positional"


@dataclass(frozen=True)
class PriorLocked:
    """Model committed at L2–3; its native answer is prior-driven, so the probe
    override is high-value here. (mc_selection_two_stage: bimodal decision layer.)"""


@dataclass(frozen=True)
class Deliberated:
    """Model committed late with real cross-option competition."""
    winner_margin: float


SelectionShape = PriorLocked | Deliberated


# ── The Task sum type: the parsed shape of a prompt ───────────────────────────

@dataclass(frozen=True)
class Arithmetic:
    """A symbolic arithmetic prompt. Holding this means parse already evaluated
    it — `value` is the answer."""
    expr: str
    value: str


@dataclass(frozen=True)
class MultipleChoice:
    """A multiple-choice prompt. `gather`/`selection` are populated by the
    model-based detectors (future); cheap structural parse leaves them None."""
    options: list[str]                          # option letters, e.g. ["A", "B"]
    gather: Optional[Gather] = None
    selection: Optional[SelectionShape] = None


@dataclass(frozen=True)
class TruthChain:
    """A web-of-lies truth-propagation prompt. Deterministically solved at parse
    time (like Arithmetic) — holding this means the answer is already known."""
    answer: str           # "Yes" / "No"
    query: str


@dataclass(frozen=True)
class Spatial:
    """A navigate prompt. Deterministically simulated at parse time — holding
    this means the path was walked and the answer is known."""
    answer: str           # "Yes" / "No"


@dataclass(frozen=True)
class FreeForm:
    """No structured variant matched — delegate to the legacy blackboard."""


Task = Arithmetic | MultipleChoice | TruthChain | Spatial | FreeForm


# ── Answer + Context ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Answer:
    text: str
    source: str
    confidence: float = 1.0
    proof: Optional[str] = None


@dataclass
class Ctx:
    """Runtime resources the model-touching leaves need."""
    model: Any = None
    tokenizer: Any = None
    device: str = "cpu"
    choice_artifact: Any = None      # ProbeArtifact (mode="per_option")
    legacy_registry: Any = None      # blackboard Registry for FreeForm fallback


_OPTION_RE = re.compile(r"^\(([A-Z])\)", re.MULTILINE)


# ── parse: raw prompt -> typed Task ("parse, don't validate") ─────────────────

def parse(prompt: str, ctx: Ctx) -> Task:
    """Classify a prompt into exactly one Task variant using cheap detectors.
    Arithmetic is pure; multiple-choice is a structural marker check. The
    expensive model-based work (scoring, gather/selection classification)
    happens in solve(), not here."""
    res = parse_expression(prompt)
    if res is not None:
        expr, value = res
        return Arithmetic(expr=expr, value=str(value))
    res = parse_arithmetic(prompt)
    if res is not None:
        a, b, op, value = res
        return Arithmetic(expr=f"{a}{op}{b}", value=str(value))

    # truth-chain before multiple-choice: web_of_lies prompts often carry
    # (A) Yes / (B) No options, but the deterministic solver is more specific.
    from turnstyle.ir import parse_scene, _wol_solve, _navigate_solve
    scene = parse_scene(prompt)
    wol = _wol_solve(scene.body, scene.question)
    if wol is not None:
        return TruthChain(answer=wol, query=scene.question or "")
    nav = _navigate_solve(scene.body)
    if nav is not None:
        return Spatial(answer=nav)

    letters = _OPTION_RE.findall(prompt)
    if len(letters) >= 2:
        return MultipleChoice(options=letters)

    return FreeForm()


# ── solve: total dispatch over Task ───────────────────────────────────────────

def solve(task: Task, prompt: str, ctx: Ctx) -> Answer:
    match task:
        case Arithmetic(expr, value):
            return Answer(text=value, source="arithmetic", proof=f"{expr} = {value}")

        case TruthChain(answer, query):
            return Answer(text=answer, source="truth_chain",
                          proof=f"truth-propagation; query={query!r}")

        case Spatial(answer):
            return Answer(text=answer, source="navigate", proof="coordinate simulation")

        case MultipleChoice(options, gather, selection):
            return _solve_choice(prompt, options, gather, selection, ctx)

        case FreeForm():
            return _solve_freeform(prompt, ctx)

        case _:
            assert_never(task)   # pyright errors here if a Task variant is unhandled


def run(prompt: str, ctx: Ctx) -> Answer:
    """parse → enrich → solve."""
    return solve(enrich(parse(prompt, ctx), prompt, ctx), prompt, ctx)


# ── enrich: model-based detectors populate the typed fields ───────────────────

def enrich(task: Task, prompt: str, ctx: Ctx) -> Task:
    """Cheap parse leaves model-derived fields None; enrich fills them with a
    forward pass. Kept separate from parse so the cheap path stays cheap."""
    match task:
        case MultipleChoice(options, gather, None) if ctx.model is not None:
            return MultipleChoice(options=options, gather=gather,
                                  selection=detect_selection_shape(prompt, options, ctx))
        case _:
            return task


def detect_selection_shape(prompt: str, options: list[str],
                           ctx: Ctx) -> SelectionShape:
    """Per-prompt decision layer (logit-lens at an `Answer: (` position): if the
    model commits early it's prior-locked (native answer is bias-driven, low
    trust); if late it deliberated. Productizes experiments/option_decision_layer.py."""
    import torch

    tok, mdl = ctx.tokenizer, ctx.model
    n_layers = mdl.config.num_hidden_layers
    cand = torch.tensor([tok.encode(c, add_special_tokens=False)[0] for c in options],
                        device=ctx.device)
    enc = tok(prompt + "\nAnswer: (", return_tensors="pt").to(ctx.device)
    with torch.no_grad():
        hs = mdl(**enc, output_hidden_states=True).hidden_states

        def lens(layer: int) -> "torch.Tensor":
            return mdl.lm_head(mdl.model.norm(hs[layer][0, -1]))[cand]

        picks = [int(torch.argmax(lens(l)).item()) for l in range(n_layers + 1)]
        final = picks[-1]
        decision_layer = n_layers
        for l in range(n_layers, -1, -1):
            if picks[l] == final:
                decision_layer = l
            else:
                break

        if decision_layer <= n_layers // 4:      # bimodal: prior-lock ≈L2-3, else ≈L18+
            return PriorLocked()
        probs = torch.softmax(lens(n_layers), dim=-1)
        top = torch.topk(probs, min(2, len(probs))).values
        margin = float(top[0] - (top[1] if len(top) > 1 else top[0] * 0))
        return Deliberated(winner_margin=margin)


# ── Model-touching leaves: adapt the existing primitives ──────────────────────

def _solve_choice(prompt: str, options: list[str],
                  gather: Optional[Gather], selection: Optional[SelectionShape],
                  ctx: Ctx) -> Answer:
    if ctx.choice_artifact is None or ctx.model is None:
        return _solve_freeform(prompt, ctx)

    from turnstyle.blackboard import Blackboard
    from turnstyle.primitives.choice_probe import ChoiceProbe

    bb = Blackboard(prompt=prompt, context={
        "model": ctx.model, "tokenizer": ctx.tokenizer, "device": ctx.device})
    ChoiceProbe(ctx.choice_artifact).fire(bb)
    ans = bb.terminal_answer()
    if ans is None:
        return _solve_freeform(prompt, ctx)

    # selection shape modulates how much we trust the probe vs. the model:
    # a prior-locked model contributes no real signal, so the probe stands alone;
    # a deliberated model with a healthy margin corroborates it.
    match selection:
        case PriorLocked():
            trust = "model prior-locked (native answer is bias-driven); probe stands alone"
        case Deliberated(margin):
            trust = f"model deliberated (margin={margin:.2f}); corroborates probe"
        case None:
            trust = "selection shape not detected"
        case _:
            assert_never(selection)
    return Answer(text=ans.payload["answer"], source="choice_probe",
                  confidence=ans.confidence,
                  proof=f"scores={ans.payload['scores']} | {trust}")


def _solve_freeform(prompt: str, ctx: Ctx) -> Answer:
    if ctx.legacy_registry is None:
        return Answer(text="", source="abstain", confidence=0.0)
    from turnstyle.blackboard import Blackboard, dispatch as bb_dispatch

    bb = Blackboard(prompt=prompt, context={
        "model": ctx.model, "tokenizer": ctx.tokenizer, "device": ctx.device})
    bb_dispatch(bb, ctx.legacy_registry)
    ans = bb.terminal_answer()
    if ans is None:
        return Answer(text="", source="abstain", confidence=0.0)
    return Answer(text=ans.payload.get("answer", ""), source=ans.source,
                  confidence=ans.confidence)


__all__ = [
    "Gather", "PriorLocked", "Deliberated", "SelectionShape",
    "Arithmetic", "MultipleChoice", "TruthChain", "Spatial", "FreeForm", "Task",
    "Answer", "Ctx", "parse", "enrich", "solve", "run",
    "detect_selection_shape",
]


if __name__ == "__main__":
    # Pure-path smoke (no model): proves dispatch runs and the match is total.
    ctx = Ctx()
    for p in ["What is 3 * (4 + 5)?", "12 - 7 ="]:
        print(f"{p!r:30s} -> {run(p, ctx)}")
    mc = parse("Which is sarcastic?\n(A) foo\n(B) bar", ctx)
    print(f"\nparsed MC: {mc}")
    print(f"solve (no artifact -> abstain): {solve(mc, 'x', ctx)}")
