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
from typing import Any, Optional
try:
    from typing import assert_never
except ImportError:  # Python < 3.11
    from typing_extensions import assert_never

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


# Pure symbolic variants — each evaluated at parse time by its shipped solver.
@dataclass(frozen=True)
class Boolean:
    answer: str           # "True" / "False"


@dataclass(frozen=True)
class Dyck:
    answer: str           # closing-bracket sequence


@dataclass(frozen=True)
class Sorting:
    answer: str           # space-joined sorted words


@dataclass(frozen=True)
class DateCalc:
    """Deterministic-SELECTION variant: parse_bbh_date computes the date and
    matches it to an option, so the answer is an option letter, not a date."""
    answer: str           # option letter, e.g. "(B)"


@dataclass(frozen=True)
class Ordering:
    """logical_deduction: constraint extraction + permutation search yields the
    unique ordering, mapped to an option letter. Deterministic-selection (the
    entity-token probe is a research alternative, not the production path)."""
    answer: str           # option letter


@dataclass(frozen=True)
class FormalFallacy:
    """formal_fallacies: NL syllogism → FOL → validity check by interpretation
    enumeration. valid/invalid is mapped to the prompt's option letter, so this
    is deterministic-selection like DateCalc/Ordering."""
    answer: str           # option letter


@dataclass(frozen=True)
class Hyperbaton:
    """hyperbaton: two adjective orderings (permutations); the correct one is sorted
    by decreasing subjectivity. Deterministic-selection via the subjectivity axis."""
    answer: str           # option letter


@dataclass(frozen=True)
class Tabular:
    """penguins_in_a_table: a bespoke data table + a query. Parsed structurally
    (CSV records + add/delete mutations) into SQLite; the model writes the SQL,
    which is executed and matched to an option. Holding this means the SQL solve
    succeeded and the answer (option letter) is known. Knowledge questions (e.g.
    'first name of a famous jazzman') fall through to FreeForm by abstaining."""
    answer: str           # option letter, e.g. "(A)"


@dataclass(frozen=True)
class Tracking:
    """tracking_shuffled_objects: initial actor→item assignments + a chain of
    pairwise swaps. Deterministically simulated at parse time (regex parse of the
    init state + swaps, final state matched to an option) — holding this means
    the swap chain was walked and the answer (option letter) is known."""
    answer: str           # option letter, e.g. "(A)"


@dataclass(frozen=True)
class ObjectCount:
    """object_counting: 'I have a flute, … how many musical instruments?'. The item
    list is parsed structurally (quantities from a closed number-word set); category
    membership is decided by the MODEL's world knowledge (a yes/no scorer), not a
    hardcoded category set; the count is deterministic. The integer answer is free-
    form (no options)."""
    answer: str           # integer as a string, e.g. "8"


@dataclass(frozen=True)
class ColoredObjects:
    """reasoning_about_colored_objects: a row of colored objects + a spatial/color
    query. Deterministically solved at parse time — color is parsed positionally
    (no color list), the query operator is structural, the answer is mapped to an
    option (color/count/yes-no) read from the prompt's own options."""
    answer: str           # option letter, e.g. "(A)"


@dataclass(frozen=True)
class FrameOrdering:
    """A superlative over an IMPLICIT perceptual attribute ('which is the biggest?')
    with no numeric column — answered by synthesizing the column from a semantic frame
    (FrameLibrary.rank). A fallback below the explicit-data solvers; commits only when
    the attribute routes to a known frame. Deterministic-selection (answer = letter)."""
    answer: str           # option letter, e.g. "(A)"


@dataclass(frozen=True)
class FreeForm:
    """No structured variant matched — delegate to the legacy blackboard."""


Task = (Arithmetic | MultipleChoice | TruthChain | Spatial
        | Boolean | Dyck | Sorting | DateCalc | Ordering | FormalFallacy
        | Hyperbaton | Tracking | Tabular | ColoredObjects | ObjectCount
        | FrameOrdering | FreeForm)


# ── Answer + Context ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Answer:
    text: str
    source: str
    confidence: float = 1.0
    proof: Optional[str] = None


@dataclass(frozen=True)
class Abstain:
    """No variant produced an answer — the consumer should escalate (fall back to
    the model, try another solver, …). A real type rather than an empty Answer,
    because a consumer now branches on got-an-answer-vs-not (see DispatchTurnstyle)."""
    reason: str = ""


Result = Answer | Abstain


@dataclass
class Ctx:
    """Runtime resources the model-touching leaves need."""
    model: Any = None
    tokenizer: Any = None
    device: str = "cpu"
    choice_artifact: Any = None      # ProbeArtifact (mode="per_option")
    legacy_registry: Any = None      # blackboard Registry for FreeForm fallback
    polarity_probe: Any = None       # PolarityProbe for the Ordering scalar-adjective poles
    pole_cache: Any = None           # {root: pole} memo, reused across prompts
    subjectivity_axis: Any = None    # deprecated (old 1-axis hyperbaton); kept for profile back-compat
    ordering_classifier: Any = None  # OrderingClassifier for Hyperbaton (lazy-fit, cached)
    sql_turnstyle: Any = None        # cached SQLTurnstyle for the Tabular (penguins) path
    frame_library: Any = None        # FrameLibrary for the FrameOrdering implicit-attribute path


_OPTION_RE = re.compile(r"^\(([A-Z])\)", re.MULTILINE)

# A value-solver (arithmetic/boolean) must EXPLAIN the prompt, not extract a
# fragment — otherwise it trips on incidental numbers/keywords in prose (snarks,
# causal_judgement). commitment = coverage = matched-span / prompt length.
# See memory commitment_coverage_routing.
_VALUE_COVERAGE = 0.5


def _covers(span: str, prompt: str) -> bool:
    p = prompt.strip()
    return bool(p) and len(span) >= _VALUE_COVERAGE * len(p)


# ── parse: raw prompt -> typed Task ("parse, don't validate") ─────────────────

def parse(prompt: str, ctx: Ctx) -> Task:
    """Classify a prompt into exactly one Task variant using cheap detectors.
    Arithmetic is pure; multiple-choice is a structural marker check. The
    expensive model-based work (scoring, gather/selection classification)
    happens in solve(), not here."""
    # dates first: parse_bbh_date is tightly guarded (needs Options: + a parseable
    # "today" + the date question), and must precede arithmetic, which otherwise
    # grabs the digits in a date prompt.
    from turnstyle.dates import parse_bbh_date
    if (letter := parse_bbh_date(prompt)) is not None:
        return DateCalc(answer=letter)

    res = parse_expression(prompt)
    if res is not None and _covers(res[0], prompt):
        expr, value = res
        return Arithmetic(expr=expr, value=str(value))
    res = parse_arithmetic(prompt)
    if res is not None:
        a, b, op, value = res
        expr = f"{a}{op}{b}"
        if _covers(expr, prompt):
            return Arithmetic(expr=expr, value=str(value))

    # other pure symbolic solvers — each returns its answer (or a tuple ending
    # in it) or None. Smart constructors: the variant only builds on success.
    from turnstyle.boolean import parse_boolean
    from turnstyle.dyck import parse_dyck
    from turnstyle.sorting import parse_sorting

    if (r := parse_boolean(prompt)) is not None and _covers(r[0], prompt):
        return Boolean(answer=r[2])
    if (r := parse_dyck(prompt)) is not None:
        return Dyck(answer=r[2])
    # A prompt carrying (A)..(E) option lines is explained by the MultipleChoice
    # frame; don't let a stray "in alphabetical order" phrase in such a table
    # prompt (penguins) mis-commit to a Sorting answer. word_sorting is never MC.
    # See commitment_coverage_routing.
    is_mc = len(_OPTION_RE.findall(prompt)) >= 2
    if not is_mc and (r := parse_sorting(prompt)) is not None:
        return Sorting(answer=r[2])

    # tracking_shuffled_objects: deterministic swap-chain simulation. The solver is
    # tightly gated (needs known actor names + an "At the start" init it can fully
    # parse + a query actor + a matching option), so it's a no-op on other prompts.
    from turnstyle.object_tracking import _solve_tracking
    if (letter := _solve_tracking(prompt)) is not None:
        return Tracking(answer=letter)

    # reasoning_about_colored_objects: deterministic positional-color + structural
    # spatial query. Returns None unless it parses a colored-object scene AND a
    # recognized query that maps to an option, so it's a no-op on other prompts.
    from turnstyle.colored_objects import solve_colored_objects
    if (letter := solve_colored_objects(prompt)) is not None:
        return ColoredObjects(answer=letter)

    # object_counting: cheap structural gate (does "how many X do I have" parse?)
    # then model-classified category membership. Free-answer integer, no options.
    from turnstyle.object_counting import parse_item_list, solve_object_counting
    if parse_item_list(prompt) is not None and ctx.model is not None:
        count = solve_object_counting(prompt, ctx.model, ctx.tokenizer, ctx.device)
        if count is not None:
            return ObjectCount(answer=count)

    # penguins_in_a_table: cheap structural gate (does the bespoke table parse?)
    # runs before the expensive model SQL solve, so this is a no-op on non-table
    # prompts. The SQL solve abstains (None) on knowledge questions → FreeForm.
    if ctx.model is not None:
        from turnstyle.penguins import parse_penguins_tables, solve_penguins
        if parse_penguins_tables(prompt) is not None:
            if ctx.sql_turnstyle is None:
                from turnstyle.sql import SQLTurnstyle
                ctx.sql_turnstyle = SQLTurnstyle(
                    ctx.model, ctx.tokenizer, ctx.device,
                    parse_tables_fn=parse_penguins_tables, probe_label="penguins")
            letter = solve_penguins(prompt, ctx.model, ctx.tokenizer, ctx.device,
                                    sql_turnstyle=ctx.sql_turnstyle)
            if letter is not None:
                return Tabular(answer=letter)

    # logical_deduction: structural frames + symbolic solve; scalar-adjective poles
    # come from the polarity probe (if calibrated) else the regex lexicon fallback.
    from turnstyle.comparison_solver import solve_comparison
    if (letter := solve_comparison(
            prompt, model=ctx.model, tokenizer=ctx.tokenizer, device=ctx.device,
            probe=ctx.polarity_probe, pole_cache=ctx.pole_cache)) is not None:
        return Ordering(answer=letter)

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

    # formal_fallacies: FOL validity check. BBH target is the word valid/invalid
    # (options are dash-bullets, not (A)/(B)), so the verdict IS the answer.
    from turnstyle.formal_fallacies import solve_formal_fallacy
    verdict = solve_formal_fallacy(prompt)
    if verdict is not None:
        return FormalFallacy(answer=verdict)

    # hyperbaton: a permutation-pair of adjective orderings. The cheap structural
    # gate (same words, different order) runs before any model forward, so this is
    # a no-op on other MC prompts. Needs the calibrated subjectivity axis.
    from turnstyle.hyperbaton import (is_hyperbaton, fit_ordering_classifier,
                                      solve_hyperbaton)
    if ctx.model is not None and is_hyperbaton(prompt):
        if ctx.ordering_classifier is None:        # lazy-fit once, cache on ctx
            ctx.ordering_classifier = fit_ordering_classifier(
                ctx.model, ctx.tokenizer, ctx.device)
        if (letter := solve_hyperbaton(prompt, ctx.model, ctx.tokenizer, ctx.device,
                                       ctx.ordering_classifier)) is not None:
            return Hyperbaton(answer=letter)

    # frame-as-column: superlative over an implicit perceptual attribute, no numeric
    # column (the explicit-data solvers above already committed if data was present).
    # Gated on the attribute routing to a known frame, so it's a no-op otherwise.
    if ctx.frame_library is not None and ctx.model is not None:
        from turnstyle.frame_ordering import solve_frame_ordering
        letter = solve_frame_ordering(prompt, ctx.frame_library,
                                      ctx.model, ctx.tokenizer, ctx.device)
        if letter is not None:
            return FrameOrdering(answer=letter)

    letters = _OPTION_RE.findall(prompt)
    if len(letters) >= 2:
        return MultipleChoice(options=letters)

    return FreeForm()


# ── solve: total dispatch over Task ───────────────────────────────────────────

def solve(task: Task, prompt: str, ctx: Ctx) -> Result:
    match task:
        case Arithmetic(expr, value):
            return Answer(text=value, source="arithmetic", proof=f"{expr} = {value}")

        case TruthChain(answer, query):
            return Answer(text=answer, source="truth_chain",
                          proof=f"truth-propagation; query={query!r}")

        case Spatial(answer):
            return Answer(text=answer, source="navigate", proof="coordinate simulation")

        case Boolean(answer):
            return Answer(text=answer, source="boolean")
        case Dyck(answer):
            return Answer(text=answer, source="dyck")
        case Sorting(answer):
            return Answer(text=answer, source="sorting")
        case DateCalc(answer):
            return Answer(text=answer, source="dates", proof="date computed, option matched")
        case Ordering(answer):
            return Answer(text=answer, source="logical_deduction",
                          proof="constraint solve, unique ordering")
        case FormalFallacy(answer):
            return Answer(text=answer, source="formal_fallacies",
                          proof="FOL validity by interpretation enumeration")
        case Hyperbaton(answer):
            return Answer(text=answer, source="hyperbaton",
                          proof="adjectives sorted by decreasing subjectivity")
        case Tracking(answer):
            return Answer(text=answer, source="object_tracking",
                          proof="swap-chain simulation, final state matched")
        case Tabular(answer):
            return Answer(text=answer, source="penguins",
                          proof="structural table parse + text-to-SQL")
        case ColoredObjects(answer):
            return Answer(text=answer, source="colored_objects",
                          proof="positional-color scene + structural spatial query")
        case ObjectCount(answer):
            return Answer(text=answer, source="object_counting",
                          proof="structural item parse + model-classified membership")
        case FrameOrdering(answer):
            return Answer(text=answer, source="frame_ordering",
                          proof="implicit attribute ranked via a semantic frame")

        case MultipleChoice(options, gather, selection):
            return _solve_choice(prompt, options, gather, selection, ctx)

        case FreeForm():
            return _solve_freeform(prompt, ctx)

        case _:
            assert_never(task)   # pyright errors here if a Task variant is unhandled


def run(prompt: str, ctx: Ctx) -> Result:
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
                  ctx: Ctx) -> Result:
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


def _solve_freeform(prompt: str, ctx: Ctx) -> Result:
    if ctx.legacy_registry is None:
        return Abstain(reason="no_handler")
    from turnstyle.blackboard import Blackboard, dispatch as bb_dispatch

    bb = Blackboard(prompt=prompt, context={
        "model": ctx.model, "tokenizer": ctx.tokenizer, "device": ctx.device})
    bb_dispatch(bb, ctx.legacy_registry)
    ans = bb.terminal_answer()
    if ans is None:
        return Abstain(reason="legacy_empty")
    return Answer(text=ans.payload.get("answer", ""), source=ans.source,
                  confidence=ans.confidence)


__all__ = [
    "Gather", "PriorLocked", "Deliberated", "SelectionShape",
    "Arithmetic", "MultipleChoice", "TruthChain", "Spatial",
    "Boolean", "Dyck", "Sorting", "DateCalc", "Ordering", "FormalFallacy",
    "Hyperbaton", "Tracking", "Tabular", "ColoredObjects", "ObjectCount",
    "FrameOrdering", "FreeForm", "Task",
    "Answer", "Abstain", "Result", "Ctx", "parse", "enrich", "solve", "run",
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
