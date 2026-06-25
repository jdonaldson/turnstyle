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
class GeometricShape:
    """geometric_shapes: an SVG `<path d="…"/>` is classified deterministically by
    modeling it as a graph (nodes=unique points, edges=typed line/arc) and reading the
    shape off the walk (cycle length / edge kinds / 4-gon geometry). Purely structural;
    the parser doubles as the coverage gate. Answer is an option letter."""
    answer: str           # option letter, e.g. "(B)"


@dataclass(frozen=True)
class FreeForm:
    """No structured variant matched — delegate to the legacy blackboard."""


Task = (Arithmetic | MultipleChoice | TruthChain | Spatial
        | Boolean | Dyck | Sorting | DateCalc | Ordering | FormalFallacy
        | Hyperbaton | Tracking | Tabular | ColoredObjects | ObjectCount
        | FrameOrdering | GeometricShape | FreeForm)


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
    # Position-marginalized choice scoring: score each option over every slot it can
    # occupy (cyclic shifts) and average, so the per-option probe's POSITION component
    # cancels. The probe reads a contextualized last-token hidden state, so its score
    # depends on option order (snarks: 100% natural-order but 37.5% reordered — order
    # luck). Marginalizing makes the answer order-INVARIANT (snarks → 92.5% robustly).
    # Surfaced by the perturbation harness reorder delta. Capped (O(N) passes).
    marginalize_choice: bool = True
    marginalize_cap: int = 8         # skip marginalization above this many options (cost)
    # Zero-shot MC floor: when an MC prompt has NO calibrated probe (and no
    # deterministic variant claimed it), score each option's CONTENT by domain-
    # conditional PMI (logP(content|question) - logP(content|neutral)) and pick the
    # argmax, instead of abstaining to ~chance free generation. Scores content not
    # letters (no letter prior / surface-form competition) and is order-invariant by
    # construction (each option scored as an isolated continuation). The generalizable
    # replacement for swollm's BBH-specific 3-shot poll. O(2N) passes, capped.
    zeroshot_floor: bool = True
    zeroshot_floor_cap: int = 12


_OPTION_RE = re.compile(r"^\(([A-Z])\)", re.MULTILINE)

# Marker-agnostic option intake. Every downstream parser assumes the canonical
# "(A)" glyph; real prompts use many ("A.", "[A]", "A:", "1."). Canonicalize a
# consecutive option block of any of these to "(A)" at the single dispatch entry so
# the solvers generalize across surface format instead of abstaining on it. Purely
# STRUCTURAL (consecutive labels starting at A/1 + a delimiter) — no keyword list;
# idempotent on already-canonical prompts. Surfaced by the perturbation harness
# (turnstyle.perturb): the committed BBH number was conditional on the "(A)" glyph.
_OPT_CAND = re.compile(
    r"^([ \t]*)(?:\(([A-Za-z\d]{1,2})\)|\[([A-Za-z\d]{1,2})\]|([A-Za-z\d]{1,2})[.:])\s+(.*)$")


def _valid_label_seq(labels: list[str]) -> bool:
    """True iff labels are A,B,C… (case-insensitive) or 1,2,3… in order."""
    up = [l.upper() for l in labels]
    if all(len(l) == 1 and l.isalpha() for l in up):
        return up == [chr(ord("A") + k) for k in range(len(up))]
    if all(l.isdigit() for l in labels):
        return [int(l) for l in labels] == list(range(1, len(labels) + 1))
    return False


def normalize_option_markers(prompt: str) -> str:
    """Rewrite a consecutive option block in any supported marker style to canonical
    "(A) …". Returns the prompt unchanged if no valid block is found (idempotent)."""
    lines = prompt.split("\n")
    cand = {}
    for i, ln in enumerate(lines):
        m = _OPT_CAND.match(ln)
        if m:
            cand[i] = (m.group(2) or m.group(3) or m.group(4), m.group(5))
    if not cand:
        return prompt
    idxs = sorted(cand)
    best = None
    j = 0
    while j < len(idxs):                     # walk maximal contiguous runs
        run = [idxs[j]]
        k = j + 1
        while k < len(idxs) and idxs[k] == run[-1] + 1:
            run.append(idxs[k]); k += 1
        if len(run) >= 2 and _valid_label_seq([cand[i][0] for i in run]) \
                and (best is None or len(run) > len(best)):
            best = run
        j = k
    if best is None:
        return prompt
    for pos, i in enumerate(best):
        lines[i] = f"({chr(ord('A') + pos)}) {cand[i][1]}"
    return "\n".join(lines)

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

    # geometric_shapes: deterministic SVG-path classifier (no model). The parser IS the
    # coverage gate — returns None unless a `d="…"` path parses to a shape present in the
    # options, so it's a no-op on every other prompt.
    from turnstyle.geometric_shapes import solve_geometric_shapes
    if (letter := solve_geometric_shapes(prompt)) is not None:
        return GeometricShape(answer=letter)

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
        case GeometricShape(answer):
            return Answer(text=answer, source="geometric_shapes",
                          proof="SVG path classified by graph-walk")

        case MultipleChoice(options, gather, selection):
            return _solve_choice(prompt, options, gather, selection, ctx)

        case FreeForm():
            return _solve_freeform(prompt, ctx)

        case _:
            assert_never(task)   # pyright errors here if a Task variant is unhandled


def run(prompt: str, ctx: Ctx) -> Result:
    """parse → enrich → solve. Option markers are canonicalized to "(A)" first so
    every parser is marker-agnostic (see normalize_option_markers)."""
    prompt = normalize_option_markers(prompt)
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

def _choice_scores(prompt: str, ctx: Ctx) -> Optional[dict]:
    """One ChoiceProbe pass → {label: P(positive)}, or None if it didn't fire."""
    from turnstyle.blackboard import Blackboard
    from turnstyle.primitives.choice_probe import ChoiceProbe
    bb = Blackboard(prompt=prompt, context={
        "model": ctx.model, "tokenizer": ctx.tokenizer, "device": ctx.device})
    ChoiceProbe(ctx.choice_artifact).fire(bb)
    ans = bb.terminal_answer()
    return ans.payload["scores"] if ans is not None else None


def _split_canonical_options(prompt: str):
    """Split a canonicalized prompt into (head, [contents]) for its (A).. block, or
    None. Assumes normalize_option_markers already ran (markers are "(A)")."""
    ms = list(_OPTION_RE.finditer(prompt))
    if len(ms) < 2:
        return None
    head = prompt[: ms[0].start()]
    contents = [prompt[m.end(): (ms[i + 1].start() if i + 1 < len(ms) else len(prompt))].strip()
                for i, m in enumerate(ms)]
    return head, contents


def _score_choice_marginalized(prompt: str, ctx: Ctx):
    """Position-marginalized scoring: present the options in N cyclic shifts (each
    option lands in each slot once), average each option's P(positive) across the
    slots it occupied, and pick the argmax. Returns (answer_text, {label: avg}) or
    None. Order-invariant by construction → strips the probe's position component."""
    parsed = _split_canonical_options(prompt)
    if parsed is None:
        return None
    head, contents = parsed
    n = len(contents)
    agg = [0.0] * n
    for k in range(n):                        # shift k: slot p holds content (p+k)%n
        shifted = [contents[(p + k) % n] for p in range(n)]
        sp = head + "\n".join(f"({chr(ord('A') + p)}) {c}" for p, c in enumerate(shifted))
        scores = _choice_scores(sp, ctx)
        if scores is None:
            return None
        for p in range(n):
            agg[(p + k) % n] += scores.get(chr(ord("A") + p), 0.0)
    avg = {chr(ord("A") + i): agg[i] / n for i in range(n)}
    best = max(avg, key=lambda key: avg[key])
    return ctx.choice_artifact._format(best), avg


# Question-agnostic prior context for domain-conditional PMI: measures each option's
# intrinsic likelihood (length/frequency) so it cancels out of the score.
_PMI_NEUTRAL = "Answer with the correct option."


def _lm_logprob(model, tokenizer, device, context_text: str, continuation: str) -> float:
    """sum logP(continuation tokens | context_text). context_text is already
    chat-templated; continuation is scored token-by-token conditioned on it."""
    import torch
    ctx_ids = tokenizer(context_text, return_tensors="pt").input_ids
    cont_ids = tokenizer(continuation, add_special_tokens=False, return_tensors="pt").input_ids
    if cont_ids.shape[1] == 0:
        return 0.0
    full = torch.cat([ctx_ids, cont_ids], dim=1).to(device)
    with torch.no_grad():
        logp = model(full).logits[0].float().log_softmax(-1)
    n_ctx = ctx_ids.shape[1]
    total = 0.0
    for i in range(cont_ids.shape[1]):                 # logits[t] predicts token t+1
        total += float(logp[n_ctx - 1 + i, cont_ids[0, i]])
    return total


def _score_options_pmi(prompt: str, ctx: Ctx):
    """Domain-conditional PMI per option: logP(content|question) - logP(content|neutral),
    argmax. Returns (answer_text, {label: pmi}) or None. Order-invariant (each option
    scored as an isolated continuation, never inside the option list)."""
    parsed = _split_canonical_options(prompt)
    if parsed is None:
        return None
    head, contents = parsed
    question = head.rsplit("Options:", 1)[0].strip() or head.strip()
    tok = ctx.tokenizer
    q_ctx = tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    n_ctx = tok.apply_chat_template(
        [{"role": "user", "content": _PMI_NEUTRAL}], tokenize=False, add_generation_prompt=True)
    pmis = {}
    for i, c in enumerate(contents):
        cont = " " + c
        pmis[chr(ord("A") + i)] = (_lm_logprob(ctx.model, tok, ctx.device, q_ctx, cont)
                                   - _lm_logprob(ctx.model, tok, ctx.device, n_ctx, cont))
    best = max(pmis, key=lambda key: pmis[key])
    return f"({best})", pmis


def _solve_choice(prompt: str, options: list[str],
                  gather: Optional[Gather], selection: Optional[SelectionShape],
                  ctx: Ctx) -> Result:
    if ctx.choice_artifact is None or ctx.model is None:
        # No calibrated probe → zero-shot content-PMI floor before abstaining.
        if (ctx.zeroshot_floor and ctx.model is not None
                and 2 <= len(options) <= ctx.zeroshot_floor_cap):
            floor = _score_options_pmi(prompt, ctx)
            if floor is not None:
                answer_text, pmis = floor
                return Answer(text=answer_text, source="pmi_floor", confidence=1.0,
                              proof=f"zero-shot content-PMI scores={pmis}")
        return _solve_freeform(prompt, ctx)

    # Position-marginalized path (default): order-invariant, O(N) passes, small N only.
    if ctx.marginalize_choice and 2 <= len(options) <= ctx.marginalize_cap:
        marg = _score_choice_marginalized(prompt, ctx)
        if marg is not None:
            answer_text, scores = marg
            trust = _selection_trust(selection)
            return Answer(text=answer_text, source="choice_probe",
                          confidence=max(scores.values()),
                          proof=f"position-marginalized scores={scores} | {trust}")

    from turnstyle.blackboard import Blackboard
    from turnstyle.primitives.choice_probe import ChoiceProbe

    bb = Blackboard(prompt=prompt, context={
        "model": ctx.model, "tokenizer": ctx.tokenizer, "device": ctx.device})
    ChoiceProbe(ctx.choice_artifact).fire(bb)
    ans = bb.terminal_answer()
    if ans is None:
        return _solve_freeform(prompt, ctx)

    return Answer(text=ans.payload["answer"], source="choice_probe",
                  confidence=ans.confidence,
                  proof=f"scores={ans.payload['scores']} | {_selection_trust(selection)}")


def _selection_trust(selection: Optional[SelectionShape]) -> str:
    """How much to trust the probe vs. the model, given the selection shape: a
    prior-locked model contributes no real signal (probe stands alone); a deliberated
    model with a healthy margin corroborates it."""
    match selection:
        case PriorLocked():
            return "model prior-locked (native answer is bias-driven); probe stands alone"
        case Deliberated(margin):
            return f"model deliberated (margin={margin:.2f}); corroborates probe"
        case None:
            return "selection shape not detected"
        case _:
            assert_never(selection)


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
