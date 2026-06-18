"""Blackboard architecture substrate for solver primitives.

A Blackboard is shared state. Primitives watch it via Selectors; when a
selector matches, the primitive fires and emits Facts back into the state.
A Registry owns a list of primitives and a dispatch loop runs until the
state quiesces or a terminal answer appears.

This is the standard blackboard pattern (Hearsay-II, 1970s) — primitives are
"knowledge sources", the Blackboard is shared working memory, and dispatch
is the control component. Picked over a flat solver-chain because (a) the
fact graph carries provenance for free via parent_indices, and (b) selectors
make dispatch logic visible in the registry instead of buried inside each
primitive's `applies()` method.

Decisions baked into v1:

  * **kind="answer" with `mode` payload field.** A single answer kind covers
    both picked-from-options ("choice") and computed-result ("result").
    Provenance via parent_indices distinguishes them, and selectors can
    filter via `where=lambda f: f.payload["mode"] == "choice"` if needed.
    Less vocabulary, same expressive power.
  * **Each primitive fires at most once per blackboard.** Strong constraint
    but holds for one-prompt-one-blackboard usage. Multi-fire is a future
    problem.
  * **One primitive per dispatch pass, re-evaluate after.** Slower than
    firing all applicables in a single pass, but correct: primitive B sees
    primitive A's emitted facts. Loop terminates in O(n_primitives) passes.
  * **Priority for ordering, registration order for ties.** Higher priority
    fires first when multiple primitives match in the same pass.
  * **Model/tokenizer/device live in `blackboard.context`.** Primitives that
    need a forward pass pull them from there; primitives that don't (regex
    finders, AST evaluators) ignore the context entirely.

The substrate ships no domain-specific primitives — those live with their
callers (e.g., swollm/primitives/). Reusable structural primitives that
ship with turnstyle live in turnstyle/primitives/ if/when they're added.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ── Facts ────────────────────────────────────────────────────────────────────

@dataclass
class Fact:
    """A piece of evidence emitted by a primitive into the blackboard.

    Facts are kind-tagged (so selectors can match on `.kind`), carry a typed
    payload, record their source primitive, and link to parent facts they
    depend on. Once emitted, a Fact is treated as immutable — primitives that
    refine an earlier fact emit a new one with parent_indices pointing back.
    """
    kind: str
    payload: dict[str, Any]
    source: str
    timestamp: int
    confidence: float = 1.0
    parent_indices: list[int] = field(default_factory=list)

    def __repr__(self):
        parents = f" parents={self.parent_indices}" if self.parent_indices else ""
        conf = f" conf={self.confidence:.2f}" if self.confidence < 1.0 else ""
        return f"Fact[{self.timestamp}] {self.kind} ←{self.source}{parents}{conf}"


# ── Blackboard ───────────────────────────────────────────────────────────────

@dataclass
class Blackboard:
    """Shared working memory: the prompt under solve, the accumulated fact
    graph, and a context dict for runtime resources (model, tokenizer, etc.)
    that primitives may need but aren't worth threading through fact payloads.
    """
    prompt: str
    facts: list[Fact] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def emit(self, kind: str, payload: dict, source: str,
             parent_indices: Optional[list[int]] = None,
             confidence: float = 1.0) -> Fact:
        f = Fact(kind=kind, payload=payload, source=source,
                 timestamp=len(self.facts),
                 parent_indices=parent_indices or [],
                 confidence=confidence)
        self.facts.append(f)
        return f

    def has(self, kind: str) -> list[Fact]:
        return [f for f in self.facts if f.kind == kind]

    def first(self, kind: str) -> Optional[Fact]:
        return next((f for f in self.facts if f.kind == kind), None)

    def terminal_answer(self) -> Optional[Fact]:
        """The most recent answer fact, if any. Dispatch terminates on this."""
        for f in reversed(self.facts):
            if f.kind == "answer":
                return f
        return None


# ── Selectors ────────────────────────────────────────────────────────────────

class Selector(ABC):
    """A predicate over a Blackboard. Subclasses implement matches()."""

    @abstractmethod
    def matches(self, state: Blackboard) -> bool:
        ...


@dataclass
class Has(Selector):
    """Matches when ≥ min_count facts of `kind` exist (optionally filtered
    by a payload predicate)."""
    kind: str
    min_count: int = 1
    where: Optional[Callable[[Fact], bool]] = None

    def matches(self, state: Blackboard) -> bool:
        facts = state.has(self.kind)
        if self.where is not None:
            facts = [f for f in facts if self.where(f)]
        return len(facts) >= self.min_count


@dataclass
class Not(Selector):
    """Negation."""
    inner: Selector

    def matches(self, state: Blackboard) -> bool:
        return not self.inner.matches(state)


@dataclass
class And(Selector):
    """All sub-selectors must match."""
    selectors: list[Selector]

    def matches(self, state: Blackboard) -> bool:
        return all(s.matches(state) for s in self.selectors)


@dataclass
class Or(Selector):
    """Any sub-selector matches."""
    selectors: list[Selector]

    def matches(self, state: Blackboard) -> bool:
        return any(s.matches(state) for s in self.selectors)


class Always(Selector):
    """Unconditional. Used for fallback primitives that should always have a
    chance to fire (logit poll, generate)."""

    def matches(self, state: Blackboard) -> bool:
        del state
        return True


# ── Primitives ───────────────────────────────────────────────────────────────

@dataclass
class Primitive(ABC):
    """A unit of work that watches the blackboard and emits facts when its
    selector matches. Subclasses override fire().

    `priority`: higher fires first when multiple primitives match in the same
    dispatch pass. Default 0; use negative values for fallbacks (logit poll
    at -10) and positive for high-precision primitives (deterministic
    arithmetic at +10).
    """
    name: str
    selector: Selector
    priority: int = 0

    @abstractmethod
    def fire(self, state: Blackboard) -> None:
        ...


# ── Registry + dispatch ──────────────────────────────────────────────────────

@dataclass
class Registry:
    """Holds primitives. Order matters as a tiebreaker (registration order
    when priorities are equal). Caller-side code (e.g. swollm/registry.py)
    instantiates this with task-specific configurations."""
    primitives: list[Primitive] = field(default_factory=list)

    def register(self, primitive: Primitive) -> None:
        self.primitives.append(primitive)


def dispatch(state: Blackboard, registry: Registry,
             max_passes: Optional[int] = None) -> Blackboard:
    """Run primitives against the blackboard until quiescence or a terminal
    answer is emitted.

    Each primitive fires at most once per blackboard. In each pass we collect
    the set of primitives that haven't fired yet and whose selector currently
    matches; we fire the highest-priority one (registration order breaks
    ties), then loop. This means primitive B sees primitive A's effects.

    `max_passes` defaults to `len(registry.primitives) + 1` — enough for one
    fire per primitive plus one quiescence check.
    """
    if max_passes is None:
        max_passes = len(registry.primitives) + 1

    fired_names: set[str] = set()
    for _ in range(max_passes):
        if state.terminal_answer() is not None:
            return state
        candidates = [
            (i, p) for i, p in enumerate(registry.primitives)
            if p.name not in fired_names and p.selector.matches(state)
        ]
        if not candidates:
            return state
        candidates.sort(key=lambda ip: (-ip[1].priority, ip[0]))
        _, p = candidates[0]
        p.fire(state)
        fired_names.add(p.name)
    return state


__all__ = [
    "Fact",
    "Blackboard",
    "Selector",
    "Has",
    "Not",
    "And",
    "Or",
    "Always",
    "Primitive",
    "Registry",
    "dispatch",
]
