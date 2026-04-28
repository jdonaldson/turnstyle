#!/usr/bin/env python3
"""Prototype: state-aware primitives for the Paskian solver framework.

Goal: prove out the ConversationState + Fact data model BEFORE building any
selector engine. Three primitives that emit facts into shared state:

  L0: OptionBoundaryDetector       — finds option markers
  L1: LongestOptionChoice          — placeholder categorical solver (picks
                                      longest option; correctness doesn't
                                      matter for this test, only fact flow)
  L2: ArithmeticEvaluator          — parses + evaluates an expression

Dispatch is a straw model: fire every applicable primitive in level order.
No selectors yet. Goal is to verify:

  1. The state graph is informative — looking at it tells you what happened.
  2. Primitives compose — L1 consumes L0 facts without explicit coupling.
  3. Dispatch is debuggable — every fact has provenance.
  4. Same state shape works across categorical (snarks-style) and procedural
     (arithmetic) tasks.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/conversation_state_prototype.py
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Fact:
    """A piece of evidence emitted by a primitive into the conversation state.

    Facts are kind-tagged (so future selectors can match on `.kind`), carry a
    typed payload, record their source primitive, and link to parent facts
    they depend on.
    """
    kind: str                                 # "options" | "choice" | …
    payload: dict[str, Any]
    source: str                               # primitive name
    timestamp: int                            # 0..N order of emission
    confidence: float = 1.0
    parent_indices: list[int] = field(default_factory=list)

    def __repr__(self):
        parents = f" parents={self.parent_indices}" if self.parent_indices else ""
        conf = f" conf={self.confidence:.2f}" if self.confidence < 1.0 else ""
        return f"Fact[{self.timestamp}] {self.kind} ←{self.source}{parents}{conf}"


@dataclass
class ConversationState:
    """The accumulated graph of facts produced by primitives. Selectors (later)
    will match patterns over this graph; dispatch (now) just inspects it."""
    prompt: str
    facts: list[Fact] = field(default_factory=list)

    def emit(self, kind: str, payload: dict, source: str,
             parent_indices: list[int] | None = None,
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

    def terminal_answer(self) -> Optional[str]:
        """Return the answer from the most-recently-emitted answer-bearing fact."""
        for f in reversed(self.facts):
            if "answer" in f.payload:
                return str(f.payload["answer"])
        return None


# ── Primitive base ────────────────────────────────────────────────────────────

@dataclass
class Primitive:
    """A unit that emits facts into a ConversationState. Subclasses override
    `applies(state)` and `fire(state)`."""
    level: str = ""    # "L0" | "L1" | "L2" | "meta"
    name: str = ""

    def applies(self, state: ConversationState) -> bool:
        raise NotImplementedError

    def fire(self, state: ConversationState) -> None:
        raise NotImplementedError


# ── Three concrete primitives ─────────────────────────────────────────────────

class OptionBoundaryDetector(Primitive):
    """L0: detect (A)/(B)/... option markers and emit per-option spans."""

    OPTION_RE = re.compile(r"\(([A-Z])\)\s+(.+?)(?=\n\(|\Z)", flags=re.S)

    def __init__(self):
        self.level = "L0"
        self.name = "option_boundary"

    def applies(self, state):
        return self.OPTION_RE.search(state.prompt) is not None

    def fire(self, state):
        positions = []
        for m in self.OPTION_RE.finditer(state.prompt):
            positions.append({
                "label": m.group(1),
                "content": m.group(2).strip(),
                "char_start": m.start(),
                "char_end": m.end(),
            })
        state.emit(
            kind="options",
            payload={"positions": positions, "count": len(positions)},
            source=self.name,
        )


class LongestOptionChoice(Primitive):
    """L1: pick the longest option as the answer.

    This is a placeholder for a real categorical-choice primitive (which
    would be a probe). The point here is to demonstrate that an L1 primitive
    consumes an L0 fact without explicit coupling — it only knows it needs
    something of kind 'options' in state."""

    def __init__(self):
        self.level = "L1"
        self.name = "longest_option_choice"

    def applies(self, state):
        return state.first("options") is not None

    def fire(self, state):
        options_fact = state.first("options")
        opts = options_fact.payload["positions"]
        longest = max(opts, key=lambda o: len(o["content"]))
        state.emit(
            kind="choice",
            payload={
                "answer": f"({longest['label']})",
                "method": "longest_option",
                "considered": [o["label"] for o in opts],
            },
            source=self.name,
            parent_indices=[options_fact.timestamp],
            confidence=0.5,   # placeholder — heuristic, not a real classifier
        )


class ArithmeticEvaluator(Primitive):
    """L2: parse and evaluate an arithmetic expression."""

    EXPR_RE = re.compile(
        r"((?:\d+(?:\.\d+)?)(?:\s*[\+\-\*/]\s*(?:\d+(?:\.\d+)?))+)"
    )

    def __init__(self):
        self.level = "L2"
        self.name = "arithmetic_evaluator"

    def applies(self, state):
        return self.EXPR_RE.search(state.prompt) is not None

    def fire(self, state):
        m = self.EXPR_RE.search(state.prompt)
        expr_text = m.group(1)
        try:
            tree = ast.parse(expr_text, mode="eval")
            value = self._eval(tree.body)
            state.emit(
                kind="arithmetic_evaluation",
                payload={"expression": expr_text, "value": value,
                         "answer": str(value)},
                source=self.name,
            )
        except Exception as e:
            state.emit(
                kind="arithmetic_evaluation",
                payload={"expression": expr_text, "error": str(e)},
                source=self.name,
                confidence=0.0,
            )

    @staticmethod
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            l = ArithmeticEvaluator._eval(node.left)
            r = ArithmeticEvaluator._eval(node.right)
            return {ast.Add: l + r, ast.Sub: l - r,
                    ast.Mult: l * r, ast.Div: l / r}[type(node.op)]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ArithmeticEvaluator._eval(node.operand)
        raise ValueError(f"unsupported node: {ast.dump(node)}")


# ── Straw dispatch ────────────────────────────────────────────────────────────

def dispatch_verbose(state: ConversationState,
                     primitives: list[Primitive]) -> dict:
    """Fire every applicable primitive in level order. Return a trace
    suitable for printing."""
    trace = []
    for level in ("L0", "L1", "L2"):
        for p in primitives:
            if p.level != level:
                continue
            applies = p.applies(state)
            if applies:
                before = len(state.facts)
                p.fire(state)
                emitted = state.facts[before:]
                trace.append({"primitive": p.name, "level": level,
                              "fired": True, "emitted": emitted})
            else:
                trace.append({"primitive": p.name, "level": level,
                              "fired": False, "emitted": []})
    return {
        "trace": trace,
        "answer": state.terminal_answer(),
        "facts": list(state.facts),
    }


# ── Pretty-printer ───────────────────────────────────────────────────────────

def print_run(prompt: str, result: dict):
    print(f"\n{'='*70}")
    print(f"Prompt: {prompt[:80]}{'…' if len(prompt) > 80 else ''}")
    print(f"{'='*70}")
    for entry in result["trace"]:
        marker = "✓" if entry["fired"] else "·"
        suffix = ""
        if entry["fired"]:
            kinds = ", ".join(f.kind for f in entry["emitted"])
            suffix = f" → emitted [{kinds}]"
        print(f"  {marker} {entry['level']:>3}  {entry['primitive']:25s}{suffix}")

    print(f"\n  State graph ({len(result['facts'])} fact(s)):")
    for f in result["facts"]:
        keys = ", ".join(k for k in f.payload.keys())
        print(f"    {f}   payload={{{keys}}}")

    print(f"\n  Terminal answer: {result['answer']!r}")


# ── Test ──────────────────────────────────────────────────────────────────────

TEST_PROMPTS = [
    # Categorical-choice (snarks-shape)
    "Which statement is sarcastic?\nOptions:\n"
    "(A) He's a generous person\n"
    "(B) He's a terrible person doing the same kind charity work",
    # Arithmetic
    "What is 2 + 3 * 4?",
    # Plain text — neither primitive applies
    "This is a plain text prompt without any structure.",
    # MIXED: has both options AND arithmetic. What do we get?
    "Compute the answer:\n"
    "Options:\n"
    "(A) Twelve\n"
    "(B) Ten\n"
    "What is 2 + 5 * 2?",
]


def main():
    primitives = [
        OptionBoundaryDetector(),
        LongestOptionChoice(),
        ArithmeticEvaluator(),
    ]
    for prompt in TEST_PROMPTS:
        state = ConversationState(prompt=prompt)
        result = dispatch_verbose(state, primitives)
        print_run(prompt, result)


if __name__ == "__main__":
    main()
