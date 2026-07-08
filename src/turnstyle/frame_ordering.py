"""frame-as-column: answer superlative questions over IMPLICIT perceptual attributes
by synthesizing the missing numeric column from a semantic frame.

"Which is the biggest? (A) ant (B) whale (C) mouse" has no numeric column to ORDER BY —
SQL can't touch it. A FrameLibrary can: route the attribute ("big") to its frame (size),
project the options onto it (a synthesized column), and pick the extreme. This is the
generalization of the polarity-probe ordering path from adjectives to arbitrary entities.

Guardrails: it only commits when the attribute ROUTES to a known frame (frame membership —
non-attribute words like "forest"/"honest" route to None → abstain) AND there are options.
It is a FALLBACK: explicit numeric data (penguins' table) is handled earlier and must win;
frame coordinates are noisy, so they're for *missing* columns only.
"""
from __future__ import annotations

import re

from turnstyle.sql import extract_options, extract_question

_SUPERLATIVE = re.compile(r"\b(\w{3,}est)\b", re.IGNORECASE)
# (?<![-\w]) — a plain \b matches after a hyphen, so "right-most object" parsed
# as "most object" (and 'object' is a concreteness-frame member → junk commits
# on colored_objects; surfaced by the negation BBH conflict sweep).
_MOST_LEAST = re.compile(r"(?<![-\w])(most|least)\s+(\w+)", re.IGNORECASE)


def _candidates(question: str):
    """(attribute_word, is_least) pairs from superlative surface forms. Over-matching is
    fine — route() filters non-attributes downstream."""
    out = []
    for m in _SUPERLATIVE.finditer(question):
        out.append((m.group(1), False))
    for m in _MOST_LEAST.finditer(question):
        out.append((m.group(2), m.group(1).lower() == "least"))
    return out


def solve_frame_ordering(prompt: str, library, model, tokenizer, device,
                         detector=None) -> str | None:
    """Return the option letter "(A)" whose entity is extreme on the queried implicit
    attribute, or None to abstain.

    `detector` (turnstyle.operators.NegationDetector, optional) makes the path
    negation-aware: negation is applied SYMBOLICALLY — the frame projects the bare
    attribute (frames read early layers, which are negation-blind; the model itself
    does not compose negation), and a hedge-probe verdict on the attribute token
    selects the complement instead. Three-way and precision-first: a detected
    negation commits the complement only when unique; a mid-band verdict abstains
    the whole path (a false verdict in either direction flips the answer; an
    abstain only costs the fallback)."""
    if library is None or not getattr(library, "frames", None):
        return None
    options = extract_options(prompt)
    question = extract_question(prompt) or prompt
    if len(options) < 2:
        return None
    # only rank short, entity-like options — never sentence options (logical_deduction,
    # snarks). Ranking a sentence on a perceptual frame is meaningless; abstain instead.
    if any(len(t.split()) > 4 for t in options.values()):
        return None

    for attr, least in _candidates(question):
        routed = library.route(attr)              # membership routing (no model needed)
        if routed is None:
            continue
        negated = False
        if detector is not None and model is not None:
            verdict = detector.detect_word(model, tokenizer, device, question, attr)
            if verdict is None:                   # mid band: negation status uncertain
                return None
            negated = verdict
        frame, sign = routed
        descending = (sign > 0) != least          # toward the queried pole
        texts = list(options.values())
        ranked = library.rank(texts, frame, model, tokenizer, device,
                              descending=descending)
        if not ranked:
            return None
        winner = ranked[0][0]
        if negated:
            rest = [t for t in texts if t != winner]
            if len(rest) != 1:                    # complement unique or abstain
                return None
            winner = rest[0]
        for letter, text in options.items():
            if text == winner:
                return f"({letter})"
        return None
    return None


__all__ = ["solve_frame_ordering"]
