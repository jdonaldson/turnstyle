"""Invariance / perturbation harness — the overfitting meter.

A capability that fit the *structure* of a task survives an answer-preserving change
to its *surface*; one that fit the surface drops. This module generates such variants
so `turnstyle.bbh` can report `acc` next to `acc_perturbed` — the delta IS the
surface-fitting measure, reported per perturbation so you see *which* invariance breaks.

Design rules (so the harness itself stays clean and generalizable):
  - Variants are ANSWER-PRESERVING by construction (the target is recomputed, never
    guessed) — a non-zero delta is real fragility, never a mislabelled key.
  - The default set is DETERMINISTIC and MODEL-FREE (seeded `random.Random`), so the
    same seed reproduces the same variants and the no-model tests pin them exactly.
  - NO keyword/gazetteer lists. Entity-rename and paraphrase want NER / a model to be
    answer-preserving without a name list; introducing a gazetteer here would defeat
    the harness's whole purpose, so they're left as future model-backed perturbations.

v1 ships the two highest-signal MC perturbations:
  - MarkerRestyle: `(A)` → `A.` / `[A]` / `A:` (catches option-format fitting; the
    letter and target are unchanged — only the marker glyph moves)
  - OptionReorder: shuffle option *contents*, re-letter A.. in order, remap the target
    (catches letter-position / answer-letter-prior fitting)

Free-form tasks (arithmetic, word_sorting, navigate) have no options → both return
None and those examples are simply not perturbed (reported as coverage). Answer-
mutating perturbations for them (renumber-and-recompute) are necessarily task-aware
and live with their solvers, not here.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Protocol

# splits on an option marker "(A) "; the trailing \s* is consumed so captured
# content starts at the option text and keeps its own trailing whitespace.
_OPT_SPLIT = re.compile(r"\(([A-Z])\)\s*")
_TARGET_RE = re.compile(r"\(([A-Z])\)")

# marker renderers — each maps a letter to its surface glyph. "paren" is identity.
STYLES = {
    "paren": lambda L: f"({L})",
    "dotted": lambda L: f"{L}.",
    "bracket": lambda L: f"[{L}]",
    "colon": lambda L: f"{L}:",
}


@dataclass
class Variant:
    """An answer-preserving surface variant of an example."""
    name: str       # perturbation id, e.g. "marker:dotted" / "reorder"
    input: str
    target: str


def parse_options(text: str):
    """Parse a trailing MC option block. Returns (head, letters, raw) where `raw[i]`
    is option i's content INCLUDING its trailing whitespace (so a paren re-render is
    byte-identical), or None if there isn't a clean A,B,C,… block of >=2 options."""
    matches = list(_OPT_SPLIT.finditer(text))
    if len(matches) < 2:
        return None
    letters = [m.group(1) for m in matches]
    if letters != [chr(ord("A") + i) for i in range(len(letters))]:
        return None  # non-consecutive ⇒ a stray "(X)" in the body, not an option block
    head = text[: matches[0].start()]
    raw = [text[m.end(): (matches[i + 1].start() if i + 1 < len(matches) else len(text))]
           for i, m in enumerate(matches)]
    return head, letters, raw


class Perturbation(Protocol):
    name: str
    def apply(self, input: str, target: str, rng) -> Variant | None: ...


@dataclass
class MarkerRestyle:
    """Re-render option markers in a different glyph, keeping letters, content, and the
    target unchanged. Isolates option-format fitting: paren-render is the identity."""
    style: str = "dotted"

    @property
    def name(self) -> str:
        return f"marker:{self.style}"

    def apply(self, input: str, target: str, _rng=None) -> Variant | None:
        parsed = parse_options(input)
        if parsed is None or self.style not in STYLES:
            return None
        head, letters, raw = parsed
        render = STYLES[self.style]
        body = head + "".join(f"{render(L)} {c}" for L, c in zip(letters, raw))
        if body == input:
            return None  # paren on already-paren options: no change to test
        return Variant(self.name, body, target)


@dataclass
class OptionReorder:
    """Shuffle option *contents*, re-letter A.. in order, and remap the target to the
    letter that now holds the originally-correct content. Catches answer-letter-prior /
    position fitting. Answer-preserving by construction (target recomputed)."""
    name: str = "reorder"

    def apply(self, input: str, target: str, rng=None) -> Variant | None:
        parsed = parse_options(input)
        if parsed is None:
            return None
        m = _TARGET_RE.fullmatch(target.strip())
        if not m:
            return None  # only MC-letter targets can be remapped answer-preservingly
        rng = rng or random.Random()
        head, letters, raw = parsed
        contents = [c.strip() for c in raw]
        old_idx = letters.index(m.group(1))
        n = len(contents)
        order = list(range(n))
        for _ in range(8):  # avoid the identity permutation when a real shuffle exists
            rng.shuffle(order)
            if order != list(range(n)):
                break
        else:
            return None
        new_contents = [contents[j] for j in order]
        new_letter = letters[order.index(old_idx)]
        body = head.rstrip() + "\n" + "\n".join(
            f"({L}) {new_contents[i]}" for i, L in enumerate(letters))
        return Variant(self.name, body, f"({new_letter})")


def default_perturbations() -> list:
    """The v1 deterministic, model-free, answer-preserving MC perturbation set."""
    return [MarkerRestyle("dotted"), MarkerRestyle("bracket"), OptionReorder()]


__all__ = ["Variant", "Perturbation", "MarkerRestyle", "OptionReorder",
           "parse_options", "default_perturbations", "STYLES"]
