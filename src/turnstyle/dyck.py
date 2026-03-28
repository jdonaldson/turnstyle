"""Dyck language turnstyle — grounds bracket completion in stack computation.

Handles:
    "Complete the brackets: ( ( ) [ ]"
    "Close the brackets: { [ ( )"
"""

from __future__ import annotations

import re

from turnstyle.core import SequenceLogitsProcessor, Turnstyle
from turnstyle.extract import ExtractionSpec, FieldSpec

_OPEN_TO_CLOSE = {'(': ')', '[': ']', '{': '}', '<': '>'}
_CLOSE_TO_OPEN = {v: k for k, v in _OPEN_TO_CLOSE.items()}
_ALL_BRACKETS = set(_OPEN_TO_CLOSE.keys()) | set(_OPEN_TO_CLOSE.values())


def parse_dyck(text: str) -> tuple[str, str, str] | None:
    """Extract a bracket sequence and compute the closing brackets.

    Returns (open_sequence, closing_sequence, full_result) or None.
    """
    lower = text.lower()

    # "Complete the brackets: ( ( ) [ ]"
    # "Close the brackets: { [ ( )"
    m = re.search(
        r'(?:complete|close|finish|balance)(?:\s+the)?\s+'
        r'(?:brackets?|parenthes[ei]s|braces)\s*[:=]?\s*'
        r'([\(\)\[\]\{\}<>\s]+)',
        lower,
    )
    if not m:
        return None

    bracket_str = m.group(1).strip()
    brackets = [ch for ch in bracket_str if ch in _ALL_BRACKETS]

    if not brackets:
        return None

    # Compute what's needed to close
    stack = []
    for ch in brackets:
        if ch in _OPEN_TO_CLOSE:
            stack.append(ch)
        elif ch in _CLOSE_TO_OPEN:
            if stack and stack[-1] == _CLOSE_TO_OPEN[ch]:
                stack.pop()
            else:
                # Mismatched close bracket — invalid input
                return None

    if not stack:
        # Already balanced
        return None

    # Close in reverse order (stack top first)
    closing = "".join(_OPEN_TO_CLOSE[ch] for ch in reversed(stack))
    open_seq = " ".join(brackets)
    return open_seq, closing, closing


def _filter_brackets(raw: str) -> str:
    """Keep only bracket characters from raw text."""
    return "".join(ch for ch in raw if ch in _ALL_BRACKETS)


def _assemble_dyck(fields: dict) -> tuple[str, str, str]:
    """Assemble dyck extraction fields into parse() tuple format."""
    raw = fields["brackets"]
    brackets = [ch for ch in raw if ch in _ALL_BRACKETS]

    if not brackets:
        raise ValueError("No brackets found")

    stack = []
    for ch in brackets:
        if ch in _OPEN_TO_CLOSE:
            stack.append(ch)
        elif ch in _CLOSE_TO_OPEN:
            if stack and stack[-1] == _CLOSE_TO_OPEN[ch]:
                stack.pop()
            else:
                raise ValueError(f"Mismatched bracket: {ch}")

    if not stack:
        raise ValueError("Brackets already balanced")

    closing = "".join(_OPEN_TO_CLOSE[ch] for ch in reversed(stack))
    open_seq = " ".join(brackets)
    return open_seq, closing, closing


DYCK_EXTRACTION_SPEC = ExtractionSpec(
    fields=[
        FieldSpec(
            name="brackets",
            prompt_template=(
                "Extract the bracket sequence from this text. "
                "Return only the brackets.\n"
                "Text: {input}\nBrackets:"
            ),
            postprocess=_filter_brackets,
        ),
    ],
    assemble=_assemble_dyck,
)


class DyckTurnstyle(Turnstyle):
    """Grounds bracket completion in stack-based computation.

        t = DyckTurnstyle(model, tokenizer, device)
        text, proof = t.generate("Complete the brackets: ( ( ) [ ]")
    """

    probe_label = "dyck"
    extraction_spec = DYCK_EXTRACTION_SPEC

    def parse(self, prompt: str):
        return None  # routing via probe, fields via extraction

    def make_processor(self, parsed, max_new_tokens: int):
        open_seq, closing, closing_str = parsed
        answer_ids = self.tokenizer.encode(closing_str, add_special_tokens=False)
        return SequenceLogitsProcessor(
            self.tokenizer, answer_ids, expression=open_seq,
            answer_str=closing_str, bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens)
