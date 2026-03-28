"""Dyck language turnstyle — grounds bracket completion in stack computation.

Handles:
    "Complete the brackets: ( ( ) [ ]"
    "Close the brackets: { [ ( )"
"""

from __future__ import annotations

import re

from turnstyle.core import SequenceLogitsProcessor, Turnstyle

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


class DyckTurnstyle(Turnstyle):
    """Grounds bracket completion in stack-based computation.

        t = DyckTurnstyle(model, tokenizer, device)
        text, proof = t.generate("Complete the brackets: ( ( ) [ ]")
    """

    probe_label = "dyck"

    def parse(self, prompt: str):
        return parse_dyck(prompt)

    def make_processor(self, parsed, max_new_tokens: int):
        open_seq, closing, closing_str = parsed
        answer_ids = self.tokenizer.encode(closing_str, add_special_tokens=False)
        return SequenceLogitsProcessor(
            self.tokenizer, answer_ids, expression=open_seq,
            answer_str=closing_str, bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens)
