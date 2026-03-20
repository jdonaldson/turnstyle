"""Counting turnstyle — grounds string counting in exact computation.

Handles:
    "How many vowels in 'mississippi'?"
    "How many r's in 'strawberry'?"
    "How many words in 'the quick brown fox'?"
    "How many letters in 'hello world'?"
    "How many characters in 'hello world'?"
"""

from __future__ import annotations

import re

from turnstyle.arithmetic import ArithmeticLogitsProcessor
from turnstyle.core import Turnstyle

_VOWELS = set('aeiouAEIOU')


def _short(text: str, max_len: int = 20) -> str:
    """Truncate text for expression display."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "\u2026"


def parse_counting(text: str):
    """Extract counting query from text.

    Returns (target, count_type, result, expression) or None.
    """
    lower = text.lower()

    # Extract target: quoted string after "in"
    m = re.search(r'\bin\s+["\'](.+?)["\']', text)
    if not m:
        # Try double quotes
        m = re.search(r'\bin\s+"(.+?)"', text)
        if not m:
            return None
    target = m.group(1)
    short = _short(target)

    # Keyword-based counting (check these before single-char)
    if re.search(r'\bvowels?\b', lower):
        count = sum(1 for c in target if c in _VOWELS)
        return target, 'vowels', count, f"vowels({short})"

    if re.search(r'\bconsonants?\b', lower):
        count = sum(1 for c in target if c.isalpha() and c not in _VOWELS)
        return target, 'consonants', count, f"consonants({short})"

    if re.search(r'\bwords?\b', lower):
        count = len(target.split())
        return target, 'words', count, f"words({short})"

    if re.search(r'\bletters?\b', lower):
        count = sum(1 for c in target if c.isalpha())
        return target, 'letters', count, f"letters({short})"

    if re.search(r'\bcharacters?\b', lower):
        count = len(target)
        return target, 'characters', count, f"chars({short})"

    # Single character counting: "How many r's in 'strawberry'?"
    m2 = re.search(r'how many\s+(\w)\'?s?\s', lower)
    if m2:
        char = m2.group(1)
        if len(char) == 1 and char.isalpha():
            count = target.lower().count(char)
            return target, f"'{char}'", count, f"count({char},{short})"

    # "Count the r's in 'strawberry'"
    m2 = re.search(r'count\s+(?:the\s+)?(\w)\'?s?\s', lower)
    if m2:
        char = m2.group(1)
        if len(char) == 1 and char.isalpha():
            count = target.lower().count(char)
            return target, f"'{char}'", count, f"count({char},{short})"

    return None


class CountingTurnstyle(Turnstyle):
    """Grounds string counting in exact computation.

        t = CountingTurnstyle(model, tokenizer, device)
        text, proof = t.generate("How many r's in 'strawberry'?")
    """

    def parse(self, prompt: str):
        return parse_counting(prompt)

    def make_processor(self, parsed, max_new_tokens: int):
        _, _, result, expr = parsed
        answer_digits = [int(d) for d in str(result)]

        return ArithmeticLogitsProcessor(
            self.tokenizer, answer_digits, expr, result,
            self.bias_strength, max_new_tokens=max_new_tokens)
