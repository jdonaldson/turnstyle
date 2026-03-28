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
from turnstyle.extract import ExtractionSpec, FieldSpec

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


_COUNT_TYPES = ["vowels", "consonants", "words", "letters", "characters", "specific_letter"]


def _assemble_counting(fields: dict) -> tuple[str, str, int, str]:
    """Assemble counting extraction fields into parse() tuple format."""
    target = fields["target"].strip().strip("'\"")
    count_type = fields["count_type"]
    short = _short(target)

    if count_type == "vowels":
        count = sum(1 for c in target if c in _VOWELS)
        return target, "vowels", count, f"vowels({short})"
    elif count_type == "consonants":
        count = sum(1 for c in target if c.isalpha() and c not in _VOWELS)
        return target, "consonants", count, f"consonants({short})"
    elif count_type == "words":
        count = len(target.split())
        return target, "words", count, f"words({short})"
    elif count_type == "letters":
        count = sum(1 for c in target if c.isalpha())
        return target, "letters", count, f"letters({short})"
    elif count_type == "characters":
        count = len(target)
        return target, "characters", count, f"chars({short})"
    elif count_type == "specific_letter":
        letter = fields["specific_letter"].lower()
        count = target.lower().count(letter)
        return target, f"'{letter}'", count, f"count({letter},{short})"
    else:
        raise ValueError(f"Unknown count_type: {count_type}")


COUNTING_EXTRACTION_SPEC = ExtractionSpec(
    fields=[
        FieldSpec(
            name="target",
            prompt_template=(
                "What is the string being analyzed in this text? "
                "Return only the target string, no quotes.\n"
                "Text: {input}\nTarget:"
            ),
        ),
        FieldSpec(
            name="count_type",
            prompt_template=(
                "What type of counting is being asked for?\n"
                "Text: {input}\nType:"
            ),
            options=_COUNT_TYPES,
        ),
        FieldSpec(
            name="specific_letter",
            prompt_template=(
                "What specific letter is being counted? "
                "If no specific letter, answer 'a'.\n"
                "Text: {input}\nLetter:"
            ),
            options=list("abcdefghijklmnopqrstuvwxyz"),
        ),
    ],
    assemble=_assemble_counting,
)


class CountingTurnstyle(Turnstyle):
    """Grounds string counting in exact computation.

        t = CountingTurnstyle(model, tokenizer, device)
        text, proof = t.generate("How many r's in 'strawberry'?")
    """

    probe_label = "counting"
    extraction_spec = COUNTING_EXTRACTION_SPEC

    def parse(self, prompt: str):
        return None  # routing via probe, fields via extraction

    def make_processor(self, parsed, max_new_tokens: int):
        _, _, result, expr = parsed
        answer_digits = [int(d) for d in str(result)]

        return ArithmeticLogitsProcessor(
            self.tokenizer, answer_digits, expr, result,
            self.bias_strength, max_new_tokens=max_new_tokens)
