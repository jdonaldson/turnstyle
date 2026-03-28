"""Word sorting turnstyle — grounds alphabetical sorting in exact computation.

Handles:
    "Sort the following words: banana apple cherry"
    "Sort [cherry, banana, apple] alphabetically"
"""

from __future__ import annotations

import re

from turnstyle.core import SequenceLogitsProcessor, Turnstyle
from turnstyle.extract import ExtractionSpec, FieldSpec


def parse_sorting(text: str) -> tuple[list[str], list[str], str] | None:
    """Extract a word list from text and return sorted version.

    Returns (original_words, sorted_words, sorted_str) or None.
    """
    lower = text.lower()

    # "Sort the following words: banana apple cherry"
    # "Sort [cherry, banana, apple]"
    # "Sort: banana, apple, cherry"
    m = re.search(
        r'sort(?:ed)?(?:\s+(?:the\s+)?(?:following\s+)?(?:words|list))?'
        r'[\s:]*\[?([a-z][a-z, ]+[a-z])\]?',
        lower,
    )
    if not m:
        return None

    raw = m.group(1)
    # Split on commas and/or spaces
    words = [w.strip() for w in re.split(r'[,\s]+', raw) if w.strip()]

    if len(words) < 2:
        return None

    sorted_words = sorted(words)
    sorted_str = " ".join(sorted_words)
    return words, sorted_words, sorted_str


def _assemble_sorting(fields: dict) -> tuple[list[str], list[str], str]:
    """Assemble sorting extraction fields into parse() tuple format."""
    raw = fields["words"]
    words = [w.strip() for w in re.split(r'[,\s]+', raw) if w.strip()]
    if len(words) < 2:
        raise ValueError("Need at least 2 words to sort")
    sorted_words = sorted(words)
    return words, sorted_words, " ".join(sorted_words)


SORTING_EXTRACTION_SPEC = ExtractionSpec(
    fields=[
        FieldSpec(
            name="words",
            prompt_template=(
                "Extract the list of words to sort from this text. "
                "Return only the words, comma-separated.\n"
                "Text: {input}\nWords:"
            ),
        ),
    ],
    assemble=_assemble_sorting,
)


class SortingTurnstyle(Turnstyle):
    """Grounds alphabetical sorting in exact computation.

        t = SortingTurnstyle(model, tokenizer, device)
        text, proof = t.generate("Sort the following words: banana apple cherry")
    """

    probe_label = "sorting"
    extraction_spec = SORTING_EXTRACTION_SPEC

    def parse(self, prompt: str):
        return None  # routing via probe, fields via extraction

    def make_processor(self, parsed, max_new_tokens: int):
        original_words, sorted_words, sorted_str = parsed
        answer_ids = self.tokenizer.encode(sorted_str, add_special_tokens=False)
        expression = " ".join(original_words)
        return SequenceLogitsProcessor(
            self.tokenizer, answer_ids, expression=expression,
            answer_str=sorted_str, bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens)
