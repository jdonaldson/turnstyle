"""Boolean expression turnstyle — grounds boolean evaluation in exact computation.

Handles:
    "True and False"
    "not True or False"
    "True and not False and True"
"""

from __future__ import annotations

import re

from turnstyle.core import SequenceLogitsProcessor, Turnstyle


def parse_boolean(text: str) -> tuple[str, bool, str] | None:
    """Extract a boolean expression from text and evaluate it.

    Returns (expression, result, result_str) or None.
    Safe evaluation: only allows True, False, and, or, not, parentheses.
    """
    # Find boolean expression in the text
    # Match sequences of True/False connected by and/or/not with optional parens
    pattern = r'(?:(?:not\s+)?(?:True|False)(?:\s+(?:and|or)\s+(?:not\s+)?(?:True|False))*)'
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None

    expr = m.group(0)
    # Normalize case
    normalized = expr
    normalized = re.sub(r'\btrue\b', 'True', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bfalse\b', 'False', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\band\b', 'and', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bor\b', 'or', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bnot\b', 'not', normalized, flags=re.IGNORECASE)

    # Validate: only allowed tokens
    tokens = set(normalized.split()) - {'True', 'False', 'and', 'or', 'not', '(', ')'}
    if tokens:
        return None

    try:
        result = eval(normalized, {"__builtins__": {}}, {  # noqa: S307
            "True": True, "False": False,
        })
    except Exception:
        return None

    if not isinstance(result, bool):
        return None

    return normalized, result, str(result)


class BooleanTurnstyle(Turnstyle):
    """Grounds boolean expression evaluation in exact computation.

        t = BooleanTurnstyle(model, tokenizer, device)
        text, proof = t.generate("What is True and False?")
    """

    probe_label = "boolean"

    def parse(self, prompt: str):
        return parse_boolean(prompt)

    def make_processor(self, parsed, max_new_tokens: int):
        expression, result, result_str = parsed
        answer_ids = self.tokenizer.encode(result_str, add_special_tokens=False)
        return SequenceLogitsProcessor(
            self.tokenizer, answer_ids, expression=expression,
            answer_str=result_str, bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens)
