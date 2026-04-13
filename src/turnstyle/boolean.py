"""Boolean expression turnstyle — grounds boolean evaluation in exact computation.

Handles:
    "True and False"
    "not True or False"
    "True and not False and True"
"""

from __future__ import annotations

import re

from turnstyle.core import SequenceLogitsProcessor, Turnstyle
from turnstyle.extract import ExtractionSpec, FieldSpec


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


def _normalize_boolean_expr(raw: str) -> str:
    """Normalize a boolean expression to valid Python."""
    normalized = raw.strip()
    normalized = re.sub(r'\btrue\b', 'True', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bfalse\b', 'False', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\band\b', 'and', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bor\b', 'or', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bnot\b', 'not', normalized, flags=re.IGNORECASE)
    return normalized


def _assemble_boolean(fields: dict) -> tuple[str, bool, str]:
    """Assemble boolean extraction fields into parse() tuple format."""
    raw = fields["expression"]
    normalized = _normalize_boolean_expr(raw)

    # Validate: only allowed tokens
    tokens = set(normalized.split()) - {'True', 'False', 'and', 'or', 'not', '(', ')'}
    if tokens:
        raise ValueError(f"Invalid tokens in expression: {tokens}")

    result = eval(normalized, {"__builtins__": {}}, {  # noqa: S307
        "True": True, "False": False,
    })
    if not isinstance(result, bool):
        raise ValueError(f"Expression did not evaluate to bool: {result}")

    return normalized, result, str(result)


BOOLEAN_EXTRACTION_SPEC = ExtractionSpec(
    fields=[
        FieldSpec(
            name="expression",
            prompt_template=(
                "Extract the boolean expression (using True, False, and, or, not) "
                "from this text. Return only the expression.\n"
                "Text: {input}\nExpression:"
            ),
        ),
    ],
    assemble=_assemble_boolean,
)


class BooleanTurnstyle(Turnstyle):
    """Grounds boolean expression evaluation in exact computation.

        t = BooleanTurnstyle(model, tokenizer, device)
        text, proof = t.generate("What is True and False?")
    """

    probe_label = "boolean"
    extraction_spec = BOOLEAN_EXTRACTION_SPEC
    examples = [
        'not ( True ) and ( True ) is',
        'True and not not ( not False ) is',
        'not True or False or ( False ) is',
        'False or not ( True ) and False is',
        'True or not False and True and False is',
        'False or not not not False and True is',
        'not True and ( False or True ) is',
        'True and not False or ( True ) is',
        'not True or ( False and True ) is',
        'not True or ( True or False ) is',
        'False or ( False ) and not False is',
        'not False or True and False and False is',
        'not True or False or not not True is',
        'True and True and False and not True is',
        'not not True and not True or True is',
        'not not not ( True and False ) is',
        'not not False and not not not False is',
        '( True and not True and False ) is',
        'False and False and False or not False is',
        'False or ( False and not False ) is',
        'True and False or ( not True ) is',
        'not not ( True ) and not False is',
        'not False or ( True ) and True is',
        'not ( True ) or False or True is',
        '( True and not not not True ) is',
        '( False or not False or False ) is',
        'False and False or True and not False is',
        'not not False or not False or True is',
        'True and not True or False or True is',
        'not False or True and False or False is',
    ]

    def parse(self, prompt: str):
        return None  # routing via probe, fields via extraction

    def make_processor(self, parsed, max_new_tokens: int):
        expression, result, result_str = parsed
        answer_ids = self.tokenizer.encode(result_str, add_special_tokens=False)
        return SequenceLogitsProcessor(
            self.tokenizer, answer_ids, expression=expression,
            answer_str=result_str, bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens)
