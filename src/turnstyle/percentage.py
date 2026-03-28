"""Percentage turnstyle — grounds percentage calculations in exact computation.

Handles:
    "What is 15% of 230?"
    "What percentage is 45 of 180?"
    "20% tip on $85"
    "25% off 200"
"""

from __future__ import annotations

import re

from turnstyle.arithmetic import ArithmeticLogitsProcessor
from turnstyle.core import Turnstyle
from turnstyle.extract import ExtractionSpec, FieldSpec


def parse_percentage(text: str):
    """Extract percentage calculation from text.

    Returns (pct_or_value, base, operation, result, expression) or None.
    Operations: 'of', 'is_pct', 'off', 'tip'
    """
    lower = text.lower()

    patterns = [
        # "What percentage is 45 of 180?" / "What percent is 45 of 180?"
        (r'what percent(?:age)?\s+is\s+([\d.,]+)\s+of\s+\$?([\d.,]+)', 'is_pct'),
        # "15% of 230" / "What is 15% of $230?"
        (r'([\d.,]+)\s*%\s*of\s+\$?([\d.,]+)', 'of'),
        # "15% off 200" / "15% discount on 200"
        (r'([\d.,]+)\s*%\s*(?:off|discount\s+on)\s+\$?([\d.,]+)', 'off'),
        # "20% tip on $85"
        (r'([\d.,]+)\s*%\s*tip\s+on\s+\$?([\d.,]+)', 'tip'),
    ]

    for pattern, op in patterns:
        m = re.search(pattern, lower)
        if not m:
            continue

        a = float(m.group(1).replace(',', ''))
        b = float(m.group(2).replace(',', ''))

        if op == 'of' or op == 'tip':
            result = round(a / 100 * b, 2)
            expr = f"{a}%\u00d7{b}"
        elif op == 'is_pct':
            if b == 0:
                return None
            result = round(a / b * 100, 2)
            expr = f"{a}/{b}\u00d7100"
        elif op == 'off':
            result = round(b * (1 - a / 100), 2)
            expr = f"{b}-{a}%"
        else:
            continue

        return a, b, op, result, expr

    return None


_PERCENTAGE_OPS = ["of", "is_pct", "off", "tip"]


def _assemble_percentage(fields: dict) -> tuple[float, float, str, float, str]:
    """Assemble percentage extraction fields into parse() tuple format."""
    a = float(fields["value_a"].replace(",", "").strip().lstrip("$"))
    b = float(fields["value_b"].replace(",", "").strip().lstrip("$"))
    op = fields["operation"]

    if op == "of" or op == "tip":
        result = round(a / 100 * b, 2)
        expr = f"{a}%\u00d7{b}"
    elif op == "is_pct":
        if b == 0:
            raise ValueError("Division by zero in percentage calculation")
        result = round(a / b * 100, 2)
        expr = f"{a}/{b}\u00d7100"
    elif op == "off":
        result = round(b * (1 - a / 100), 2)
        expr = f"{b}-{a}%"
    else:
        raise ValueError(f"Unknown percentage operation: {op}")

    return a, b, op, result, expr


PERCENTAGE_EXTRACTION_SPEC = ExtractionSpec(
    fields=[
        FieldSpec(
            name="value_a",
            prompt_template=(
                "What is the first number (percentage or value) in this text? "
                "Return just the number.\nText: {input}\nNumber:"
            ),
        ),
        FieldSpec(
            name="value_b",
            prompt_template=(
                "What is the second number (base value) in this text? "
                "Return just the number.\nText: {input}\nNumber:"
            ),
        ),
        FieldSpec(
            name="operation",
            prompt_template=(
                "What percentage operation is being asked?\n"
                "- of: X% of Y\n"
                "- is_pct: what percentage is X of Y\n"
                "- off: X% off Y (discount)\n"
                "- tip: X% tip on Y\n"
                "Text: {input}\nOperation:"
            ),
            options=_PERCENTAGE_OPS,
        ),
    ],
    assemble=_assemble_percentage,
)


class PercentageTurnstyle(Turnstyle):
    """Grounds percentage calculations in exact computation.

        t = PercentageTurnstyle(model, tokenizer, device)
        text, proof = t.generate("What is 15% of 230?")
    """

    probe_label = "percentage"
    extraction_spec = PERCENTAGE_EXTRACTION_SPEC

    def parse(self, prompt: str):
        return None  # routing via probe, fields via extraction

    def make_processor(self, parsed, max_new_tokens: int):
        _, _, _, result, expr = parsed

        if isinstance(result, float) and result != int(result):
            answer_str = f"{result:.2f}".rstrip('0').rstrip('.')
        else:
            answer_str = str(int(result))

        answer_digits = [int(d) for d in answer_str if d.isdigit()]

        proc = ArithmeticLogitsProcessor(
            self.tokenizer, answer_digits, expr, result,
            self.bias_strength, max_new_tokens=max_new_tokens)
        proc.proof.answer_str = answer_str
        return proc
