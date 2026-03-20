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


class PercentageTurnstyle(Turnstyle):
    """Grounds percentage calculations in exact computation.

        t = PercentageTurnstyle(model, tokenizer, device)
        text, proof = t.generate("What is 15% of 230?")
    """

    def parse(self, prompt: str):
        return parse_percentage(prompt)

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
