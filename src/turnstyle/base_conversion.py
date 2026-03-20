"""Base conversion turnstyle — grounds number base conversions in exact computation.

Handles:
    "What is 255 in binary?"
    "What is 255 in hex?"
    "Convert 1010 from binary to decimal"
    "What is 0xFF in decimal?"
    "42 to octal"
"""

from __future__ import annotations

import re

from turnstyle.arithmetic import ArithmeticLogitsProcessor
from turnstyle.core import Turnstyle

# ── base name lookup ──────────────────────────────────────────────────

_BASE_NAMES: dict[str, int] = {
    'binary': 2, 'bin': 2, 'base 2': 2, 'base2': 2,
    'octal': 8, 'oct': 8, 'base 8': 8, 'base8': 8,
    'decimal': 10, 'dec': 10, 'base 10': 10, 'base10': 10,
    'hex': 16, 'hexadecimal': 16, 'base 16': 16, 'base16': 16,
}


def _parse_base(name: str) -> int | None:
    return _BASE_NAMES.get(name.lower().strip())


def _format_result(value: int, base: int) -> str:
    if base == 2:
        return bin(value)[2:]
    elif base == 8:
        return oct(value)[2:]
    elif base == 16:
        return hex(value)[2:]
    else:
        return str(value)


def _parse_number(text: str, base: int | None = None) -> tuple[int, int] | None:
    """Parse a number with optional prefix. Returns (value, detected_base)."""
    text = text.strip()
    if text.startswith(('0x', '0X')):
        try:
            return int(text, 16), 16
        except ValueError:
            return None
    if text.startswith(('0b', '0B')):
        try:
            return int(text, 2), 2
        except ValueError:
            return None
    if text.startswith(('0o', '0O')):
        try:
            return int(text, 8), 8
        except ValueError:
            return None
    if base and base != 10:
        try:
            return int(text, base), base
        except ValueError:
            return None
    try:
        return int(text), 10
    except ValueError:
        return None


# ── expression parsing ────────────────────────────────────────────────

def parse_base_conversion(text: str):
    """Extract base conversion from text.

    Returns (decimal_value, from_base, to_base, result_str, expression) or None.
    """
    lower = text.lower().strip().rstrip('?.')

    # "Convert 1010 from binary to decimal"
    m = re.search(r'convert\s+([\da-fx]+)\s+from\s+(\w+)\s+to\s+(\w+)', lower)
    if m:
        num_text, from_name, to_name = m.group(1), m.group(2), m.group(3)
        from_base = _parse_base(from_name)
        to_base = _parse_base(to_name)
        if from_base and to_base:
            parsed = _parse_number(num_text, from_base)
            if parsed:
                value, _ = parsed
                result_str = _format_result(value, to_base)
                expr = f"{num_text}\u2082{from_base}\u2192\u2082{to_base}"
                return value, from_base, to_base, result_str, expr

    # "Convert 255 to binary" / "Convert 0xFF to decimal"
    m = re.search(r'convert\s+([\da-fx]+)\s+to\s+(\w+)', lower)
    if m:
        num_text, to_name = m.group(1), m.group(2)
        to_base = _parse_base(to_name)
        if to_base:
            parsed = _parse_number(num_text)
            if parsed:
                value, from_base = parsed
                if from_base != to_base:
                    result_str = _format_result(value, to_base)
                    expr = f"{num_text}\u2192base{to_base}"
                    return value, from_base, to_base, result_str, expr

    # "What is 255 in binary?" / "What is 0xFF in decimal?"
    m = re.search(r'what is\s+([\da-fx]+)\s+in\s+(\w+)', lower)
    if m:
        num_text, to_name = m.group(1), m.group(2)
        to_base = _parse_base(to_name)
        if to_base:
            parsed = _parse_number(num_text)
            if parsed:
                value, from_base = parsed
                if from_base != to_base:
                    result_str = _format_result(value, to_base)
                    expr = f"{num_text}\u2192base{to_base}"
                    return value, from_base, to_base, result_str, expr

    # "255 to binary" / "0xff to decimal" / "42 to hex"
    m = re.search(r'([\da-fx]+)\s+to\s+(\w+)', lower)
    if m:
        num_text, to_name = m.group(1), m.group(2)
        to_base = _parse_base(to_name)
        if to_base:
            parsed = _parse_number(num_text)
            if parsed:
                value, from_base = parsed
                if from_base != to_base:
                    result_str = _format_result(value, to_base)
                    expr = f"{num_text}\u2192base{to_base}"
                    return value, from_base, to_base, result_str, expr

    return None


# ── processor with hex support ────────────────────────────────────────

class BaseConversionProcessor(ArithmeticLogitsProcessor):
    """Extends digit biasing to include hex characters a-f."""

    def __init__(self, tokenizer, answer_digits: list[int], expression: str,
                 answer_value: int, bias_strength: float = 15.0,
                 max_new_tokens: int = 50):
        super().__init__(tokenizer, answer_digits, expression, answer_value,
                         bias_strength, max_new_tokens)
        # Add hex letter mappings (10=a, 11=b, ..., 15=f)
        for i, ch in enumerate('abcdef'):
            val = 10 + i
            ids = tokenizer.encode(ch, add_special_tokens=False)
            if ids:
                self.digit_to_token[val] = ids[0]
                self.token_to_digit[ids[0]] = val


class BaseConversionTurnstyle(Turnstyle):
    """Grounds number base conversions in exact computation.

        t = BaseConversionTurnstyle(model, tokenizer, device)
        text, proof = t.generate("What is 255 in binary?")
    """

    def parse(self, prompt: str):
        return parse_base_conversion(prompt)

    def make_processor(self, parsed, max_new_tokens: int):
        decimal_value, from_base, to_base, result_str, expr = parsed

        # Map result characters to digit values (0-15 for hex)
        answer_digits = []
        for ch in result_str:
            if ch.isdigit():
                answer_digits.append(int(ch))
            elif ch.lower() in 'abcdef':
                answer_digits.append(ord(ch.lower()) - ord('a') + 10)

        # Use hex-aware processor when output contains a-f
        if to_base == 16:
            proc = BaseConversionProcessor(
                self.tokenizer, answer_digits, expr, decimal_value,
                self.bias_strength, max_new_tokens)
            proc.proof.answer_charset = "0123456789abcdef"
        else:
            proc = ArithmeticLogitsProcessor(
                self.tokenizer, answer_digits, expr, decimal_value,
                self.bias_strength, max_new_tokens)

        proc.proof.answer_str = result_str
        return proc
