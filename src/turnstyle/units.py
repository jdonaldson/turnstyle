"""Unit conversion turnstyle — grounds unit conversions in exact computation.

Handles:
    "How many km is 26.2 miles?"
    "Convert 100 fahrenheit to celsius"
    "What is 5 kg in pounds?"
    "150 lbs to kg"
"""

from __future__ import annotations

import re

from turnstyle.arithmetic import ArithmeticLogitsProcessor
from turnstyle.core import CoprocessorDiagnostic, Turnstyle

# ── conversion table ─────────────────────────────────────────────────
# Each entry: (from_unit, to_unit) → multiply factor
# Reverse conversions are auto-generated.

_CONVERSIONS: dict[tuple[str, str], float] = {
    # Distance
    ('mile', 'km'): 1.60934,
    ('mile', 'meter'): 1609.34,
    ('foot', 'meter'): 0.3048,
    ('foot', 'cm'): 30.48,
    ('inch', 'cm'): 2.54,
    ('inch', 'mm'): 25.4,
    ('yard', 'meter'): 0.9144,
    ('km', 'meter'): 1000.0,
    ('meter', 'cm'): 100.0,
    ('cm', 'mm'): 10.0,
    # Weight
    ('pound', 'kg'): 0.453592,
    ('pound', 'gram'): 453.592,
    ('ounce', 'gram'): 28.3495,
    ('kg', 'gram'): 1000.0,
    ('ton', 'kg'): 907.185,       # US short ton
    ('tonne', 'kg'): 1000.0,      # metric ton
    # Volume
    ('gallon', 'liter'): 3.78541,
    ('quart', 'liter'): 0.946353,
    ('cup', 'ml'): 236.588,
    ('pint', 'ml'): 473.176,
    ('liter', 'ml'): 1000.0,
    ('tablespoon', 'ml'): 14.787,
    ('teaspoon', 'ml'): 4.929,
    # Speed
    ('mph', 'kph'): 1.60934,
    ('knot', 'kph'): 1.852,
}

# Auto-generate reverse conversions
_REVERSE = {}
for (a, b), factor in _CONVERSIONS.items():
    _REVERSE[(b, a)] = 1.0 / factor
_CONVERSIONS.update(_REVERSE)

# Temperature is special (not a simple multiply)
_TEMP_CONVERSIONS = {
    ('fahrenheit', 'celsius'): lambda f: (f - 32) * 5 / 9,
    ('celsius', 'fahrenheit'): lambda c: c * 9 / 5 + 32,
    ('celsius', 'kelvin'): lambda c: c + 273.15,
    ('kelvin', 'celsius'): lambda k: k - 273.15,
    ('fahrenheit', 'kelvin'): lambda f: (f - 32) * 5 / 9 + 273.15,
    ('kelvin', 'fahrenheit'): lambda k: (k - 273.15) * 9 / 5 + 32,
}

# ── unit name normalization ──────────────────────────────────────────

_ALIASES: dict[str, str] = {
    'miles': 'mile', 'mi': 'mile',
    'kilometers': 'km', 'kilometre': 'km', 'kilometres': 'km', 'kilometer': 'km',
    'meters': 'meter', 'metre': 'meter', 'metres': 'meter', 'm': 'meter',
    'centimeters': 'cm', 'centimetre': 'cm', 'centimetres': 'cm', 'centimeter': 'cm',
    'millimeters': 'mm', 'millimetre': 'mm', 'millimetres': 'mm', 'millimeter': 'mm',
    'feet': 'foot', 'ft': 'foot',
    'inches': 'inch', 'in': 'inch',
    'yards': 'yard', 'yd': 'yard',
    'pounds': 'pound', 'lbs': 'pound', 'lb': 'pound',
    'kilograms': 'kg', 'kilogram': 'kg', 'kgs': 'kg',
    'grams': 'gram', 'g': 'gram',
    'ounces': 'ounce', 'oz': 'ounce',
    'tons': 'ton',
    'tonnes': 'tonne', 'metric ton': 'tonne', 'metric tons': 'tonne',
    'gallons': 'gallon', 'gal': 'gallon',
    'quarts': 'quart', 'qt': 'quart',
    'cups': 'cup',
    'pints': 'pint', 'pt': 'pint',
    'liters': 'liter', 'litre': 'liter', 'litres': 'liter', 'l': 'liter',
    'milliliters': 'ml', 'milliliter': 'ml', 'millilitre': 'ml',
    'tablespoons': 'tablespoon', 'tbsp': 'tablespoon',
    'teaspoons': 'teaspoon', 'tsp': 'teaspoon',
    'kph': 'kph', 'km/h': 'kph', 'kmh': 'kph',
    'mph': 'mph',
    'knots': 'knot', 'kt': 'knot',
    'fahrenheit': 'fahrenheit', 'f': 'fahrenheit', '°f': 'fahrenheit',
    'celsius': 'celsius', 'c': 'celsius', '°c': 'celsius', 'centigrade': 'celsius',
    'kelvin': 'kelvin', 'k': 'kelvin',
}


def _normalize_unit(text: str) -> str:
    text = text.lower().strip().strip('°')
    return _ALIASES.get(text, text)


def _convert(value: float, from_unit: str, to_unit: str) -> float | None:
    """Convert value between units. Returns None if conversion unknown."""
    key = (from_unit, to_unit)

    # Temperature (special formulas)
    if key in _TEMP_CONVERSIONS:
        return _TEMP_CONVERSIONS[key](value)

    # Linear conversions
    if key in _CONVERSIONS:
        return value * _CONVERSIONS[key]

    return None


# ── expression parsing ───────────────────────────────────────────────

def parse_unit_conversion(text: str):
    """Extract unit conversion from text.

    Returns (value, from_unit, to_unit, result, expression) or None.
    """
    lower = text.lower()

    patterns = [
        # "How many km is 26.2 miles?"
        r'how many\s+(\w+)\s+(?:is|are|in)\s+([\d.,]+)\s+(\w+)',
        # "Convert 100 fahrenheit to celsius"
        r'convert\s+([\d.,]+)\s+(\w+)\s+to\s+(\w+)',
        # "What is 5 kg in pounds?"
        r'what (?:is|are)\s+([\d.,]+)\s+(\w+)\s+in\s+(\w+)',
        # "26.2 miles to km" / "26.2 miles in km"
        r'([\d.,]+)\s+(\w+)\s+(?:to|in)\s+(\w+)',
    ]

    for pattern in patterns:
        m = re.search(pattern, lower)
        if not m:
            continue

        groups = m.groups()
        if len(groups) == 3:
            if groups[0][0].isdigit():
                # value, from, to
                value = float(groups[0].replace(',', ''))
                from_unit = _normalize_unit(groups[1])
                to_unit = _normalize_unit(groups[2])
            else:
                # to, value, from (e.g., "how many km is 26.2 miles")
                to_unit = _normalize_unit(groups[0])
                value = float(groups[1].replace(',', ''))
                from_unit = _normalize_unit(groups[2])

            result = _convert(value, from_unit, to_unit)
            if result is not None:
                result = round(result, 2)
                expr = f"{value}{from_unit}\u2192{to_unit}"
                return value, from_unit, to_unit, result, expr

    return None


class UnitTurnstyle(Turnstyle):
    """Grounds unit conversions in exact computation.

        t = UnitTurnstyle(model, tokenizer, device)
        text, proof = t.generate("How many km is 26.2 miles?")
    """

    def parse(self, prompt: str):
        return parse_unit_conversion(prompt)

    def make_processor(self, parsed, max_new_tokens: int):
        value, from_unit, to_unit, result, expr = parsed
        # Format result: strip trailing zeros but keep at least one decimal
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
