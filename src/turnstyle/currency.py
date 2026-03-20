"""Currency conversion turnstyle — grounds exchange rate math in exact computation.

Handles:
    "How much is 100 USD in EUR?"
    "Convert 500 GBP to JPY"
    "1000 yen to dollars"

Rates are configurable — pass your own dict to CurrencyTurnstyle.
Defaults are approximate 2026 rates for common pairs.
"""

from __future__ import annotations

import re

from turnstyle.arithmetic import ArithmeticLogitsProcessor
from turnstyle.core import Turnstyle

# ── default rates (approximate, vs USD) ──────────────────────────────

_DEFAULT_RATES: dict[str, float] = {
    'usd': 1.0,
    'eur': 0.92,
    'gbp': 0.79,
    'jpy': 150.0,
    'cny': 7.25,
    'cad': 1.36,
    'aud': 1.53,
    'chf': 0.88,
    'inr': 83.5,
    'mxn': 17.2,
    'brl': 4.95,
    'krw': 1330.0,
    'sek': 10.5,
    'nok': 10.7,
    'dkk': 6.85,
    'sgd': 1.34,
    'hkd': 7.82,
    'nzd': 1.63,
}

# ── currency name normalization ──────────────────────────────────────

_ALIASES: dict[str, str] = {
    'dollar': 'usd', 'dollars': 'usd', 'us dollar': 'usd', 'us dollars': 'usd',
    'euro': 'eur', 'euros': 'eur',
    'pound': 'gbp', 'pounds': 'gbp', 'british pound': 'gbp', 'sterling': 'gbp',
    'yen': 'jpy', 'japanese yen': 'jpy',
    'yuan': 'cny', 'renminbi': 'cny', 'rmb': 'cny',
    'canadian dollar': 'cad', 'canadian dollars': 'cad', 'cad': 'cad',
    'australian dollar': 'aud', 'australian dollars': 'aud', 'aud': 'aud',
    'swiss franc': 'chf', 'francs': 'chf',
    'rupee': 'inr', 'rupees': 'inr', 'indian rupee': 'inr',
    'peso': 'mxn', 'pesos': 'mxn', 'mexican peso': 'mxn',
    'real': 'brl', 'reais': 'brl', 'brazilian real': 'brl',
    'won': 'krw', 'korean won': 'krw',
    'krona': 'sek', 'kronor': 'sek', 'swedish krona': 'sek',
    'krone': 'nok', 'norwegian krone': 'nok',
    'singapore dollar': 'sgd', 'singapore dollars': 'sgd',
    'hong kong dollar': 'hkd', 'hong kong dollars': 'hkd',
    'new zealand dollar': 'nzd', 'new zealand dollars': 'nzd',
}


def _normalize_currency(text: str) -> str:
    text = text.lower().strip()
    # Check 3-letter codes first
    if text in _DEFAULT_RATES:
        return text
    return _ALIASES.get(text, text)


# ── expression parsing ───────────────────────────────────────────────

def parse_currency_conversion(text: str, rates: dict[str, float] | None = None):
    """Extract currency conversion from text.

    Returns (amount, from_cur, to_cur, result, expression) or None.
    """
    rates = rates or _DEFAULT_RATES
    lower = text.lower()

    patterns = [
        # "How much is 100 USD in EUR?"
        r'how much\s+(?:is|are)\s+([\d.,]+)\s+(\w+)\s+in\s+(\w+)',
        # "Convert 500 GBP to JPY"
        r'convert\s+([\d.,]+)\s+(\w+)\s+to\s+(\w+)',
        # "100 USD to EUR" / "100 dollars in euros"
        r'([\d.,]+)\s+(\w+)\s+(?:to|in)\s+(\w+)',
    ]

    for pattern in patterns:
        m = re.search(pattern, lower)
        if not m:
            continue

        amount = float(m.group(1).replace(',', ''))
        from_cur = _normalize_currency(m.group(2))
        to_cur = _normalize_currency(m.group(3))

        if from_cur in rates and to_cur in rates:
            # Convert via USD as base
            usd_amount = amount / rates[from_cur]
            result = round(usd_amount * rates[to_cur], 2)
            expr = f"{amount}{from_cur.upper()}\u2192{to_cur.upper()}"
            return amount, from_cur, to_cur, result, expr

    return None


class CurrencyTurnstyle(Turnstyle):
    """Grounds currency conversion in exact exchange rate computation.

    Pass custom rates as a dict mapping currency codes to USD values:

        rates = {'usd': 1.0, 'eur': 0.92, 'gbp': 0.79}
        t = CurrencyTurnstyle(model, tokenizer, device, rates=rates)
        text, proof = t.generate("How much is 100 USD in EUR?")
    """

    def __init__(self, model, tokenizer, device, bias_strength=15.0,
                 rates: dict[str, float] | None = None):
        super().__init__(model, tokenizer, device, bias_strength)
        self.rates = rates or _DEFAULT_RATES

    def parse(self, prompt: str):
        return parse_currency_conversion(prompt, self.rates)

    def make_processor(self, parsed, max_new_tokens: int):
        amount, from_cur, to_cur, result, expr = parsed

        if isinstance(result, float) and result != int(result):
            answer_str = f"{result:.2f}"
        else:
            answer_str = str(int(result))

        answer_digits = [int(d) for d in answer_str if d.isdigit()]

        proc = ArithmeticLogitsProcessor(
            self.tokenizer, answer_digits, expr, result,
            self.bias_strength, max_new_tokens=max_new_tokens)
        proc.proof.answer_str = answer_str
        return proc
