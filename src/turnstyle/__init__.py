"""⊢ Turnstyle — symbolic coprocessors for LLM generation.

Ground model outputs in exact computation via logit biasing.
The model handles language. Turnstyles handle facts.

    from turnstyle import ArithmeticTurnstyle, DateTurnstyle, UnitTurnstyle

    t = ArithmeticTurnstyle(model, tokenizer, device)
    text, proof = t.generate("What is 445 + 152?")
    # proof.inline() = "⊢ 445+152=5̲97 ∎"
"""

from turnstyle.core import (
    SYMBOL,
    QED,
    Diagnostic,
    DIAGNOSTIC_LABELS,
    DigitAudit,
    CoprocessorDiagnostic,
    Turnstyle,
    extract_number,
)
from turnstyle.arithmetic import (
    ArithmeticTurnstyle,
    ArithmeticLogitsProcessor,
    parse_arithmetic,
)
from turnstyle.dates import DateTurnstyle, parse_date_arithmetic
from turnstyle.units import UnitTurnstyle, parse_unit_conversion
from turnstyle.currency import CurrencyTurnstyle, parse_currency_conversion

__all__ = [
    "SYMBOL",
    "QED",
    "Diagnostic",
    "DIAGNOSTIC_LABELS",
    "DigitAudit",
    "CoprocessorDiagnostic",
    "Turnstyle",
    "ArithmeticTurnstyle",
    "ArithmeticLogitsProcessor",
    "parse_arithmetic",
    "DateTurnstyle",
    "parse_date_arithmetic",
    "UnitTurnstyle",
    "parse_unit_conversion",
    "CurrencyTurnstyle",
    "parse_currency_conversion",
    "extract_number",
]
