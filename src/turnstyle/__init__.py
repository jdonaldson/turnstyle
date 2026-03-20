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
from turnstyle.percentage import PercentageTurnstyle, parse_percentage
from turnstyle.counting import CountingTurnstyle, parse_counting
from turnstyle.base_conversion import (
    BaseConversionTurnstyle,
    BaseConversionProcessor,
    parse_base_conversion,
)
from turnstyle.sandbox import SandboxTurnstyle, parse_sandbox_code
from turnstyle.sandbox_backend import (
    SandboxResult,
    SandboxBackend,
    DenoPyodideBackend,
    MockBackend,
)

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
    "PercentageTurnstyle",
    "parse_percentage",
    "CountingTurnstyle",
    "parse_counting",
    "BaseConversionTurnstyle",
    "BaseConversionProcessor",
    "parse_base_conversion",
    "extract_number",
    "SandboxTurnstyle",
    "parse_sandbox_code",
    "SandboxResult",
    "SandboxBackend",
    "DenoPyodideBackend",
    "MockBackend",
]
