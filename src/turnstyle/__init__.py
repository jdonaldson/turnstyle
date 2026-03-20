"""⊢ Turnstyle — symbolic coprocessors for LLM generation.

Ground model outputs in exact computation via logit biasing.
The model handles language. Turnstyles handle facts.

    from turnstyle import Turnstyle, ArithmeticTurnstyle

    t = ArithmeticTurnstyle(model, tokenizer, device)
    text, proof = t.generate("What is 445 + 152?")
    # text = "The sum of 445 and 152 is 597."
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
    "extract_number",
]
