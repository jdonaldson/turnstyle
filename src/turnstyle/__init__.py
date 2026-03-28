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
    TokenAudit,
    SequenceLogitsProcessor,
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
from turnstyle.number_theory import (
    GCDTurnstyle,
    LCMTurnstyle,
    FractionTurnstyle,
    parse_number_theory,
)
from turnstyle.boolean import BooleanTurnstyle, parse_boolean
from turnstyle.sorting import SortingTurnstyle, parse_sorting
from turnstyle.dyck import DyckTurnstyle, parse_dyck
from turnstyle.sandbox import SandboxTurnstyle, parse_sandbox_code
from turnstyle.sandbox_backend import (
    SandboxResult,
    SandboxBackend,
    DenoPyodideBackend,
    WasmtimeBackend,
    MockBackend,
)
from turnstyle.probe import TurnstyleProbe, IntentProbe, RoutingTurnstyle

try:
    from turnstyle.sweep import (
        probe_sweep,
        generate_prompts,
        generate_intent_prompts,
        intent_sweep,
        SweepResult,
        IntentSweepResult,
    )
except ImportError:
    pass

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
    "TokenAudit",
    "SequenceLogitsProcessor",
    "GCDTurnstyle",
    "LCMTurnstyle",
    "FractionTurnstyle",
    "parse_number_theory",
    "BooleanTurnstyle",
    "parse_boolean",
    "SortingTurnstyle",
    "parse_sorting",
    "DyckTurnstyle",
    "parse_dyck",
    "SandboxTurnstyle",
    "parse_sandbox_code",
    "SandboxResult",
    "SandboxBackend",
    "DenoPyodideBackend",
    "WasmtimeBackend",
    "MockBackend",
    "TurnstyleProbe",
    "IntentProbe",
    "RoutingTurnstyle",
    "probe_sweep",
    "generate_prompts",
    "generate_intent_prompts",
    "intent_sweep",
    "SweepResult",
    "IntentSweepResult",
]
