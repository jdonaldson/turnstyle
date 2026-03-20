"""Core diagnostics, annotations, and base class for turnstyles.

A turnstyle wraps a causal LM with a symbolic oracle. The model generates
freely until a trigger word, then the turnstyle biases digit logits toward
the oracle's answer. Every intervention is audited.

Annotation marks:
    underline (̲)   = digit corrected by coprocessor
    circumflex (̂)  = digit the model never emitted (added by coprocessor)

Symbols:
    ⊢  = "derived" (turnstile — opening delimiter)
    ∎  = "QED" (halmos — closing delimiter)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto

import torch

SYMBOL = "\u22a2"  # ⊢
QED = "\u220e"     # ∎


# ════════════════════════════════════════════════════════════════════════
# Diagnostics
# ════════════════════════════════════════════════════════════════════════

class Diagnostic(Enum):
    """Error classes a turnstyle can detect."""
    NO_TRIGGER = auto()
    DIGITS_TOO_FEW = auto()
    DIGITS_TOO_MANY = auto()
    HIGH_CORRECTION = auto()
    CONFIDENT_WRONG = auto()
    OPERAND_ECHO_ERROR = auto()
    NO_ANSWER_EMITTED = auto()
    LATE_TRIGGER = auto()


DIAGNOSTIC_LABELS = {
    Diagnostic.NO_TRIGGER:         "no trigger word detected",
    Diagnostic.DIGITS_TOO_FEW:     "too few digits emitted",
    Diagnostic.DIGITS_TOO_MANY:    "too many digits emitted",
    Diagnostic.HIGH_CORRECTION:    "high correction ratio (model guessing)",
    Diagnostic.CONFIDENT_WRONG:    "model confidently wrong",
    Diagnostic.OPERAND_ECHO_ERROR: "operand restatement mismatch",
    Diagnostic.NO_ANSWER_EMITTED:  "no answer digits after trigger",
    Diagnostic.LATE_TRIGGER:       "trigger late in generation",
}


@dataclass
class DigitAudit:
    """Per-digit record of what the coprocessor did."""
    position: int
    correct: int
    model_predicted: int
    bias_applied: float
    model_logit: float
    top_logit: float
    corrected: bool

    @property
    def confidence(self) -> float:
        """0-1: how confident the model was in the WRONG answer."""
        if not self.corrected:
            return 0.0
        gap = self.top_logit - self.model_logit
        return min(gap / 20.0, 1.0)


@dataclass
class CoprocessorDiagnostic:
    """Complete audit trail for a symbolically-assisted generation."""
    expression: str
    answer: int | float
    answer_str: str = ""  # formatted answer; defaults to str(abs(answer))
    answer_charset: str = "0123456789"  # characters treated as biasable answer positions
    expected_digits: int = 0
    digits: list[DigitAudit] = field(default_factory=list)
    extra_digits_after_done: int = 0
    final_state: str = "WAITING"
    trigger_step: int = -1
    total_steps: int = 0
    max_steps: int = 0
    echo_digits: list[int] = field(default_factory=list)
    operand_digits: list[int] = field(default_factory=list)

    # ── properties ───────────────────────────────────────────────────

    @property
    def any_corrected(self) -> bool:
        return any(d.corrected for d in self.digits)

    @property
    def num_corrected(self) -> int:
        return sum(1 for d in self.digits if d.corrected)

    @property
    def correction_ratio(self) -> float:
        if not self.digits:
            return 0.0
        return self.num_corrected / len(self.digits)

    @property
    def max_confidence(self) -> float:
        if not self.digits:
            return 0.0
        return max(d.confidence for d in self.digits)

    # ── diagnostics ──────────────────────────────────────────────────

    CONFIDENT_WRONG_THRESHOLD = 0.25
    HIGH_CORRECTION_THRESHOLD = 0.5
    LATE_TRIGGER_THRESHOLD = 0.8

    @property
    def diagnostics(self) -> list[Diagnostic]:
        issues = []

        if self.final_state == "WAITING":
            issues.append(Diagnostic.NO_TRIGGER)

        emitted = len(self.digits) + self.extra_digits_after_done
        if emitted > self.expected_digits and self.expected_digits > 0:
            issues.append(Diagnostic.DIGITS_TOO_MANY)
        elif len(self.digits) < self.expected_digits and self.final_state != "WAITING":
            issues.append(Diagnostic.DIGITS_TOO_FEW)

        if self.final_state in ("TRIGGERED",) and len(self.digits) == 0:
            issues.append(Diagnostic.NO_ANSWER_EMITTED)

        if self.digits and self.correction_ratio > self.HIGH_CORRECTION_THRESHOLD:
            issues.append(Diagnostic.HIGH_CORRECTION)

        if self.max_confidence > self.CONFIDENT_WRONG_THRESHOLD:
            issues.append(Diagnostic.CONFIDENT_WRONG)

        if self.echo_digits and self.operand_digits:
            if self.echo_digits != self.operand_digits:
                issues.append(Diagnostic.OPERAND_ECHO_ERROR)

        if (self.trigger_step >= 0 and self.max_steps > 0
                and self.trigger_step > self.max_steps * self.LATE_TRIGGER_THRESHOLD):
            issues.append(Diagnostic.LATE_TRIGGER)

        return issues

    @property
    def is_clean(self) -> bool:
        return not self.diagnostics and not self.any_corrected

    # ── formatting ───────────────────────────────────────────────────

    def diagnostic_summary(self) -> str:
        diags = self.diagnostics
        if not diags:
            return ""
        return " | ".join(DIAGNOSTIC_LABELS[d] for d in diags)

    @property
    def _display(self) -> str:
        return self.answer_str or str(abs(self.answer))

    def _mark_digits(self) -> str:
        """Apply underline/hat marks to answer characters, passing through others."""
        display = self._display
        audited = len(self.digits)
        charset = set(self.answer_charset)
        parts = []
        digit_idx = 0
        for ch in display:
            if ch.lower() in charset:
                audit = next((d for d in self.digits if d.position == digit_idx), None)
                if audit and audit.corrected:
                    parts.append(f"{ch}\u0332")  # underline = changed
                elif digit_idx >= audited:
                    parts.append(f"{ch}\u0302")  # circumflex = missing
                else:
                    parts.append(ch)
                digit_idx += 1
            else:
                parts.append(ch)  # decimal point, comma, minus, etc.
        return ''.join(parts)

    def inline(self) -> str:
        """⊢ 445+152=5̲97 ∎ — corrected digits underlined, missing digits hatted."""
        return f"{SYMBOL} {self.expression}={self._mark_digits()} {QED}"

    def summary(self) -> str:
        """One-line audit summary."""
        parts = [f"{SYMBOL} {self.expression}={self.answer}"]
        n = len(self.digits)
        if self.any_corrected:
            parts.append(f"{self.num_corrected}/{n} corrected")
            parts.append(f"\u0394={self.max_confidence:.2f}")
        else:
            parts.append(f"0/{n} corrected")
        diag_str = self.diagnostic_summary()
        if diag_str:
            parts.append(diag_str)
        return "  ".join(parts)

    def detail(self) -> str:
        """Multi-line forensic detail."""
        lines = [self.summary()]
        for d in self.digits:
            if d.corrected:
                lines.append(
                    f"  d{d.position}: [{d.model_predicted}\u2192{d.correct}]"
                    f"  logit_gap={d.top_logit - d.model_logit:+.1f}")
        lines.append(
            f"  trigger@step {self.trigger_step}/{self.total_steps}"
            f"  state={self.final_state}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════════════

def extract_number(text: str) -> int | None:
    """Extract the last integer from text, handling comma-separated thousands."""
    numbers = re.findall(r'-?\d[\d,]*', text)
    if not numbers:
        return None
    return int(numbers[-1].replace(',', ''))


# ════════════════════════════════════════════════════════════════════════
# Base class
# ════════════════════════════════════════════════════════════════════════

class Turnstyle:
    """Base class for symbolic coprocessors.

    Subclasses implement:
        parse(prompt) -> oracle result or None
        make_processor(parsed, max_new_tokens) -> LogitsProcessor
    """

    def __init__(self, model, tokenizer, device, bias_strength=15.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bias_strength = bias_strength

    def parse(self, prompt: str):
        raise NotImplementedError

    def make_processor(self, parsed, max_new_tokens: int):
        raise NotImplementedError

    def generate(self, prompt: str, max_new_tokens: int = 50):
        """Generate with symbolic grounding. Returns (text, diagnostic)."""
        parsed = self.parse(prompt)

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        if parsed is None:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip()
            return text, None

        processor = self.make_processor(parsed, max_new_tokens)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                logits_processor=[processor],
            )

        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()

        return text, processor.proof

    @staticmethod
    def annotate(text: str, proof: CoprocessorDiagnostic | None) -> str:
        """Replace the answer in model text with marked-up digits."""
        if proof is None or not proof.any_corrected:
            return text
        answer_str = proof._display
        marked = proof._mark_digits()
        return text.replace(answer_str, marked, 1)
