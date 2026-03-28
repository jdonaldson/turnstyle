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
from transformers import LogitsProcessor

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

    def inline(self, plain: bool = False) -> str:
        """⊢ 445+152=5̲97 ∎ — corrected digits underlined, missing digits hatted.

        If plain=True, returns '445+152=597' with no symbols or marks.
        """
        answer = self._display if plain else self._mark_digits()
        if plain:
            return f"{self.expression}={answer}"
        return f"{SYMBOL} {self.expression}={answer} {QED}"

    def summary(self, plain: bool = False) -> str:
        """One-line audit summary."""
        prefix = "" if plain else f"{SYMBOL} "
        parts = [f"{prefix}{self.expression}={self.answer}"]
        n = len(self.digits)
        if self.any_corrected:
            parts.append(f"{self.num_corrected}/{n} corrected")
            if not plain:
                parts.append(f"\u0394={self.max_confidence:.2f}")
        else:
            parts.append(f"0/{n} corrected")
        diag_str = self.diagnostic_summary()
        if diag_str:
            parts.append(diag_str)
        return "  ".join(parts)

    def detail(self, plain: bool = False) -> str:
        """Multi-line forensic detail."""
        lines = [self.summary(plain=plain)]
        for d in self.digits:
            if d.corrected:
                arrow = "->" if plain else "\u2192"
                lines.append(
                    f"  d{d.position}: [{d.model_predicted}{arrow}{d.correct}]"
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
# Token-level audit + sequence processor
# ════════════════════════════════════════════════════════════════════════

@dataclass
class TokenAudit:
    """Per-token record of what the coprocessor did (for arbitrary token biasing)."""
    position: int
    correct_token_id: int
    model_top_token_id: int
    bias_applied: float
    model_logit: float
    top_logit: float
    corrected: bool


class SequenceLogitsProcessor(LogitsProcessor):
    """Biases logits toward a pre-tokenized answer sequence.

    State machine: WAITING → INJECTING → DONE

    Unlike ArithmeticLogitsProcessor (digit-only), this handles arbitrary
    token sequences — boolean answers, sorted word lists, bracket sequences, etc.

    Usage:
        tokens = tokenizer.encode("True", add_special_tokens=False)
        proc = SequenceLogitsProcessor(
            tokenizer, tokens, expression="True and False",
            answer_str="True", bias_strength=15.0)
    """

    def __init__(
        self,
        tokenizer,
        answer_token_ids: list[int],
        expression: str,
        answer_str: str,
        bias_strength: float = 15.0,
        max_new_tokens: int = 50,
        trigger_texts: set[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.answer_token_ids = answer_token_ids
        self.bias_strength = bias_strength
        self.trigger_texts = trigger_texts or {'is', '=', 'equals', ':'}
        self.state = "WAITING"
        self.token_idx = 0
        self.step_count = 0

        self.audits: list[TokenAudit] = []
        self.proof = CoprocessorDiagnostic(
            expression=expression,
            answer=0,
            answer_str=answer_str,
            expected_digits=len(answer_token_ids),
            max_steps=max_new_tokens,
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        self.step_count += 1
        last_id = input_ids[0, -1].item()
        last_text = self.tokenizer.decode([last_id]).strip().lower()

        if self.state == "WAITING":
            if last_text in self.trigger_texts:
                self.state = "INJECTING"
                self.proof.trigger_step = self.step_count

        elif self.state == "INJECTING":
            if self.token_idx < len(self.answer_token_ids):
                scores = self._bias_token(scores)
            else:
                self.state = "DONE"

        if self.state == "DONE":
            # Force EOS to prevent model from generating extra tokens
            # after the biased answer sequence.
            eos_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_id is not None:
                scores[0, :] -= 100.0
                scores[0, eos_id] += 100.0

        self.proof.final_state = self.state
        self.proof.total_steps = self.step_count
        return scores

    def _bias_token(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        correct_id = self.answer_token_ids[self.token_idx]
        top_id = int(torch.argmax(scores, dim=-1)[0])

        model_logit = scores[0, correct_id].item()
        top_logit = scores[0, top_id].item()

        scores[0, correct_id] += self.bias_strength

        new_top_id = int(torch.argmax(scores, dim=-1)[0])
        corrected = new_top_id != top_id

        self.audits.append(TokenAudit(
            position=self.token_idx,
            correct_token_id=correct_id,
            model_top_token_id=top_id,
            bias_applied=self.bias_strength,
            model_logit=model_logit,
            top_logit=top_logit,
            corrected=corrected,
        ))

        # Also record as DigitAudit for proof compatibility
        self.proof.digits.append(DigitAudit(
            position=self.token_idx,
            correct=correct_id,
            model_predicted=top_id,
            bias_applied=self.bias_strength,
            model_logit=model_logit,
            top_logit=top_logit,
            corrected=corrected,
        ))

        self.token_idx += 1
        return scores


# ════════════════════════════════════════════════════════════════════════
# Base class
# ════════════════════════════════════════════════════════════════════════

class Turnstyle:
    """Base class for symbolic coprocessors.

    Subclasses implement:
        parse(prompt) -> oracle result or None
        make_processor(parsed, max_new_tokens) -> LogitsProcessor
    """

    intent_probe = None  # IntentProbe, set by sweep or manually

    def __init__(self, model, tokenizer, device, bias_strength=15.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bias_strength = bias_strength

    def parse(self, prompt: str):
        raise NotImplementedError

    def parse_from_hidden(self, hidden_state):
        """Probe-based parsing from model hidden states.

        Returns same format as parse(), or None if not supported/confident enough.
        Subclasses override to convert probe-extracted intents into parsed format.
        """
        return None

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
