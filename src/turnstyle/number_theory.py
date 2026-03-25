"""Number theory turnstyles — GCD, LCM, and fraction simplification.

Handles:
    "GCD of 24 and 36", "gcd(24, 36)"
    "LCM of 4 and 6", "lcm(4, 6)"
    "Simplify 6/4", "reduce 18/12"
"""

from __future__ import annotations

import math
import re

import torch

from turnstyle.arithmetic import ArithmeticLogitsProcessor
from turnstyle.core import CoprocessorDiagnostic, DigitAudit, Turnstyle


def parse_number_theory(text: str):
    """Extract a number theory operation from text.

    Returns (a, b, operation, result, expression) or None.
    - operation: 'gcd', 'lcm', or 'simplify'
    - result: int for gcd/lcm, "n/d" string for simplify
    """
    lower = text.lower().strip()

    # GCD patterns
    m = re.search(
        r'(?:gcd|greatest common (?:divisor|factor)).*?(\d+).*?(\d+)', lower
    )
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        result = math.gcd(a, b)
        return a, b, 'gcd', result, f"gcd({a},{b})"

    # LCM patterns
    m = re.search(
        r'(?:lcm|least common multiple).*?(\d+).*?(\d+)', lower
    )
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        result = a * b // math.gcd(a, b)
        return a, b, 'lcm', result, f"lcm({a},{b})"

    # Fraction simplification patterns
    # "Simplify 6/4" or "What is 8/12 in simplest form?"
    m = re.search(
        r'(?:simplify|reduce|lowest terms?|simplest form).*?(\d+)\s*/\s*(\d+)',
        lower,
    )
    if not m:
        # Try fraction-first: "What is 8/12 in simplest form?"
        m = re.search(
            r'(\d+)\s*/\s*(\d+).*?(?:simplify|reduce|lowest terms?|simplest form)',
            lower,
        )
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            return None
        g = math.gcd(a, b)
        result = f"{a // g}/{b // g}"
        return a, b, 'simplify', result, f"simplify({a}/{b})"

    return None


class GCDTurnstyle(Turnstyle):
    """Grounds GCD computation in exact math.gcd.

        t = GCDTurnstyle(model, tokenizer, device)
        text, proof = t.generate("What is the GCD of 24 and 36?")
    """

    probe_label = "number_theory"

    def parse(self, prompt: str):
        parsed = parse_number_theory(prompt)
        if parsed is not None and parsed[2] == 'gcd':
            return parsed
        return None

    def make_processor(self, parsed, max_new_tokens: int):
        a, b, _, result, expr = parsed
        answer_digits = [int(d) for d in str(result)]
        return ArithmeticLogitsProcessor(
            self.tokenizer, answer_digits, expr, result,
            self.bias_strength, max_new_tokens=max_new_tokens)


class LCMTurnstyle(Turnstyle):
    """Grounds LCM computation in exact arithmetic.

        t = LCMTurnstyle(model, tokenizer, device)
        text, proof = t.generate("What is the LCM of 4 and 6?")
    """

    probe_label = "number_theory"

    def parse(self, prompt: str):
        parsed = parse_number_theory(prompt)
        if parsed is not None and parsed[2] == 'lcm':
            return parsed
        return None

    def make_processor(self, parsed, max_new_tokens: int):
        a, b, _, result, expr = parsed
        answer_digits = [int(d) for d in str(result)]
        return ArithmeticLogitsProcessor(
            self.tokenizer, answer_digits, expr, result,
            self.bias_strength, max_new_tokens=max_new_tokens)


class FractionLogitsProcessor(ArithmeticLogitsProcessor):
    """Biases digit logits for fraction output: numerator/denominator.

    State machine: WAITING → TRIGGERED → NUMERATOR → SLASH → DENOMINATOR → DONE
    """

    def __init__(self, tokenizer, numerator_digits: list[int],
                 denominator_digits: list[int], expression: str,
                 answer_str: str, bias_strength: float = 15.0,
                 max_new_tokens: int = 50):
        # Initialize parent with numerator digits as the "answer"
        super().__init__(
            tokenizer, numerator_digits, expression, 0,
            bias_strength, max_new_tokens)

        self.numerator_digits = numerator_digits
        self.denominator_digits = denominator_digits
        self.denom_idx = 0
        self.fraction_state = "WAITING"  # separate from parent state machine

        # Find the slash token
        slash_ids = tokenizer.encode("/", add_special_tokens=False)
        self.slash_token = slash_ids[0] if slash_ids else None

        # Override proof for fraction display
        self.proof = CoprocessorDiagnostic(
            expression=expression, answer=0,
            expected_digits=len(numerator_digits) + len(denominator_digits),
            max_steps=max_new_tokens)
        self.proof.answer_str = answer_str

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step_count += 1
        last_id = input_ids[0, -1].item()
        last_text = self.tokenizer.decode([last_id]).strip().lower()

        if self.fraction_state == "WAITING":
            if last_text in self.trigger_texts:
                self.fraction_state = "TRIGGERED"
                self.proof.trigger_step = self.step_count

        elif self.fraction_state == "TRIGGERED":
            top_id = int(torch.argmax(scores, dim=-1)[0])
            if top_id in self.token_to_digit and self.digit_idx < len(self.numerator_digits):
                scores, _ = self._audit_and_bias(scores)
                self.fraction_state = "NUMERATOR"

        elif self.fraction_state == "NUMERATOR":
            top_id = int(torch.argmax(scores, dim=-1)[0])
            if top_id in self.token_to_digit and self.digit_idx < len(self.numerator_digits):
                scores, _ = self._audit_and_bias(scores)
            else:
                # Done with numerator, bias toward slash
                if self.slash_token is not None:
                    scores[0, self.slash_token] += self.bias_strength
                self.fraction_state = "SLASH"

        elif self.fraction_state == "SLASH":
            # After slash, start denominator
            top_id = int(torch.argmax(scores, dim=-1)[0])
            if top_id in self.token_to_digit and self.denom_idx < len(self.denominator_digits):
                scores = self._bias_denom(scores)
                self.fraction_state = "DENOMINATOR"

        elif self.fraction_state == "DENOMINATOR":
            top_id = int(torch.argmax(scores, dim=-1)[0])
            if top_id in self.token_to_digit and self.denom_idx < len(self.denominator_digits):
                scores = self._bias_denom(scores)
            else:
                self.fraction_state = "DONE"

        self.proof.final_state = self.fraction_state
        self.proof.total_steps = self.step_count
        return scores

    def _bias_denom(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Bias toward the next denominator digit."""
        correct_digit = self.denominator_digits[self.denom_idx]
        correct_token = self.digit_to_token[correct_digit]

        top_id = int(torch.argmax(scores, dim=-1)[0])
        model_digit = self.token_to_digit.get(top_id, -1)

        model_logit = scores[0, correct_token].item()
        top_logit = scores[0, top_id].item()

        scores[0, correct_token] += self.bias_strength

        new_top_id = int(torch.argmax(scores, dim=-1)[0])
        corrected = (new_top_id != top_id)

        self.proof.digits.append(DigitAudit(
            position=len(self.numerator_digits) + self.denom_idx,
            correct=correct_digit,
            model_predicted=model_digit,
            bias_applied=self.bias_strength,
            model_logit=model_logit,
            top_logit=top_logit,
            corrected=corrected,
        ))

        self.denom_idx += 1
        return scores


class FractionTurnstyle(Turnstyle):
    """Grounds fraction simplification in exact computation.

        t = FractionTurnstyle(model, tokenizer, device)
        text, proof = t.generate("Simplify 6/4")
    """

    probe_label = "number_theory"

    def parse(self, prompt: str):
        parsed = parse_number_theory(prompt)
        if parsed is not None and parsed[2] == 'simplify':
            return parsed
        return None

    def make_processor(self, parsed, max_new_tokens: int):
        a, b, _, result_str, expr = parsed
        num_str, den_str = result_str.split("/")
        num_digits = [int(d) for d in num_str]
        den_digits = [int(d) for d in den_str]

        return FractionLogitsProcessor(
            self.tokenizer, num_digits, den_digits, expr, result_str,
            self.bias_strength, max_new_tokens)
