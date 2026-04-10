"""Arithmetic turnstyle — grounds LLM digit generation in exact computation.

Supports +, -, *, / (integer division).
"""

from __future__ import annotations

import re

import torch
from transformers import LogitsProcessor

from turnstyle.core import (
    CoprocessorDiagnostic,
    DigitAudit,
    Turnstyle,
)


def parse_arithmetic(text: str) -> tuple[int, int, str, int] | None:
    """Extract a binary arithmetic expression from text."""
    m = re.search(r'(\d+)\s*(\+|-|\*|/)\s*(\d+)', text)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    ops = {'+': a + b, '-': a - b, '*': a * b, '/': a // b if b != 0 else None}
    result = ops.get(op)
    if result is None:
        return None
    return a, b, op, result


class ArithmeticLogitsProcessor(LogitsProcessor):
    """Biases digit logits toward the symbolically-computed correct answer.

    State machine: WAITING -> TRIGGERED -> INJECTING -> DONE
    """

    def __init__(self, tokenizer, answer_digits: list[int], expression: str,
                 answer_value: int, bias_strength: float = 15.0,
                 max_new_tokens: int = 50, operand_digits: list[int] | None = None):
        self.tokenizer = tokenizer
        self.answer_digits = answer_digits
        self.bias_strength = bias_strength

        self.digit_to_token = {}
        self.token_to_digit = {}
        for d in range(10):
            ids = tokenizer.encode(str(d), add_special_tokens=False)
            if ids:
                self.digit_to_token[d] = ids[0]
                self.token_to_digit[ids[0]] = d

        self.trigger_texts = {'is', '=', 'equals'}
        self.state = "WAITING"
        self.digit_idx = 0
        self.step_count = 0

        self.proof = CoprocessorDiagnostic(
            expression=expression, answer=answer_value,
            expected_digits=len(answer_digits),
            max_steps=max_new_tokens,
            operand_digits=operand_digits or [])

    def _audit_and_bias(self, scores: torch.FloatTensor) -> tuple[torch.FloatTensor, bool]:
        top_id = int(torch.argmax(scores, dim=-1)[0])
        if top_id not in self.token_to_digit or self.digit_idx >= len(self.answer_digits):
            return scores, False

        model_digit = self.token_to_digit[top_id]
        correct_digit = self.answer_digits[self.digit_idx]
        correct_token = self.digit_to_token[correct_digit]

        model_logit_for_correct = scores[0, correct_token].item()
        top_logit = scores[0, top_id].item()

        scores[0, correct_token] += self.bias_strength

        new_top_id = int(torch.argmax(scores, dim=-1)[0])
        corrected = (new_top_id != top_id)

        self.proof.digits.append(DigitAudit(
            position=self.digit_idx,
            correct=correct_digit,
            model_predicted=model_digit,
            bias_applied=self.bias_strength,
            model_logit=model_logit_for_correct,
            top_logit=top_logit,
            corrected=corrected,
        ))

        self.digit_idx += 1
        return scores, True

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step_count += 1
        last_id = input_ids[0, -1].item()
        last_text = self.tokenizer.decode([last_id]).strip().lower()

        if self.state == "WAITING":
            if last_id in self.token_to_digit:
                self.proof.echo_digits.append(self.token_to_digit[last_id])
            if last_text in self.trigger_texts:
                self.state = "TRIGGERED"
                self.proof.trigger_step = self.step_count

        elif self.state == "TRIGGERED":
            top_id = int(torch.argmax(scores, dim=-1)[0])
            top_text = self.tokenizer.decode([top_id]).strip()
            if top_id in self.token_to_digit and self.digit_idx < len(self.answer_digits):
                scores, _ = self._audit_and_bias(scores)
                self.state = "INJECTING"
            elif top_text == '':
                pass
            else:
                pass

        elif self.state == "INJECTING":
            top_id = int(torch.argmax(scores, dim=-1)[0])
            top_text = self.tokenizer.decode([top_id]).strip()
            if top_id in self.token_to_digit and self.digit_idx < len(self.answer_digits):
                scores, _ = self._audit_and_bias(scores)
            elif top_text in (',', '.'):
                pass
            else:
                self.state = "DONE"

        elif self.state == "DONE":
            last_id_check = input_ids[0, -1].item()
            if last_id_check in self.token_to_digit:
                self.proof.extra_digits_after_done += 1

        self.proof.final_state = self.state
        self.proof.total_steps = self.step_count

        return scores


class ArithmeticTurnstyle(Turnstyle):
    """Grounds arithmetic in symbolic computation.

        t = ArithmeticTurnstyle(model, tokenizer, device)
        text, proof = t.generate("What is 445 + 152?")
        print(proof.inline())  # ⊢ 445+152=5̲97 ∎
    """

    probe_label = "arithmetic"
    examples = [
        "What is 445 + 152?",
        "What is 100 - 34?",
        "What is 8 * 7?",
        "What is 144 / 12?",
        "What is 250 + 375?",
        "What is 1000 - 537?",
        "What is 9 * 13?",
        "What is 56 / 7?",
        "Calculate 55 plus 37",
        "Add 200 and 150 together",
        "Subtract 39 from 100",
        "Multiply 7 by 9",
        "What do you get when you add 45 to 87?",
        "Find the difference between 500 and 237",
        "What's the product of 6 and 13?",
        "Compute 48 divided by 6",
        "How much is 123 times 4?",
        "What is 777 + 888?",
        "What is 500 * 3?",
        "What is 300 / 15?",
        "What is 99 - 44?",
        "What is 12 * 12?",
        "What is 81 / 9?",
        "What is 256 + 128?",
        "What is 1000 - 1?",
        "What is 17 * 18?",
        "What is 240 / 8?",
        "What is 450 + 550?",
        "What is 75 - 25?",
        "What is 11 * 11?",
    ]

    def parse(self, prompt: str):
        return parse_arithmetic(prompt)

    def parse_from_hidden(self, hidden_state):
        """Extract operation and operands from hidden states via IntentProbe."""
        if self.intent_probe is None:
            return None

        intent = self.intent_probe.predict(hidden_state)

        op_label, op_conf = intent.get("operation", (None, 0))
        a_label, a_conf = intent.get("operand_a", (None, 0))
        b_label, b_conf = intent.get("operand_b", (None, 0))

        # Confidence gate
        min_conf = min(op_conf, a_conf, b_conf)
        if min_conf < 0.7:
            return None

        op_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        op = op_map.get(op_label)
        if op is None:
            return None

        try:
            a, b = int(a_label), int(b_label)
        except (ValueError, TypeError):
            return None

        if op == "/" and b == 0:
            return None

        result = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
        return a, b, op, result

    def make_processor(self, parsed, max_new_tokens: int):
        a, b, op, answer = parsed
        answer_digits = [int(d) for d in str(abs(answer))]
        expression = f"{a}{op}{b}"
        operand_digits = [int(d) for d in str(a)] + [int(d) for d in str(b)]
        return ArithmeticLogitsProcessor(
            self.tokenizer, answer_digits, expression, answer,
            self.bias_strength, max_new_tokens=max_new_tokens,
            operand_digits=operand_digits)
