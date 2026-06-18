"""Arithmetic turnstyle — grounds LLM digit generation in exact computation.

Supports +, -, *, / (integer division).
"""

from __future__ import annotations

import ast
import re

import torch
from transformers import LogitsProcessor

from turnstyle.core import (
    CoprocessorDiagnostic,
    DigitAudit,
    SequenceLogitsProcessor,
    Turnstyle,
)


def parse_arithmetic(text: str) -> tuple[int, int, str, int] | None:
    """Extract a binary arithmetic expression from text.

    Returns None for date-like strings (e.g. MM/DD/YYYY).
    """
    # Reject date-like prompts before regex to avoid false matches on "7/9/1972"
    if re.search(r'\b\d{1,2}/\d{1,2}/\d{4}\b', text):
        return None
    m = re.search(r'(\d+)\s*(\+|-|\*|/)\s*(\d+)', text)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    ops = {'+': a + b, '-': a - b, '*': a * b, '/': a // b if b != 0 else None}
    result = ops.get(op)
    if result is None:
        return None
    return a, b, op, result


def _eval_node(node: ast.AST) -> int | None:
    """Recursively evaluate an AST node. Returns int or None."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, int) else None
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        return None
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if left is None or right is None:
            return None
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left // right if right != 0 else None
        if isinstance(op, ast.FloorDiv):
            return left // right if right != 0 else None
        if isinstance(op, ast.Mod):
            return left % right if right != 0 else None
        if isinstance(op, ast.Pow):
            return left ** right
        return None
    return None


def safe_eval(expr_str: str) -> int | None:
    """Evaluate an arithmetic expression safely. Returns int or None."""
    try:
        tree = ast.parse(expr_str, mode='eval')
    except SyntaxError:
        return None
    return _eval_node(tree)


def parse_expression(text: str) -> tuple[str, int] | None:
    """Extract and evaluate a general arithmetic expression from text.

    Returns (expression_str, result) or None.
    Priority 1: BBH format ``<expr> =``
    Priority 2: longest evaluable numeric sub-expression
    """
    # Priority 1 — BBH format: expression followed by =
    m = re.search(r'([\d(][\d+\-*/() ]+)\s*=', text)
    if m:
        expr = m.group(1).strip()
        result = safe_eval(expr)
        if result is not None:
            return expr, result

    # Priority 2 — general: find all candidate sub-expressions, try longest first
    candidates = re.findall(r'[\d(][\d+\-*/() ]+[\d)]', text)
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        result = safe_eval(candidate.strip())
        if result is not None:
            return candidate.strip(), result

    return None


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
        '((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) =',
        '((-9 * -5 - 6 + -2) - (-8 - -6 * -3 * 1)) =',
        '((3 * -3 * 6 + -5) - (-2 + -7 - 7 - -7)) =',
        '((6 * -6 * 8 * 1) * (-1 * 7 * -6 + -2)) =',
        '((-6 - -4 + 9 + 0) + (1 + -4 - -9 * 6)) =',
        '((-6 - 4 * 2 - 6) + (1 + -2 * 1 * 7)) =',
        '((1 - 0 + 1 - 4) - (-3 * 1 - -6 * -8)) =',
        '((1 + 7 * -9 + -5) + (3 + -5 * 2 - 6)) =',
        '((-7 * -9 + 8 * -3) * (5 + -7 - 4 * -5)) =',
        '((-9 - 1 * 5 * -5) - (6 + -3 - -1 * -7)) =',
        '((6 - 0 * 5 + -3) * (6 - -7 + -2 - -7)) =',
        '((2 - -2 + -7 * 8) * (-7 * -8 * 3 - -2)) =',
        '((8 - 2 + -2 * 6) * (8 + -6 + -8 + -1)) =',
        '((-6 + -9 - -6 + -4) * (-1 - -6 + -4 - 3)) =',
        '((-5 - 4 * -8 + 8) * (4 + 3 - 9 * 7)) =',
        '((-5 * -7 * -6 + 9) * (-2 - 8 + -5 + 7)) =',
        '((8 + 9 - 4 - -9) + (8 + 7 - 6 * 1)) =',
        '((5 * -1 + -6 * -3) + (-1 + -8 - 5 + 3)) =',
        '((-6 - 6 + 7 - 7) - (5 + 3 - 9 * -8)) =',
        '((7 - 4 + -3 * 4) - (5 + -8 - 6 + -5)) =',
        '((-6 * -1 - 2 + -2) + (9 - 4 + -1 - 7)) =',
        '((8 * -6 + 6 * 1) - (-3 * 7 * 0 - 7)) =',
        '((-6 - 8 - -7 * -2) - (-9 - 5 + 7 + 1)) =',
        '((1 - 7 - -8 * 3) + (-7 - -2 + -3 * 6)) =',
        '((5 * -8 - -5 * -9) * (2 - -7 * 6 - 4)) =',
        '((6 + 1 - 4 - 3) - (-4 * -6 * -3 + 1)) =',
        '((-9 - -9 + 0 + -3) + (-2 - -1 - 1 + 2)) =',
        '((9 - 3 + 2 + -1) - (5 - -1 - -6 * -4)) =',
        '((-8 + 6 * -2 + 4) * (-4 * 5 + 2 - 8)) =',
        '((3 + -1 * 7 * -6) - (-7 * -1 + -5 - -3)) =',
    ]

    def parse(self, prompt: str):
        result = parse_expression(prompt)
        if result is not None:
            return result
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
        if len(parsed) == 2:
            # General expression from parse_expression: (expr_str, result)
            expression, answer = parsed
            answer_str = str(answer)
            answer_ids = self.tokenizer.encode(answer_str, add_special_tokens=False)
            return SequenceLogitsProcessor(
                self.tokenizer, answer_ids, expression=expression,
                answer_str=answer_str, bias_strength=self.bias_strength,
                max_new_tokens=max_new_tokens, immediate=True)
        # Binary expression from parse_arithmetic: (a, b, op, answer)
        a, b, op, answer = parsed
        answer_digits = [int(d) for d in str(abs(answer))]
        expression = f"{a}{op}{b}"
        operand_digits = [int(d) for d in str(a)] + [int(d) for d in str(b)]
        return ArithmeticLogitsProcessor(
            self.tokenizer, answer_digits, expression, answer,
            self.bias_strength, max_new_tokens=max_new_tokens,
            operand_digits=operand_digits)
