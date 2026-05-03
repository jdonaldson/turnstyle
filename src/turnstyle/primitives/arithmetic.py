"""ArithmeticEvaluator — deterministic AST-based arithmetic primitive.

No model needed. Tries `parse_expression` first (handles the BBH `<expr> =`
format), falls back to `parse_arithmetic` (binary `a op b`). Emits an
`answer` fact with `mode="result"` when an expression is found and
evaluated, or no facts when the prompt has no arithmetic.

Priority defaults high (10) so it preempts the forward-pass-bearing
primitives (OptionDetector, ChoiceProbe) for prompts that are pure
arithmetic — no point loading hidden states for a prompt we can solve with
the AST.
"""
from __future__ import annotations

from turnstyle.arithmetic import parse_arithmetic, parse_expression
from turnstyle.blackboard import Blackboard, Has, Not, Primitive


class ArithmeticEvaluator(Primitive):
    """Solve arithmetic prompts via AST evaluation. Selector: no answer yet."""

    def __init__(self, name: str = "arithmetic_evaluator", priority: int = 10):
        super().__init__(
            name=name,
            selector=Not(Has("answer")),
            priority=priority,
        )

    def fire(self, state: Blackboard) -> None:
        # Priority 1: BBH `<expr> =` format or longest evaluable sub-expression.
        result = parse_expression(state.prompt)
        if result is not None:
            expr, value = result
            state.emit(
                kind="answer",
                payload={
                    "mode": "result",
                    "answer": str(value),
                    "expression": expr,
                },
                source=self.name,
            )
            return

        # Fallback: binary `a op b` form.
        bin_result = parse_arithmetic(state.prompt)
        if bin_result is not None:
            a, b, op, value = bin_result
            state.emit(
                kind="answer",
                payload={
                    "mode": "result",
                    "answer": str(value),
                    "expression": f"{a}{op}{b}",
                },
                source=self.name,
            )
