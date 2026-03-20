"""Sandbox turnstyle — grounds LLM generation in arbitrary Python execution.

Extracts Python code from prompts, executes in a WASM sandbox (Deno + Pyodide),
and biases digit logits toward the computed numeric result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from turnstyle.arithmetic import ArithmeticLogitsProcessor
from turnstyle.core import Turnstyle
from turnstyle.sandbox_backend import (
    DenoPyodideBackend,
    SandboxBackend,
)


@dataclass
class SandboxParsed:
    """Extracted code and description from a prompt."""
    code: str
    description: str


def parse_sandbox_code(text: str) -> SandboxParsed | None:
    """Extract Python code from a prompt.

    Matches (first match wins):
    1. Fenced code blocks: ```python\\n...\\n```
    2. "What does `expr` return?" / "What is the output of `expr`?"
    3. Inline backtick: `expr` (must look like code, not bare arithmetic)
    4. "Execute: expr" / "Evaluate: expr"

    Returns SandboxParsed(code, description) or None.
    Does NOT match bare arithmetic — that's ArithmeticTurnstyle's domain.
    """

    # 1. Fenced code blocks: ```python\n...\n```
    m = re.search(r'```(?:python)?\s*\n(.+?)```', text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code:
            return SandboxParsed(code=code, description="fenced code block")

    # 2. "What does `expr` return?" / "What is the output of `expr`?"
    m = re.search(
        r'(?:what\s+does|what\s+is\s+the\s+(?:output|result)\s+of)\s+`([^`]+)`',
        text, re.IGNORECASE,
    )
    if m:
        code = m.group(1).strip()
        if code:
            return SandboxParsed(code=code, description=f"inline: {code}")

    # 3. Inline backtick `expr` — must look like code, not bare arithmetic
    m = re.search(r'`([^`]+)`', text)
    if m:
        code = m.group(1).strip()
        if _looks_like_code(code):
            return SandboxParsed(code=code, description=f"inline: {code}")

    # 4. "Execute: expr" / "Evaluate: expr" / "Run: expr"
    m = re.search(
        r'(?:execute|evaluate|run)\s*:\s*(.+)',
        text, re.IGNORECASE,
    )
    if m:
        code = m.group(1).strip()
        if code:
            return SandboxParsed(code=code, description=f"directive: {code}")

    return None


def _looks_like_code(text: str) -> bool:
    """Return True if text looks like Python code, not bare arithmetic.

    Bare arithmetic like "445 + 152" is ArithmeticTurnstyle's domain.
    Code indicators: function calls, imports, keywords, subscripts, etc.
    """
    # Reject bare arithmetic: just numbers and operators
    if re.fullmatch(r'[\d\s\+\-\*/\(\)\.%]+', text):
        return False
    # Accept if it has code-like features
    code_patterns = [
        r'\w+\(',       # function call
        r'import\s',    # import statement
        r'\bfor\b',     # for loop
        r'\bif\b',      # conditional
        r'\bwhile\b',   # while loop
        r'\bdef\b',     # function definition
        r'\bclass\b',   # class definition
        r'\[.*\]',      # list/subscript
        r'\{.*\}',      # dict/set
        r'\blen\b',     # builtin
        r'\brange\b',   # range
        r'\bprint\b',   # print
        r'\bTrue\b',    # boolean
        r'\bFalse\b',   # boolean
        r'\bNone\b',    # None
        r'\blambda\b',  # lambda
        r'=',           # assignment
        r'\.\w+',       # attribute access
    ]
    return any(re.search(p, text) for p in code_patterns)


class SandboxTurnstyle(Turnstyle):
    """Grounds LLM generation in arbitrary Python execution via WASM sandbox.

        backend = DenoPyodideBackend()
        t = SandboxTurnstyle(model, tokenizer, device, backend=backend)
        text, proof = t.generate("What does `sum(range(101))` return?")
        # proof.answer == 5050
    """

    def __init__(self, model, tokenizer, device, backend: SandboxBackend | None = None,
                 timeout: float = 5.0, bias_strength: float = 15.0):
        super().__init__(model, tokenizer, device, bias_strength)
        self.backend = backend or DenoPyodideBackend()
        self.timeout = timeout

    def parse(self, prompt: str):
        parsed = parse_sandbox_code(prompt)
        if parsed is None:
            return None

        result = self.backend.execute(parsed.code, timeout=self.timeout)

        if result.error is not None or result.numeric_value is None:
            return None

        return parsed, result

    def make_processor(self, parsed, max_new_tokens: int):
        sandbox_parsed, result = parsed

        answer = result.numeric_value
        answer_str = f"{answer:.6g}" if isinstance(answer, float) else str(answer)

        # Split answer into digits for biasing
        answer_digits = [int(d) for d in answer_str if d.isdigit()]

        expression = sandbox_parsed.code
        if len(expression) > 40:
            expression = expression[:37] + "..."

        proc = ArithmeticLogitsProcessor(
            self.tokenizer, answer_digits, expression, answer,
            self.bias_strength, max_new_tokens=max_new_tokens,
        )
        proc.proof.answer_str = answer_str
        return proc
