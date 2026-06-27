"""Human-readable step traces for the deterministic solvers — the "show your work"
behind a committed Answer. `explain(task, prompt)` returns a multi-line proof string
(or None to fall back to the solver's one-line method label). Each tracer reuses its
solver's own primitives so the narration can't drift from the computed answer.
"""
from __future__ import annotations

import ast
import re
from typing import Optional

_OPS = {ast.Add: "+", ast.Sub: "−", ast.Mult: "×", ast.Div: "÷",
        ast.FloorDiv: "÷", ast.Mod: "mod", ast.Pow: "^"}


def _trace_arith(node, steps):
    if isinstance(node, ast.Expression):
        return _trace_arith(node.body, steps)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        v = _trace_arith(node.operand, steps)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        l = _trace_arith(node.left, steps)
        r = _trace_arith(node.right, steps)
        sym = _OPS[type(node.op)]
        res = {"+": l + r, "−": l - r, "×": l * r, "÷": l // r if r else 0,
               "mod": l % r if r else 0, "^": l ** r}[sym]
        steps.append(f"{l} {sym} {r} = {res}")
        return res
    raise ValueError("unsupported node")


def arithmetic_steps(expr: str) -> Optional[list[str]]:
    """Order-of-operations trace (post-order = inner expressions first)."""
    try:
        tree = ast.parse(expr, mode="eval")
        steps: list[str] = []
        _trace_arith(tree, steps)
    except Exception:
        return None
    return steps or None


def dyck_steps(prompt: str) -> Optional[list[str]]:
    """Bracket-stack resolution: scan the input, show the unclosed stack and the
    closers needed (top of stack first)."""
    from turnstyle.dyck import parse_dyck, _OPEN_TO_CLOSE, _CLOSE_TO_OPEN
    parsed = parse_dyck(prompt)
    if parsed is None:
        return None
    stack: list[str] = []
    for ch in parsed[0].split():           # open_seq is space-separated bracket tokens
        if ch in _OPEN_TO_CLOSE:
            stack.append(ch)
        elif ch in _CLOSE_TO_OPEN and stack and stack[-1] == _CLOSE_TO_OPEN[ch]:
            stack.pop()
    if not stack:
        return None
    closers = " ".join(_OPEN_TO_CLOSE[c] for c in reversed(stack))
    return [
        f"unmatched open brackets (bottom→top): {' '.join(stack)}",
        f"close in reverse, innermost first: {closers}",
    ]


def tracking_steps(prompt: str) -> Optional[list[str]]:
    """Swap-chain simulation: initial assignment → each swap → final → matched option."""
    from turnstyle.object_tracking import (
        _detect_actors, _parse_init_sent, _ACTION_RE, _QUERY_RE, _OPTIONS_RE)
    lines = [l.strip() for l in prompt.split(".") if l.strip()]
    actors = _detect_actors(lines[0] if lines else prompt)
    if not actors:
        return None
    init_sent = next((l for l in lines if re.search(r"At the start", l, re.I)), None)
    if not init_sent:
        return None
    state = _parse_init_sent(init_sent, actors)
    if len(state) < len(actors):
        return None

    def show():
        return ", ".join(f"{a}:{state[a]}" for a in actors if a in state)

    steps = [f"start: {show()}"]
    for line in lines:
        m = _ACTION_RE.search(line)
        if m:
            a1, a2 = m.group(1), m.group(2)
            if a1 in state and a2 in state:
                state[a1], state[a2] = state[a2], state[a1]
                steps.append(f"{a1} ↔ {a2}  →  {show()}")
    qm = _QUERY_RE.search(prompt)
    if not qm or qm.group(1) not in state:
        return None
    who = qm.group(1)
    steps.append(f"final: {who} has {state[who]}")
    return steps


def explain(task, prompt: str) -> Optional[str]:
    """Dispatch on the committed Task; return a worked-step proof or None."""
    from turnstyle.dispatch import Arithmetic, BracketMatch, StateTracking
    steps = None
    if isinstance(task, Arithmetic):
        steps = arithmetic_steps(task.expr)
    elif isinstance(task, BracketMatch):
        steps = dyck_steps(prompt)
    elif isinstance(task, StateTracking):
        steps = tracking_steps(prompt)
    return "\n".join(steps) if steps else None
