"""No-model tests for the solver step-proofs."""
from turnstyle import proofs
from turnstyle.bbh import load_task


def test_arithmetic_steps_order_and_result():
    from turnstyle.arithmetic import parse_expression
    expr, _ = parse_expression("((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) =")
    steps = proofs.arithmetic_steps(expr)   # dispatch passes the cleaned expr
    assert steps is not None
    assert steps[-1].endswith("= 24")       # post-order trace ends at the top-level result
    assert "9 × 5 = 45" in steps            # multiplication shown as its own step


def test_arithmetic_steps_bad_expr():
    assert proofs.arithmetic_steps("not an expression") is None


def test_dyck_steps_closers_match_target():
    ex = load_task("dyck_languages")[3]
    steps = proofs.dyck_steps(ex["input"])
    assert steps is not None
    # the closers line must spell the target answer
    closers = steps[-1].split(":", 1)[1].strip()
    assert closers == ex["target"].strip()


def test_tracking_steps_chain():
    ex = load_task("tracking_shuffled_objects_three_objects")[0]
    steps = proofs.tracking_steps(ex["input"])
    assert steps is not None
    assert steps[0].startswith("start:")
    assert steps[-1].startswith("final:")
    assert any("↔" in s for s in steps)   # at least one swap narrated


def test_explain_dispatches_by_task():
    from turnstyle.dispatch import Arithmetic, BracketMatch
    assert "= 24" in proofs.explain(Arithmetic(expr="(20 + 4)", value="24"), "")
    # non-traced task → None (falls back to the solver's label)
    assert proofs.explain(object(), "") is None
