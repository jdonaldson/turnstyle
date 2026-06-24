"""BBH evaluation with regex fast paths disabled — forces LLM fallback paths.

Tests SmolLM2's SQL/IR extraction quality on tasks where regex normally handles 100%.

Note: Monkey-patches regex solvers BEFORE importing run functions — import order matters.
"""
import time

# Monkey-patch regex solvers to return None BEFORE importing run functions
from swollm.solvers import (
    tracking_shuffled,
    navigate,
    web_of_lies,
    logical_deduction,
    object_counting,
    colored_objects,
)

tracking_shuffled._regex_solve = lambda *a, **kw: None
navigate._regex_solve = lambda *a, **kw: None
web_of_lies._regex_solve = lambda *a, **kw: None
logical_deduction._parse_constraints = lambda *a, **kw: []
object_counting._regex_solve = lambda *a, **kw: None
colored_objects._co_regex_solve = lambda *a, **kw: None

from swollm.bench.bbh import load_task, run_bypass, run_baseline, get_exemplars
from swollm.solvers import SOLVER_MAP, LLM_FALLBACK_TASKS
from swollm.solvers.turnstyles import get_turnstyle_solver

from common import load_model, DEFAULT_MODEL

# Tasks with LLM fallback paths (the ones we patched)
TASKS = [
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "navigate",
    "web_of_lies",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "object_counting",
    "reasoning_about_colored_objects",
]


def main():
    print(f"Loading model: {DEFAULT_MODEL}", flush=True)
    tokenizer, model, device = load_model()
    print(f"Device: {device}", flush=True)

    print(f"\n{'Task':<50} {'Base':>6} {'NoRegex':>8} {'Fails':>6}  Time", flush=True)
    print("─" * 85, flush=True)

    totals = {"base": 0, "bypass": 0, "n": 0}

    for task_name in TASKS:
        examples = load_task(task_name)
        solver = SOLVER_MAP[task_name]
        exemplar_idx = get_exemplars(examples)

        # Build LLM fallback
        fallback = None
        if task_name in LLM_FALLBACK_TASKS:
            fallback = get_turnstyle_solver(task_name, model, tokenizer, device)

        t0 = time.time()

        # Baseline (logit poll, no solver)
        base_acc, _n = run_baseline(model, tokenizer, device, examples, exemplar_idx)

        # Bypass with regex disabled — forces LLM fallback
        bypass_acc, _n2, parse_fails = run_bypass(
            model, tokenizer, device, examples, exemplar_idx,
            solver, llm_fallback=fallback,
        )

        elapsed = time.time() - t0
        print(
            f"  {task_name:<48} {base_acc:5.1f}% {bypass_acc:7.1f}%  {parse_fails:>5}  {elapsed:.0f}s",
            flush=True,
        )

        totals["base"] += base_acc
        totals["bypass"] += bypass_acc
        totals["n"] += 1

    n = totals["n"]
    print("─" * 85)
    print(f"  {'AVERAGE':<48} {totals['base']/n:5.1f}% {totals['bypass']/n:7.1f}%", flush=True)


if __name__ == "__main__":
    main()
