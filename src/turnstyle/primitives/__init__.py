"""Reusable structural primitives that ship with turnstyle.

These wrap the artifacts in `turnstyle/data/` (structural probes, etc.) and
the autoprobe library into Primitives that operate on a Blackboard. Domain-
specific primitives (e.g., per-task choice probes, IR extractors) live with
their callers (e.g., swollm/primitives/), not here.
"""
from turnstyle.primitives.arithmetic import ArithmeticEvaluator
from turnstyle.primitives.choice_probe import ChoiceProbe
from turnstyle.primitives.logit_poll import LogitPoll
from turnstyle.primitives.option_detector import OptionDetector

__all__ = [
    "ArithmeticEvaluator",
    "ChoiceProbe",
    "LogitPoll",
    "OptionDetector",
]
