# Turnstyle Project

Neurosymbolic library for structured LLM intervention — logit biasing, hidden-state probes, and symbolic solvers that compose into generalizable pipelines.

## Design Philosophy

**BBH is a test harness, not the objective.** The 27-task BBH suite provides ground-truth labels and structural variety for validating generalizable tools. Every component should work beyond BBH:

- **Scene parsing** should detect structure (context → question → options) from model representations, not regex on "Options:". The regex parser is the baseline to beat.
- **Task routing** should use hidden-state probes to classify input type, not keyword matching. `_detect_task` heuristics validate that the signal exists; the probe replaces them.
- **Solvers** should compose from reusable primitives (table parsing, constraint satisfaction, navigation simulation), not be one-off per-task scripts.
- **Metacognitive gates** should generalize the "does the model know it can't do this?" signal across tasks and models.

When building a new capability: first prove it works on BBH with full accuracy, then strip out the BBH-specific parts and test what generalizes.

## Architecture

- **turnstyle**: Core library — probes, logit biasing, intervention primitives
- **swollm**: Evaluation harness — `swollm verify` / `swollm evaluate`, BBH-specific solvers, benchmarking

## Current Frontier

Model-based split detection: use per-token hidden states to identify phase transitions (context → question → options) instead of regex. The regex parser (`parse_scene`) achieves 100% on BBH — it's the ground truth for training and validating a probe-based alternative.
