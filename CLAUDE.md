# Turnstyle Project

Neurosymbolic library for structured LLM intervention — logit biasing, hidden-state probes, and symbolic solvers that compose into generalizable pipelines.

## Design Philosophy

**BBH is a test harness, not the objective.** The 27-task BBH suite provides ground-truth labels and structural variety for validating generalizable tools. Every component should work beyond BBH:

- **Scene parsing**: per-token split probe at L8 detects body→options transitions (100% accuracy, cross-task generalized). Regex `parse_scene` remains the offline/no-model fallback.
- **Task routing**: `route_solver()` uses hidden-state probes for single-pass classification + scene splitting. `_detect_task` heuristics are the baseline it replaces.
- **Solvers**: compose from reusable primitives — SQL generation, logit polling, knowledge decomposition. Task-specific solvers live in swollm.
- **Metacognitive gates**: explored; gate is redundant when fallback chain is sequential ("try A, then B on failure" = same routing). Gate only matters for commit-before-trying scenarios.

When building a new capability: first prove it works on BBH with full accuracy, then strip out the BBH-specific parts and test what generalizes.

## Architecture

```
src/turnstyle/
  core.py           Base Turnstyle class, SequenceLogitsProcessor
  probe.py          TurnstyleProbe, MultiPositionProbe, IntentProbe,
                    MetacognitiveProbe, StrategyRouter, RoutingTurnstyle
  extract.py        LLM extraction, ExtractionSpec
  sweep.py          Probe training infrastructure (optional, needs sklearn)
  sql.py            SQLTurnstyle — text-to-SQL + probe routing + logit poll
  formal_fallacies.py  NL→FOL with probe-dispatched parser
  sandbox.py        SandboxTurnstyle + backends (Deno/Pyodide, Wasmtime)
  ...               15+ task-specific solvers (arithmetic, boolean, dates, etc.)
```

**Downstream**: [swollm](~/Projects/swollm) — BBH evaluation harness. See ecosystem map in `~/Projects/CLAUDE.md`.

## Current Frontier

- **Penguins at 99.3%** (142/143) via SQL → knowledge poll → logit poll → free generation. Single remaining failure is a unit conversion issue (0.6 m → 60 cm).
- **repair_sql v3**: 90.9% → 99.3% (+8.4pp). New patterns: COUNT(*) UNION, explicit table name triggers, superlative→MAX/MIN, ordinal ROWID, "next to last", inverted comparisons, multi-table ROWID ordering.
- **Meta-schema solver**: intercepts "how many species" (→ table count) and "column number" (→ column position) before SQL generation.
- **`--diagnose` flag**: `swollm evaluate --diagnose` captures per-example diagnostics (tier, SQL, errors, options) for debugging.
- **Selective few-shot conditioning**: intent probe at L4 routes COMPARISON queries to few-shot hints (+2.8pp). Other intents are net-negative when hinted.
- **Bonsai 8B**: 1-bit 8B model smoke-tested (194.8 tok/s). Not integrated. MLX path blocked on Python 3.14.

## Known Issues

- `transformers` 5.4.0 (from Bonsai exploration) breaks `sentence-transformers` (wants <5.0.0). Pin back if embeddings needed.
