# Turnstyle Project

Neurosymbolic library for structured LLM intervention — logit biasing, hidden-state probes, and symbolic solvers that compose into generalizable pipelines.


## Design Philosophy

**BBH is a test harness, not the objective.** The 27-task BBH suite provides ground-truth labels and structural variety for validating generalizable tools. Every component should work beyond BBH:

- **Scene parsing**: per-token split probe at L8 detects body→options transitions (100% accuracy, cross-task generalized). `parse_scene()` is the offline/no-model fallback — sentences-first, returns `Scene(body, question, options)` dataclass, no "Options:" keyword dependency.
- **Task routing**: `route_solver()` uses hidden-state probes for single-pass classification + scene splitting. `_detect_task` heuristics are the baseline it replaces.
- **Solvers**: compose from reusable primitives — SQL generation, logit polling, knowledge decomposition. Task-specific solvers live in swollm.
- **Metacognitive gates**: explored; gate is redundant when fallback chain is sequential ("try A, then B on failure" = same routing). Gate only matters for commit-before-trying scenarios.

**No semantic keyword lists in solvers.** If a solver contains adjective lists, vocabulary enumerations, or multi-case regex for semantic parsing, it must be replaced with probe routing + LLM extraction. The only legitimate regex uses are structural patterns with no natural-language synonyms: arithmetic/boolean operators, digit patterns, bracket matching. When you see a keyword list in a solver, replace it — don't add a fallback around it.

When building a new capability: first prove it works on BBH with full accuracy, then strip out the BBH-specific parts and test what generalizes.

## Architecture

```
src/turnstyle/
  core.py           Base Turnstyle class, SequenceLogitsProcessor
  probe.py          TurnstyleProbe, MultiPositionProbe, IntentProbe,
                    MetacognitiveProbe, StrategyRouter, RoutingTurnstyle
  extract.py        LLM extraction, ExtractionSpec
  ir.py             Scene, parse_scene(); IRSpec + IRSolver — single-pass JSON extraction + deterministic compute; SentenceIRSpec
  sweep.py          Probe training infrastructure (optional, needs sklearn)
  sql.py            SQLTurnstyle — text-to-SQL + probe routing + logit poll
  formal_fallacies.py  NL→FOL with probe-dispatched parser
  sandbox.py        SandboxTurnstyle + backends (Deno/Pyodide, Wasmtime)
  ...               15+ task-specific solvers (arithmetic, boolean, dates, etc.)
```

**Downstream**: [swollm](~/Projects/swollm) — BBH evaluation harness. See ecosystem map in `~/Projects/CLAUDE.md`.

## Current Frontier

- **T0 eval: 100%** (2026-04-14): All 4 symbolic tasks at 40/40 on SmolLM2. `parse_expression()` handles nested arithmetic via `ast.parse` + `_eval_node` (no `eval()`), routed through `SequenceLogitsProcessor(immediate=True)`. Sorting regex broadened to `\S` for special chars (`it&t`). See `experiments/smol_capability_eval_report.md`.
- **Scene/SolverResult API** (2026-04-10): `parse_scene()` returns `Scene(body, question, options)` — sentences-first, no keyword deps. `SolverResult(text, proof, solver, sentences)` is the structured output type. `RoutingTurnstyle.solve()` → `list[SolverResult]`; `generate()` preserved for compat. `_solve_one()` internal method tracks which solver handled the prompt.
- **Solver generalization**: 7 tasks wired with SQL/IR fallback paths behind regex fast paths. Regex remains primary (100% on BBH); fallbacks activate on regex parse failure for out-of-distribution resilience. SQL-first: object_counting, colored_objects, tracking_shuffled (×3). IR extraction: navigate, web_of_lies. LLM_FALLBACK_TASKS: 20/27 tasks.
- **IRSpec/IRSolver** (`ir.py`): generic infrastructure for single-pass JSON extraction via LLM + deterministic compute. Used by navigate (coordinate simulation) and web_of_lies (truth propagation).
- **`_sql_solve()` free-answer support**: extended to handle tasks without multiple-choice options (returns raw SQL result string).
- **Penguins at 99.3%** (142/143) via SQL → knowledge poll → logit poll → free generation. Single remaining failure is a unit conversion issue (0.6 m → 60 cm).
- **repair_sql v3**: 90.9% → 99.3% (+8.4pp). New patterns: COUNT(*) UNION, explicit table name triggers, superlative→MAX/MIN, ordinal ROWID, "next to last", inverted comparisons, multi-table ROWID ordering.
- **Meta-schema solver**: intercepts "how many species" (→ table count) and "column number" (→ column position) before SQL generation.
- **`--diagnose` flag**: `swollm evaluate --diagnose` captures per-example diagnostics (tier, SQL, errors, options) for debugging.
- **Selective few-shot conditioning**: intent probe at L4 routes COMPARISON queries to few-shot hints (+2.8pp). Other intents are net-negative when hinted.
- **Relational transfer capability** (2026-04-04): 4-model probe study (Qwen 1.5B, SmolLM2 1.7B, Qwen 3B, Phi 3.8B) shows relational abstraction is **training-data-dependent, not size-dependent**. Non-monotonic scaling: Qwen 1.5B (76%) > Qwen 3B (50%) > SmolLM2 1.7B (41%). Phi 3.8B has deepest/most stable features (79%, L12+). Qwen's early-layer transfer (L3) is likely surface patterns; Phi's mid-layer transfer (L12) is genuinely abstract. SmolLM2 should stay deterministic; for backbone upgrades prioritize structured-data-trained models (Qwen, Phi). See memory `relationship_transfer_probe.md`.
- **Relational transcription pipeline** (2026-04-07): no-regex 5-task experiment with Qwen 1.5B. L1 boundary probe (100% in-distribution) replaces LLM segmenter. Flexible accumulator (structural typing by field presence, not field names) + accumulator-side fallbacks. Suite average 75.8% → **96.9%** (+21.1pp). Per-task: navigate 100%, web_of_lies 98.8%, **tracking_shuffled 92.4%** (+31.2pp), **object_counting 100%** (+35.6pp), **logical_deduction 93.2%** (+20.0pp). All gains from diagnose-then-fix-consumer: run a failure-categorizing diagnostic, identify which structural variants Qwen produces (e.g. `compared_item` instead of `item2`, `type: number` instead of `type: item`, `position: newest` instead of `position: second-newest`), then extend the accumulator's field fallback chains and add segment-text overrides. **Key lesson (confirmed three times)**: don't fight Qwen's learned output format — diagnose, then fix the consumer. Prompt-side attempts (list-output, pluralized container) lost ≥10pp. See `experiments/` and memory `relational_transcription.md`.
- **Intercept-and-correct architecture** (2026-04-09): CoT generation + swap extraction + Python simulation on SmolLM2 (tracking_shuffled). T2 swap extraction: 96/92/68% (three/five/seven objects). T3 intercept+correct: **98/94/84%** — within 2pp of 100% deterministic ceiling on three_objects, without any task-specific regex. Key finding: model transcribes structure accurately (96% swap extraction) but fails catastrophically at in-context state propagation (40/8/12% CoT answer). SQL generation failed entirely (~5%) — 1.5B models can't generate chained CTEs. Seven-objects T2 failure mode: `gt=7 cot=8` (hallucinated extra swap). See `experiments/tracking_cot.py`.
- **Simulation route architecture** (2026-04-09): 4 BBH route types fit "model transcribes, code simulates" — OBJECT_TRACKING (×3, proven), SPATIAL_NAVIGATION (navigate, directly parseable), TRUTH_CHAIN (web_of_lies, directly parseable), COMPARISON_ORDERING (logical_deduction ×3, entity-token probe path). Navigate/web_of_lies are directly parseable from problem text; CoT interception most valuable for OBJECT_TRACKING; entity-token probe is the right path for COMPARISON_ORDERING (SmolLM2 cannot produce orderings via CoT).
- **Entity-token probe for logical_deduction** (2026-04-10): Last-occurrence hidden state at item tokens (L13-15) encodes correct position with 88-91% item-level accuracy. Probe → `answer_from_ordering()` achieves **87.3%** answer accuracy vs 37% logit poll baseline. Cross-task: works only for COMPARISON_ORDERING (static constraints locally encodable). Fails for TRUTH_CHAIN (requires transitive chain) and OBJECT_TRACKING (requires future-swap awareness). Key bug: BPE splits multi-syllable words — use token-ID subsequence search, not first-word string matching. Script: `/tmp/entity_probe_sweep.py`.
- **Negative results** (2026-04-10): (1) **Probe prefill hurts** — injecting correct ordering as text before Options: 32% vs 37% baseline; SmolLM2 cannot use injected ordering statements. (2) **Metacognitive probe: no signal** — last-token hidden state AUC ≈ 0.50 at all 25 layers; model cannot predict its own CoT success.
- **Route classification probe** (2026-04-10): Last-token hidden state at **L1** classifies 5 route types with 100% 5-fold CV and 92-100% LOO for multi-task families (COMP×3: 92-100%, TRACK×3: 100%). Signal at L1 = structural/syntactic, not semantic. Single-task classes (TRUTH, NAV) need ≥1 training example (LOO zeroes are a data-coverage artifact, not a representation failure). **Replaces keyword routing** — vocabulary-independent, no code changes needed for new phrasing. Script: `/tmp/route_probe_sweep.py`. See memory `route_classification_probe.md`.
- **Vocabulary routing** (2026-04-09, updated 2026-04-10): 18/18 BBH tasks correctly routed. logical_deduction per-example confidence: 76-81% → 97-100% after adding tournament-finish vocabulary. Superseded by route classification probe for production; kept as fast offline fallback. See `experiments/vocab_routing.py`.
- **Self-describing solvers + auto-fit routing** (2026-04-12): `RoutingTurnstyle.build(solvers, model, tokenizer)` trains L1 last-token route probe from each solver's `examples` attribute — 30 real BBH inputs per class. **11/11 routing accuracy** on held-out BBH (index 31, outside training window). Three integration bugs found during wiring: (1) `_navigate_solve()` always returned "Yes"/"No" — added step/turn pattern guard so it returns `None` on non-navigate prompts; (2) `parse_arithmetic()` matched date strings like `7/9/1972` — added early-reject for `\d{1,2}/\d{1,2}/\d{4}`; (3) `SQLTurnstyle.examples` used synthetic pipe-format while all actual BBH penguins are CSV — replaced with `load_task("penguins_in_a_table")`. Rule: solver `examples` must be drawn from `load_task()`, not invented, because format drift is invisible until routing fails.
- **Parser fast-path reliability rule**: any `parse()` that cannot return `None` is a bug — it becomes a catch-all that preempts probe routing for every task type. All parsers must fail gracefully on non-matching input.
- **Untested**: LLM fallback paths never fire on BBH (regex handles 100%). SmolLM2 JSON extraction quality unknown — needs GPU validation.
- **Bonsai 8B**: 1-bit 8B model smoke-tested (194.8 tok/s). Not integrated. MLX path blocked on Python 3.14.
- **BitNet-b1.58-2B-4T**: partner=87.8%@L15, owner=100%@L17 — comparable to SmolLM2-1.7B. Requires bfloat16 (float16 overflows to NaN). Not integrated; staying with SmolLM2.

## Known Issues

- `transformers` 5.4.0 (from Bonsai exploration) breaks `sentence-transformers` (wants <5.0.0). Pin back if embeddings needed.
