# Experiments

Standalone scripts for empirical investigations. Not part of the package — these are
reproducible experiments that produced findings recorded in `CLAUDE.md` or memory.

## Index

| Script | Result | Date |
|---|---|---|
| `accumulator_fix_experiment.py` | tracking_shuffled 78.0% → 92.4% (+14.4pp) via accumulator-side fix (no prompt change) | 2026-04-06 |
| `objcount_diag.py` | object_counting failure diagnostic: 63 extraction_lost_item + 3 category_lookup_failed at baseline 73.6% | 2026-04-06 |
| `objcount_accumulator_fix.py` | object_counting 73.6% → 100.0% (+26.4pp) via accumulator-side fix (no prompt change) | 2026-04-06 |
| `ld_diag.py` | logical_deduction failure diagnostic: 51 failures at baseline 79.6% across 4 categories | 2026-04-07 |
| `ld_accumulator_fix.py` | logical_deduction 79.6% → 93.2% (+13.6pp) via accumulator-side fix (field fallback + segment override + preamble skip) | 2026-04-07 |
| `route_probe_sweep.py` | L1 last-token probe classifies 5 route types: 100% 5-fold CV, 92-100% LOO. Replaces keyword routing. | 2026-04-10 |
| `bitnet_probe.py` | BitNet-2B partner=87.8%@L15, owner=100%@L17. Comparable to SmolLM2, requires bfloat16. | 2026-04-11 |
| `hub_accuracy_test.py` | End-to-end RoutingTurnstyle validation on held-out BBH (indices 30-39). 11/11 routing accuracy. | 2026-04-12 |
| `bbh_no_regex.py` | LLM fallback stress test: disables all regex fast paths, forces SmolLM2 JSON extraction. | 2026-04-13 |
| `smol_capability_eval.py` | Full T0–T3 capability eval on SmolLM2. T0=100% (40/40). See `smol_capability_eval_report.md`. | 2026-04-14 |
| `temporal_encoding_probe.py` | Exp 1: time-token RSA — linear ordinal encoding, peaks L8 (r=0.93). Exp 2: last-token answer decodability — 66.4% at L20. | 2026-04-19 |
| `temporal_option_probe.py` | Option start-time token probe: 94.8% answer accuracy at L14 (vs 66.4% last-token, 25% chance). Mirrors entity-ordering probe pattern. | 2026-04-19 |
| `temporal_probe_diagnostic.py` | Structural diagnostic: n_overlap=0 — free-slot start NEVER equals a constraint start in the dataset. Probe learns set-membership, not semantic reasoning. | 2026-04-19 |
| `temporal_multimodel_probe.py` | Multi-model sweep: signal universal (93–99.2% across 5 models). Phi-4-mini 99.2% at L8 (25% depth). L/N varies 0.25–0.66; not architecture-invariant. | 2026-04-19 |
| `temporal_rsa_pearson.py` | Weber's Law reanalysis with Pearson r (Spearman is rank-invariant, can't distinguish linear vs log). Result: log wins 17/25 layers — Weber encoding confirmed for clock time. Best L8: R²-lin=0.834, R²-log=0.847. | 2026-04-19 |
| `social_affect_embedding.md` | **Design spec (not yet built):** Monkey-Sphere Affect Embedding — social emotion as perceived affect-flux on an affect-weighted relational graph (built on `SemanticFrame`). Transfer-operator spectral treatment, egocentric vs allocentric, monkey-sphere as a rank budget. First falsifier F1 = synthetic 3-node graphs. | 2026-06-21 |
| `epa_theory_horse_race.py` | Ternary affect-theory horse race (phase 1, intrinsic). SmolLM2 across 25 layers: **PAD most independent** (axes |cos|≈0.06), **EPA most encodable** (loo 0.86); third-axis race **Potency > Dominance > Attention-Rejection > Tension** (Potency enc 0.79, cos\|valence 0.04); 4th axis (novelty) independent (\|cos\|0.13) AND encodable (0.87) → ternary may be insufficient. Intrinsic/circular — phase 2 needs NRC-VAD/ACT. | 2026-06-21 |
| `epa_external_validation.py` | Phase 2, anti-circular: ridge probes from SmolLM2 acts → **human** ratings (Warriner V/A/D + ACT E/P/A), held-out 5-fold CV on 1000 common words. **EPA wins all 3 axes:** Evaluation 0.83 ≈ Valence 0.82, **Activity 0.72 > Arousal 0.63**, **Potency 0.81 > Dominance 0.73** — confirms phase 1 against independent data. E≈V near-tie controls for source noise. Corrects phase-1's "Potency weak" (anchor artifact). | 2026-06-21 |

## Common Setup

`common.py` provides shared model loading and solver initialization:

```python
from common import load_hub
tok, mdl, solvers, hub = load_hub()
```

Functions:
- `load_model(model_id=DEFAULT_MODEL)` → `(tok, mdl, device)` — auto-detects MPS/CPU
- `make_solvers(mdl, tok, device)` → `list` — all 11 core solvers
- `load_hub(model_id=DEFAULT_MODEL)` → `(tok, mdl, solvers, hub)` — full pipeline with routing

Default model: `SmolLM2-1.7B-Instruct`.

## Running

```bash
.venv/bin/python experiments/<script>.py
```

Newer scripts (Apr 10+) use `common.py` and default to SmolLM2-1.7B-Instruct.
Older scripts may expect Qwen2.5-1.5B-Instruct — check the top of each file.
