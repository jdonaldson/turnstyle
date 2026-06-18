# Phase 0 — Feasibility Gate

Model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, no-regex mode.

## Sequential vs Oracle

| task | n | sequential | oracle | gap |
|---|---|---|---|---|
| `penguins_in_a_table` | 143 | 0.944 | 0.958 | +0.014 |
| `tracking_shuffled_objects_three_objects` | 247 | 0.300 | 0.615 | +0.316 |
| `object_counting` | 247 | 0.271 | 0.336 | +0.065 |
| `navigate` | 247 | 0.798 | 0.862 | +0.065 |
| `web_of_lies` | 247 | 0.486 | 0.486 | +0.000 |

## Per-tier breakdown

### `penguins_in_a_table`

| tier | answered | when answered | overall |
|---|---|---|---|
| sql | 93.7% | 97.0% | 90.9% |
| knowledge_poll | 26.6% | 31.6% | 8.4% |
| logit_poll | 100.0% | 23.1% | 23.1% |
| baseline | 100.0% | 23.1% | 23.1% |

### `tracking_shuffled_objects_three_objects`

| tier | answered | when answered | overall |
|---|---|---|---|
| sql | 0.0% | 0.0% | 0.0% |
| logit_poll | 100.0% | 30.0% | 30.0% |
| baseline | 100.0% | 31.6% | 31.6% |

### `object_counting`

| tier | answered | when answered | overall |
|---|---|---|---|
| sql | 100.0% | 27.1% | 27.1% |
| baseline | 100.0% | 6.5% | 6.5% |

### `navigate`

| tier | answered | when answered | overall |
|---|---|---|---|
| ir | 94.7% | 82.1% | 77.7% |
| baseline | 100.0% | 51.0% | 51.0% |

### `web_of_lies`

| tier | answered | when answered | overall |
|---|---|---|---|
| ir | 0.0% | 0.0% | 0.0% |
| baseline | 100.0% | 48.6% | 48.6% |

## Gate

Max gap: **+0.316**. Threshold +0.02. **PASSED.**
