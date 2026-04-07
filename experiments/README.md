# Experiments

Standalone scripts for empirical investigations. Not part of the package — these are
reproducible experiments that produced findings recorded in `CLAUDE.md` or memory.

## Index

| Script | Result | Date |
|---|---|---|
| `accumulator_fix_experiment.py` | tracking_shuffled 78.0% → 92.4% (+14.4pp) via accumulator-side fix (no prompt change) | 2026-04-06 |
| `objcount_diag.py` | object_counting failure diagnostic: 63 extraction_lost_item + 3 category_lookup_failed at baseline 73.6% | 2026-04-06 |
| `objcount_accumulator_fix.py` | object_counting 73.6% → 100.0% (+26.4pp) via accumulator-side fix (no prompt change) | 2026-04-06 |

## Running

```bash
.venv/bin/python experiments/<script>.py
```

Most scripts expect Qwen2.5-1.5B-Instruct and `BBH_CACHE` paths to be configured at
the top of the file.
