"""No-model diagnostic: run the deterministic date solver over date_understanding,
print every failure (or abstention) with its input so we can see the missing forms.

  .venv/bin/python experiments/date_fail_diag.py
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from turnstyle.bbh import load_task, answer_matches
from turnstyle.dates import parse_bbh_date

exs = load_task("date_understanding")[:40]
ok = abstain = wrong = 0
for i, ex in enumerate(exs):
    tgt = ex["target"].strip()
    got = parse_bbh_date(ex["input"])
    if got is None:
        abstain += 1
        verdict = "ABSTAIN"
    elif answer_matches(got, tgt):
        ok += 1
        continue
    else:
        wrong += 1
        verdict = "WRONG"
    print(f"\n--- [{i}] {verdict}  got={got} tgt={tgt} ---")
    print(ex["input"])

print(f"\n=== ok={ok} wrong={wrong} abstain={abstain} / {len(exs)} ===")
