"""Measure SmolLM2's text-to-SQL path on penguins_in_a_table (limit-40).

Runs solve_penguins() with per-example diag and categorizes outcomes:
  correct / wrong_option / no_table / no_question / sql_error / no_match.
Prints the question type (superlative / count / lookup / sort / other) per failure
so we can see whether superlatives are the weak spot the polarity probe would fix.
"""
from __future__ import annotations

import re
import sys

from turnstyle.bbh import load_task
from turnstyle.penguins import parse_penguins_tables, solve_penguins
from turnstyle.sql import SQLTurnstyle


def qtype(q: str) -> str:
    ql = (q or "").lower()
    if re.search(r"\b(oldest|youngest|tallest|shortest|heaviest|lightest|largest|smallest|biggest)\b", ql):
        return "superlative"
    if re.search(r"\bolder|younger|taller|shorter|heavier|lighter|more than|less than\b", ql):
        return "comparison"
    if "how many" in ql:
        return "count"
    if "cumulated" in ql or "average" in ql or "total" in ql:
        return "aggregate"
    if "sorted by" in ql or "alphabet" in ql:
        return "sort"
    if re.search(r"what is the name|what is the age|which .* cm|tall", ql):
        return "lookup"
    return "other"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    sqlt = SQLTurnstyle(mdl, tok, dev, parse_tables_fn=parse_penguins_tables,
                        probe_label="penguins")

    ex = load_task("penguins_in_a_table")[:limit]
    from collections import Counter
    by_type_total, by_type_ok = Counter(), Counter()
    cats = Counter()
    ncorrect = 0
    for i, e in enumerate(ex):
        diag: dict = {}
        ans = solve_penguins(e["input"], mdl, tok, dev, sql_turnstyle=sqlt, diag=diag)
        tgt = e["target"].strip()
        ok = ans is not None and ans.strip().lower() == tgt.lower()
        qt = qtype(diag.get("question", ""))
        by_type_total[qt] += 1
        if ok:
            ncorrect += 1
            by_type_ok[qt] += 1
            cat = "correct"
        elif not diag.get("tables_parsed"):
            cat = "no_table"
        elif not diag.get("question"):
            cat = "no_question"
        elif diag.get("sql_error"):
            cat = "sql_error"
        elif ans is None:
            cat = "no_match"
        else:
            cat = "wrong_option"
        cats[cat] += 1
        flag = "✓" if ok else "✗"
        if not ok or i < 5:
            print(f"[{i:2d}] {flag} {qt:11s} ans={str(ans):5s} tgt={tgt:5s} "
                  f"sql={str(diag.get('repaired_sql') or diag.get('raw_sql'))[:60]!r}",
                  flush=True)

    n = len(ex)
    print("\n--- penguins SQL path ---")
    print(f"accuracy: {ncorrect}/{n} = {ncorrect/n*100:.1f}%")
    print("outcomes:", dict(cats))
    print("by question type (correct/total):")
    for qt in sorted(by_type_total):
        print(f"  {qt:11s} {by_type_ok[qt]}/{by_type_total[qt]}")


if __name__ == "__main__":
    main()
