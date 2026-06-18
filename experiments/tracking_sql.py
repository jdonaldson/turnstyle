#!/usr/bin/env python3
"""SQL approach for tracking_shuffled_objects.

Pipeline:
  1. Deterministic extraction → state(actor, item) + swaps(seq, actor1, actor2)
  2. Load tables into SQLite (the sandbox)
  3. LLM sees schema description + full data + question, generates SQL
  4. Execute SQL, match against options

The LLM must generate SQL that simulates the swap sequence to compute
final state — it is NOT given pre-computed answers.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/tracking_sql.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, "/Users/jdonaldson/Projects/turnstyle/src")
sys.path.insert(0, str(Path(__file__).parent))

from swollm.bench.bbh import load_task
from turnstyle.sql import get_schema_description, generate_sql, load_into_sqlite
from tracking_deterministic import (
    detect_actors, parse_init, parse_action, parse_query, parse_options,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# ── model loading ──────────────────────────────────────────────────────────

def load_model():
    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"Loading {MODEL_ID} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device).eval()
    return mdl, tok, device


# ── extraction ─────────────────────────────────────────────────────────────

def extract(example: dict) -> dict | None:
    """Extract structured tables from tracking example."""
    text = example["input"]
    lines = [l.strip() for l in text.split(".") if l.strip()]

    actors = detect_actors(lines[0] if lines else text)
    if not actors:
        return None

    init_sent = next((l for l in lines if re.search(r"At the start", l, re.I)), None)
    if not init_sent:
        return None

    initial = parse_init(init_sent, actors)
    if len(initial) < len(actors):
        return None

    swaps = []
    for line in lines:
        pair = parse_action(line)
        if pair:
            swaps.append(pair)

    query_actor = parse_query(text)
    opts = parse_options(text)

    return {
        "initial": initial,
        "swaps": swaps,
        "query_actor": query_actor,
        "options": opts,
    }


# ── table building ─────────────────────────────────────────────────────────

def build_tables(data: dict) -> dict:
    """Build {table_name: (columns, rows)} for load_into_sqlite."""
    state_rows = [(actor, item) for actor, item in data["initial"].items()]
    swap_rows  = [(i + 1, a1, a2) for i, (a1, a2) in enumerate(data["swaps"])]
    return {
        "state": (["actor", "item"], state_rows),
        "swaps": (["seq", "actor1", "actor2"], swap_rows),
    }


# ── few-shot hint ──────────────────────────────────────────────────────────
# One worked example showing the chained-CTE swap pattern.

FEW_SHOT = """\
Example:
  state: Alice/X, Bob/Y  |  swaps: (1,Alice,Bob)
  Q: What does Bob have at the end?
  SQL: SELECT item FROM (
    SELECT actor,
      CASE WHEN actor='Alice' THEN (SELECT item FROM state WHERE actor='Bob')
           WHEN actor='Bob'   THEN (SELECT item FROM state WHERE actor='Alice')
           ELSE item END AS item
    FROM state
  ) WHERE actor='Bob'
  -> X
"""


# ── solve ──────────────────────────────────────────────────────────────────

def solve(example: dict, model, tokenizer, device) -> tuple[str | None, str | None]:
    """Returns (sql, answer_letter) or (None, None)."""
    data = extract(example)
    if data is None or data["query_actor"] is None:
        return None, None

    tables = build_tables(data)
    conn   = load_into_sqlite(tables)
    schema = get_schema_description(conn)

    query_actor = data["query_actor"]
    question    = f"What does {query_actor} have after all swaps?"

    sql = generate_sql(model, tokenizer, device, schema, question,
                       examples=FEW_SHOT, max_tokens=200)

    try:
        cursor = conn.execute(sql)
        row    = cursor.fetchone()
        result = str(row[0]).strip() if row else None
    except Exception as e:
        conn.close()
        return sql, None
    finally:
        conn.close()

    if result is None:
        return sql, None

    # Match to option letter
    opts = data["options"]
    for letter, val in opts.items():
        if val.strip().lower() == result.lower():
            return sql, f"({letter})"
        if result.lower() in val.lower() or val.lower() in result.lower():
            return sql, f"({letter})"

    return sql, None


# ── evaluation ─────────────────────────────────────────────────────────────

def evaluate(task: str, model, tokenizer, device, n: int = 50) -> None:
    examples = load_task(task)[:n]
    correct = wrong = sql_fail = extract_fail = 0

    for i, ex in enumerate(examples):
        sql, pred = solve(ex, model, tokenizer, device)
        target = ex["target"]

        if sql is None:
            extract_fail += 1
            status = "EXTRACT_FAIL"
        elif pred is None:
            sql_fail += 1
            status = f"SQL_FAIL  sql={sql[:60]!r}"
        elif pred == target:
            correct += 1
            status = "ok"
        else:
            wrong += 1
            status = f"WRONG pred={pred} target={target}"

        # Verbose for first 10, then quiet
        if i < 10 or status != "ok":
            print(f"  [{i:3d}] {status}", flush=True)

    total = correct + wrong + sql_fail + extract_fail
    print(
        f"\n{task:<48}  {correct}/{total} ({100*correct/total:.1f}%)  "
        f"wrong={wrong}  sql_fail={sql_fail}  extract_fail={extract_fail}"
    )


def main() -> None:
    model, tokenizer, device = load_model()
    for task in [
        "tracking_shuffled_objects_three_objects",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
    ]:
        print(f"\n=== {task} ===")
        evaluate(task, model, tokenizer, device, n=50)


if __name__ == "__main__":
    main()
