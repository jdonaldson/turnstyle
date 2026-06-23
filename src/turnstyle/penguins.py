"""Penguins-in-a-table: structural table glue for the text-to-SQL pipeline.

The generalizable machinery (SQLite load, model SQL generation, repair, option
matching) lives in `turnstyle.sql.SQLTurnstyle`. The only task-specific code here
is `parse_penguins_tables` — a *structural* parser for BBH's bespoke table format
(inline comma-separated records, newline records, add/delete mutations). It carries
no semantic keyword lists: column/value parsing is by delimiter and type-inference;
the only verbs matched (add/new/delete/remove) are structural table mutations, not
content semantics.

`solve_penguins(prompt, model, tokenizer, device)` is the ADT entry point: it
builds a SQLTurnstyle around the structural parser and runs `_sql_solve`, returning
the matched option letter "(A)" or None on any failure (so the dispatcher abstains
cleanly rather than mis-committing).
"""
from __future__ import annotations

import re


def _clean_columns(raw_cols: list[str]) -> list[str]:
    """Clean raw column strings into SQL-safe names (drop parenthetical units)."""
    out = []
    for col in raw_cols:
        c = re.sub(r"\s*\([^)]*\)", "", col)
        out.append(c.strip().replace(" ", "_").lower())
    return out


def _type_row(vals: list[str]) -> list:
    """Type-infer string values → int/float/str."""
    typed = []
    for v in vals:
        v = v.strip()
        try:
            typed.append(int(v))
        except ValueError:
            try:
                typed.append(float(v))
            except ValueError:
                typed.append(v)
    return typed


def _parse_inline_table(inline_text: str):
    records = re.split(r"(?<=[\d)])\s+(?=[A-Z])", inline_text.strip())
    if len(records) < 2:
        return None, None
    columns = _clean_columns(records[0].split(","))
    n_cols = len(columns)
    if n_cols < 2:
        return None, None
    rows = []
    for rec in records[1:]:
        vals = [v.strip() for v in rec.split(",")]
        if len(vals) == n_cols:
            rows.append(_type_row(vals))
    return columns, rows


def _parse_newline_table(block: str):
    lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return None, None
    columns = _clean_columns(lines[0].split(","))
    n_cols = len(columns)
    rows = []
    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == n_cols:
            rows.append(_type_row(vals))
    return columns, rows


def parse_penguins_tables(text: str) -> dict | None:
    """Parse all tables from BBH penguins text into {name: (columns, rows)}.

    Structural only: tables are located by their fixed framing phrases, parsed by
    delimiter, and mutated by add/delete patterns. Returns None if nothing parsed."""
    tables: dict[str, tuple[list[str], list[list]]] = {}
    peng_m = re.search(r"is a penguin:\s{2,}(.+?)\s{2,}(?:For example|$)", text)
    if peng_m:
        cols, rows = _parse_inline_table(peng_m.group(1))
        if cols and rows:
            tables["penguins"] = (cols, list(rows))
    gir_m = re.search(
        r"listing giraffes:\n(.+?)(?=\n(?:Which|What|How|We|$))", text, re.DOTALL)
    if gir_m:
        cols, rows = _parse_newline_table(gir_m.group(1))
        if cols and rows:
            tables["giraffes"] = (cols, list(rows))

    add_m = re.search(
        r"(?:add|new)\s+(?:a\s+)?(\w+).*?:\n([A-Z].+?)(?:\n|$)", text, re.IGNORECASE)
    if add_m:
        entity = add_m.group(1).lower()
        row_text = add_m.group(2).strip()
        target = next((t for t in tables if entity in t or t in entity), None)
        if target is None and tables:
            target = next(iter(tables))
        if target:
            n_cols = len(tables[target][0])
            vals = [v.strip() for v in row_text.split(",")]
            if len(vals) == n_cols:
                tables[target][1].append(_type_row(vals))

    del_m = re.search(
        r"(?:delete|remove)\s+the\s+\w+\s+named\s+(\w+)", text, re.IGNORECASE)
    if del_m:
        name = del_m.group(1)
        for tname in list(tables):
            cols, rows = tables[tname]
            tables[tname] = (cols, [r for r in rows if r[0] != name])

    return tables or None


def solve_penguins(prompt: str, model, tokenizer, device, *,
                   sql_turnstyle=None, diag: dict | None = None) -> str | None:
    """ADT entry point: structural table parse + text-to-SQL → option letter.

    Returns "(A)" or None. `sql_turnstyle` may be passed to reuse a constructed
    SQLTurnstyle (cheap to build — it only wraps model/tok/device)."""
    if model is None:
        return None
    if sql_turnstyle is None:
        from turnstyle.sql import SQLTurnstyle
        sql_turnstyle = SQLTurnstyle(
            model, tokenizer, device, parse_tables_fn=parse_penguins_tables,
            probe_label="penguins")
    result = sql_turnstyle._sql_solve(prompt, diag=diag)
    if result is None:
        return None
    _, answer = result
    return answer


__all__ = ["parse_penguins_tables", "solve_penguins"]
