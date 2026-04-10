"""SQL turnstyle — grounds structured-data reasoning via text-to-SQL.

Scene/data → SQLite table → LLM generates SQL → execute → match to option → logit bias.
Regex fast path via _regex_solve(); SQL fallback via _sql_solve() (model-generated SQL).

Provides a generalized base class for any task where structured data can be
loaded into SQLite and questions answered via SQL. Infrastructure adapted from
swollm/solvers/penguins.py.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import torch

from turnstyle.core import SequenceLogitsProcessor, Turnstyle


# ════════════════════════════════════════════════════════════════════════
# Number word mapping (for answer matching)
# ════════════════════════════════════════════════════════════════════════

_NUM_TO_WORD = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}


# ════════════════════════════════════════════════════════════════════════
# Schema specification for model-based extraction
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SchemaSpec:
    """Declares how to extract structured data from a prompt via model generation.

    The model outputs JSON objects with the declared columns (plus count_key for
    row expansion). The library handles expansion, positional detection, SQL
    table creation, and query generation.

    Args:
        table_name: Name of the SQLite table to create.
        columns: Column names in the SQL table (excluding count_key).
        extraction_prompt: Template with {scene} placeholder for the model.
        count_key: JSON key for row expansion (consumed, not stored in table).
        positional_detector: Callable returning True when position column needed.
        max_extract_tokens: Max tokens for model extraction generation.
    """
    table_name: str
    columns: list[str]
    extraction_prompt: str
    count_key: str = "count"
    positional_detector: Callable[[str], bool] | None = None
    max_extract_tokens: int = 300


# ════════════════════════════════════════════════════════════════════════
# SQL infrastructure
# ════════════════════════════════════════════════════════════════════════

def load_into_sqlite(tables: dict) -> sqlite3.Connection:
    """Load parsed tables into in-memory SQLite.

    Args:
        tables: {table_name: (columns, rows)} where columns is a list of
                column name strings and rows is a list of tuples/lists.

    Returns:
        sqlite3.Connection to the in-memory database.
    """
    conn = sqlite3.connect(":memory:")
    for tname, (columns, rows) in tables.items():
        col_defs = []
        if rows:
            for col, val in zip(columns, rows[0]):
                if isinstance(val, int):
                    col_defs.append(f'"{col}" INTEGER')
                elif isinstance(val, float):
                    col_defs.append(f'"{col}" REAL')
                else:
                    col_defs.append(f'"{col}" TEXT')
        else:
            col_defs = [f'"{col}" TEXT' for col in columns]
        conn.execute(f"CREATE TABLE {tname} ({', '.join(col_defs)})")
        ph = ", ".join(["?"] * len(columns))
        for row in rows:
            conn.execute(f"INSERT INTO {tname} VALUES ({ph})", row)
    conn.commit()
    return conn


def get_schema_description(conn: sqlite3.Connection) -> str:
    """Schema + full table contents for LLM prompting."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = [row[0] for row in cursor.fetchall()]
    parts = []
    for tname in table_names:
        cursor = conn.execute(f"PRAGMA table_info({tname})")
        cols = [row[1] for row in cursor.fetchall()]
        parts.append(f"Table: {tname} (columns: {', '.join(cols)})")
        cursor = conn.execute(f"SELECT * FROM {tname}")
        for row in cursor.fetchall():
            parts.append(f"  {row}")
        parts.append("")
    parts.append(
        'Row order matches the listing above. '
        '"first" = ROWID 1, "last" = highest ROWID.'
    )
    return "\n".join(parts)


def generate_sql(
    model, tokenizer, device, schema: str, question: str,
    examples: str = "", max_tokens: int = 100,
) -> str:
    """Use an LLM to generate SQL from question + schema."""
    prompt = (
        f"Database contents:\n{schema}\n\n"
        f"{examples}\n"
        f"Q: {question}\n"
        f"SQL: SELECT"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    text += "SELECT"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False, temperature=1.0)
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True).strip()
    sql = "SELECT " + response
    # Stop at semicolon or blank line
    lines = []
    for line in sql.split("\n"):
        stripped = line.strip()
        if not stripped:
            break
        if ";" in stripped:
            lines.append(stripped.split(";")[0].strip())
            break
        lines.append(stripped)
    sql = " ".join(lines)
    return sql


def auto_sql_examples(
    table_name: str, columns: list[str], positional: bool = False,
    rows: list[tuple] | None = None,
    query_hints: dict[str, str] | None = None,
) -> str:
    """Generate few-shot SQL examples from column names and sample data.

    When rows are provided, uses actual values from the data so the model
    sees which values belong to which columns (e.g. 'red' → color, 'pen' → type).
    Falls back to generic placeholders when no data is available.
    """
    lines = ["Example queries:"]
    text_cols = [c for c in columns if c != "position"]

    # Build sample values per text column from actual data
    samples: dict[str, list[str]] = {}
    if rows and text_cols:
        col_idx = {c: i for i, c in enumerate(columns)}
        for col in text_cols:
            idx = col_idx[col]
            seen = []
            for row in rows:
                val = str(row[idx])
                if val not in seen:
                    seen.append(val)
                if len(seen) >= 3:
                    break
            samples[col] = seen

    def _sv(col: str, n: int = 0) -> str:
        """Get sample value n for a column, or a placeholder."""
        vals = samples.get(col, [])
        if n < len(vals):
            return vals[n]
        return chr(ord('X') + n) if n < 3 else 'V'

    # Single-column TEXT patterns
    for col in text_cols:
        v = _sv(col, 0)
        lines.append(
            f"Q: How many rows have {col} equal to {v}?\n"
            f"SQL: SELECT COUNT(*) FROM {table_name} WHERE {col} = '{v}'"
        )

    # NOT IN pattern for first text column
    if text_cols:
        col = text_cols[0]
        v1, v2 = _sv(col, 0), _sv(col, 1)
        lines.append(
            f"Q: How many rows have {col} not in {v1} or {v2}?\n"
            f"SQL: SELECT COUNT(*) FROM {table_name} WHERE {col} NOT IN ('{v1}', '{v2}')"
        )

    # Removal pattern: exclude by col1, count by col2
    if len(text_cols) >= 2:
        c1, c2 = text_cols[0], text_cols[1]
        v1 = _sv(c1, 0)
        v2 = _sv(c2, 0)
        lines.append(
            f"Q: If I remove all the {v1} items, how many {v2} remain?\n"
            f"SQL: SELECT COUNT(*) FROM {table_name} WHERE {c1} != '{v1}' AND {c2} = '{v2}'"
        )

    # Cross-column: SELECT col1 WHERE col2 = ?
    if len(text_cols) >= 2:
        c1, c2 = text_cols[0], text_cols[1]
        v2 = _sv(c2, 0)
        v1 = _sv(c1, 0)
        lines.append(
            f"Q: What is the {c1} of the {v2}?\n"
            f"SQL: SELECT {c1} FROM {table_name} WHERE {c2} = '{v2}'"
        )
        lines.append(
            f"Q: Is the {v2} {v1}?\n"
            f"SQL: SELECT CASE WHEN {c1} = '{v1}' THEN 'yes' ELSE 'no' END "
            f"FROM {table_name} WHERE {c2} = '{v2}'"
        )

    # Positional patterns
    if positional and text_cols:
        c_last = text_cols[-1]
        v_last = _sv(c_last, 0)
        lines.append(
            f"Q: What is at the first position?\n"
            f"SQL: SELECT * FROM {table_name} WHERE position = "
            f"(SELECT MIN(position) FROM {table_name})"
        )
        lines.append(
            f"Q: What is directly to the left of the {v_last}?\n"
            f"SQL: SELECT * FROM {table_name} WHERE position = "
            f"(SELECT position - 1 FROM {table_name} WHERE {c_last} = '{v_last}')"
        )
        lines.append(
            f"Q: What is directly to the right of the {v_last}?\n"
            f"SQL: SELECT * FROM {table_name} WHERE position = "
            f"(SELECT position + 1 FROM {table_name} WHERE {c_last} = '{v_last}')"
        )
        lines.append(
            f"Q: What is furthest from the {v_last}?\n"
            f"SQL: SELECT * FROM {table_name} ORDER BY ABS(position - "
            f"(SELECT position FROM {table_name} WHERE {c_last} = '{v_last}')) DESC LIMIT 1"
        )

    # ── Intent-specific examples from probe hints ──
    if query_hints:
        intent = query_hints.get("intent", "")
        where_type = query_hints.get("where_type", "")

        if intent == "MAX_MIN" and text_cols:
            c = text_cols[0]
            # Need a numeric column for MAX/MIN; use any available
            num_col = next(
                (col for col in columns if col not in text_cols and col != "position"),
                text_cols[-1],
            )
            lines.append(
                f"Q: Which {c} has the highest {num_col}?\n"
                f"SQL: SELECT {c} FROM {table_name} WHERE {num_col} = "
                f"(SELECT MAX({num_col}) FROM {table_name})"
            )
            lines.append(
                f"Q: Which {c} has the lowest {num_col}?\n"
                f"SQL: SELECT {c} FROM {table_name} WHERE {num_col} = "
                f"(SELECT MIN({num_col}) FROM {table_name})"
            )

        if intent == "ORDER_BY" and text_cols:
            c = text_cols[0]
            lines.append(
                f"Q: What is the last {c}?\n"
                f"SQL: SELECT {c} FROM {table_name} ORDER BY ROWID DESC LIMIT 1"
            )
            lines.append(
                f"Q: What is the first {c} alphabetically?\n"
                f"SQL: SELECT {c} FROM {table_name} ORDER BY {c} ASC LIMIT 1"
            )

        if intent == "SUM" and text_cols:
            num_col = next(
                (col for col in columns if col not in text_cols and col != "position"),
                text_cols[-1],
            )
            lines.append(
                f"Q: What is the total {num_col}?\n"
                f"SQL: SELECT SUM({num_col}) FROM {table_name}"
            )

        if intent == "AVG" and text_cols:
            num_col = next(
                (col for col in columns if col not in text_cols and col != "position"),
                text_cols[-1],
            )
            lines.append(
                f"Q: What is the average {num_col}?\n"
                f"SQL: SELECT AVG({num_col}) FROM {table_name}"
            )

        if intent == "COMPARISON" and len(text_cols) >= 2:
            c1, c2 = text_cols[0], text_cols[1]
            num_col = next(
                (col for col in columns if col not in text_cols and col != "position"),
                text_cols[-1],
            )
            v = _sv(c1, 0)
            lines.append(
                f"Q: Which {c1} has a higher {num_col} than {v}?\n"
                f"SQL: SELECT {c1} FROM {table_name} WHERE {num_col} > "
                f"(SELECT {num_col} FROM {table_name} WHERE {c1} = '{v}')"
            )

        if where_type == "inequality" and text_cols:
            num_col = next(
                (col for col in columns if col not in text_cols and col != "position"),
                text_cols[-1],
            )
            c = text_cols[0]
            lines.append(
                f"Q: Which {c} has {num_col} greater than 5?\n"
                f"SQL: SELECT {c} FROM {table_name} WHERE {num_col} > 5"
            )

        if where_type == "compound" and len(text_cols) >= 2:
            c1, c2 = text_cols[0], text_cols[1]
            v1, v2 = _sv(c1, 0), _sv(c2, 0)
            lines.append(
                f"Q: Which rows have {c1} equal to {v1} and {c2} equal to {v2}?\n"
                f"SQL: SELECT * FROM {table_name} WHERE {c1} = '{v1}' AND {c2} = '{v2}'"
            )

    return "\n\n".join(lines) + "\n"




def repair_sql(raw_sql: str, conn: sqlite3.Connection, tables: dict,
               question: str | None = None) -> str:
    """Fix common SQL generation errors using schema knowledge."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    valid_columns = set()
    for tname_row in cursor.fetchall():
        try:
            cursor2 = conn.execute(f"PRAGMA table_info({tname_row[0]})")
            valid_columns.update(row[1] for row in cursor2.fetchall())
        except Exception:
            pass

    repaired = raw_sql
    repaired = re.sub(
        r"\bWHERE\s+ROWID\s*=\s*last\b",
        "ORDER BY ROWID DESC LIMIT 1",
        repaired, flags=re.IGNORECASE)
    name_dot = re.search(r"\b([A-Z][a-z]+)\.([\w]+)", repaired)
    if name_dot:
        entity_name = name_dot.group(1)
        col_name = name_dot.group(2)
        if col_name in valid_columns:
            for tname in tables:
                cols = tables[tname][0]
                if "name" in cols and col_name in cols:
                    subq = (
                        f"(SELECT {col_name} FROM {tname} "
                        f"WHERE name = '{entity_name}')"
                    )
                    repaired = repaired.replace(name_dot.group(0), subq)
                    break

    # Fix: spurious WHERE name='X' on aggregate queries (MIN/MAX/SUM/AVG).
    # Pattern: SELECT AGG(col) FROM table WHERE name = 'X' → remove WHERE.
    # An aggregate with a single-row WHERE is always a bug — the aggregate
    # is meaningless when constrained to one row.
    agg_where = re.match(
        r"(SELECT\s+(?:MIN|MAX|SUM|AVG)\s*\([^)]+\)\s+FROM\s+\w+)"
        r"\s+WHERE\s+\w+\s*=\s*'[^']*'",
        repaired, flags=re.IGNORECASE)
    if agg_where:
        repaired = agg_where.group(1)

    # ── Single-pass repairs (operate on flat SQL, before UNION restructure) ──

    # Fix: inverted comparison operators for "younger/shorter than X and older/taller than Y"
    if question:
        q_lower = question.lower()
        inv_m = re.match(
            r".*(?:younger|shorter|lighter|smaller)\s+than\s+(\w+)"
            r"\s+and\s+(?:older|taller|heavier|larger)\s+than\s+(\w+)",
            q_lower)
        if inv_m:
            name_a, name_b = inv_m.group(1), inv_m.group(2)
            cmp_fix = re.match(
                r"(SELECT\s+\w+\s+FROM\s+\w+\s+WHERE\s+\w+\s*)"
                r"(>)(\s*\(SELECT\s+\w+\s+FROM\s+\w+\s+WHERE\s+\w+\s*=\s*'"
                + re.escape(name_a.capitalize()) + r"'\))"
                r"(\s+AND\s+\w+\s*)"
                r"(<)(\s*\(SELECT\s+\w+\s+FROM\s+\w+\s+WHERE\s+\w+\s*=\s*'"
                + re.escape(name_b.capitalize()) + r"'\))",
                repaired, flags=re.IGNORECASE)
            if cmp_fix:
                repaired = (
                    cmp_fix.group(1) + "<" + cmp_fix.group(3)
                    + cmp_fix.group(4) + ">" + cmp_fix.group(6)
                )

    # Fix: "next to last" with ORDER BY ASC → should be DESC LIMIT 1 OFFSET 1
    if question and "next to last" in question.lower():
        ntl_m = re.match(
            r"(SELECT\s+\w+\s+FROM\s+\w+\s+ORDER\s+BY\s+\w+)\s+ASC\s+"
            r"LIMIT\s+\d+\s+OFFSET\s+\d+",
            repaired, flags=re.IGNORECASE)
        if ntl_m:
            repaired = f"{ntl_m.group(1)} DESC LIMIT 1 OFFSET 1"

    # Fix: ordinal ROWID off-by-one ("second" → ROWID=2, not 1)
    if question:
        _ORDINAL_MAP = {
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        }
        q_lower = question.lower()
        for word, expected_pos in _ORDINAL_MAP.items():
            if word in q_lower:
                rowid_fix = re.match(
                    r"(SELECT\s+\w+\s+FROM\s+\w+\s+WHERE\s+ROWID\s*=\s*)(\d+)",
                    repaired, flags=re.IGNORECASE)
                if rowid_fix:
                    actual_pos = int(rowid_fix.group(2))
                    if actual_pos != expected_pos:
                        repaired = rowid_fix.group(1) + str(expected_pos)
                break

    # Fix: superlative questions with hardcoded values instead of MAX/MIN.
    # Skip when question contains ordinals ("second youngest" ≠ "youngest").
    if question:
        q_lower = question.lower()
        _ORDINALS = {"first", "second", "third", "fourth", "fifth",
                     "sixth", "seventh", "eighth", "ninth", "tenth",
                     "next to last"}
        has_ordinal = any(o in q_lower for o in _ORDINALS)
        _SUPERLATIVE_MAX = [
            "oldest", "tallest", "heaviest", "largest", "biggest",
            "highest", "longest", "most", "taller than the other",
            "older than the other", "heavier than the other",
        ]
        _SUPERLATIVE_MIN = [
            "youngest", "shortest", "lightest", "smallest",
            "lowest", "least",
        ]
        sup_dir = None
        if not has_ordinal:
            if any(s in q_lower for s in _SUPERLATIVE_MAX):
                sup_dir = "MAX"
            elif any(s in q_lower for s in _SUPERLATIVE_MIN):
                sup_dir = "MIN"

        if sup_dir:
            # Pattern A: col > (subquery for name='X') → col = (SELECT MAX/MIN)
            sup_compare = re.match(
                r"(SELECT\s+)(\w+(?:\s*,\s*\w+)*)(\s+FROM\s+)(\w+)"
                r"\s+WHERE\s+(\w+)\s*>\s*\(SELECT\s+\w+\s+FROM\s+\w+"
                r"\s+WHERE\s+\w+\s*=\s*'[^']+'\)",
                repaired, flags=re.IGNORECASE)
            if sup_compare:
                sel_col = sup_compare.group(2)
                tbl = sup_compare.group(4)
                cmp_col = sup_compare.group(5)
                repaired = (
                    f"SELECT {sel_col} FROM {tbl} "
                    f"WHERE {cmp_col} = (SELECT {sup_dir}({cmp_col}) FROM {tbl})"
                )
            else:
                # Pattern B: col = 'literal' (no nested subquery) → col = (SELECT MAX/MIN)
                if "(SELECT" not in repaired.upper():
                    sup_literal = re.match(
                        r"(SELECT\s+\w+(?:\s*,\s*\w+)*\s+FROM\s+)(\w+)"
                        r"(\s+WHERE\s+)(\w+)\s*=\s*'([^']+)'",
                        repaired, flags=re.IGNORECASE)
                    if sup_literal:
                        tbl = sup_literal.group(2)
                        col = sup_literal.group(4)
                        repaired = (
                            f"{sup_literal.group(1)}{tbl}{sup_literal.group(3)}"
                            f"{col} = (SELECT {sup_dir}({col}) FROM {tbl})"
                        )

    # ── Multi-table UNION (runs AFTER single-pass repairs on flat SQL) ──

    if question and len(tables) > 1:
        q_lower = question.lower()
        generic_terms = ["animal", "animals", "creature", "creatures"]
        table_names_mentioned = sum(
            1 for tname in tables if tname.rstrip("s") in q_lower or tname in q_lower
        )
        needs_union = (
            any(term in q_lower for term in generic_terms)
            or table_names_mentioned >= 2
        )
        if needs_union:
            # Pattern 1: SELECT AGG(col_or_*) FROM table → UNION the inner column
            agg_single = re.match(
                r"(SELECT\s+)(SUM|MIN|MAX|AVG|COUNT)"
                r"(\s*\(\s*)(\w+|\*)(\s*\)\s+FROM\s+)(\w+)(.*)",
                repaired, flags=re.IGNORECASE)
            if agg_single:
                agg_col = agg_single.group(4)
                queried_table = agg_single.group(6)
                if queried_table in tables:
                    if agg_col == "*" or all(agg_col in t[0] for t in tables.values()):
                        union = " UNION ALL ".join(
                            f"SELECT {agg_col} FROM {t}" for t in tables)
                        repaired = (
                            f"{agg_single.group(1)}{agg_single.group(2)}"
                            f"{agg_single.group(3)}{agg_col}"
                            f"{agg_single.group(5)}({union})"
                            f"{agg_single.group(7)}"
                        )

            # Pattern 2: SELECT col FROM table [ORDER BY/WHERE ...] → UNION
            if not agg_single:
                single_table = re.match(
                    r"(SELECT\s+)(\w+)(\s+FROM\s+)(\w+)(.*)",
                    repaired, flags=re.IGNORECASE)
                if single_table:
                    col = single_table.group(2)
                    queried_table = single_table.group(4)
                    rest = single_table.group(5)
                    if (queried_table in tables
                            and all(col in t[0] for t in tables.values())
                            and "rowid" not in repaired.lower()):
                        # If rest has WHERE on other columns, use SELECT *
                        inner_col = col
                        if re.search(r"\bWHERE\b", rest, re.IGNORECASE):
                            inner_col = "*"
                        union = " UNION ALL ".join(
                            f"SELECT {inner_col} FROM {t}" for t in tables)
                        repaired = (
                            f"{single_table.group(1)}{col}"
                            f"{single_table.group(3)}({union}){rest}"
                        )

            # Pattern 3: ROWID-based single-table query → last table's last row
            if "rowid" in repaired.lower() and agg_single is None:
                rowid_m = re.match(
                    r"(SELECT\s+)(\w+)(\s+FROM\s+)(\w+)"
                    r"\s+WHERE\s+ROWID\s*=\s*\(SELECT\s+MAX\(ROWID\)\s+FROM\s+\w+\)",
                    repaired, flags=re.IGNORECASE)
                if rowid_m:
                    col = rowid_m.group(2)
                    if all(col in t[0] for t in tables.values()):
                        last_table = list(tables.keys())[-1]
                        repaired = (
                            f"SELECT {col} FROM {last_table} "
                            f"ORDER BY ROWID DESC LIMIT 1"
                        )

    return repaired


def _try_execute(conn: sqlite3.Connection, sql: str):
    """Try to execute SQL, return (result_val, error_str)."""
    try:
        cursor = conn.execute(sql)
        result = cursor.fetchone()
        return (result[0] if result else None), None
    except Exception as e:
        return None, str(e)


def match_result_to_option(result, options: dict) -> str | None:
    """Match a SQL result to one of the BBH options.

    Handles: direct string match, numeric match, number-word match (2 → "two").
    """
    if result is None:
        return None
    result_str = str(result).strip()
    for letter, opt_val in options.items():
        opt_clean = opt_val.strip()
        # Direct match
        if result_str.lower() == opt_clean.lower():
            return f"({letter})"
        # Numeric match
        try:
            if float(result_str) == float(opt_clean):
                return f"({letter})"
        except (ValueError, TypeError):
            pass
        # Number word match (2 → "two")
        try:
            num = int(float(result_str))
            if _NUM_TO_WORD.get(num, "").lower() == opt_clean.lower():
                return f"({letter})"
        except (ValueError, TypeError):
            pass
    return None


# ════════════════════════════════════════════════════════════════════════
# Scene / question / options extraction (generic BBH-style)
# ════════════════════════════════════════════════════════════════════════

def extract_scene_text(text: str) -> str:
    """Extract the scene/data portion from a BBH-style prompt.

    Takes all text before the first '?' (the question), strips any
    'Options:' block. Returns the descriptive scene text.
    """
    # Take text before first question mark
    q_idx = text.find("?")
    if q_idx >= 0:
        scene = text[:q_idx]
    else:
        scene = text.split("Options:")[0] if "Options:" in text else text

    # Find the last sentence boundary before the question
    # (strip trailing question-sentence fragment)
    sentences = re.split(r'(?<=[.!])\s+', scene)
    # Drop fragments that look like question starts
    kept = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Stop if this looks like a question sentence
        if re.match(r'(?:How|What|Which|Is|Are|If|Do|Does|Where|Who)\b', s, re.IGNORECASE) and len(kept) > 0:
            break
        kept.append(s)

    result = " ".join(kept).strip()
    # Clean trailing punctuation
    result = result.rstrip(".")
    return result

def extract_question(text: str) -> str | None:
    """Extract the question sentence from BBH-style text.

    Finds the last sentence ending with '?' before 'Options:'.
    """
    q_text = text.split("Options:")[0]
    q_idx = q_text.rfind("?")
    if q_idx < 0:
        return None
    q_start = q_text.rfind(".", 0, q_idx)
    if q_start < 0:
        q_start = q_text.rfind("  ", 0, q_idx)
    raw = q_text[q_start + 1:q_idx + 1].strip()
    if "\n" in raw:
        for line in reversed(raw.split("\n")):
            line = line.strip()
            if line and "?" in line:
                return line
    return raw


def extract_options(text: str) -> dict:
    """Extract answer options from BBH-style text.

    Matches patterns like '(A) some text' with multi-word values.
    """
    options = {}
    for m in re.finditer(r"\(([A-R])\)\s+(.+?)(?=\s*\([A-R]\)|\s*$)", text):
        options[m.group(1)] = m.group(2).strip()
    return options


# ════════════════════════════════════════════════════════════════════════
# JSON parsing for model extraction
# ════════════════════════════════════════════════════════════════════════

def _meta_schema_solve(
    question: str, tables: dict, options: dict,
) -> tuple[str, str] | None:
    """Answer meta-schema questions from table structure alone.

    Handles:
    - "How many species are listed" → number of tables
    - "What is the number of the column with X" → column position (1-indexed)
    """
    q_lower = question.lower()

    # "How many species" → count of tables
    species_m = re.match(
        r"how many (?:species|types|kinds).+(?:listed|in the table)",
        q_lower)
    if species_m:
        n_tables = len(tables)
        answer = match_result_to_option(n_tables, options)
        if answer:
            return f"meta_schema: {n_tables} species (tables)", answer

    # "What is the number of the column with X" → column position
    col_num_m = re.match(
        r"what is the (?:number|position|index) of the column (?:with |for |of )?(?:the )?(\w+)",
        q_lower)
    if col_num_m:
        target_col = col_num_m.group(1).rstrip("s")  # "weights" → "weight"
        for _tname, (columns, _rows) in tables.items():
            for i, col in enumerate(columns, 1):
                if col.startswith(target_col) or target_col in col:
                    answer = match_result_to_option(i, options)
                    if answer:
                        return f"meta_schema: column '{col}' is #{i}", answer

    return None


def parse_markdown_table(text: str) -> tuple[list[str], list[tuple]] | None:
    """Parse a markdown table from model output.

    Finds header row, separator row (|---|), and data rows.
    Returns (columns, rows) or None.

    Tolerant of surrounding text, extra whitespace, and missing outer pipes.
    """
    lines = text.strip().splitlines()

    # Find the separator row (contains |---|)
    sep_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'\|?\s*-[-\s|]*-\s*\|?$', stripped) and '-' in stripped:
            sep_idx = i
            break

    if sep_idx is None or sep_idx == 0:
        return None

    # Header is the line before separator
    header_line = lines[sep_idx - 1].strip()
    # Split on pipe, strip whitespace, filter empty
    header_cells = [c.strip() for c in header_line.split('|')]
    columns = [c for c in header_cells if c]
    if not columns:
        return None

    # Data rows are everything after separator
    rows = []
    for line in lines[sep_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            break
        cells = [c.strip() for c in stripped.split('|')]
        cells = [c for c in cells if c != '']
        if not cells:
            break
        # Pad or truncate to match column count
        if len(cells) < len(columns):
            cells.extend([''] * (len(columns) - len(cells)))
        elif len(cells) > len(columns):
            cells = cells[:len(columns)]
        # Try to convert numeric strings to int
        typed = []
        for c in cells:
            try:
                typed.append(int(c))
            except ValueError:
                typed.append(c)
        rows.append(tuple(typed))

    if not rows:
        return None

    return columns, rows


# ════════════════════════════════════════════════════════════════════════
# Generic JSON extraction with count expansion (no SchemaSpec needed)
# ════════════════════════════════════════════════════════════════════════

_GENERIC_JSON_EXTRACTION_PROMPT = (
    'Extract every entity from the scene as a JSON array.\n'
    'Each object has "count", plus attribute columns. Use "count" for duplicates.\n\n'
    'Scene: On the floor, there are two red pens and one blue cup.\n'
    'JSON: [{{"count": 2, "color": "red", "type": "pen"}},'
    ' {{"count": 1, "color": "blue", "type": "cup"}}]\n\n'
    'Scene: On the desk, you see a set of items arranged in a row:'
    ' a gold plate, a silver ball, and a mauve jug.\n'
    'JSON: [{{"count": 1, "color": "gold", "type": "plate"}},'
    ' {{"count": 1, "color": "silver", "type": "ball"}},'
    ' {{"count": 1, "color": "mauve", "type": "jug"}}]\n\n'
    'Scene: {scene}\n'
    'JSON:'
)


def _model_extract_table(model, tokenizer, device, scene: str,
                         prompt_text: str | None = None,
                         positional: bool = False,
                         max_tokens: int = 300) -> dict | None:
    """Extract structured data from a scene via model-generated JSON.

    Generic extraction — no SchemaSpec needed. The model outputs a JSON
    array with "count" keys; objects are expanded into rows. Columns are
    inferred from the JSON keys.

    Returns {table_name: (columns, rows)} or None.
    """
    prompt = (prompt_text or _GENERIC_JSON_EXTRACTION_PROMPT).format(scene=scene)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, temperature=1.0)
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True).strip()

    objects = _parse_json_array(response)
    if objects is None:
        return None

    # Infer columns from object keys (excluding "count")
    all_keys: list[str] = []
    for obj in objects:
        for k in obj:
            if k != "count" and k not in all_keys:
                all_keys.append(k)
    if not all_keys:
        return None

    # Expand by count, optionally prepend position
    rows: list[tuple] = []
    pos = 1
    for obj in objects:
        count = obj.pop("count", 1)
        if isinstance(count, str):
            try:
                count = int(count)
            except ValueError:
                count = 1
        for _ in range(count):
            row_vals: list = []
            if positional:
                row_vals.append(pos)
                pos += 1
            for col in all_keys:
                row_vals.append(obj.get(col, ""))
            rows.append(tuple(row_vals))

    if not rows:
        return None

    columns = (["position"] + all_keys) if positional else list(all_keys)
    return {"data": (columns, rows)}


def _parse_json_array(text: str) -> list[dict] | None:
    """Extract a JSON array from model output, tolerant of surrounding text.

    Finds the first '[' and last ']' in text, parses the content between them.
    Returns a list of dicts or None on failure.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        result = json.loads(text[start:end + 1])
        if isinstance(result, list) and all(isinstance(x, dict) for x in result):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        return None


# ════════════════════════════════════════════════════════════════════════
# SQLTurnstyle base class
# ════════════════════════════════════════════════════════════════════════

class SQLTurnstyle(Turnstyle):
    """Base class for turnstyles that solve via text-to-SQL.

    Scene/data → SQLite table → LLM generates SQL → execute → match to option → logit bias.
    Regex fast path via _regex_solve(); SQL fallback via _sql_solve().

    Subclasses implement:
        parse_tables(text) -> {table_name: (columns, rows)} or None
        _regex_solve(prompt) -> (summary, answer_str) or None

    Alternatively, pass parse_tables_fn to the constructor, or set
    schema_spec (SchemaSpec) for model-based extraction.
    """

    schema_spec: SchemaSpec | None = None
    probe_label = "tabular"
    examples = [
        # penguins_in_a_table style
        "Here is a table that describes some penguins and attributes about them.\n|name|age|height (cm)|weight (kg)|bill length (mm)|flipper length (cm)|\n|Herman|10|47|29|39|18|\n|Gwen|15|45|28|38|19|\n|Nora|9|40|22|32|16|\nWhich penguin is the oldest?\nOptions:\n(A) Herman\n(B) Gwen\n(C) Nora",
        "Here is a table that describes some penguins and attributes about them.\n|name|age|height (cm)|weight (kg)|\n|Amara|7|42|24|\n|Floyd|14|50|31|\n|Kwame|11|46|27|\nWhich penguin weighs the most?\nOptions:\n(A) Amara\n(B) Floyd\n(C) Kwame",
        "Here is a table that describes some penguins and attributes about them.\n|name|age|height (cm)|bill length (mm)|\n|Olga|6|43|35|\n|Peter|13|48|40|\n|Amy|9|41|33|\nWhich penguin has the longest bill?\nOptions:\n(A) Olga\n(B) Peter\n(C) Amy",
        # reasoning_about_colored_objects style
        "On the nightstand, there is a red fidget spinner, a mauve dog leash, a blue teddy bear, and a burgundy cup. How many non-burgundy things are on the nightstand?\nOptions:\n(A) zero\n(B) one\n(C) two\n(D) three\n(E) four",
        "On the table, there is a purple paperclip, a red plate, a green mug, and a yellow cup. How many objects are on the table?\nOptions:\n(A) two\n(B) three\n(C) four\n(D) five",
        "On the desk, there is a blue pencil, a green notebook, and a red eraser. How many non-blue things are on the desk?\nOptions:\n(A) zero\n(B) one\n(C) two\n(D) three",
        "On the floor, there is a brown ball, a black shoe, a white sock, and a gray hat. How many items are not brown?\nOptions:\n(A) one\n(B) two\n(C) three\n(D) four",
        "On the shelf, there is a silver mirror, a gold frame, and a bronze clock. What color is the mirror?\nOptions:\n(A) silver\n(B) gold\n(C) bronze",
        # object_counting style
        "I have a flower, a cup, two pencils, and three books. How many objects do I have?\nOptions:\n(A) five\n(B) six\n(C) seven\n(D) eight",
        "There are four apples, two oranges, and one banana on the table. How many fruits are there?\nOptions:\n(A) five\n(B) six\n(C) seven\n(D) eight",
        "I have three dogs and two cats. How many pets do I have?\nOptions:\n(A) four\n(B) five\n(C) six",
        "On the table there are two cups, three plates, one fork, and four spoons. How many items are on the table?\nOptions:\n(A) eight\n(B) nine\n(C) ten\n(D) eleven",
        "I bought one apple, two bananas, three oranges, and four grapes. How many fruit items did I buy?\nOptions:\n(A) eight\n(B) nine\n(C) ten\n(D) eleven",
        "Here is a table about species of birds.\n|name|wingspan (cm)|weight (g)|\n|Robin|26|19|\n|Sparrow|21|30|\n|Finch|24|22|\nWhich bird has the widest wingspan?\nOptions:\n(A) Robin\n(B) Sparrow\n(C) Finch",
        "Here is a table about gemstones.\n|name|hardness|carat|price|\n|Diamond|10|2.1|5000|\n|Ruby|9|1.5|3000|\n|Emerald|7.5|2.0|2500|\nWhich gemstone is the hardest?\nOptions:\n(A) Diamond\n(B) Ruby\n(C) Emerald",
        "On the windowsill, there is a green plant, a red vase, a blue bowl, a yellow cup, and a white candle. How many objects are on the windowsill?\nOptions:\n(A) three\n(B) four\n(C) five\n(D) six",
        "There are five red cars and three blue cars in the parking lot. How many cars are there in total?\nOptions:\n(A) six\n(B) seven\n(C) eight\n(D) nine",
        "I have two tables, four chairs, one sofa, and three lamps. How many furniture items do I have?\nOptions:\n(A) eight\n(B) nine\n(C) ten\n(D) eleven",
        "Here is a table that describes some fish.\n|name|length (cm)|weight (g)|color|\n|Nemo|10|50|orange|\n|Dory|20|120|blue|\n|Gill|15|80|gray|\nWhich fish is the longest?\nOptions:\n(A) Nemo\n(B) Dory\n(C) Gill",
        "On the counter there is a purple bowl, a green mug, and an orange plate. How many non-purple items are on the counter?\nOptions:\n(A) zero\n(B) one\n(C) two\n(D) three",
        "I see one moon, eight planets, and countless stars. How many distinct celestial bodies did I list?\nOptions:\n(A) seven\n(B) eight\n(C) nine\n(D) ten",
        "Here is a table about athletes.\n|name|age|sport|\n|Alice|25|swimming|\n|Bob|30|cycling|\n|Carol|27|running|\nWhich athlete is oldest?\nOptions:\n(A) Alice\n(B) Bob\n(C) Carol",
        "There are two apples, zero oranges, and four bananas. How many fruits are there in total?\nOptions:\n(A) four\n(B) five\n(C) six\n(D) seven",
        "On the desk there are three pens and two pencils. How many writing instruments are on the desk?\nOptions:\n(A) three\n(B) four\n(C) five\n(D) six",
        "Here is a table about coffee shops.\n|name|rating|price|\n|Starbucks|4.2|high|\n|BlueBottle|4.5|high|\n|Dunkin|3.8|low|\nWhich shop has the highest rating?\nOptions:\n(A) Starbucks\n(B) BlueBottle\n(C) Dunkin",
        "On the floor, I see a red ball, two green cubes, and a blue pyramid. How many objects are there in total?\nOptions:\n(A) three\n(B) four\n(C) five",
        "I have five coins, three bills, and two checks. How many financial instruments do I have?\nOptions:\n(A) eight\n(B) nine\n(C) ten\n(D) eleven",
        "Here is a table that lists some cities and their populations.\n|city|population|\n|Alpha|50000|\n|Beta|120000|\n|Gamma|75000|\nWhich city has the largest population?\nOptions:\n(A) Alpha\n(B) Beta\n(C) Gamma",
    ]

    def __init__(self, model, tokenizer, device, bias_strength=15.0,
                 schema_spec: SchemaSpec | None = None,
                 parse_tables_fn: Callable[[str], dict | None] | None = None,
                 probe_label: str | None = None,
                 intent_probe=None,
                 intent_probe_layer: int = 4,
                 logit_poll_fallback: bool = False):
        super().__init__(model, tokenizer, device, bias_strength)
        if schema_spec is not None:
            self.schema_spec = schema_spec
        self._parse_tables_fn = parse_tables_fn
        if probe_label is not None:
            self.probe_label = probe_label
        self.intent_probe = intent_probe
        self.intent_probe_layer = intent_probe_layer
        self.logit_poll_fallback = logit_poll_fallback
        self._logit_prior: dict[str, float] | None = None

    def parse_tables(self, text: str) -> dict | None:
        """Parse structured data into {table_name: (columns, rows)}.

        Delegates to parse_tables_fn if set, otherwise returns None.
        Subclasses can override via normal MRO.
        """
        if self._parse_tables_fn is not None:
            return self._parse_tables_fn(text)
        return None

    def _regex_solve(self, prompt: str) -> tuple[str, str] | None:
        """Override: regex fast path. Returns (summary, answer_str) or None."""
        return None

    def parse(self, prompt: str):
        """Regex fast path — delegates to _regex_solve()."""
        return self._regex_solve(prompt)

    def _model_extract(self, prompt: str) -> dict | None:
        """Extract structured data from prompt via model generation.

        If schema_spec is set, uses it for JSON extraction with count expansion.
        Otherwise, uses generic JSON extraction with automatic column inference.

        Returns {table_name: (columns, rows)} or None on failure.
        """
        scene = extract_scene_text(prompt)
        if not scene:
            return None

        spec = self.schema_spec
        if spec is None:
            # Generic JSON extraction with count expansion
            positional = "in a row" in prompt
            return _model_extract_table(
                self.model, self.tokenizer, self.device, scene,
                positional=positional)

        # SchemaSpec-based JSON extraction
        extraction_prompt = spec.extraction_prompt.format(scene=scene)
        messages = [{"role": "user", "content": extraction_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=spec.max_extract_tokens,
                do_sample=False, temperature=1.0)
        response = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()

        # Parse JSON array from response (tolerant of surrounding text)
        objects = _parse_json_array(response)
        if objects is None:
            return None

        # Determine if positional
        positional = (spec.positional_detector is not None
                      and spec.positional_detector(prompt))

        # Expand objects by count_key and collect rows
        rows = []
        pos = 1
        for obj in objects:
            count = obj.pop(spec.count_key, 1)
            if isinstance(count, str):
                try:
                    count = int(count)
                except ValueError:
                    count = 1
            for _ in range(count):
                row_vals = []
                if positional:
                    row_vals.append(pos)
                    pos += 1
                for col in spec.columns:
                    row_vals.append(obj.get(col, ""))
                rows.append(tuple(row_vals))

        if not rows:
            return None

        columns = (["position"] + spec.columns) if positional else list(spec.columns)
        return {spec.table_name: (columns, rows)}

    def _predict_query_hints(self, prompt: str) -> dict[str, str] | None:
        """Use intent_probe to classify question type from hidden states.

        Runs a forward pass, finds the question token (last '?' before
        'Options:'), extracts the hidden state at intent_probe_layer, and
        returns probe predictions as {dimension: label}.
        """
        if self.intent_probe is None:
            return None

        # Tokenize and forward pass
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]

        with torch.no_grad():
            outputs = self.model(
                **inputs, output_hidden_states=True)

        # Find question token: last '?' before 'Options:'
        decoded_tokens = [
            self.tokenizer.decode([tid]) for tid in input_ids
        ]
        # Find 'Options' token position as upper bound
        options_pos = len(decoded_tokens)
        for i, tok in enumerate(decoded_tokens):
            if "Options" in tok:
                options_pos = i
                break

        # Find last '?' before Options
        q_pos = None
        for i in range(options_pos - 1, -1, -1):
            if "?" in decoded_tokens[i]:
                q_pos = i
                break

        if q_pos is None:
            q_pos = -1  # fall back to last token

        # Extract hidden state at probe layer (cpu + fp32 for probe weights)
        hidden = outputs.hidden_states[self.intent_probe_layer + 1]  # +1: index 0 is embeddings
        h = hidden[0, q_pos].cpu().float()  # (hidden_dim,)

        # Predict: {dim_name: (label, confidence)}
        predictions = self.intent_probe.predict(h)

        # Selective conditioning: only inject hints for categories where
        # they empirically help. ORDER_BY/MAX_MIN/SUM hints cause pattern
        # copying that overrides correct contextual reasoning.
        _HELPFUL_INTENTS = {"COMPARISON"}
        _HELPFUL_WHERE = {"inequality"}

        hints = {}
        if "intent" in predictions:
            label, _conf = predictions["intent"]
            if label in _HELPFUL_INTENTS:
                hints["intent"] = label
        if "where_type" in predictions:
            label, _conf = predictions["where_type"]
            if label in _HELPFUL_WHERE:
                hints["where_type"] = label

        return hints if hints else None

    def _compute_logit_prior(self) -> dict[str, float]:
        """Compute prior logits for option letters from a neutral prompt.

        Cached after first call. Used to calibrate logit polling by removing
        positional bias (e.g. model always favoring 'A' over 'B').
        """
        if self._logit_prior is not None:
            return self._logit_prior
        neutral = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Please select one option."}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(neutral, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1]
        prior = {}
        for letter in "ABCDEFGHIJKLMNOPQR":
            token_ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if token_ids:
                prior[letter] = logits[token_ids[0]].item()
        self._logit_prior = prior
        return prior

    def _knowledge_poll(self, question: str, options: dict) -> tuple[str, str] | None:
        """Decompose into per-option yes/no knowledge queries.

        Runs two prompt strategies and picks the more discriminative one:
        1. General: "{question}\\nIs the answer '{value}'?"
        2. Predicate: "Is '{value}' {predicate}?" (when extractable)

        The general prompt preserves question context; the predicate prompt
        isolates the knowledge query. Picking the higher-gap strategy gets
        the best of both.

        Returns (summary, answer_str) if one option has a clear positive
        margin, or None if no option stands out.
        """
        yes_ids = self.tokenizer.encode("yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("no", add_special_tokens=False)
        if not yes_ids or not no_ids:
            return None
        yes_id, no_id = yes_ids[0], no_ids[0]

        def _score(prompt_text: str) -> float:
            messages = [{"role": "user", "content": prompt_text}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]
            return logits[yes_id].item() - logits[no_id].item()

        # Strategy 1: general prompt (always available)
        gen_margins: dict[str, float] = {}
        for letter, value in options.items():
            q = f"{question}\nIs the answer '{value}'? Answer yes or no."
            gen_margins[letter] = _score(q)

        # Strategy 2: predicate prompt (when extractable)
        predicate = None
        m = re.match(
            r"Which\s+\w+\s+(?:has|have)\s+(.+?)\??\s*$",
            question, re.IGNORECASE)
        if m:
            predicate = m.group(1)
        else:
            m = re.match(
                r"Which\s+\w+\s+(?:is|are)\s+(.+?)\??\s*$",
                question, re.IGNORECASE)
            if m:
                predicate = m.group(1)

        pred_margins: dict[str, float] | None = None
        if predicate:
            pred_margins = {}
            for letter, value in options.items():
                q = f"Is '{value}' {predicate}? Answer yes or no."
                pred_margins[letter] = _score(q)

        def _pick(margins: dict[str, float]):
            s = sorted(margins.items(), key=lambda x: x[1], reverse=True)
            best_l, best_m = s[0]
            second_m = s[1][1] if len(s) > 1 else float("-inf")
            return best_l, best_m, best_m - second_m

        gen_letter, gen_margin, gen_gap = _pick(gen_margins)

        # Pick the strategy with the higher gap
        if pred_margins:
            pred_letter, pred_margin, pred_gap = _pick(pred_margins)
            if pred_gap > gen_gap:
                best_letter, best_margin, gap = pred_letter, pred_margin, pred_gap
                src = "pred"
            else:
                best_letter, best_margin, gap = gen_letter, gen_margin, gen_gap
                src = "gen"
        else:
            best_letter, best_margin, gap = gen_letter, gen_margin, gen_gap
            src = "gen"

        # Require: best option has positive yes-no margin AND clear gap
        if best_margin <= 0 or gap < 0.5:
            return None

        answer = f"({best_letter})"
        return (
            f"knowledge_poll[{src}]: {best_letter} "
            f"(margin={best_margin:.2f}, gap={gap:.2f})",
            answer,
        )

    def _logit_poll(self, prompt: str, options: dict) -> tuple[str, str] | None:
        """Score each option by prior-corrected logit, return best match.

        For each option letter, measures the model's logit for that letter
        as the next token, subtracts the prior (neutral-prompt) logit to
        remove positional bias. Returns (summary, answer_str) or None.
        """
        prior = self._compute_logit_prior()
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1]

        best_letter = None
        best_score = float("-inf")
        for letter in options:
            token_ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if token_ids:
                score = logits[token_ids[0]].item() - prior.get(letter, 0.0)
                if score > best_score:
                    best_score = score
                    best_letter = letter

        if best_letter is None:
            return None
        answer = f"({best_letter})"
        return f"logit_poll: {best_letter} (score={best_score:.2f})", answer

    def _sql_solve(self, prompt: str, diag: dict | None = None) -> tuple[str, str] | None:
        """Text-to-SQL fallback.

        Tries parse_tables() first (deterministic), then _model_extract()
        (SchemaSpec or generic markdown). Generates SQL via the model.
        If intent_probe is set, predicts query hints to condition few-shot examples.

        Works with or without multiple-choice options. When options are present,
        matches the SQL result to an option letter. When absent (free-answer tasks
        like object_counting), returns the raw SQL result string.

        When diag is provided, populates it with intermediate state for debugging.
        """
        tables = self.parse_tables(prompt)

        # Model-based extraction fallback (SchemaSpec or generic markdown)
        if tables is None:
            tables = self._model_extract(prompt)

        if diag is not None:
            diag["tables_parsed"] = tables is not None
            diag["table_names"] = list(tables.keys()) if tables else []

        if not tables:
            return None

        options = extract_options(prompt)

        question = extract_question(prompt)
        if diag is not None:
            diag["question"] = question
            diag["options"] = options

        if not question:
            if diag is not None:
                diag["error"] = "no_question"
            return None

        # Meta-schema questions: can be answered from table structure alone
        if options:
            meta_result = _meta_schema_solve(question, tables, options)
            if meta_result is not None:
                if diag is not None:
                    diag["meta_schema"] = True
                return meta_result

        # Predict query hints from hidden states (if probe available)
        query_hints = self._predict_query_hints(prompt)
        if diag is not None:
            diag["query_hints"] = query_hints

        conn = load_into_sqlite(tables)
        tbl_name = next(iter(tables))
        tbl_columns, tbl_rows = tables[tbl_name]

        schema = get_schema_description(conn)
        if self.schema_spec is not None:
            spec = self.schema_spec
            positional = (spec.positional_detector is not None
                          and spec.positional_detector(prompt))
            examples = auto_sql_examples(
                spec.table_name, spec.columns, positional,
                query_hints=query_hints)
        else:
            positional = "position" in tbl_columns
            examples = auto_sql_examples(
                tbl_name, tbl_columns, positional, rows=tbl_rows,
                query_hints=query_hints)

        raw_sql = generate_sql(
            self.model, self.tokenizer, self.device,
            schema, question, examples)
        sql = repair_sql(raw_sql, conn, tables, question=question)

        if diag is not None:
            diag["raw_sql"] = raw_sql
            diag["repaired_sql"] = sql if sql != raw_sql else None

        result, err = _try_execute(conn, sql)
        conn.close()

        if diag is not None:
            diag["sql_result"] = result
            diag["sql_error"] = err

        if err:
            return None

        # With options: match result to option letter
        if options:
            answer = match_result_to_option(result, options)
            if diag is not None:
                diag["matched_option"] = answer
            if answer is None:
                return None
            return f"SQL: {sql} -> {result}", answer

        # Without options (free-answer): return raw result
        answer = str(result).strip() if result is not None else None
        if diag is not None:
            diag["raw_answer"] = answer
        if answer is None:
            return None
        return f"SQL: {sql} -> {result}", answer

    def generate(self, prompt: str, max_new_tokens: int = 50):
        """Regex → SQL → logit poll → free generate."""
        parsed = self.parse(prompt)

        # SQL fallback if regex didn't match
        if parsed is None:
            parsed = self._sql_solve(prompt)

        # Extraction fallback (from base class)
        if parsed is None and self.extraction_spec is not None:
            from turnstyle.extract import extract
            result = extract(prompt, self, self.extraction_spec)
            if result is not None and result.parsed is not None:
                parsed = result.parsed

        # Knowledge decomposition: per-option yes/no queries (world knowledge)
        if parsed is None and self.logit_poll_fallback:
            question = extract_question(prompt)
            options = extract_options(prompt)
            if options and question:
                parsed = self._knowledge_poll(question, options)

        # Calibrated logit poll fallback when knowledge poll isn't confident
        if parsed is None and self.logit_poll_fallback:
            options = extract_options(prompt)
            if options:
                parsed = self._logit_poll(prompt, options)

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        if parsed is None:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip()
            return text, None

        processor = self.make_processor(parsed, max_new_tokens)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                logits_processor=[processor],
            )

        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()

        return text, processor.proof

    def make_processor(self, parsed, max_new_tokens: int):
        summary, answer_str = parsed
        answer_ids = self.tokenizer.encode(answer_str, add_special_tokens=False)
        return SequenceLogitsProcessor(
            self.tokenizer, answer_ids, expression=summary,
            answer_str=answer_str, bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens,
        )
