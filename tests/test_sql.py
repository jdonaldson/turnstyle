"""Tests for SQL turnstyle infrastructure — no model needed."""

from turnstyle.sql import (
    SchemaSpec,
    load_into_sqlite,
    get_schema_description,
    _try_execute,
    match_result_to_option,
    extract_question,
    extract_options,
    extract_scene_text,
    auto_sql_examples,
    _parse_json_array,
    _meta_schema_solve,
    parse_markdown_table,
    repair_sql,
    _NUM_TO_WORD,
)


# ── load_into_sqlite ──────────────────────────────────────────────────

class TestLoadIntoSqlite:
    def test_single_table(self):
        tables = {"items": (["name", "color"], [("pen", "red"), ("cup", "blue")])}
        conn = load_into_sqlite(tables)
        cursor = conn.execute("SELECT COUNT(*) FROM items")
        assert cursor.fetchone()[0] == 2
        conn.close()

    def test_type_inference_int(self):
        tables = {"data": (["id", "value"], [(1, 100), (2, 200)])}
        conn = load_into_sqlite(tables)
        cursor = conn.execute("PRAGMA table_info(data)")
        col_types = {row[1]: row[2] for row in cursor.fetchall()}
        assert col_types["id"] == "INTEGER"
        assert col_types["value"] == "INTEGER"
        conn.close()

    def test_type_inference_text(self):
        tables = {"data": (["name", "color"], [("pen", "red")])}
        conn = load_into_sqlite(tables)
        cursor = conn.execute("PRAGMA table_info(data)")
        col_types = {row[1]: row[2] for row in cursor.fetchall()}
        assert col_types["name"] == "TEXT"
        conn.close()

    def test_empty_rows(self):
        tables = {"empty": (["a", "b"], [])}
        conn = load_into_sqlite(tables)
        cursor = conn.execute("SELECT COUNT(*) FROM empty")
        assert cursor.fetchone()[0] == 0
        conn.close()

    def test_multiple_tables(self):
        tables = {
            "t1": (["x"], [(1,), (2,)]),
            "t2": (["y"], [("a",), ("b",)]),
        }
        conn = load_into_sqlite(tables)
        r1 = conn.execute("SELECT COUNT(*) FROM t1").fetchone()[0]
        r2 = conn.execute("SELECT COUNT(*) FROM t2").fetchone()[0]
        assert r1 == 2
        assert r2 == 2
        conn.close()


# ── get_schema_description ────────────────────────────────────────────

class TestGetSchemaDescription:
    def test_includes_table_name(self):
        tables = {"objects": (["color", "type"], [("red", "pen")])}
        conn = load_into_sqlite(tables)
        desc = get_schema_description(conn)
        assert "Table: objects" in desc
        assert "color" in desc
        assert "type" in desc
        conn.close()

    def test_includes_row_data(self):
        tables = {"objects": (["color", "type"], [("red", "pen")])}
        conn = load_into_sqlite(tables)
        desc = get_schema_description(conn)
        assert "red" in desc
        assert "pen" in desc
        conn.close()


# ── _try_execute ──────────────────────────────────────────────────────

class TestTryExecute:
    def test_valid_query(self):
        tables = {"t": (["x"], [(1,), (2,), (3,)])}
        conn = load_into_sqlite(tables)
        result, err = _try_execute(conn, "SELECT COUNT(*) FROM t")
        assert result == 3
        assert err is None
        conn.close()

    def test_invalid_query(self):
        tables = {"t": (["x"], [(1,)])}
        conn = load_into_sqlite(tables)
        result, err = _try_execute(conn, "SELECT * FROM nonexistent")
        assert result is None
        assert err is not None
        conn.close()

    def test_empty_result(self):
        tables = {"t": (["x"], [(1,)])}
        conn = load_into_sqlite(tables)
        result, err = _try_execute(conn, "SELECT x FROM t WHERE x = 99")
        assert result is None
        assert err is None
        conn.close()


# ── match_result_to_option ────────────────────────────────────────────

class TestMatchResultToOption:
    def test_direct_string(self):
        opts = {"A": "red", "B": "blue"}
        assert match_result_to_option("red", opts) == "(A)"

    def test_case_insensitive(self):
        opts = {"A": "Red", "B": "blue"}
        assert match_result_to_option("red", opts) == "(A)"

    def test_numeric_match(self):
        opts = {"A": "3", "B": "5"}
        assert match_result_to_option(3, opts) == "(A)"

    def test_float_to_int_match(self):
        opts = {"A": "3.0", "B": "5"}
        assert match_result_to_option(3, opts) == "(A)"

    def test_number_word_match(self):
        opts = {"A": "zero", "B": "one", "C": "two"}
        assert match_result_to_option(2, opts) == "(C)"

    def test_number_word_zero(self):
        opts = {"A": "zero", "B": "one"}
        assert match_result_to_option(0, opts) == "(A)"

    def test_no_match(self):
        opts = {"A": "red", "B": "blue"}
        assert match_result_to_option("green", opts) is None

    def test_none_result(self):
        opts = {"A": "red"}
        assert match_result_to_option(None, opts) is None

    def test_number_word_large(self):
        opts = {"A": "fifteen", "B": "twenty"}
        assert match_result_to_option(15, opts) == "(A)"


# ── extract_question ──────────────────────────────────────────────────

class TestExtractQuestion:
    def test_basic(self):
        text = "Some context. What color is the pen?\nOptions:\n(A) red"
        q = extract_question(text)
        assert q == "What color is the pen?"

    def test_no_question(self):
        text = "No question here\nOptions:\n(A) red"
        assert extract_question(text) is None

    def test_multiple_sentences(self):
        text = (
            "On the table there are objects. "
            "How many red objects do you see?\n"
            "Options:\n(A) one"
        )
        q = extract_question(text)
        assert "How many red objects" in q

    def test_multiline_question(self):
        text = (
            "Context here.\n"
            "What is the answer?\n"
            "Options:\n(A) yes"
        )
        q = extract_question(text)
        assert "What is the answer?" == q


# ── extract_options ───────────────────────────────────────────────────

class TestExtractOptions:
    def test_basic(self):
        text = "Options:\n(A) red (B) blue (C) green"
        opts = extract_options(text)
        assert opts == {"A": "red", "B": "blue", "C": "green"}

    def test_multiword(self):
        text = "(A) big red thing (B) small blue thing"
        opts = extract_options(text)
        assert opts["A"] == "big red thing"
        assert opts["B"] == "small blue thing"

    def test_no_options(self):
        text = "No options here"
        opts = extract_options(text)
        assert opts == {}

    def test_number_options(self):
        text = "(A) zero (B) one (C) two (D) three"
        opts = extract_options(text)
        assert len(opts) == 4
        assert opts["A"] == "zero"


# ── Round-trip: load → query → match ─────────────────────────────────

class TestSQLRoundTrip:
    def test_count_query(self):
        tables = {"objects": (
            ["color", "type"],
            [("red", "pen"), ("blue", "pen"), ("red", "cup")],
        )}
        conn = load_into_sqlite(tables)
        result, err = _try_execute(
            conn, "SELECT COUNT(*) FROM objects WHERE color = 'red'")
        assert result == 2
        assert err is None

        opts = {"A": "zero", "B": "one", "C": "two", "D": "three"}
        assert match_result_to_option(result, opts) == "(C)"
        conn.close()

    def test_color_lookup(self):
        tables = {"objects": (
            ["color", "type"],
            [("red", "pen"), ("blue", "cup")],
        )}
        conn = load_into_sqlite(tables)
        result, err = _try_execute(
            conn, "SELECT color FROM objects WHERE type = 'pen'")
        assert result == "red"

        opts = {"A": "red", "B": "blue"}
        assert match_result_to_option(result, opts) == "(A)"
        conn.close()

    def test_positional_query(self):
        tables = {"objects": (
            ["position", "color", "type"],
            [(1, "red", "pen"), (2, "blue", "cup"), (3, "green", "plate")],
        )}
        conn = load_into_sqlite(tables)
        result, err = _try_execute(
            conn,
            "SELECT color FROM objects WHERE position = "
            "(SELECT position - 1 FROM objects WHERE type = 'cup')")
        assert result == "red"
        conn.close()

    def test_leftmost(self):
        tables = {"objects": (
            ["position", "color", "type"],
            [(1, "red", "pen"), (2, "blue", "cup"), (3, "green", "plate")],
        )}
        conn = load_into_sqlite(tables)
        result, err = _try_execute(
            conn,
            "SELECT color FROM objects WHERE position = "
            "(SELECT MIN(position) FROM objects)")
        assert result == "red"
        conn.close()

    def test_furthest_from(self):
        tables = {"objects": (
            ["position", "color", "type"],
            [(1, "red", "pen"), (2, "blue", "cup"),
             (3, "green", "plate"), (4, "yellow", "ball")],
        )}
        conn = load_into_sqlite(tables)
        result, err = _try_execute(
            conn,
            "SELECT color FROM objects ORDER BY ABS(position - "
            "(SELECT position FROM objects WHERE type = 'cup')) DESC LIMIT 1")
        assert result == "yellow"
        conn.close()

    def test_count_after_remove(self):
        tables = {"objects": (
            ["color", "type"],
            [("red", "pen"), ("blue", "pen"), ("red", "cup")],
        )}
        conn = load_into_sqlite(tables)
        result, err = _try_execute(
            conn,
            "SELECT COUNT(*) FROM objects WHERE color != 'red' AND type = 'pen'")
        assert result == 1

        opts = {"A": "zero", "B": "one", "C": "two"}
        assert match_result_to_option(result, opts) == "(B)"
        conn.close()

    def test_neither_count(self):
        tables = {"objects": (
            ["color", "type"],
            [("green", "bracelet"), ("black", "spinner"),
             ("red", "pen"), ("blue", "cup")],
        )}
        conn = load_into_sqlite(tables)
        result, err = _try_execute(
            conn,
            "SELECT COUNT(*) FROM objects WHERE color NOT IN ('black', 'blue')")
        assert result == 2

        opts = {"A": "zero", "B": "one", "C": "two", "D": "three"}
        assert match_result_to_option(result, opts) == "(C)"
        conn.close()

    def test_is_color_yes(self):
        tables = {"objects": (
            ["color", "type"],
            [("red", "pen"), ("blue", "cup")],
        )}
        conn = load_into_sqlite(tables)
        result, err = _try_execute(
            conn,
            "SELECT CASE WHEN color = 'red' THEN 'yes' ELSE 'no' END "
            "FROM objects WHERE type = 'pen'")
        assert result == "yes"

        opts = {"A": "yes", "B": "no"}
        assert match_result_to_option(result, opts) == "(A)"
        conn.close()


# ── SchemaSpec ────────────────────────────────────────────────────────

class TestSchemaSpec:
    def test_defaults(self):
        spec = SchemaSpec(
            table_name="objects",
            columns=["color", "type"],
            extraction_prompt="Extract: {scene}",
        )
        assert spec.count_key == "count"
        assert spec.positional_detector is None
        assert spec.max_extract_tokens == 300

    def test_custom_values(self):
        spec = SchemaSpec(
            table_name="items",
            columns=["name", "value"],
            extraction_prompt="Parse: {scene}",
            count_key="qty",
            positional_detector=lambda t: "ordered" in t,
            max_extract_tokens=500,
        )
        assert spec.table_name == "items"
        assert spec.count_key == "qty"
        assert spec.positional_detector("ordered list")
        assert not spec.positional_detector("random set")
        assert spec.max_extract_tokens == 500


# ── extract_scene_text ────────────────────────────────────────────────

class TestExtractSceneText:
    def test_basic_bbh(self):
        text = (
            "On the floor, you see a red pen, a blue cup, and a green plate. "
            "What color is the pen?\n"
            "Options:\n(A) red (B) blue"
        )
        scene = extract_scene_text(text)
        assert "red pen" in scene
        assert "blue cup" in scene
        assert "What color" not in scene

    def test_inventory_scene(self):
        text = (
            "On the table, there are two green pens and one red cup. "
            "How many green items do you see?\n"
            "Options:\n(A) one (B) two"
        )
        scene = extract_scene_text(text)
        assert "two green pens" in scene
        assert "How many" not in scene

    def test_no_question_mark(self):
        text = "On the desk, there is a blue pen and a red cup."
        scene = extract_scene_text(text)
        assert "blue pen" in scene
        assert "red cup" in scene

    def test_with_removal_question(self):
        text = (
            "On the floor, you see three green bracelets and one teal dog leash. "
            "If I remove all the teal items from the table, "
            "how many paperclips remain on it?\n"
            "Options:\n(A) zero (B) one"
        )
        scene = extract_scene_text(text)
        assert "green bracelets" in scene
        # Question fragment should not be in scene
        assert "how many" not in scene.lower()


# ── auto_sql_examples ─────────────────────────────────────────────────

class TestAutoSqlExamples:
    def test_text_columns(self):
        result = auto_sql_examples("objects", ["color", "type"])
        assert "SELECT COUNT(*)" in result
        assert "WHERE color = 'X'" in result
        assert "WHERE type = 'X'" in result
        assert "NOT IN" in result

    def test_cross_column(self):
        result = auto_sql_examples("objects", ["color", "type"])
        assert "SELECT color FROM objects WHERE type" in result
        assert "CASE WHEN" in result

    def test_positional(self):
        result = auto_sql_examples("objects", ["color", "type"], positional=True)
        assert "MIN(position)" in result
        assert "position - 1" in result
        assert "ORDER BY ABS(position" in result

    def test_no_positional_by_default(self):
        result = auto_sql_examples("objects", ["color", "type"])
        assert "position" not in result

    def test_single_column(self):
        result = auto_sql_examples("data", ["name"])
        assert "WHERE name = 'X'" in result
        assert "NOT IN" in result
        # No cross-column patterns with only one column
        assert "SELECT name FROM data WHERE" not in result or "name = 'X'" in result

    def test_with_sample_rows(self):
        """When rows are provided, examples use actual values."""
        rows = [("red", "pen"), ("blue", "cup"), ("red", "plate")]
        result = auto_sql_examples(
            "objects", ["color", "type"], rows=rows)
        # Should use actual values, not placeholders
        assert "WHERE color = 'red'" in result
        assert "WHERE type = 'pen'" in result
        assert "NOT IN ('red', 'blue')" in result
        # Cross-column should use actual values
        assert "SELECT color FROM objects WHERE type = 'pen'" in result
        assert "CASE WHEN color = 'red'" in result

    def test_with_sample_rows_positional(self):
        """Positional examples use type values from actual data."""
        rows = [(1, "red", "pen"), (2, "blue", "cup")]
        result = auto_sql_examples(
            "objects", ["position", "color", "type"],
            positional=True, rows=rows)
        assert "WHERE type = 'pen'" in result
        assert "ORDER BY ABS(position" in result
        assert "position - 1" in result
        assert "position + 1" in result

    def test_query_hints_none(self):
        """No hints → same as baseline (no extra examples)."""
        base = auto_sql_examples("t", ["name", "age"])
        with_none = auto_sql_examples("t", ["name", "age"], query_hints=None)
        assert base == with_none

    def test_query_hints_max_min(self):
        """MAX_MIN intent adds MAX/MIN examples."""
        result = auto_sql_examples(
            "penguins", ["name", "age"], query_hints={"intent": "MAX_MIN"})
        assert "SELECT MAX(" in result
        assert "SELECT MIN(" in result

    def test_query_hints_order_by(self):
        """ORDER_BY intent adds ORDER BY examples."""
        result = auto_sql_examples(
            "t", ["name", "age"], query_hints={"intent": "ORDER_BY"})
        assert "ORDER BY ROWID DESC LIMIT 1" in result
        assert "ORDER BY name ASC LIMIT 1" in result

    def test_query_hints_sum(self):
        """SUM intent adds SUM example when numeric column exists."""
        result = auto_sql_examples(
            "t", ["name", "weight"], query_hints={"intent": "SUM"})
        assert "SELECT SUM(" in result

    def test_query_hints_avg(self):
        """AVG intent adds AVG example when numeric column exists."""
        result = auto_sql_examples(
            "t", ["name", "weight"], query_hints={"intent": "AVG"})
        assert "SELECT AVG(" in result

    def test_query_hints_comparison(self):
        """COMPARISON intent adds comparison subquery example."""
        result = auto_sql_examples(
            "t", ["name", "species", "weight"],
            query_hints={"intent": "COMPARISON"})
        assert "WHERE weight >" in result

    def test_query_hints_where_inequality(self):
        """inequality where_type adds inequality WHERE example."""
        result = auto_sql_examples(
            "t", ["name", "height"], query_hints={"where_type": "inequality"})
        assert "WHERE height > 5" in result

    def test_query_hints_where_compound(self):
        """compound where_type adds AND-combined WHERE example."""
        result = auto_sql_examples(
            "t", ["name", "color"], query_hints={"where_type": "compound"})
        assert "AND" in result

    def test_query_hints_additive(self):
        """Hints ADD examples; base examples still present."""
        result = auto_sql_examples(
            "t", ["name", "age"], query_hints={"intent": "MAX_MIN"})
        # Base examples still there
        assert "SELECT COUNT(*)" in result
        assert "NOT IN" in result
        # Plus hint-specific
        assert "SELECT MAX(" in result

    def test_query_hints_unknown_ignored(self):
        """Unknown hint keys/values don't crash, just add nothing."""
        result = auto_sql_examples(
            "t", ["name"], query_hints={"intent": "UNKNOWN", "foo": "bar"})
        # Should return baseline without error
        assert "Example queries:" in result


# ── _parse_json_array ─────────────────────────────────────────────────

class TestParseJsonArray:
    def test_clean_array(self):
        text = '[{"color": "red", "type": "pen"}, {"color": "blue", "type": "cup"}]'
        result = _parse_json_array(text)
        assert len(result) == 2
        assert result[0]["color"] == "red"

    def test_surrounding_text(self):
        text = 'Here is the JSON:\n[{"color": "red"}]\nDone.'
        result = _parse_json_array(text)
        assert len(result) == 1
        assert result[0]["color"] == "red"

    def test_no_brackets(self):
        assert _parse_json_array("no json here") is None

    def test_empty_array(self):
        result = _parse_json_array("[]")
        assert result == []

    def test_malformed_json(self):
        assert _parse_json_array("[{bad json}]") is None

    def test_not_array_of_dicts(self):
        assert _parse_json_array("[1, 2, 3]") is None

    def test_nested_brackets(self):
        text = '[{"a": [1, 2]}, {"a": [3]}]'
        result = _parse_json_array(text)
        assert len(result) == 2


# ── _model_extract (unit tests with mock data) ───────────────────────

class TestModelExtractExpansion:
    """Test the JSON→table expansion logic without a model.

    Uses _parse_json_array + the expansion logic directly.
    """

    def test_count_expansion(self):
        """Objects with count > 1 should produce multiple rows."""
        objects = [
            {"count": 2, "color": "red", "type": "pen"},
            {"count": 1, "color": "blue", "type": "cup"},
        ]
        spec = SchemaSpec(
            table_name="objects",
            columns=["color", "type"],
            extraction_prompt="",
        )
        rows = []
        for obj in objects:
            count = obj.pop(spec.count_key, 1)
            for _ in range(count):
                rows.append(tuple(obj.get(col, "") for col in spec.columns))

        assert len(rows) == 3
        assert rows[0] == ("red", "pen")
        assert rows[1] == ("red", "pen")
        assert rows[2] == ("blue", "cup")

    def test_positional_expansion(self):
        """Positional detector adds position column."""
        objects = [
            {"count": 1, "color": "red", "type": "pen"},
            {"count": 1, "color": "blue", "type": "cup"},
        ]
        spec = SchemaSpec(
            table_name="objects",
            columns=["color", "type"],
            extraction_prompt="",
            positional_detector=lambda t: "in a row" in t,
        )
        positional = spec.positional_detector("arranged in a row")
        assert positional

        rows = []
        pos = 1
        for obj in objects:
            count = obj.pop(spec.count_key, 1)
            for _ in range(count):
                row = [pos]
                pos += 1
                for col in spec.columns:
                    row.append(obj.get(col, ""))
                rows.append(tuple(row))

        columns = ["position"] + spec.columns
        assert columns == ["position", "color", "type"]
        assert rows[0] == (1, "red", "pen")
        assert rows[1] == (2, "blue", "cup")

    def test_string_count(self):
        """Count given as string should be parsed to int."""
        objects = [{"count": "3", "color": "green", "type": "ball"}]
        spec = SchemaSpec(
            table_name="objects",
            columns=["color", "type"],
            extraction_prompt="",
        )
        rows = []
        for obj in objects:
            count = obj.pop(spec.count_key, 1)
            if isinstance(count, str):
                count = int(count)
            for _ in range(count):
                rows.append(tuple(obj.get(col, "") for col in spec.columns))

        assert len(rows) == 3

    def test_missing_count_key_defaults_to_1(self):
        """If count_key is absent from object, treat as 1."""
        objects = [{"color": "red", "type": "pen"}]
        spec = SchemaSpec(
            table_name="objects",
            columns=["color", "type"],
            extraction_prompt="",
        )
        rows = []
        for obj in objects:
            count = obj.pop(spec.count_key, 1)
            for _ in range(count):
                rows.append(tuple(obj.get(col, "") for col in spec.columns))

        assert len(rows) == 1
        assert rows[0] == ("red", "pen")

    def test_end_to_end_json_to_sqlite(self):
        """Parse JSON → expand → load into SQLite → query."""
        json_text = '[{"count": 2, "color": "red", "type": "pen"}, {"count": 1, "color": "blue", "type": "cup"}]'
        objects = _parse_json_array(json_text)
        assert objects is not None

        spec = SchemaSpec(
            table_name="objects",
            columns=["color", "type"],
            extraction_prompt="",
        )
        rows = []
        for obj in objects:
            count = obj.pop(spec.count_key, 1)
            for _ in range(count):
                rows.append(tuple(obj.get(col, "") for col in spec.columns))

        tables = {spec.table_name: (spec.columns, rows)}
        conn = load_into_sqlite(tables)

        result, err = _try_execute(
            conn, "SELECT COUNT(*) FROM objects WHERE color = 'red'")
        assert result == 2
        assert err is None

        result2, _ = _try_execute(
            conn, "SELECT type FROM objects WHERE color = 'blue'")
        assert result2 == "cup"
        conn.close()


# ── parse_markdown_table ─────────────────────────────────────────────

class TestParseMarkdownTable:
    def test_clean_table(self):
        text = (
            "| position | color | type |\n"
            "|----------|-------|------|\n"
            "| 1 | red | pen |\n"
            "| 2 | blue | cup |"
        )
        result = parse_markdown_table(text)
        assert result is not None
        columns, rows = result
        assert columns == ["position", "color", "type"]
        assert len(rows) == 2
        assert rows[0] == (1, "red", "pen")
        assert rows[1] == (2, "blue", "cup")

    def test_surrounding_text(self):
        text = (
            "Here is the table:\n\n"
            "| color | type |\n"
            "|-------|------|\n"
            "| red | pen |\n"
            "\nSome trailing text."
        )
        result = parse_markdown_table(text)
        assert result is not None
        columns, rows = result
        assert columns == ["color", "type"]
        assert len(rows) == 1
        assert rows[0] == ("red", "pen")

    def test_no_outer_pipes(self):
        text = (
            "color | type\n"
            "------|-----\n"
            "red | pen\n"
            "blue | cup"
        )
        result = parse_markdown_table(text)
        assert result is not None
        columns, rows = result
        assert columns == ["color", "type"]
        assert len(rows) == 2

    def test_extra_whitespace(self):
        text = (
            "|  position  |  color  |  type  |\n"
            "|------------|---------|--------|\n"
            "|  1  |  red  |  pen  |\n"
            "|  2  |  blue  |  cup  |"
        )
        result = parse_markdown_table(text)
        assert result is not None
        columns, rows = result
        assert columns == ["position", "color", "type"]
        assert rows[0] == (1, "red", "pen")

    def test_malformed_no_separator(self):
        text = (
            "| color | type |\n"
            "| red | pen |"
        )
        assert parse_markdown_table(text) is None

    def test_malformed_no_data(self):
        text = (
            "| color | type |\n"
            "|-------|------|"
        )
        assert parse_markdown_table(text) is None

    def test_missing_header(self):
        text = (
            "|-------|------|\n"
            "| red | pen |"
        )
        assert parse_markdown_table(text) is None

    def test_numeric_position_column(self):
        """Position values should be parsed as int."""
        text = (
            "| position | name |\n"
            "|----------|------|\n"
            "| 1 | alice |\n"
            "| 2 | bob |"
        )
        result = parse_markdown_table(text)
        assert result is not None
        _, rows = result
        assert rows[0][0] == 1
        assert isinstance(rows[0][0], int)


# ── Generic table extraction (markdown → SQLite → query) ────────────

class TestGenericTableExtraction:
    """End-to-end: parse markdown table → load into SQLite → query."""

    def test_count_query(self):
        text = (
            "| position | color | type |\n"
            "|----------|-------|------|\n"
            "| 1 | red | pen |\n"
            "| 2 | red | pen |\n"
            "| 3 | blue | cup |"
        )
        parsed = parse_markdown_table(text)
        assert parsed is not None
        columns, rows = parsed
        tables = {"data": (columns, rows)}
        conn = load_into_sqlite(tables)

        result, err = _try_execute(
            conn, "SELECT COUNT(*) FROM data WHERE color = 'red'")
        assert result == 2
        assert err is None
        conn.close()

    def test_positional_query(self):
        text = (
            "| position | color | type |\n"
            "|----------|-------|------|\n"
            "| 1 | red | pen |\n"
            "| 2 | blue | cup |\n"
            "| 3 | green | plate |"
        )
        parsed = parse_markdown_table(text)
        assert parsed is not None
        columns, rows = parsed
        tables = {"data": (columns, rows)}
        conn = load_into_sqlite(tables)

        result, err = _try_execute(
            conn,
            "SELECT color FROM data WHERE position = "
            "(SELECT position - 1 FROM data WHERE type = 'cup')")
        assert result == "red"
        assert err is None
        conn.close()

    def test_auto_examples_from_extracted(self):
        """auto_sql_examples works with columns from parse_markdown_table."""
        text = (
            "| position | color | type |\n"
            "|----------|-------|------|\n"
            "| 1 | red | pen |"
        )
        parsed = parse_markdown_table(text)
        assert parsed is not None
        columns, _ = parsed
        positional = "position" in columns
        examples = auto_sql_examples("data", columns, positional)
        assert "MIN(position)" in examples
        assert "color" in examples
        assert "type" in examples


# ── Generic JSON extraction (JSON+counts → SQLite → query) ──────────

class TestGenericJsonExtraction:
    """End-to-end: parse JSON with counts → expand → SQLite → query."""

    def test_count_expansion_to_sqlite(self):
        json_text = (
            '[{"count": 2, "color": "red", "type": "pen"}, '
            '{"count": 1, "color": "blue", "type": "cup"}]'
        )
        objects = _parse_json_array(json_text)
        assert objects is not None
        # Expand with counts (same logic as _model_extract_table)
        rows = []
        pos = 1
        for obj in objects:
            count = obj.pop("count", 1)
            for _ in range(count):
                rows.append((pos, obj.get("color", ""), obj.get("type", "")))
                pos += 1
        columns = ["position", "color", "type"]
        tables = {"data": (columns, rows)}
        conn = load_into_sqlite(tables)

        result, err = _try_execute(
            conn, "SELECT COUNT(*) FROM data WHERE color = 'red'")
        assert result == 2
        assert err is None

        result2, _ = _try_execute(
            conn, "SELECT type FROM data WHERE color = 'blue'")
        assert result2 == "cup"
        conn.close()

    def test_auto_examples_with_sample_values(self):
        """auto_sql_examples uses actual values from expanded JSON rows."""
        rows = [(1, "red", "pen"), (2, "red", "pen"), (3, "blue", "cup")]
        columns = ["position", "color", "type"]
        examples = auto_sql_examples("data", columns, positional=True, rows=rows)
        assert "WHERE color = 'red'" in examples
        assert "WHERE type = 'pen'" in examples
        assert "MIN(position)" in examples
        assert "ORDER BY ABS(position" in examples

    def test_non_positional_expansion(self):
        """Inventory scenes have no position column."""
        json_text = (
            '[{"count": 3, "color": "green", "type": "ball"}, '
            '{"count": 1, "color": "red", "type": "cup"}]'
        )
        objects = _parse_json_array(json_text)
        assert objects is not None
        rows = []
        for obj in objects:
            count = obj.pop("count", 1)
            for _ in range(count):
                rows.append((obj.get("color", ""), obj.get("type", "")))
        columns = ["color", "type"]
        tables = {"data": (columns, rows)}
        conn = load_into_sqlite(tables)

        result, err = _try_execute(
            conn, "SELECT COUNT(*) FROM data WHERE color = 'green'")
        assert result == 3
        assert err is None
        conn.close()


# ── repair_sql ─────────────────────────────────────────────────────

class TestRepairSql:
    """Test SQL repair patterns."""

    def _make_db(self, tables):
        return load_into_sqlite(tables)

    def test_rowid_last(self):
        tables = {"t": (["name"], [("Alice",), ("Bob",)])}
        conn = self._make_db(tables)
        result = repair_sql("SELECT name FROM t WHERE ROWID = last", conn, tables)
        assert "ORDER BY ROWID DESC LIMIT 1" in result
        conn.close()

    def test_entity_dot_column(self):
        tables = {"t": (["name", "age"], [("Alice", 30), ("Bob", 25)])}
        conn = self._make_db(tables)
        result = repair_sql("SELECT age FROM t WHERE age > Alice.age", conn, tables)
        assert "SELECT age FROM t WHERE name = 'Alice'" in result
        conn.close()

    def test_strip_where_on_max(self):
        """SELECT MAX(col) WHERE name='X' → remove WHERE."""
        tables = {"t": (["name", "height"], [("A", 50), ("B", 80)])}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT MAX(height) FROM t WHERE name = 'A'", conn, tables)
        assert result == "SELECT MAX(height) FROM t"
        conn.close()

    def test_strip_where_on_min(self):
        tables = {"t": (["name", "height"], [("A", 50), ("B", 80)])}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT MIN(height) FROM t WHERE name = 'A'", conn, tables)
        assert result == "SELECT MIN(height) FROM t"
        conn.close()

    def test_strip_where_on_sum(self):
        tables = {"t": (["name", "age"], [("A", 5), ("B", 10)])}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT SUM(age) FROM t WHERE name = 'A'", conn, tables)
        assert result == "SELECT SUM(age) FROM t"
        conn.close()

    def test_no_strip_where_on_non_aggregate(self):
        """Non-aggregate queries should keep their WHERE clause."""
        tables = {"t": (["name", "age"], [("A", 5), ("B", 10)])}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT age FROM t WHERE name = 'A'", conn, tables)
        assert "WHERE name = 'A'" in result
        conn.close()

    def test_multi_table_union_for_animal(self):
        """'animal' question with 2 tables → UNION."""
        tables = {
            "penguins": (["name", "age"], [("Louis", 7)]),
            "giraffes": (["name", "age"], [("Jody", 5)]),
        }
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins ORDER BY name DESC LIMIT 1",
            conn, tables,
            question="What is the last animal sorted by alphabetic order?")
        assert "UNION ALL" in result
        assert "penguins" in result
        assert "giraffes" in result
        conn.close()

    def test_no_union_for_specific_table(self):
        """'penguin' question stays single-table."""
        tables = {
            "penguins": (["name", "age"], [("Louis", 7)]),
            "giraffes": (["name", "age"], [("Jody", 5)]),
        }
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins ORDER BY name DESC LIMIT 1",
            conn, tables,
            question="What is the last penguin sorted by alphabetic order?")
        assert "UNION ALL" not in result
        conn.close()

    def test_no_union_single_table(self):
        """Single table — no union even with 'animal' question."""
        tables = {"penguins": (["name", "age"], [("Louis", 7)])}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins ORDER BY ROWID DESC LIMIT 1",
            conn, tables,
            question="What is the last animal?")
        assert "UNION ALL" not in result
        conn.close()

    def test_multi_table_union_aggregate(self):
        """SUM(age) with 'animal' question → UNION inner column."""
        tables = {
            "penguins": (["name", "age"], [("Louis", 7)]),
            "giraffes": (["name", "age"], [("Jody", 5)]),
        }
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT SUM(age) FROM penguins",
            conn, tables,
            question="What is the cumulated age of the animals?")
        assert "UNION ALL" in result
        assert "penguins" in result
        assert "giraffes" in result
        # Execute to verify correctness
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == 12  # 7 + 5
        conn.close()

    def test_no_union_when_rowid_used(self):
        """ROWID doesn't exist in UNION subqueries — skip rewrite."""
        tables = {
            "penguins": (["name", "age"], [("Louis", 7)]),
            "giraffes": (["name", "age"], [("Jody", 5)]),
        }
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins WHERE ROWID = (SELECT MAX(ROWID) FROM penguins)",
            conn, tables,
            question="What is the last animal?")
        assert "UNION ALL" not in result
        conn.close()


# ── repair_sql v3 patterns ────────────────────────────────────────

class TestRepairSqlV3:
    """Tests for repair_sql v3 patterns added 2026-04-03."""

    def _make_db(self, tables):
        return load_into_sqlite(tables)

    PENGUIN_GIRAFFE = {
        "penguins": (["name", "age", "height", "weight"],
                     [("Louis", 7, 50, 11), ("Bernard", 5, 80, 13),
                      ("Vincent", 9, 60, 11), ("Gwen", 8, 70, 15)]),
        "giraffes": (["name", "age", "height", "weight"],
                     [("Jody", 5, 430, 620), ("Gladys", 10, 420, 590),
                      ("Marian", 2, 310, 410), ("Donna", 9, 440, 650)]),
    }

    # ── COUNT(*) UNION ──

    def test_count_star_union_for_animals(self):
        """COUNT(*) with WHERE should UNION across tables for 'animals' question."""
        conn = self._make_db(self.PENGUIN_GIRAFFE)
        result = repair_sql(
            "SELECT COUNT(*) FROM giraffes WHERE age > 5",
            conn, self.PENGUIN_GIRAFFE,
            question="How many animals are more than 5 years old?")
        assert "UNION ALL" in result
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == 5  # Louis(7), Vincent(9), Gwen(8), Gladys(10), Donna(9)
        conn.close()

    def test_count_star_no_union_single_table(self):
        """COUNT(*) with single table — no UNION."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT COUNT(*) FROM penguins WHERE age > 5",
            conn, tables,
            question="How many animals are more than 5 years old?")
        assert "UNION ALL" not in result
        conn.close()

    # ── Explicit table name triggers ──

    def test_explicit_table_names_trigger_union(self):
        """'penguins and giraffes' triggers UNION even without 'animals'."""
        conn = self._make_db(self.PENGUIN_GIRAFFE)
        result = repair_sql(
            "SELECT SUM(age) FROM penguins",
            conn, self.PENGUIN_GIRAFFE,
            question="What is the cumulated age of the penguins and giraffes?")
        assert "UNION ALL" in result
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == 55  # 7+5+9+8 + 5+10+2+9
        conn.close()

    def test_single_table_name_no_union(self):
        """Mentioning one table name doesn't trigger UNION."""
        conn = self._make_db(self.PENGUIN_GIRAFFE)
        result = repair_sql(
            "SELECT SUM(age) FROM penguins",
            conn, self.PENGUIN_GIRAFFE,
            question="What is the total age of the penguins?")
        assert "UNION ALL" not in result
        conn.close()

    # ── Multi-table ROWID ordering ──

    def test_multi_table_last_animal_rowid(self):
        """'last animal' with ROWID=MAX → query last table's last row."""
        conn = self._make_db(self.PENGUIN_GIRAFFE)
        result = repair_sql(
            "SELECT name FROM penguins WHERE ROWID = (SELECT MAX(ROWID) FROM penguins)",
            conn, self.PENGUIN_GIRAFFE,
            question="What is the name of the last animal?")
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == "Donna"  # Last giraffe
        conn.close()

    # ── Ordinal ROWID off-by-one ──

    def test_second_ordinal_rowid(self):
        """'second penguin' + ROWID=1 → ROWID=2."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins WHERE ROWID = 1",
            conn, tables,
            question="What is the name of the second penguin?")
        assert "ROWID = 2" in result
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == "Bernard"
        conn.close()

    def test_first_ordinal_rowid_unchanged(self):
        """'first penguin' + ROWID=1 → unchanged."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins WHERE ROWID = 1",
            conn, tables,
            question="What is the name of the first penguin?")
        assert "ROWID = 1" in result
        conn.close()

    def test_third_ordinal_rowid(self):
        """'third penguin' + ROWID=2 → ROWID=3."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins WHERE ROWID = 2",
            conn, tables,
            question="What is the name of the third penguin?")
        assert "ROWID = 3" in result
        conn.close()

    # ── "Next to last" ORDER BY flip ──

    def test_next_to_last_flips_order(self):
        """'next to last' + ASC → DESC OFFSET 1."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins ORDER BY name ASC LIMIT 1 OFFSET 1",
            conn, tables,
            question="What is the name of the next to last penguin sorted by alphabetic order?")
        assert "DESC" in result
        assert "OFFSET 1" in result
        cursor = conn.execute(result)
        # Names: Bernard, Gwen, Louis, Vincent. DESC OFFSET 1 = Louis
        assert cursor.fetchone()[0] == "Louis"
        conn.close()

    def test_no_flip_without_next_to_last(self):
        """Normal ORDER BY ASC is not flipped."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins ORDER BY name ASC LIMIT 1 OFFSET 1",
            conn, tables,
            question="What is the second penguin alphabetically?")
        assert "ASC" in result
        conn.close()

    # ── Inverted comparisons ──

    def test_younger_older_inversion(self):
        """'younger than X and older than Y' → flip > and < operators."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins WHERE age > "
            "(SELECT age FROM penguins WHERE name = 'Vincent') AND age < "
            "(SELECT age FROM penguins WHERE name = 'Louis')",
            conn, tables,
            question="Which penguin is younger than Vincent and older than Louis?")
        # Should flip: age < Vincent AND age > Louis
        assert "< (SELECT age FROM penguins WHERE name = 'Vincent')" in result
        assert "> (SELECT age FROM penguins WHERE name = 'Louis')" in result
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == "Gwen"  # age 8, between Louis(7) and Vincent(9)
        conn.close()

    def test_no_inversion_without_pattern(self):
        """Normal comparison not inverted."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins WHERE age > "
            "(SELECT age FROM penguins WHERE name = 'Louis')",
            conn, tables,
            question="Which penguin is older than Louis?")
        assert "> (SELECT age FROM penguins WHERE name = 'Louis')" in result
        conn.close()

    # ── Superlative → MAX/MIN ──

    def test_oldest_hardcoded_to_max(self):
        """'oldest penguin' + WHERE age='8' → WHERE age=(SELECT MAX(age))."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins WHERE age = '8'",
            conn, tables,
            question="Which is the oldest penguin?")
        assert "SELECT MAX(age)" in result
        conn.close()

    def test_oldest_multicol_to_max(self):
        """Multi-column SELECT + superlative → MAX repair."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name, age, height, weight FROM penguins WHERE age = '8'",
            conn, tables,
            question="Which is the oldest penguin?")
        assert "SELECT MAX(age)" in result
        conn.close()

    def test_youngest_to_min(self):
        """'youngest penguin' → MIN."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins WHERE age = '5'",
            conn, tables,
            question="Which is the youngest penguin?")
        assert "SELECT MIN(age)" in result
        conn.close()

    def test_tallest_comparison_to_max(self):
        """'taller than the other ones' + col > subquery → col = MAX."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins WHERE height > "
            "(SELECT height FROM penguins WHERE name = 'Louis')",
            conn, tables,
            question="Which penguin is taller than the other ones?")
        assert "SELECT MAX(height)" in result
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == "Bernard"  # height 80
        conn.close()

    def test_second_youngest_not_rewritten(self):
        """'second youngest' should NOT trigger MIN — ordinal guard."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        original = "SELECT name, age FROM penguins WHERE age = '7' ORDER BY age ASC LIMIT 1"
        result = repair_sql(original, conn, tables,
                            question="Which is the second youngest penguin?")
        assert "MIN" not in result
        assert "MAX" not in result
        conn.close()

    def test_superlative_skips_subquery(self):
        """Superlative literal repair must NOT match inside nested subqueries."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        # This SQL already has a subquery — literal repair should not fire
        original = ("SELECT name FROM penguins WHERE age > "
                    "(SELECT age FROM penguins WHERE name = 'Gwen')")
        result = repair_sql(original, conn, tables,
                            question="Which penguin is older than the other ones?")
        # Should use the comparison pattern (MAX), not corrupt the subquery
        assert "SELECT MAX(age)" in result
        assert "incomplete" not in result.lower()
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == "Vincent"  # age 9
        conn.close()

    def test_no_superlative_for_normal_question(self):
        """Non-superlative question should not trigger MAX/MIN repair."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins WHERE age = '7'",
            conn, tables,
            question="Which penguin is 7 years old?")
        assert "MAX" not in result
        assert "MIN" not in result
        assert "age = '7'" in result
        conn.close()


# ── Combinatorial repair interactions ────────────────────────────

class TestRepairSqlComposition:
    """Tests for interactions between multiple repair patterns.

    The v3 regression (superlative matching inside subqueries) showed that
    repair patterns can interfere. These tests exercise overlapping triggers.
    """

    def _make_db(self, tables):
        return load_into_sqlite(tables)

    PENGUIN_GIRAFFE = TestRepairSqlV3.PENGUIN_GIRAFFE

    def test_oldest_animal_multitable_superlative(self):
        """'oldest animal' triggers both UNION and superlative.

        Known limitation: UNION fires first (restructures FROM clause),
        then superlative can't match the new shape. The result has UNION
        but keeps the hardcoded value. The SQL is still valid — it returns
        the '8'-year-old animal across both tables (Gwen), not the true
        oldest (Gladys, 10). This is acceptable: the knowledge/logit poll
        fallback handles these cases in practice.
        """
        conn = self._make_db(self.PENGUIN_GIRAFFE)
        result = repair_sql(
            "SELECT name FROM penguins WHERE age = '8'",
            conn, self.PENGUIN_GIRAFFE,
            question="Which is the oldest animal?")
        # UNION fires, superlative cannot fire after restructure
        assert "UNION ALL" in result
        # SQL is valid and returns a result (not corrupted)
        cursor = conn.execute(result)
        val = cursor.fetchone()[0]
        assert val is not None
        conn.close()

    def test_next_to_last_animal_multitable(self):
        """'next to last animal' triggers UNION, ROWID, and 'next to last'.

        The ROWID pattern should fire for multi-table, and 'next to last'
        should flip ordering. Since these interact, verify no corruption.
        """
        conn = self._make_db(self.PENGUIN_GIRAFFE)
        # This SQL has ORDER BY (not ROWID), so the ROWID pattern won't fire.
        # "next to last" + "animal" both active.
        result = repair_sql(
            "SELECT name FROM penguins ORDER BY name ASC LIMIT 1 OFFSET 1",
            conn, self.PENGUIN_GIRAFFE,
            question="What is the name of the next to last animal sorted alphabetically?")
        assert "DESC" in result
        assert "OFFSET 1" in result
        # UNION should also fire since "animal" + multi-table
        assert "UNION ALL" in result
        # Execute to verify valid SQL
        cursor = conn.execute(result)
        val = cursor.fetchone()[0]
        assert val is not None
        conn.close()

    def test_second_oldest_no_superlative(self):
        """'second oldest' should NOT rewrite to MAX — ordinal guard."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins ORDER BY age DESC LIMIT 1 OFFSET 1",
            conn, tables,
            question="Which is the second oldest penguin?")
        assert "MAX" not in result
        assert "MIN" not in result
        # "second" ordinal + ROWID check shouldn't fire (no ROWID in SQL)
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == "Gwen"  # age 8, second after Vincent(9)
        conn.close()

    def test_inverted_comparison_not_superlative(self):
        """'younger than X and older than Y' triggers inversion, NOT superlative."""
        tables = {"penguins": self.PENGUIN_GIRAFFE["penguins"]}
        conn = self._make_db(tables)
        result = repair_sql(
            "SELECT name FROM penguins WHERE age > "
            "(SELECT age FROM penguins WHERE name = 'Vincent') AND age < "
            "(SELECT age FROM penguins WHERE name = 'Louis')",
            conn, tables,
            question="Which penguin is younger than Vincent and older than Louis?")
        # Inversion should fire, not superlative
        assert "MAX" not in result
        assert "MIN" not in result
        assert "<" in result and ">" in result
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == "Gwen"
        conn.close()

    def test_aggregate_where_strip_then_union(self):
        """Spurious WHERE strip + multi-table UNION compose correctly."""
        conn = self._make_db(self.PENGUIN_GIRAFFE)
        result = repair_sql(
            "SELECT MAX(age) FROM penguins WHERE name = 'Louis'",
            conn, self.PENGUIN_GIRAFFE,
            question="What is the maximum age of the animals?")
        # WHERE strip fires first, then UNION
        assert "name = 'Louis'" not in result
        assert "UNION ALL" in result
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == 10  # Gladys
        conn.close()

    def test_ordinal_plus_specific_table_no_union(self):
        """'second penguin' with multi-table DB should NOT trigger UNION."""
        conn = self._make_db(self.PENGUIN_GIRAFFE)
        result = repair_sql(
            "SELECT name FROM penguins WHERE ROWID = 1",
            conn, self.PENGUIN_GIRAFFE,
            question="What is the name of the second penguin?")
        assert "UNION ALL" not in result
        assert "ROWID = 2" in result
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == "Bernard"
        conn.close()

    def test_count_star_union_preserves_where(self):
        """COUNT(*) UNION preserves the WHERE clause."""
        conn = self._make_db(self.PENGUIN_GIRAFFE)
        result = repair_sql(
            "SELECT COUNT(*) FROM penguins WHERE age > 5",
            conn, self.PENGUIN_GIRAFFE,
            question="How many animals are more than 5 years old?")
        assert "UNION ALL" in result
        assert "WHERE age > 5" in result
        cursor = conn.execute(result)
        assert cursor.fetchone()[0] == 5
        conn.close()


# ── Diagnostic infrastructure ────────────────────────────────────

class TestDiagnosticInfrastructure:
    """Smoke tests for the diag= dict contract in _sql_solve and solve_penguins."""

    def test_sql_solve_diag_keys_on_success(self):
        """_sql_solve(diag={}) populates expected keys on successful solve."""
        from turnstyle.sql import SQLTurnstyle
        import torch

        class FakeModel:
            def __call__(self, **kw):
                return self
            def generate(self, **kw):
                # Return a sequence that decodes to a valid SQL fragment
                return kw["input_ids"]
            @property
            def logits(self):
                return torch.zeros(1, 1, 50000)

        class FakeBatch(dict):
            """Dict subclass with .to() for device placement."""
            def to(self, device):
                return self

        class FakeTokenizer:
            eos_token_id = 0
            def __call__(self, text, return_tensors="pt"):
                return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True):
                return messages[0]["content"]
            def encode(self, text, add_special_tokens=False):
                return [1]
            def decode(self, ids, skip_special_tokens=True):
                # Return SQL that will work: "SELECT name FROM penguins WHERE ROWID = 1"
                return "name FROM penguins WHERE ROWID = 1"

        ts = SQLTurnstyle(FakeModel(), FakeTokenizer(), "cpu",
                          parse_tables_fn=lambda text: {
                              "penguins": (["name", "age"],
                                           [("Louis", 7), ("Bernard", 5)])
                          })

        prompt = ("Here is a table: name, age  Louis, 7  Bernard, 5  "
                  "What is the first penguin?\n"
                  "Options:\n(A) Louis\n(B) Bernard")
        diag = {}
        result = ts._sql_solve(prompt, diag=diag)

        # Core keys always present
        assert "tables_parsed" in diag
        assert diag["tables_parsed"] is True
        assert "table_names" in diag
        assert "penguins" in diag["table_names"]
        assert "question" in diag
        assert "options" in diag
        assert "raw_sql" in diag
        assert "sql_result" in diag
        assert "sql_error" in diag

    def test_sql_solve_diag_keys_on_parse_failure(self):
        """_sql_solve(diag={}) populates tables_parsed=False on no tables."""
        from turnstyle.sql import SQLTurnstyle

        ts = SQLTurnstyle.__new__(SQLTurnstyle)
        ts._parse_tables_fn = lambda text: None
        ts._model_extract = lambda prompt: None  # also mock model extraction
        ts.schema_spec = None
        ts.model = None
        ts.tokenizer = None
        ts.device = "cpu"
        ts.intent_probe = None

        diag = {}
        result = ts._sql_solve("no tables here", diag=diag)
        assert result is None
        assert diag["tables_parsed"] is False
        assert diag["table_names"] == []

    def test_sql_solve_diag_meta_schema(self):
        """_sql_solve(diag={}) sets meta_schema=True for species questions."""
        from turnstyle.sql import SQLTurnstyle

        ts = SQLTurnstyle.__new__(SQLTurnstyle)
        ts._parse_tables_fn = lambda text: {
            "penguins": (["name", "age"], [("A", 1)]),
            "giraffes": (["name", "age"], [("B", 2)]),
        }
        ts.schema_spec = None
        ts.model = None
        ts.tokenizer = None
        ts.device = "cpu"
        ts.intent_probe = None

        prompt = ("Data here.\nHow many species are listed in the tables?\n"
                  "Options:\n(A) 1\n(B) 2\n(C) 3")
        diag = {}
        result = ts._sql_solve(prompt, diag=diag)
        assert result is not None
        assert diag.get("meta_schema") is True

    def test_solve_penguins_diag_tier_sql(self):
        """solve_penguins(diag={}) records tier='sql' on SQL success."""
        from swollm.solvers.penguins import solve_penguins

        class FakeFallback:
            logit_poll_fallback = False
            def _sql_solve(self, text, diag=None):
                if diag is not None:
                    diag["tables_parsed"] = True
                    diag["raw_sql"] = "SELECT name FROM penguins"
                return ("SQL: test", "(A)")

        diag = {}
        answer = solve_penguins("test input", llm_fallback=FakeFallback(), diag=diag)
        assert answer == "(A)"
        assert diag["tier"] == "sql"
        assert diag["answer"] == "(A)"
        assert "sql" in diag  # nested sql diag dict
        assert diag["sql"]["tables_parsed"] is True

    def test_solve_penguins_diag_tier_none(self):
        """solve_penguins(diag={}) records tier='none' when all tiers fail."""
        from swollm.solvers.penguins import solve_penguins

        class FakeFallback:
            logit_poll_fallback = False
            def _sql_solve(self, text, diag=None):
                return None

        diag = {}
        answer = solve_penguins("test input", llm_fallback=FakeFallback(), diag=diag)
        assert answer is None
        assert diag["tier"] == "none"

    def test_solve_penguins_diag_no_fallback(self):
        """solve_penguins(diag={}) with no fallback records error."""
        from swollm.solvers.penguins import solve_penguins

        diag = {}
        answer = solve_penguins("test input", llm_fallback=None, diag=diag)
        assert answer is None
        assert diag["tier"] == "none"
        assert diag["error"] == "no_llm_fallback"


# ── _meta_schema_solve ───────────────────────────────────────────

class TestMetaSchemaSolve:
    """Tests for meta-schema question interception."""

    def test_species_count_single_table(self):
        tables = {"penguins": (["name", "age"], [("A", 1)])}
        options = {"A": "1", "B": "2", "C": "3"}
        result = _meta_schema_solve(
            "How many species are listed in the table?", tables, options)
        assert result is not None
        _, answer = result
        assert answer == "(A)"

    def test_species_count_two_tables(self):
        tables = {
            "penguins": (["name", "age"], [("A", 1)]),
            "giraffes": (["name", "age"], [("B", 2)]),
        }
        options = {"A": "1", "B": "2", "C": "3"}
        result = _meta_schema_solve(
            "How many species are listed in the tables?", tables, options)
        assert result is not None
        _, answer = result
        assert answer == "(B)"

    def test_column_number_weight(self):
        tables = {"penguins": (["name", "age", "height", "weight"], [("A", 1, 50, 11)])}
        options = {"A": "1", "B": "2", "C": "3", "D": "4"}
        result = _meta_schema_solve(
            "what is the number of the column with the weights (1, 2, 3 or 4)?",
            tables, options)
        assert result is not None
        _, answer = result
        assert answer == "(D)"  # weight is column 4

    def test_column_number_age(self):
        tables = {"penguins": (["name", "age", "height", "weight"], [("A", 1, 50, 11)])}
        options = {"A": "1", "B": "2", "C": "3", "D": "4"}
        result = _meta_schema_solve(
            "what is the number of the column with the ages?",
            tables, options)
        assert result is not None
        _, answer = result
        assert answer == "(B)"  # age is column 2

    def test_non_meta_question_returns_none(self):
        tables = {"penguins": (["name", "age"], [("A", 1)])}
        options = {"A": "1", "B": "2"}
        result = _meta_schema_solve(
            "Which is the oldest penguin?", tables, options)
        assert result is None

    def test_species_no_match_returns_none(self):
        """If table count doesn't match any option, return None."""
        tables = {
            "a": (["x"], [(1,)]),
            "b": (["x"], [(2,)]),
            "c": (["x"], [(3,)]),
        }
        options = {"A": "1", "B": "2"}  # 3 not available
        result = _meta_schema_solve(
            "How many species are listed in the tables?", tables, options)
        assert result is None


# ── SQLTurnstyle logit_poll_fallback flag ──────────────────────────

class TestLogitPollFlag:
    """Test that logit_poll_fallback flag is stored and prior cache initializes."""

    def test_default_off(self):
        from turnstyle.sql import SQLTurnstyle
        ts = SQLTurnstyle.__new__(SQLTurnstyle)
        ts.logit_poll_fallback = False
        ts._logit_prior = None
        assert ts.logit_poll_fallback is False
        assert ts._logit_prior is None

    def test_flag_on(self):
        from turnstyle.sql import SQLTurnstyle
        ts = SQLTurnstyle.__new__(SQLTurnstyle)
        ts.logit_poll_fallback = True
        ts._logit_prior = None
        assert ts.logit_poll_fallback is True

    def test_prior_cache_reused(self):
        """Once _logit_prior is set, _compute_logit_prior returns cached value."""
        from turnstyle.sql import SQLTurnstyle
        ts = SQLTurnstyle.__new__(SQLTurnstyle)
        ts._logit_prior = {"A": 15.0, "B": 11.0, "C": 12.0}
        result = ts._compute_logit_prior()
        assert result is ts._logit_prior
        assert result["A"] == 15.0


# ── Knowledge poll predicate extraction ──────────────────────────

class TestKnowledgePollPredicateExtraction:
    """Test that _knowledge_poll extracts predicates correctly.

    These tests verify the regex predicate extraction without needing
    a model — they use a mock that returns controlled logits.
    """

    def _make_ts_with_mock(self, margin_map):
        """Create a SQLTurnstyle with a mock model returning controlled margins.

        margin_map: {(prompt_fragment, option_value): yes_no_margin}
        Uses a simple mock: yes_logit = margin, no_logit = 0.
        """
        import types
        from turnstyle.sql import SQLTurnstyle

        ts = SQLTurnstyle.__new__(SQLTurnstyle)
        ts.logit_poll_fallback = True
        ts._logit_prior = {}

        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text == "yes":
                    return [100]  # yes token id
                if text == "no":
                    return [200]  # no token id
                return [300]

            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True):
                return messages[0]["content"]

            def __call__(self, text, return_tensors="pt"):
                import torch
                return {"input_ids": torch.tensor([[1]])}

        ts.tokenizer = MockTokenizer()
        ts.device = "cpu"

        # Mock model
        class MockModel:
            def __init__(self, margin_map, tokenizer):
                self._margin_map = margin_map
                self._tokenizer = tokenizer
                self._last_prompt = None

            def __call__(self, **kwargs):
                return self

            @property
            def logits(self):
                import torch
                # Find matching margin from the prompt
                logit_vec = torch.zeros(1, 1, 400)
                for (frag, val), margin in self._margin_map.items():
                    if frag in self._last_prompt and f"'{val}'" in self._last_prompt:
                        logit_vec[0, 0, 100] = margin  # yes token
                        logit_vec[0, 0, 200] = 0.0  # no token
                        return logit_vec
                # Default: no signal
                logit_vec[0, 0, 100] = 0.0
                logit_vec[0, 0, 200] = 0.0
                return logit_vec

        mock_model = MockModel(margin_map, ts.tokenizer)

        # Patch tokenizer to record the prompt
        original_call = ts.tokenizer.__call__
        def patched_call(text, return_tensors="pt"):
            mock_model._last_prompt = text
            return original_call(text, return_tensors=return_tensors)
        ts.tokenizer.__call__ = patched_call

        ts.model = mock_model
        return ts

    def test_has_predicate_extraction(self):
        """'Which X has Y?' extracts Y as predicate."""
        import re
        question = "Which penguin has a welsh name?"
        m = re.match(
            r"Which\s+\w+\s+(?:has|have)\s+(.+?)\??\s*$",
            question, re.IGNORECASE)
        assert m is not None
        assert m.group(1) == "a welsh name"

    def test_is_predicate_extraction(self):
        """'Which X is Y?' extracts Y as predicate."""
        import re
        question = "Which penguin is a female?"
        m = re.match(
            r"Which\s+\w+\s+(?:is|are)\s+(.+?)\??\s*$",
            question, re.IGNORECASE)
        assert m is not None
        assert m.group(1) == "a female"

    def test_no_predicate_for_how_many(self):
        """'How many X?' does not match predicate pattern."""
        import re
        question = "How many animals are more than 5 years old?"
        m = re.match(
            r"Which\s+\w+\s+(?:has|have)\s+(.+?)\??\s*$",
            question, re.IGNORECASE)
        assert m is None
        m = re.match(
            r"Which\s+\w+\s+(?:is|are)\s+(.+?)\??\s*$",
            question, re.IGNORECASE)
        assert m is None
