"""Tests for colored_objects parser and utilities — no model needed."""

from turnstyle.colored_objects import (
    _stem,
    _stem_phrase,
    parse_scene,
    colored_objects_parse_tables,
    COLORS,
)
from turnstyle.sql import SQLTurnstyle, load_into_sqlite, _try_execute, match_result_to_option


# ── Stemmer ────────────────────────────────────────────────────────────

class TestStem:
    def test_regular_plural(self):
        assert _stem("plates") == "plate"

    def test_singular(self):
        assert _stem("plate") == "plate"

    def test_sibilant_shes(self):
        assert _stem("leashes") == "leash"

    def test_sibilant_ches(self):
        assert _stem("watches") == "watch"

    def test_sibilant_xes(self):
        assert _stem("boxes") == "box"

    def test_sibilant_zes(self):
        assert _stem("buzzes") == "buzz"

    def test_sibilant_ses(self):
        assert _stem("glasses") == "glass"

    def test_short_word(self):
        assert _stem("as") == "as"

    def test_no_s(self):
        assert _stem("cat") == "cat"

    def test_necklaces(self):
        # "necklaces" -> strip 's' -> "necklace" (not "es" since no sibilant)
        assert _stem("necklaces") == "necklace"

    def test_textbooks(self):
        assert _stem("textbooks") == "textbook"


class TestStemPhrase:
    def test_multi_word(self):
        assert _stem_phrase("dog leashes") == "dog leash"

    def test_single_word(self):
        assert _stem_phrase("plates") == "plate"

    def test_pairs_of_sunglasses(self):
        assert _stem_phrase("pairs of sunglasses") == _stem_phrase("pair of sunglasses")


# ── Scene parsing ──────────────────────────────────────────────────────

class TestParseScene:
    def test_inventory_simple(self):
        text = "On the floor, there is one red ball and two blue cups."
        items, is_row = parse_scene(text)
        assert not is_row
        assert len(items) == 3
        assert items[0] == ("red", "ball")
        assert items[1] == ("blue", "cups")

    def test_inventory_with_counts(self):
        text = "On the table, there are three green pens and two red pencils."
        items, is_row = parse_scene(text)
        assert not is_row
        assert len(items) == 5

    def test_row_scene(self):
        text = "On the table, you see several objects arranged in a row: a red pen, a blue cup, a green plate."
        items, is_row = parse_scene(text)
        assert is_row
        assert len(items) == 3
        assert items[0] == ("red", "pen")
        assert items[2] == ("green", "plate")

    def test_scene_boundary(self):
        """Question text should not bleed into scene items."""
        text = (
            "On the floor, you see three green bracelets, one teal dog leash, "
            "and one green dog leash. If I remove all the teal items from the table, "
            "how many paperclips remain on it?"
        )
        items, _ = parse_scene(text)
        # Should only have 5 items (3 bracelets + 2 leashes), not extra garbage
        assert len(items) == 5
        assert all("remain" not in t for _, t in items)

    def test_multi_word_objects(self):
        text = "On the desk, there are two yellow pairs of sunglasses and one red dog leash."
        items, _ = parse_scene(text)
        assert len(items) == 3
        assert items[0][1] == "pairs of sunglasses"
        assert items[2][1] == "dog leash"


# ── parse_tables (SQLTurnstyle integration) ───────────────────────────

_colored_objects_turnstyle = SQLTurnstyle(
    None, None, None,
    parse_tables_fn=colored_objects_parse_tables,
    probe_label="colored_objects",
)


class TestParseTables:
    """Test SQLTurnstyle configured with colored_objects_parse_tables."""

    def test_inventory_scene(self):
        text = "On the floor, there is one red pen, two blue cups, and one green plate."
        tables = _colored_objects_turnstyle.parse_tables(text)
        assert tables is not None
        assert "objects" in tables
        cols, rows = tables["objects"]
        assert cols == ["color", "type"]
        assert len(rows) == 4  # 1 pen + 2 cups + 1 plate

    def test_row_scene_has_position(self):
        text = (
            "On the table, you see several objects arranged in a row: "
            "a red pen, a blue cup, and a green plate."
        )
        tables = _colored_objects_turnstyle.parse_tables(text)
        assert tables is not None
        cols, rows = tables["objects"]
        assert cols == ["position", "color", "type"]
        assert len(rows) == 3
        assert rows[0] == (1, "red", "pen")
        assert rows[2] == (3, "green", "plate")

    def test_empty_scene_returns_none(self):
        tables = _colored_objects_turnstyle.parse_tables("No objects here.")
        assert tables is None

    def test_probe_label(self):
        assert _colored_objects_turnstyle.probe_label == "colored_objects"


class TestSQLSolvesRegexQuestions:
    """Verify that SQL produces the correct answer for known BBH questions.

    Uses SQLTurnstyle configured with colored_objects_parse_tables, runs SQL
    directly (no model needed), and confirms the answer matches expected.
    """

    def _parse_and_query(self, text, sql):
        """Parse scene via configured SQLTurnstyle → SQLite → execute → match."""
        tables = _colored_objects_turnstyle.parse_tables(text)
        assert tables is not None
        conn = load_into_sqlite(tables)
        result, err = _try_execute(conn, sql)
        conn.close()
        assert err is None, f"SQL error: {err}"
        opts_text = text[text.index("Options:"):]
        options = {}
        import re
        for m in re.finditer(r"\(([A-R])\)\s+(\w+)", opts_text):
            options[m.group(1)] = m.group(2).lower()
        return match_result_to_option(result, options)

    def test_count_after_remove(self):
        text = (
            "On the floor, there is one red pen, two blue pens, and one red cup. "
            "If I remove all the red items from the floor, how many pens remain on it?\n"
            "Options:\n(A) zero\n(B) one\n(C) two\n(D) three"
        )
        sql_answer = self._parse_and_query(
            text,
            "SELECT COUNT(*) FROM objects WHERE color != 'red' AND type = 'pen'")
        assert sql_answer == "(C)"

    def test_count_neither(self):
        text = (
            "On the floor, you see a green bracelet, a black spinner, "
            "a red pen, and a blue cup. "
            "How many objects are neither black nor blue?\n"
            "Options:\n(A) zero\n(B) one\n(C) two\n(D) three"
        )
        sql_answer = self._parse_and_query(
            text,
            "SELECT COUNT(*) FROM objects WHERE color NOT IN ('black', 'blue')")
        assert sql_answer == "(C)"

    def test_left_of(self):
        text = (
            "On the table, you see several objects arranged in a row: "
            "a red pen, a blue cup, and a green plate. "
            "What color is the object directly to the left of the cup?\n"
            "Options:\n(A) red\n(B) blue\n(C) green"
        )
        sql_answer = self._parse_and_query(
            text,
            "SELECT color FROM objects WHERE position = "
            "(SELECT position - 1 FROM objects WHERE type = 'cup')")
        assert sql_answer == "(A)"

    def test_leftmost(self):
        text = (
            "On the table, you see several objects arranged in a row: "
            "a red pen, a blue cup, and a green plate. "
            "What is the color of the leftmost object?\n"
            "Options:\n(A) red\n(B) blue\n(C) green"
        )
        sql_answer = self._parse_and_query(
            text,
            "SELECT color FROM objects WHERE position = "
            "(SELECT MIN(position) FROM objects)")
        assert sql_answer == "(A)"

    def test_what_color(self):
        text = (
            "On the floor, you see a red pen and a blue cup. "
            "What color is the pen?\n"
            "Options:\n(A) red\n(B) blue"
        )
        sql_answer = self._parse_and_query(
            text,
            "SELECT color FROM objects WHERE type = 'pen'")
        assert sql_answer == "(A)"

    def test_is_color(self):
        text = (
            "On the floor, you see a red pen and a blue cup. "
            "Is the pen red?\n"
            "Options:\n(A) yes\n(B) no"
        )
        sql_answer = self._parse_and_query(
            text,
            "SELECT CASE WHEN color = 'red' THEN 'yes' ELSE 'no' END "
            "FROM objects WHERE type = 'pen'")
        assert sql_answer == "(A)"

    def test_furthest_from(self):
        text = (
            "On the table, you see several objects arranged in a row: "
            "a red pen, a blue cup, a green plate, and a yellow ball. "
            "What is the color of the object furthest from the cup?\n"
            "Options:\n(A) red\n(B) yellow\n(C) green"
        )
        sql_answer = self._parse_and_query(
            text,
            "SELECT color FROM objects ORDER BY ABS(position - "
            "(SELECT position FROM objects WHERE type = 'cup')) DESC LIMIT 1")
        assert sql_answer == "(B)"
