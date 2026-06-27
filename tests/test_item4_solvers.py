"""No-model tests for the item-4 SQL/sim ADT solvers: colored_objects (fully
deterministic), object_counting (model membership stubbed), penguins table parse,
and the dispatch wiring (cheap structural gates fire without a model)."""
from __future__ import annotations

from turnstyle.colored_objects import solve_colored_objects
from turnstyle.object_counting import parse_item_list, solve_object_counting, _singular
from turnstyle.penguins import parse_penguins_tables
from turnstyle.dispatch import parse, Ctx, SceneQuery, StateTracking


# ── colored_objects: deterministic, no model, no hardcoded color list ──────────

def _co(scene_q: str, options: list[str]) -> str | None:
    opts = "\n".join(f"({chr(65+i)}) {v}" for i, v in enumerate(options))
    return solve_colored_objects(f"{scene_q}\nOptions:\n{opts}")


def test_colored_what_color():
    s = ("On the desk, you see a red pen, a blue cup, and a green book. "
         "What color is the cup?")
    assert _co(s, ["red", "blue", "green"]) == "(B)"


def test_colored_leftmost_rightmost():
    s = ("On the table, there is a yellow pencil, a purple mug, and a black hat "
         "arranged in a row. What is the color of the left-most item?")
    assert _co(s, ["yellow", "purple", "black"]) == "(A)"


def test_colored_directly_left_of():
    s = ("On the floor, there is a red ball, a blue cube, and a green ring "
         "arranged in a row. What color is the object directly to the left of the green ring?")
    assert _co(s, ["red", "blue", "green"]) == "(B)"


def test_colored_count_absent_color_is_zero():
    # 'turquoise' is absent from the scene → count 0, even with no color list.
    s = ("On the desk, there are three yellow pens, two yellow cups, and one blue mug. "
         "If I remove all the pens from the desk, how many turquoise objects remain on it?")
    assert _co(s, ["zero", "one", "two", "three", "four", "five", "six"]) == "(A)"


def test_colored_remove_and_count_color():
    s = ("On the desk, there are three yellow pens, two blue pens, and one blue mug. "
         "If I remove all the pens from the desk, how many blue objects remain on it?")
    # only the blue mug (1) remains blue
    assert _co(s, ["zero", "one", "two", "three"]) == "(B)"


# ── object_counting: structural parse + injected membership ────────────────────

def test_object_counting_parse_structural():
    p = "I have a flute, a piano, four stoves, and two lamps. How many objects do I have?"
    cat, items = parse_item_list(p)
    assert cat == "objects"
    assert (4, "stoves") in items and (2, "lamps") in items


def test_object_counting_objects_is_deterministic_sum():
    # 'objects' needs no model: 1+1+4+2 = 8
    p = "I have a flute, a piano, four stoves, and two lamps. How many objects do I have?"
    assert solve_object_counting(p, model=None, tokenizer=None, device="cpu") == "8"


def test_object_counting_with_stub_scorer():
    p = ("I have a flute, a piano, three bananas, and a drum. "
         "How many musical instruments do I have?")
    instruments = {"flute", "piano", "drum"}
    scorer = lambda item, cat: _singular(item) in instruments
    assert solve_object_counting(p, model=object(), tokenizer=None, device="cpu",
                                 scorer=scorer) == "3"


# ── penguins: structural table parse (no model) ────────────────────────────────

_PENG = ("Here is a table where the first line is a header and each subsequent line "
         "is a penguin:  name, age, height (cm), weight (kg) Louis, 7, 50, 11 "
         "Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15  For example: the age "
         "of Louis is 7.  We now add a penguin to the table:\nJames, 12, 90, 12\n"
         "Which is the oldest penguin?\nOptions:\n(A) Louis\n(B) Bernard\n(C) Vincent\n"
         "(D) Gwen\n(E) James")


def test_penguins_parse_table_and_add():
    tables = parse_penguins_tables(_PENG)
    cols, rows = tables["penguins"]
    assert cols == ["name", "age", "height", "weight"]
    assert ["James", 12, 90, 12] in rows         # add mutation applied
    assert len(rows) == 5


def test_penguins_parse_delete():
    text = _PENG.replace("Which is the oldest penguin?",
                         "We then delete the penguin named Bernard from the table.\n"
                         "Which is the oldest penguin?")
    tables = parse_penguins_tables(text)
    names = [r[0] for r in tables["penguins"][1]]
    assert "Bernard" not in names


# ── dispatch wiring: cheap structural gates fire with no model ─────────────────

def test_dispatch_colored_commits_without_model():
    s = ("On the desk, you see a red pen, a blue cup, and a green book. "
         "What color is the cup?\nOptions:\n(A) red\n(B) blue\n(C) green")
    assert parse(s, Ctx()) == SceneQuery(answer="(B)")


def test_dispatch_tracking_commits_without_model():
    s = ("Alice, Bob, and Claire each have a ball. At the start of the day, Alice has "
         "a yellow ball, Bob has a white ball, and Claire has a green ball. Then, Alice "
         "and Bob swap balls. At the end of the day, Alice has\n"
         "Options:\n(A) yellow ball\n(B) white ball\n(C) green ball")
    assert parse(s, Ctx()) == StateTracking(answer="(B)")
