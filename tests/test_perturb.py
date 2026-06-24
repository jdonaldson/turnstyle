"""No-model tests for the invariance/perturbation harness.

Pins: clean option parsing, marker restyle isolates the glyph (paren = identity),
option reorder is ANSWER-PRESERVING (target recomputed to the moved content), variants
are deterministic under seed, and free-form prompts are correctly left alone.
"""
import random

import pytest

from turnstyle.perturb import (
    MarkerRestyle,
    OptionReorder,
    Variant,
    default_perturbations,
    parse_options,
)

MC = (
    "On a branch there are three birds: a blue jay, a quail, and a falcon. "
    "The falcon is to the right of the blue jay.\n"
    "Options:\n"
    "(A) The blue jay is second\n"
    "(B) The quail is second\n"
    "(C) The falcon is second"
)
FREEFORM = "What is 3 + 4 * 2?"


def _rng():
    return random.Random(0)


# ── parse_options ──────────────────────────────────────────────────────────
def test_parse_clean_block():
    head, letters, raw = parse_options(MC)
    assert letters == ["A", "B", "C"]
    assert head.endswith("Options:\n")
    assert [c.strip() for c in raw] == [
        "The blue jay is second", "The quail is second", "The falcon is second"]


def test_parse_rejects_freeform():
    assert parse_options(FREEFORM) is None


def test_parse_rejects_nonconsecutive():
    # a stray "(B)" in prose with no "(A)" is not an option block
    assert parse_options("foo (B) bar (D) baz") is None


# ── MarkerRestyle ──────────────────────────────────────────────────────────
def test_marker_paren_is_identity_returns_none():
    # paren on already-paren options is a no-op → not a variant worth scoring
    assert MarkerRestyle("paren").apply(MC, "(A)") is None


def test_marker_dotted_changes_only_glyph():
    v = MarkerRestyle("dotted").apply(MC, "(A)")
    assert isinstance(v, Variant) and v.name == "marker:dotted"
    assert "A. The blue jay is second" in v.input
    assert "(A)" not in v.input and "(B)" not in v.input
    assert v.target == "(A)"                    # target unchanged
    # body text and option contents preserved
    assert "blue jay is second" in v.input and "Options:" in v.input


def test_marker_bracket():
    v = MarkerRestyle("bracket").apply(MC, "(A)")
    assert "[A] The blue jay is second" in v.input


def test_marker_skips_freeform():
    assert MarkerRestyle("dotted").apply(FREEFORM, "10") is None


# ── OptionReorder (the answer-preservation invariant) ──────────────────────
def test_reorder_is_answer_preserving():
    v = OptionReorder().apply(MC, "(A)", _rng())
    assert v is not None and v.name == "reorder"
    # the letter named by the new target must hold the originally-correct content
    _, letters, raw = parse_options(v.input)
    new_letter = v.target.strip("()")
    moved = dict(zip(letters, [c.strip() for c in raw]))[new_letter]
    assert moved == "The blue jay is second"   # option A's original content


def test_reorder_actually_reorders():
    # with seed 0 and 3 options the shuffle is non-identity → target moves off A
    v = OptionReorder().apply(MC, "(A)", _rng())
    assert v.target != "(A)" or v.input != MC


def test_reorder_skips_non_letter_target():
    # a free-form / integer target can't be remapped answer-preservingly
    assert OptionReorder().apply(MC, "second") is None


def test_reorder_deterministic_under_seed():
    a = OptionReorder().apply(MC, "(A)", random.Random(7))
    b = OptionReorder().apply(MC, "(A)", random.Random(7))
    assert a.input == b.input and a.target == b.target


# ── default set ────────────────────────────────────────────────────────────
def test_default_set_shape():
    ps = default_perturbations()
    names = [p.name for p in ps]
    assert names == ["marker:dotted", "marker:bracket", "reorder"]
    # every default applies to a real MC example and yields a valid variant
    rng = _rng()
    for p in ps:
        v = p.apply(MC, "(A)", rng)
        assert isinstance(v, Variant)


def test_default_set_all_skip_freeform():
    rng = _rng()
    for p in default_perturbations():
        assert p.apply(FREEFORM, "10", rng) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
