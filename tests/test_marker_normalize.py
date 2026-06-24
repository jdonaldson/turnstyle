"""No-model tests for dispatch.normalize_option_markers — the marker-agnostic intake
the perturbation harness motivated. Canonicalizes any option-marker style to "(A)".

Round-trips the perturbation harness's own variants: a MarkerRestyle variant fed
through the normalizer must reproduce the canonical prompt (the fix closes exactly
the surface gap the harness opened).
"""
import pytest

from turnstyle.dispatch import normalize_option_markers
from turnstyle.perturb import MarkerRestyle

CANON = (
    "Which statement is sarcastic?\n"
    "Options:\n"
    "(A) He is a generous person\n"
    "(B) He is a terrible person"
)
THREE = (
    "On a branch there are three birds.\n"
    "Options:\n"
    "(A) The blue jay is second\n"
    "(B) The quail is second\n"
    "(C) The falcon is second"
)


def test_canonical_is_idempotent():
    assert normalize_option_markers(CANON) == CANON


def test_dotted_to_canonical():
    dotted = CANON.replace("(A)", "A.").replace("(B)", "B.")
    assert normalize_option_markers(dotted) == CANON


def test_bracket_to_canonical():
    bracket = CANON.replace("(A)", "[A]").replace("(B)", "[B]")
    assert normalize_option_markers(bracket) == CANON


def test_colon_to_canonical():
    colon = CANON.replace("(A)", "A:").replace("(B)", "B:")
    assert normalize_option_markers(colon) == CANON


def test_numeric_remaps_to_letters():
    numeric = THREE.replace("(A)", "1.").replace("(B)", "2.").replace("(C)", "3.")
    assert normalize_option_markers(numeric) == THREE


def test_lowercase_letters_uppercased():
    lower = CANON.replace("(A)", "(a)").replace("(B)", "(b)")
    assert normalize_option_markers(lower) == CANON


def test_no_options_unchanged():
    txt = "What is 3 + 4 * 2? Answer with a number."
    assert normalize_option_markers(txt) == txt


def test_stray_marker_in_prose_not_rewritten():
    # a single "(B)"-like token in prose is not a >=2 consecutive block → untouched
    txt = "See clause (b) for details. Then compute the total."
    assert normalize_option_markers(txt) == txt


def test_does_not_touch_body_enumeration_starting_midway():
    # a non-A-start run is not a valid option label sequence → left alone
    txt = "Step C. do this\nStep D. do that"  # not parsed as markers (prefix 'Step ')
    assert normalize_option_markers(txt) == txt


def test_roundtrips_harness_marker_variants():
    # the fix must close exactly what the harness opens: every MarkerRestyle variant,
    # normalized, equals the canonical prompt.
    for style in ("dotted", "bracket"):
        v = MarkerRestyle(style).apply(THREE, "(A)")
        assert v is not None
        assert normalize_option_markers(v.input) == THREE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
