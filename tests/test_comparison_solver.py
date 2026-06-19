"""ComparisonSolver — frames + symbolic solver + pole resolution. No model needed.

Offline path uses the regex lexicon fallback; the probe path is exercised by
supplying an explicit pole_map (what the polarity probe would return), so these
tests stay model-free while covering the same code path."""

from turnstyle.comparison_solver import (
    HIGH, LOW, make_pole_map, normalize_phrase, solve_comparison,
)


def _q(body, options):
    return body + "\nOptions:\n" + "\n".join(options)


def test_normalize_keeps_surface_form_no_lemmatization():
    # NO -er/-est stripping (English morphology) — the probe classifies the
    # comparative/superlative surface form directly. Only function words drop.
    assert normalize_phrase("newer") == ("newer", 1)
    assert normalize_phrase("oldest") == ("oldest", 1)
    assert normalize_phrase("more expensive") == ("expensive", 1)
    assert normalize_phrase("less expensive") == ("expensive", -1)
    assert normalize_phrase("the second-cheapest") == ("cheapest", 1)
    assert normalize_phrase("least valuable") == ("valuable", -1)


def test_spatial_structural_no_pole_needed():
    p = _q("There are three birds: a wren, a robin, and a finch. "
           "The robin is to the right of the wren. "
           "The finch is to the right of the robin.\nWhich is the leftmost?",
           ["(A) wren", "(B) robin", "(C) finch"])
    assert solve_comparison(p) == "(A)"        # wren leftmost, no adjective at all


def test_finished_rank_structural():
    p = _q("In a race there are three runners: Amy, Bob, and Cal. "
           "Amy finished above Bob. Bob finished above Cal.\nWho finished last?",
           ["(A) Amy", "(B) Bob", "(C) Cal"])
    assert solve_comparison(p) == "(C)"


def test_age_axis_via_regex_fallback():
    p = _q("There are three cars: a sedan, a coupe, and a van. "
           "The sedan is newer than the coupe. The coupe is newer than the van."
           "\nWhich is the oldest?",
           ["(A) sedan", "(B) coupe", "(C) van"])
    assert solve_comparison(p) == "(C)"        # van oldest (low end of new axis)


def test_price_second_most_expensive():
    p = _q("A stand sells three fruits: peaches, mangoes, and apples. "
           "The peaches are more expensive than the apples. "
           "The mangoes are the cheapest.\nWhich is the second-most expensive?",
           ["(A) peaches", "(B) mangoes", "(C) apples"])
    # order low→high: mangoes(cheapest) < apples < peaches → 2nd-most-expensive = apples
    assert solve_comparison(p) == "(C)"


def test_explicit_pole_map_is_the_probe_path():
    # what word_poles(probe) would return — no regex involved
    p = _q("There are three gems: a ruby, an opal, and a jade. "
           "The ruby is brighter than the opal. The opal is brighter than the jade."
           "\nWhich is the dullest?",
           ["(A) ruby", "(B) opal", "(C) jade"])
    # 'bright'/'dull' aren't in the regex lexicon → offline path can't solve it
    assert solve_comparison(p) is None
    # but a probe that knows the surface forms resolves it (no lemmatization)
    pm = {"brighter": HIGH, "dullest": LOW}
    assert solve_comparison(p, pole_map=pm) == "(C)"


def test_make_pole_map_regex_fallback():
    p = _q("There are three cars: a sedan, a coupe, and a van. "
           "The sedan is newer than the coupe.\nWhich is the oldest?",
           ["(A) sedan", "(B) coupe", "(C) van"])
    pm = make_pole_map(p)            # no model → regex lexicon
    assert pm["newer"] == HIGH       # surface form, no lemmatization


def test_unsolvable_returns_none():
    assert solve_comparison("totally unrelated prose with no items") is None
