"""Dispatch Ordering path wired to comparison_solver. No model needed.

Covers the regex-fallback route and the pole_cache mechanism that the polarity
probe populates (probe stays None; a pre-seeded cache stands in for it)."""

from turnstyle.dispatch import Answer, Ctx, Ordering, parse, run


def _q(body, options):
    return body + "\nOptions:\n" + "\n".join(options)


_AGE = _q("There are three cars: a sedan, a coupe, and a van. "
          "The sedan is newer than the coupe. The coupe is newer than the van."
          "\nWhich is the oldest?",
          ["(A) sedan", "(B) coupe", "(C) van"])

# bright/dull are NOT in the regex lexicon → only resolvable via probe/cache
_SHINE = _q("There are three gems: a ruby, an opal, and a jade. "
            "The ruby is brighter than the opal. The opal is brighter than the jade."
            "\nWhich is the dullest?",
            ["(A) ruby", "(B) opal", "(C) jade"])


def test_ordering_routes_via_regex_fallback():
    task = parse(_AGE, Ctx())
    assert isinstance(task, Ordering)
    assert task.answer == "(C)"


def test_ordering_run_returns_answer():
    ans = run(_AGE, Ctx())
    assert isinstance(ans, Answer)
    assert ans.text == "(C)"
    assert ans.source == "logical_deduction"


def test_out_of_lexicon_needs_pole_source():
    # no probe, no cache → 'bright'/'dull' unresolved → not an Ordering
    task = parse(_SHINE, Ctx())
    assert not isinstance(task, Ordering)


def test_pole_cache_drives_ordering():
    # the polarity probe would populate pole_cache with these surface forms
    ctx = Ctx(pole_cache={"brighter": 1, "dullest": -1})
    task = parse(_SHINE, ctx)
    assert isinstance(task, Ordering)
    assert task.answer == "(C)"


def test_pole_cache_is_reused_across_prompts():
    # a fresh cache gets populated by the regex fallback and persists
    ctx = Ctx(pole_cache={})
    parse(_AGE, ctx)
    assert ctx.pole_cache.get("newer") == 1     # surface form resolved + memoized
