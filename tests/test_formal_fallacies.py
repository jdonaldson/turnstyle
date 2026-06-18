"""Tests for formal_fallacies parser and solver — no model needed."""

from turnstyle.formal_fallacies import (
    Pred, Not, And, Or, Implies, ForAll, Exists,
    evaluate, check_validity, collect_predicates, collect_individuals,
    parse_pred_atom, parse_pred_expr, parse_sentence,
    probe_parse_sentence, classify_pattern, QUANTIFIER_WORDS,
    extract_and_split, solve_formal_fallacy,
    FormalFallaciesTurnstyle,
)


# ── FOL evaluation ──────────────────────────────────────────────────

class TestEvaluate:
    """Test the model-theoretic evaluation function."""

    def _model(self, **kwargs):
        """Build a model with domain {0, 1} and predicate extensions."""
        m = {"__domain__": [0, 1]}
        m.update(kwargs)
        return m

    def test_pred_true(self):
        model = self._model(tall={0, 1})
        assert evaluate(Pred("tall", 0), model, {})

    def test_pred_false(self):
        model = self._model(tall={0})
        assert not evaluate(Pred("tall", 1), model, {})

    def test_not(self):
        model = self._model(tall={0})
        assert evaluate(Not(Pred("tall", 1)), model, {})

    def test_and(self):
        model = self._model(tall={0, 1}, smart={0})
        assert evaluate(And(Pred("tall", 0), Pred("smart", 0)), model, {})
        assert not evaluate(And(Pred("tall", 1), Pred("smart", 1)), model, {})

    def test_or(self):
        model = self._model(tall={0}, smart={1})
        assert evaluate(Or(Pred("tall", 0), Pred("smart", 0)), model, {})
        assert evaluate(Or(Pred("tall", 1), Pred("smart", 1)), model, {})
        assert not evaluate(Or(Pred("tall", 1), Pred("smart", 0)), model, {})

    def test_implies(self):
        model = self._model(tall={0}, smart={0, 1})
        # tall(0) → smart(0): T → T = T
        assert evaluate(Implies(Pred("tall", 0), Pred("smart", 0)), model, {})
        # tall(1) → smart(1): F → T = T
        assert evaluate(Implies(Pred("tall", 1), Pred("smart", 1)), model, {})

    def test_forall(self):
        model = self._model(tall={0, 1})
        assert evaluate(ForAll("x", Pred("tall", "x")), model, {})
        model2 = self._model(tall={0})
        assert not evaluate(ForAll("x", Pred("tall", "x")), model2, {})

    def test_exists(self):
        model = self._model(tall={0})
        assert evaluate(Exists("x", Pred("tall", "x")), model, {})
        model2 = self._model(tall=set())
        assert not evaluate(Exists("x", Pred("tall", "x")), model2, {})

    def test_variable_binding(self):
        # ∀x: tall(x) → smart(x), model where tall={0,1}, smart={0}
        model = self._model(tall={0, 1}, smart={0})
        f = ForAll("x", Implies(Pred("tall", "x"), Pred("smart", "x")))
        assert not evaluate(f, model, {})  # x=1 is tall but not smart


# ── Validity checker ────────────────────────────────────────────────

class TestCheckValidity:
    def test_simple_valid_syllogism(self):
        # All A are B, All B are C ⊨ All A are C
        p1 = ForAll("x", Implies(Pred("a", "x"), Pred("b", "x")))
        p2 = ForAll("x", Implies(Pred("b", "x"), Pred("c", "x")))
        conc = ForAll("x", Implies(Pred("a", "x"), Pred("c", "x")))
        assert check_validity([p1, p2], conc)

    def test_simple_invalid_syllogism(self):
        # All A are B, All C are B ⊭ All A are C
        p1 = ForAll("x", Implies(Pred("a", "x"), Pred("b", "x")))
        p2 = ForAll("x", Implies(Pred("c", "x"), Pred("b", "x")))
        conc = ForAll("x", Implies(Pred("a", "x"), Pred("c", "x")))
        assert not check_validity([p1, p2], conc)

    def test_modus_ponens_valid(self):
        # ∀x: A(x) → B(x), A(0) ⊨ B(0)
        p1 = ForAll("x", Implies(Pred("a", "x"), Pred("b", "x")))
        p2 = Pred("a", 0)
        conc = Pred("b", 0)
        assert check_validity([p1, p2], conc)

    def test_affirming_consequent_invalid(self):
        # ∀x: A(x) → B(x), B(0) ⊭ A(0)
        p1 = ForAll("x", Implies(Pred("a", "x"), Pred("b", "x")))
        p2 = Pred("b", 0)
        conc = Pred("a", 0)
        assert not check_validity([p1, p2], conc)

    def test_existential_valid(self):
        # ∃x: A(x) ∧ B(x), ∀x: B(x) → C(x) ⊨ ∃x: A(x) ∧ C(x)
        p1 = Exists("x", And(Pred("a", "x"), Pred("b", "x")))
        p2 = ForAll("x", Implies(Pred("b", "x"), Pred("c", "x")))
        conc = Exists("x", And(Pred("a", "x"), Pred("c", "x")))
        assert check_validity([p1, p2], conc)

    def test_negation_valid(self):
        # ∀x: A(x) → ¬B(x), A(0) ⊨ ¬B(0)
        p1 = ForAll("x", Implies(Pred("a", "x"), Not(Pred("b", "x"))))
        p2 = Pred("a", 0)
        conc = Not(Pred("b", 0))
        assert check_validity([p1, p2], conc)


# ── Predicate expression parser ────────────────────────────────────

class TestParsePredExpr:
    def test_atomic(self):
        f, preds = parse_pred_atom("friend of Tom", "x")
        assert f is not None
        assert f.name == "friend of tom"

    def test_not(self):
        f, _ = parse_pred_expr("not a friend of Tom", "x")
        assert isinstance(f, Not)

    def test_both_and(self):
        f, _ = parse_pred_expr(
            "both a cousin of Tom and a friend of Mary", "x",
        )
        assert isinstance(f, And)

    def test_neither_nor(self):
        f, _ = parse_pred_expr(
            "neither a cousin of Tom nor a friend of Mary", "x",
        )
        assert isinstance(f, And)
        assert isinstance(f.left, Not)
        assert isinstance(f.right, Not)

    def test_or(self):
        f, _ = parse_pred_expr(
            "a cousin of Tom or a friend of Mary", "x",
        )
        assert isinstance(f, Or)


# ── Sentence-level parser ──────────────────────────────────────────

class TestParseSentence:
    def test_every_is(self):
        f, _ = parse_sentence("Every cousin of Tom is a friend of Mary")
        assert isinstance(f, ForAll)
        assert isinstance(f.body, Implies)

    def test_no_is(self):
        f, _ = parse_sentence("No cousin of Tom is a friend of Mary")
        assert isinstance(f, ForAll)
        assert isinstance(f.body, Implies)
        assert isinstance(f.body.right, Not)

    def test_whoever_is(self):
        f, _ = parse_sentence(
            "Whoever is a cousin of Tom is also a friend of Mary",
        )
        assert isinstance(f, ForAll)

    def test_being_necessary(self):
        f, _ = parse_sentence(
            "Being a cousin of Tom is necessary for being a friend of Mary",
        )
        assert isinstance(f, ForAll)
        # necessary: Y → X, so body is Implies(friend_of_mary, cousin_of_tom)
        assert isinstance(f.body, Implies)

    def test_being_sufficient(self):
        f, _ = parse_sentence(
            "Being a cousin of Tom is sufficient for being a friend of Mary",
        )
        assert isinstance(f, ForAll)
        # sufficient: X → Y, so body is Implies(cousin_of_tom, friend_of_mary)
        assert isinstance(f.body, Implies)

    def test_individual_assertion(self):
        f, _ = parse_sentence("Tom is a cousin of Mary")
        assert isinstance(f, Pred)
        assert f.var == "tom"

    def test_individual_negation(self):
        f, _ = parse_sentence("Tom is not a cousin of Mary")
        assert isinstance(f, Not)
        assert isinstance(f.inner, Pred)

    def test_some_is(self):
        f, _ = parse_sentence("Some cousin of Tom is a friend of Mary")
        assert isinstance(f, Exists)

    def test_it_is_not_the_case(self):
        f, _ = parse_sentence(
            "It is not the case that Tom is a friend of Mary",
        )
        assert isinstance(f, Not)

    def test_everyone_who(self):
        f, _ = parse_sentence(
            "Everyone who is a cousin of Tom is a friend of Mary",
        )
        assert isinstance(f, ForAll)

    def test_not_every(self):
        f, _ = parse_sentence(
            "Not every cousin of Tom is a friend of Mary",
        )
        assert isinstance(f, Not)
        assert isinstance(f.inner, ForAll)

    def test_there_exists(self):
        f, _ = parse_sentence(
            "There exists a cousin of Tom who is a friend of Mary",
        )
        assert isinstance(f, Exists)

    def test_nobody_neither_nor(self):
        f, _ = parse_sentence(
            "Nobody is neither a cousin of Tom nor a friend of Mary",
        )
        assert isinstance(f, ForAll)
        assert isinstance(f.body, Or)


# ── Argument extraction ─────────────────────────────────────────────

class TestExtractAndSplit:
    def test_basic_argument(self):
        text = (
            '"Every cousin of Tom is a friend of Mary. '
            'No friend of Mary is a classmate of Joe. '
            'Therefore, no cousin of Tom is a classmate of Joe."'
        )
        premises, conclusion = extract_and_split(text)
        assert premises is not None
        assert conclusion is not None
        assert len(premises) == 2
        assert "cousin of tom" in conclusion.lower()

    def test_preamble_stripped(self):
        text = '''"Consider the following argument: Every cat is a mammal. Therefore, every cat is a mammal."'''
        premises, conclusion = extract_and_split(text)
        assert premises is not None
        assert conclusion is not None

    def test_no_quotes_returns_none(self):
        premises, conclusion = extract_and_split("No quotes here")
        assert premises is None
        assert conclusion is None

    def test_no_conclusion_returns_none(self):
        text = '''"Every cat is a mammal. Every mammal breathes."'''
        premises, conclusion = extract_and_split(text)
        assert conclusion is None


# ── End-to-end solver ───────────────────────────────────────────────

class TestSolveFormalFallacy:
    def test_valid_syllogism(self):
        text = (
            '"Every schoolmate of Tom is a cousin of Mary. '
            'Every cousin of Mary is a friend of Joe. '
            'Therefore, every schoolmate of Tom is a friend of Joe."'
        )
        assert solve_formal_fallacy(text) == "valid"

    def test_invalid_syllogism(self):
        text = (
            '"Every schoolmate of Tom is a cousin of Mary. '
            'Every friend of Joe is a cousin of Mary. '
            'Therefore, every schoolmate of Tom is a friend of Joe."'
        )
        assert solve_formal_fallacy(text) == "invalid"

    def test_necessary_sufficient(self):
        text = (
            '"Being a cousin of Tom is necessary for being a friend of Mary. '
            'Joe is a friend of Mary. '
            'Therefore, Joe is a cousin of Tom."'
        )
        assert solve_formal_fallacy(text) == "valid"

    def test_parse_failure_returns_none(self):
        assert solve_formal_fallacy("Not a valid input at all") is None

    def test_negation_valid(self):
        text = (
            '"No cousin of Tom is a friend of Mary. '
            'Joe is a cousin of Tom. '
            'Therefore, Joe is not a friend of Mary."'
        )
        assert solve_formal_fallacy(text) == "valid"

    def test_negation_invalid(self):
        text = (
            '"No cousin of Tom is a friend of Mary. '
            'Joe is not a cousin of Tom. '
            'Therefore, Joe is a friend of Mary."'
        )
        assert solve_formal_fallacy(text) == "invalid"


# ── Turnstyle interface ─────────────────────────────────────────────

class TestFormalFallaciesTurnstyle:
    def test_parse_valid(self):
        t = FormalFallaciesTurnstyle.__new__(FormalFallaciesTurnstyle)
        text = (
            '"Every cousin of Tom is a friend of Mary. '
            'Every friend of Mary is a classmate of Joe. '
            'Therefore, every cousin of Tom is a classmate of Joe."'
        )
        result = t.parse(text)
        assert result is not None
        summary, answer = result
        assert answer == "valid"
        assert "2 premises" in summary

    def test_parse_invalid(self):
        t = FormalFallaciesTurnstyle.__new__(FormalFallaciesTurnstyle)
        text = (
            '"Every cousin of Tom is a friend of Mary. '
            'Every classmate of Joe is a friend of Mary. '
            'Therefore, every cousin of Tom is a classmate of Joe."'
        )
        result = t.parse(text)
        assert result is not None
        _, answer = result
        assert answer == "invalid"

    def test_parse_failure(self):
        t = FormalFallaciesTurnstyle.__new__(FormalFallaciesTurnstyle)
        result = t.parse("What is 2+2?")
        assert result is None

    def test_probe_label(self):
        assert FormalFallaciesTurnstyle.probe_label == "formal_fallacies"


# ── Helper utilities ────────────────────────────────────────────────

class TestHelpers:
    def test_collect_predicates(self):
        f = ForAll("x", Implies(Pred("tall", "x"), Pred("smart", "x")))
        preds = collect_predicates([f])
        assert preds == {"tall", "smart"}

    def test_collect_individuals(self):
        f = And(Pred("tall", "tom"), Pred("smart", "mary"))
        indivs = collect_individuals([f])
        assert indivs == {"tom", "mary"}

    def test_collect_individuals_ignores_x(self):
        f = ForAll("x", Pred("tall", "x"))
        indivs = collect_individuals([f])
        assert indivs == set()


# ── Pattern classifier ─────────────────────────────────────────────

class TestClassifyPattern:
    def test_every_is(self):
        assert classify_pattern("Every cousin of Tom is a friend of Mary") == "every_is"

    def test_every_who_is(self):
        assert classify_pattern(
            "Every cousin of Tom who is a friend of Mary is a classmate of Joe"
        ) == "every_who_is"

    def test_no_is(self):
        assert classify_pattern("No cousin of Tom is a friend of Mary") == "no_is"

    def test_whoever_is(self):
        assert classify_pattern(
            "Whoever is a cousin of Tom is also a friend of Mary"
        ) == "whoever_is"

    def test_being_necessary(self):
        assert classify_pattern(
            "Being a cousin of Tom is necessary for being a friend of Mary"
        ) == "being_necessary"

    def test_being_sufficient(self):
        assert classify_pattern(
            "Being a cousin of Tom is sufficient for being a friend of Mary"
        ) == "being_sufficient"

    def test_name_is(self):
        assert classify_pattern("Tom is a cousin of Mary") == "name_is"

    def test_name_is_not(self):
        assert classify_pattern("Tom is not a cousin of Mary") == "name_is_not"

    def test_quantifier_word_not_name(self):
        # "Every..." should never classify as name_is
        assert classify_pattern("Every cousin of Tom is a friend of Mary") != "name_is"
        assert classify_pattern("No cousin of Tom is a friend of Mary") != "name_is"

    def test_it_is_not(self):
        assert classify_pattern(
            "It is not the case that Tom is a friend of Mary"
        ) == "it_is_not"

    def test_some_is(self):
        assert classify_pattern("Some cousin of Tom is a friend of Mary") == "some_is"

    def test_there_exists(self):
        assert classify_pattern(
            "There exists a cousin of Tom who is a friend of Mary"
        ) == "there_exists"


# ── Probe-dispatched parser ────────────────────────────────────────

class TestProbeParsesSentence:
    def test_every_is_with_label(self):
        f, _ = probe_parse_sentence(
            "Every cousin of Tom is a friend of Mary", "every_is",
        )
        assert isinstance(f, ForAll)
        assert isinstance(f.body, Implies)

    def test_no_is_with_label(self):
        f, _ = probe_parse_sentence(
            "No cousin of Tom is a friend of Mary", "no_is",
        )
        assert isinstance(f, ForAll)
        assert isinstance(f.body, Implies)
        assert isinstance(f.body.right, Not)

    def test_name_is_with_label(self):
        f, _ = probe_parse_sentence("Tom is a cousin of Mary", "name_is")
        assert isinstance(f, Pred)
        assert f.var == "tom"

    def test_guard_rejects_name_is_for_quantified(self):
        """Guard prevents catch-all name_is from corrupting quantified sentences."""
        f, _ = probe_parse_sentence(
            "Every cousin of Tom is a friend of Mary", "name_is",
        )
        assert f is None

    def test_guard_rejects_name_is_not_for_quantified(self):
        f, _ = probe_parse_sentence(
            "No cousin of Tom is a friend of Mary", "name_is_not",
        )
        assert f is None

    def test_guard_allows_name_is_for_actual_names(self):
        f, _ = probe_parse_sentence("Tom is a cousin of Mary", "name_is")
        assert f is not None

    def test_none_label_uses_regex_classifier(self):
        """Without a label, falls back to regex classify_pattern."""
        f, _ = probe_parse_sentence("Every cousin of Tom is a friend of Mary")
        assert isinstance(f, ForAll)

    def test_wrong_label_returns_none(self):
        """Wrong label that doesn't match should return None."""
        f, _ = probe_parse_sentence(
            "Every cousin of Tom is a friend of Mary", "being_necessary",
        )
        assert f is None

    def test_matches_regex_parser(self):
        """Probe-dispatched parser with oracle labels matches regex parser."""
        sentences = [
            "Every cousin of Tom is a friend of Mary",
            "No cousin of Tom is a friend of Mary",
            "Tom is a cousin of Mary",
            "Tom is not a cousin of Mary",
            "Being a cousin of Tom is necessary for being a friend of Mary",
            "Whoever is a cousin of Tom is also a friend of Mary",
            "Some cousin of Tom is a friend of Mary",
            "It is not the case that Tom is a friend of Mary",
        ]
        for sent in sentences:
            regex_f, _ = parse_sentence(sent)
            probe_f, _ = probe_parse_sentence(sent)
            assert repr(regex_f) == repr(probe_f), (
                f"Mismatch on: {sent}"
            )

    def test_solve_with_classifier(self):
        """solve_formal_fallacy accepts a custom classifier."""
        text = (
            '"Every schoolmate of Tom is a cousin of Mary. '
            'Every cousin of Mary is a friend of Joe. '
            'Therefore, every schoolmate of Tom is a friend of Joe."'
        )
        # Default classifier (regex)
        assert solve_formal_fallacy(text) == "valid"
        # Explicit classifier
        assert solve_formal_fallacy(text, classifier=classify_pattern) == "valid"

    def test_solve_with_wrong_classifier_falls_back(self):
        """If classifier returns wrong label, fallback to regex cascade."""
        text = (
            '"Every schoolmate of Tom is a cousin of Mary. '
            'Every cousin of Mary is a friend of Joe. '
            'Therefore, every schoolmate of Tom is a friend of Joe."'
        )
        # Classifier that always returns a wrong label
        def bad_classifier(sent):
            return "being_necessary"
        # Should still work via parse_sentence fallback
        assert solve_formal_fallacy(text, classifier=bad_classifier) == "valid"
