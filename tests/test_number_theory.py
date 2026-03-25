"""Tests for number theory parsers — no model needed."""

from turnstyle.number_theory import parse_number_theory


# ── GCD ──────────────────────────────────────────────────────────────


class TestGCD:
    def test_basic(self):
        result = parse_number_theory("GCD of 24 and 36")
        assert result is not None
        a, b, op, answer, expr = result
        assert op == 'gcd'
        assert answer == 12

    def test_functional_notation(self):
        result = parse_number_theory("gcd(24, 36)")
        assert result is not None
        _, _, op, answer, _ = result
        assert op == 'gcd'
        assert answer == 12

    def test_natural_language(self):
        result = parse_number_theory(
            "What is the greatest common divisor of 48 and 18?")
        assert result is not None
        _, _, op, answer, _ = result
        assert op == 'gcd'
        assert answer == 6

    def test_greatest_common_factor(self):
        result = parse_number_theory(
            "Find the greatest common factor of 100 and 75")
        assert result is not None
        _, _, op, answer, _ = result
        assert op == 'gcd'
        assert answer == 25

    def test_coprime(self):
        result = parse_number_theory("GCD of 7 and 13")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 1

    def test_zero_operand(self):
        result = parse_number_theory("GCD of 0 and 5")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 5

    def test_large_numbers(self):
        result = parse_number_theory("GCD of 123456 and 789012")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 12


# ── LCM ──────────────────────────────────────────────────────────────


class TestLCM:
    def test_basic(self):
        result = parse_number_theory("LCM of 4 and 6")
        assert result is not None
        a, b, op, answer, expr = result
        assert op == 'lcm'
        assert answer == 12

    def test_functional_notation(self):
        result = parse_number_theory("lcm(4, 6)")
        assert result is not None
        _, _, op, answer, _ = result
        assert op == 'lcm'
        assert answer == 12

    def test_natural_language(self):
        result = parse_number_theory(
            "What is the least common multiple of 12 and 8?")
        assert result is not None
        _, _, op, answer, _ = result
        assert op == 'lcm'
        assert answer == 24

    def test_coprime(self):
        """LCM of coprimes is their product."""
        result = parse_number_theory("LCM of 7 and 13")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 91

    def test_same_number(self):
        """LCM of n and n is n."""
        result = parse_number_theory("LCM of 5 and 5")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 5


# ── Fraction simplification ──────────────────────────────────────────


class TestFractionSimplify:
    def test_basic(self):
        result = parse_number_theory("Simplify 6/4")
        assert result is not None
        a, b, op, answer, expr = result
        assert op == 'simplify'
        assert answer == "3/2"

    def test_reduce(self):
        result = parse_number_theory("Reduce 18/12")
        assert result is not None
        _, _, op, answer, _ = result
        assert op == 'simplify'
        assert answer == "3/2"

    def test_already_reduced(self):
        result = parse_number_theory("Simplify 3/7")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == "3/7"

    def test_natural_language(self):
        result = parse_number_theory("What is 8/12 in simplest form?")
        assert result is not None
        _, _, op, answer, _ = result
        assert op == 'simplify'
        assert answer == "2/3"

    def test_lowest_terms(self):
        result = parse_number_theory("Reduce 50/100 to lowest terms")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == "1/2"

    def test_zero_denominator(self):
        result = parse_number_theory("Simplify 5/0")
        assert result is None


# ── Unified parser edge cases ────────────────────────────────────────


class TestParseNumberTheory:
    def test_nonsense(self):
        assert parse_number_theory("What color is the sky?") is None

    def test_no_numbers(self):
        assert parse_number_theory("GCD of apples and oranges") is None

    def test_expression_format_gcd(self):
        result = parse_number_theory("gcd(15, 25)")
        assert result is not None
        _, _, _, _, expr = result
        assert expr == "gcd(15,25)"

    def test_expression_format_lcm(self):
        result = parse_number_theory("lcm(3, 7)")
        assert result is not None
        _, _, _, _, expr = result
        assert expr == "lcm(3,7)"

    def test_expression_format_simplify(self):
        result = parse_number_theory("simplify 10/4")
        assert result is not None
        _, _, _, _, expr = result
        assert expr == "simplify(10/4)"
