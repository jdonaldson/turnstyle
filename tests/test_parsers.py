"""Tests for all parsers — no model needed."""

from datetime import date
from turnstyle.dates import parse_date_arithmetic, _parse_date
from turnstyle.units import parse_unit_conversion
from turnstyle.currency import parse_currency_conversion
from turnstyle.percentage import parse_percentage
from turnstyle.counting import parse_counting
from turnstyle.base_conversion import parse_base_conversion


# ── date parsing ─────────────────────────────────────────────────────

class TestDateParsing:
    def test_parse_iso_date(self):
        assert _parse_date("2026-03-20") == date(2026, 3, 20)

    def test_parse_us_date(self):
        assert _parse_date("3/20/2026") == date(2026, 3, 20)

    def test_parse_written_date(self):
        assert _parse_date("March 20, 2026") == date(2026, 3, 20)

    def test_parse_eu_date(self):
        assert _parse_date("20 March 2026") == date(2026, 3, 20)

    def test_parse_short_month(self):
        assert _parse_date("Jan 1, 2026") == date(2026, 1, 1)

    def test_days_between(self):
        result = parse_date_arithmetic(
            "How many days between 2026-01-01 and 2026-01-31?")
        assert result is not None
        expr, answer, unit = result
        assert answer == 30
        assert unit == "day"

    def test_days_between_written(self):
        result = parse_date_arithmetic(
            "How many days between March 1, 2026 and March 31, 2026?")
        assert result is not None
        _, answer, _ = result
        assert answer == 30

    def test_weeks_between(self):
        result = parse_date_arithmetic(
            "How many weeks between 2026-01-01 and 2026-03-01?")
        assert result is not None
        _, answer, unit = result
        assert unit == "week"
        assert answer == 8  # 59 days // 7

    def test_days_from_to(self):
        result = parse_date_arithmetic(
            "How many days from 2026-06-01 to 2026-12-31?")
        assert result is not None
        _, answer, _ = result
        assert answer == 213


# ── unit conversion ──────────────────────────────────────────────────

class TestUnitConversion:
    def test_miles_to_km(self):
        result = parse_unit_conversion("How many km is 26.2 miles?")
        assert result is not None
        _, _, _, answer, _ = result
        assert abs(answer - 42.16) < 0.01

    def test_convert_fahrenheit_celsius(self):
        result = parse_unit_conversion("Convert 212 fahrenheit to celsius")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 100.0

    def test_kg_to_pounds(self):
        result = parse_unit_conversion("What is 1 kg in pounds?")
        assert result is not None
        _, _, _, answer, _ = result
        assert abs(answer - 2.2) < 0.1

    def test_liters_to_gallons(self):
        result = parse_unit_conversion("3.78541 liters to gallons")
        assert result is not None
        _, _, _, answer, _ = result
        assert abs(answer - 1.0) < 0.01

    def test_freezing_point(self):
        result = parse_unit_conversion("Convert 32 fahrenheit to celsius")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 0.0

    def test_unknown_unit_returns_none(self):
        assert parse_unit_conversion("Convert 5 frobbles to widgets") is None


# ── currency conversion ──────────────────────────────────────────────

class TestCurrencyConversion:
    def test_usd_to_eur(self):
        result = parse_currency_conversion("How much is 100 USD in EUR?")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 92.0  # 100 * 0.92

    def test_convert_gbp_to_jpy(self):
        result = parse_currency_conversion("Convert 100 GBP to JPY")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer > 10000  # roughly 100 / 0.79 * 150

    def test_natural_language(self):
        result = parse_currency_conversion("100 dollars to euros")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 92.0

    def test_custom_rates(self):
        rates = {'usd': 1.0, 'btc': 0.000015}
        result = parse_currency_conversion(
            "Convert 1000 USD to BTC", rates=rates)
        assert result is not None
        _, _, _, answer, _ = result
        assert abs(answer - 0.02) < 0.01

    def test_unknown_currency_returns_none(self):
        assert parse_currency_conversion("100 zorkmids to USD") is None


# ── percentage ──────────────────────────────────────────────────────

class TestPercentage:
    def test_percent_of(self):
        result = parse_percentage("What is 15% of 230?")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 34.5

    def test_what_percentage(self):
        result = parse_percentage("What percentage is 45 of 180?")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 25.0

    def test_tip(self):
        result = parse_percentage("20% tip on $85")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 17.0

    def test_discount(self):
        result = parse_percentage("25% off 200")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 150.0

    def test_with_dollar_sign(self):
        result = parse_percentage("What is 10% of $500?")
        assert result is not None
        _, _, _, answer, _ = result
        assert answer == 50.0

    def test_nonsense_returns_none(self):
        assert parse_percentage("What color is the sky?") is None


# ── counting ────────────────────────────────────────────────────────

class TestCounting:
    def test_vowels(self):
        result = parse_counting("How many vowels in 'mississippi'?")
        assert result is not None
        _, _, count, _ = result
        assert count == 4

    def test_consonants(self):
        result = parse_counting("How many consonants in 'python'?")
        assert result is not None
        _, _, count, _ = result
        assert count == 5  # p, y, t, h, n

    def test_words(self):
        result = parse_counting("How many words in 'the quick brown fox'?")
        assert result is not None
        _, _, count, _ = result
        assert count == 4

    def test_letters(self):
        result = parse_counting("How many letters in 'hello world'?")
        assert result is not None
        _, _, count, _ = result
        assert count == 10  # excludes space

    def test_characters(self):
        result = parse_counting("How many characters in 'hello world'?")
        assert result is not None
        _, _, count, _ = result
        assert count == 11  # includes space

    def test_specific_char(self):
        result = parse_counting("How many r's in 'strawberry'?")
        assert result is not None
        _, _, count, _ = result
        assert count == 3

    def test_specific_char_enterprise(self):
        result = parse_counting('How many e\'s in "enterprise"?')
        assert result is not None
        _, _, count, _ = result
        assert count == 3

    def test_no_quoted_text_returns_none(self):
        assert parse_counting("How many vowels in the sky?") is None


# ── base conversion ─────────────────────────────────────────────────

class TestBaseConversion:
    def test_decimal_to_binary(self):
        result = parse_base_conversion("What is 255 in binary?")
        assert result is not None
        _, _, _, result_str, _ = result
        assert result_str == "11111111"

    def test_decimal_to_hex(self):
        result = parse_base_conversion("What is 255 in hex?")
        assert result is not None
        _, _, _, result_str, _ = result
        assert result_str == "ff"

    def test_decimal_to_octal(self):
        result = parse_base_conversion("42 to octal")
        assert result is not None
        _, _, _, result_str, _ = result
        assert result_str == "52"

    def test_binary_to_decimal(self):
        result = parse_base_conversion("Convert 1010 from binary to decimal")
        assert result is not None
        decimal_value, _, _, result_str, _ = result
        assert decimal_value == 10
        assert result_str == "10"

    def test_hex_prefix_to_decimal(self):
        result = parse_base_conversion("What is 0xff in decimal?")
        assert result is not None
        decimal_value, _, _, result_str, _ = result
        assert decimal_value == 255
        assert result_str == "255"

    def test_convert_to_binary(self):
        result = parse_base_conversion("Convert 10 to binary")
        assert result is not None
        _, _, _, result_str, _ = result
        assert result_str == "1010"

    def test_unknown_base_returns_none(self):
        assert parse_base_conversion("What color is blue?") is None
