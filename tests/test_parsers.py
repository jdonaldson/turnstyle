"""Tests for date, unit, and currency parsers — no model needed."""

from datetime import date
from turnstyle.dates import parse_date_arithmetic, _parse_date
from turnstyle.units import parse_unit_conversion
from turnstyle.currency import parse_currency_conversion


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
