"""Tests for turnstyle core — no model needed."""

from turnstyle.core import (
    CoprocessorDiagnostic,
    Diagnostic,
    DigitAudit,
    extract_number,
)
from turnstyle.arithmetic import parse_arithmetic


def test_parse_arithmetic():
    assert parse_arithmetic("What is 445 + 152?") == (445, 152, '+', 597)
    assert parse_arithmetic("99 * 99") == (99, 99, '*', 9801)
    assert parse_arithmetic("100 / 4") == (100, 4, '/', 25)
    assert parse_arithmetic("500 - 501") == (500, 501, '-', -1)
    assert parse_arithmetic("no math here") is None


def test_extract_number():
    assert extract_number("The answer is 597.") == 597
    assert extract_number("10,000") == 10000
    assert extract_number("result: -42") == -42
    assert extract_number("no numbers") is None
    assert extract_number("1,234,567") == 1234567


def test_diagnostic_clean():
    proof = CoprocessorDiagnostic(
        expression="2+2", answer=4, expected_digits=1,
        digits=[DigitAudit(0, 4, 4, 15.0, 10.0, 10.0, False)],
        final_state="DONE", trigger_step=3, total_steps=5, max_steps=50)
    assert proof.is_clean
    assert proof.diagnostics == []


def test_diagnostic_correction():
    proof = CoprocessorDiagnostic(
        expression="445+152", answer=597, expected_digits=3,
        digits=[
            DigitAudit(0, 5, 6, 15.0, 9.0, 10.0, True),
            DigitAudit(1, 9, 9, 15.0, 10.0, 10.0, False),
            DigitAudit(2, 7, 7, 15.0, 10.0, 10.0, False),
        ],
        final_state="DONE", trigger_step=10, total_steps=15, max_steps=50)
    assert proof.any_corrected
    assert proof.num_corrected == 1
    assert not proof.is_clean
    assert Diagnostic.HIGH_CORRECTION not in proof.diagnostics


def test_diagnostic_too_few_digits():
    proof = CoprocessorDiagnostic(
        expression="12345*6789", answer=83810205, expected_digits=8,
        digits=[DigitAudit(i, i, i, 15.0, 10.0, 10.0, False) for i in range(7)],
        final_state="DONE", trigger_step=5, total_steps=20, max_steps=50)
    assert Diagnostic.DIGITS_TOO_FEW in proof.diagnostics


def test_diagnostic_too_many_digits():
    proof = CoprocessorDiagnostic(
        expression="1+7654321", answer=8888888, expected_digits=7,
        digits=[DigitAudit(i, 8, 8, 15.0, 10.0, 10.0, False) for i in range(7)],
        extra_digits_after_done=1,
        final_state="DONE", trigger_step=5, total_steps=20, max_steps=50)
    assert Diagnostic.DIGITS_TOO_MANY in proof.diagnostics


def test_inline_formatting():
    proof = CoprocessorDiagnostic(
        expression="445+152", answer=597, expected_digits=3,
        digits=[
            DigitAudit(0, 5, 6, 15.0, 9.0, 10.0, True),
            DigitAudit(1, 9, 9, 15.0, 10.0, 10.0, False),
            DigitAudit(2, 7, 7, 15.0, 10.0, 10.0, False),
        ],
        final_state="DONE")
    inline = proof.inline()
    assert "\u22a2" in inline  # ⊢
    assert "\u220e" in inline  # ∎
    assert "5\u0332" in inline  # 5̲


def test_inline_missing_digits():
    proof = CoprocessorDiagnostic(
        expression="10+90", answer=100, expected_digits=3,
        digits=[
            DigitAudit(0, 1, 1, 15.0, 10.0, 10.0, False),
            DigitAudit(1, 0, 0, 15.0, 10.0, 10.0, False),
        ],
        final_state="DONE")
    inline = proof.inline()
    assert "0\u0302" in inline  # 0̂


def test_plain_inline():
    proof = CoprocessorDiagnostic(
        expression="445+152", answer=597, expected_digits=3,
        digits=[
            DigitAudit(0, 5, 6, 15.0, 9.0, 10.0, True),
            DigitAudit(1, 9, 9, 15.0, 10.0, 10.0, False),
            DigitAudit(2, 7, 7, 15.0, 10.0, 10.0, False),
        ],
        final_state="DONE")
    plain = proof.inline(plain=True)
    assert plain == "445+152=597"
    assert "\u22a2" not in plain
    assert "\u220e" not in plain
    assert "\u0332" not in plain


def test_plain_summary():
    proof = CoprocessorDiagnostic(
        expression="445+152", answer=597, expected_digits=3,
        digits=[
            DigitAudit(0, 5, 6, 15.0, 9.0, 10.0, True),
            DigitAudit(1, 9, 9, 15.0, 10.0, 10.0, False),
            DigitAudit(2, 7, 7, 15.0, 10.0, 10.0, False),
        ],
        final_state="DONE")
    summary = proof.summary(plain=True)
    assert "\u22a2" not in summary
    assert "1/3 corrected" in summary
    assert "\u0394" not in summary


def test_plain_detail():
    proof = CoprocessorDiagnostic(
        expression="445+152", answer=597, expected_digits=3,
        digits=[
            DigitAudit(0, 5, 6, 15.0, 9.0, 10.0, True),
            DigitAudit(1, 9, 9, 15.0, 10.0, 10.0, False),
            DigitAudit(2, 7, 7, 15.0, 10.0, 10.0, False),
        ],
        final_state="DONE", trigger_step=10, total_steps=15, max_steps=50)
    detail = proof.detail(plain=True)
    assert "->" in detail
    assert "\u2192" not in detail
