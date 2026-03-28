"""Tests for turnstyle core — no model needed."""

import torch

from turnstyle.core import (
    CoprocessorDiagnostic,
    Diagnostic,
    DigitAudit,
    SequenceLogitsProcessor,
    TokenAudit,
    extract_number,
)
from unittest.mock import MagicMock

from turnstyle.arithmetic import ArithmeticTurnstyle, parse_arithmetic
from turnstyle.probe import IntentProbe, TurnstyleProbe


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


# ── TokenAudit tests ────────────────────────────────────────────────


def test_token_audit_fields():
    a = TokenAudit(
        position=0, correct_token_id=42, model_top_token_id=99,
        bias_applied=15.0, model_logit=3.0, top_logit=5.0, corrected=True)
    assert a.position == 0
    assert a.correct_token_id == 42
    assert a.corrected


# ── SequenceLogitsProcessor tests ───────────────────────────────────


class _FakeTokenizer:
    """Minimal tokenizer for testing SequenceLogitsProcessor."""

    def __init__(self, vocab: dict[str, int], eos_token_id: int = 0):
        self._vocab = vocab
        self._inv = {v: k for k, v in vocab.items()}
        self.eos_token_id = eos_token_id

    def encode(self, text, add_special_tokens=False):
        return [self._vocab[text]] if text in self._vocab else []

    def decode(self, ids):
        return " ".join(self._inv.get(i, "?") for i in ids)


def _make_seq_processor(answer_text="True", trigger="is"):
    """Build a SequenceLogitsProcessor with a fake tokenizer."""
    vocab = {"is": 1, "True": 2, "False": 3, "=": 4}
    tok = _FakeTokenizer(vocab)
    answer_ids = tok.encode(answer_text)
    proc = SequenceLogitsProcessor(
        tok, answer_ids, expression="True and True",
        answer_str=answer_text, bias_strength=15.0)
    return proc, vocab


def test_seq_processor_waiting_state():
    proc, vocab = _make_seq_processor()
    assert proc.state == "WAITING"
    # Non-trigger token doesn't change state
    ids = torch.tensor([[vocab["False"]]])
    scores = torch.zeros(1, 5)
    out = proc(ids, scores)
    assert proc.state == "WAITING"


def test_seq_processor_trigger_to_injecting():
    proc, vocab = _make_seq_processor()
    # Feed trigger token
    ids = torch.tensor([[vocab["is"]]])
    scores = torch.zeros(1, 5)
    out = proc(ids, scores)
    assert proc.state == "INJECTING"


def test_seq_processor_biases_correct_token():
    proc, vocab = _make_seq_processor()
    # Trigger
    ids = torch.tensor([[vocab["is"]]])
    scores = torch.zeros(1, 5)
    proc(ids, scores)
    assert proc.state == "INJECTING"

    # Next call should bias toward "True" (id=2)
    scores2 = torch.zeros(1, 5)
    scores2[0, vocab["False"]] = 1.0  # model wants "False"
    out = proc(ids, scores2)
    assert out[0, vocab["True"]].item() > out[0, vocab["False"]].item()
    assert proc.token_idx == 1


def test_seq_processor_done_after_all_tokens():
    proc, vocab = _make_seq_processor()
    ids = torch.tensor([[vocab["is"]]])
    # Trigger
    proc(ids, torch.zeros(1, 5))
    # Inject the one token
    proc(ids, torch.zeros(1, 5))
    # Now done
    proc(ids, torch.zeros(1, 5))
    assert proc.state == "DONE"


def test_seq_processor_proof_populated():
    proc, vocab = _make_seq_processor()
    ids = torch.tensor([[vocab["is"]]])
    proc(ids, torch.zeros(1, 5))
    proc(ids, torch.zeros(1, 5))
    assert proc.proof.trigger_step == 1
    assert len(proc.proof.digits) == 1
    assert len(proc.audits) == 1


def test_seq_processor_forces_eos_after_done():
    proc, vocab = _make_seq_processor()
    eos_id = 0  # _FakeTokenizer default
    ids = torch.tensor([[vocab["is"]]])
    # Trigger
    proc(ids, torch.zeros(1, 5))
    # Inject the one token
    proc(ids, torch.zeros(1, 5))
    # Now DONE — should force EOS
    scores = torch.zeros(1, 5)
    out = proc(ids, scores)
    assert proc.state == "DONE"
    assert out[0, eos_id].item() > out[0, 1].item()
    assert out[0, eos_id].item() > out[0, 2].item()


def test_seq_processor_correction_tracking():
    proc, vocab = _make_seq_processor()
    ids = torch.tensor([[vocab["is"]]])
    proc(ids, torch.zeros(1, 5))  # trigger

    # Model confidently predicts wrong token
    scores = torch.zeros(1, 5)
    scores[0, vocab["False"]] = 10.0
    proc(ids, scores)

    assert proc.audits[0].corrected
    assert proc.audits[0].model_top_token_id == vocab["False"]
    assert proc.audits[0].correct_token_id == vocab["True"]


# ── ArithmeticTurnstyle.parse_from_hidden tests ────────────────────


def _make_arithmetic_intent_probe(hidden_dim=8):
    """Create an IntentProbe for arithmetic with known weights."""
    # Operation probe: add, sub, mul, div
    op_w = torch.zeros(4, hidden_dim)
    op_w[0, 0] = 5.0  # add
    op_w[1, 1] = 5.0  # sub
    op_w[2, 2] = 5.0  # mul
    op_w[3, 3] = 5.0  # div
    op_probe = TurnstyleProbe(op_w, torch.zeros(4), ["add", "sub", "mul", "div"])

    # operand_a probe: "10", "20"
    a_w = torch.zeros(2, hidden_dim)
    a_w[0, 4] = 5.0  # "10"
    a_w[1, 5] = 5.0  # "20"
    a_probe = TurnstyleProbe(a_w, torch.zeros(2), ["10", "20"])

    # operand_b probe: "5", "3"
    b_w = torch.zeros(2, hidden_dim)
    b_w[0, 6] = 5.0  # "5"
    b_w[1, 7] = 5.0  # "3"
    b_probe = TurnstyleProbe(b_w, torch.zeros(2), ["5", "3"])

    return IntentProbe({"operation": op_probe, "operand_a": a_probe, "operand_b": b_probe})


def test_parse_from_hidden_addition():
    t = ArithmeticTurnstyle(MagicMock(), MagicMock(), "cpu")
    t.intent_probe = _make_arithmetic_intent_probe()

    h = torch.zeros(8)
    h[0] = 5.0  # add
    h[4] = 5.0  # operand_a = 10
    h[6] = 5.0  # operand_b = 5

    result = t.parse_from_hidden(h)
    assert result is not None
    a, b, op, answer = result
    assert a == 10 and b == 5 and op == "+" and answer == 15


def test_parse_from_hidden_subtraction():
    t = ArithmeticTurnstyle(MagicMock(), MagicMock(), "cpu")
    t.intent_probe = _make_arithmetic_intent_probe()

    h = torch.zeros(8)
    h[1] = 5.0  # sub
    h[5] = 5.0  # operand_a = 20
    h[7] = 5.0  # operand_b = 3

    result = t.parse_from_hidden(h)
    assert result is not None
    a, b, op, answer = result
    assert a == 20 and b == 3 and op == "-" and answer == 17


def test_parse_from_hidden_no_probe():
    t = ArithmeticTurnstyle(MagicMock(), MagicMock(), "cpu")
    assert t.parse_from_hidden(torch.zeros(8)) is None


def test_parse_from_hidden_low_confidence():
    t = ArithmeticTurnstyle(MagicMock(), MagicMock(), "cpu")
    t.intent_probe = _make_arithmetic_intent_probe()

    # All zeros → sigmoid(0) = 0.5 < 0.7 threshold
    h = torch.zeros(8)
    assert t.parse_from_hidden(h) is None


def test_parse_from_hidden_division():
    t = ArithmeticTurnstyle(MagicMock(), MagicMock(), "cpu")
    t.intent_probe = _make_arithmetic_intent_probe()

    h = torch.zeros(8)
    h[3] = 5.0  # div
    h[5] = 5.0  # operand_a = 20
    h[6] = 5.0  # operand_b = 5

    result = t.parse_from_hidden(h)
    assert result is not None
    a, b, op, answer = result
    assert a == 20 and b == 5 and op == "/" and answer == 4
