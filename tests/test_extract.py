"""Tests for the extraction module — assemblers, confidence gating, fast path."""

from unittest.mock import MagicMock, patch

import pytest
import torch

# turnstyle/__init__ re-exports the `extract` function, shadowing the
# turnstyle.extract submodule attribute; `import ... as` would bind the function
# (attribute traversal), so grab the real module from sys.modules and patch that.
import sys
import turnstyle.extract  # noqa: F401  (ensure submodule is loaded)
_extract_mod = sys.modules["turnstyle.extract"]
from turnstyle.extract import (
    ExtractionMethod,
    ExtractionResult,
    ExtractionSpec,
    FieldSpec,
    classify_token,
    extract,
)
from turnstyle.sorting import _assemble_sorting, SORTING_EXTRACTION_SPEC
from turnstyle.counting import _assemble_counting, COUNTING_EXTRACTION_SPEC
from turnstyle.boolean import _assemble_boolean, BOOLEAN_EXTRACTION_SPEC
from turnstyle.dyck import _assemble_dyck, _filter_brackets, DYCK_EXTRACTION_SPEC
from turnstyle.dates import _assemble_date, DATE_EXTRACTION_SPEC
from turnstyle.percentage import _assemble_percentage, PERCENTAGE_EXTRACTION_SPEC
from turnstyle.sandbox import _assemble_sandbox, SANDBOX_EXTRACTION_SPEC, SandboxParsed


# ════════════════════════════════════════════════════════════════════════
# Assembler unit tests (no model needed)
# ════════════════════════════════════════════════════════════════════════

class TestSortingAssembler:
    def test_basic_sort(self):
        words, sorted_words, sorted_str = _assemble_sorting(
            {"words": "cherry, banana, apple"})
        assert sorted_words == ["apple", "banana", "cherry"]
        assert sorted_str == "apple banana cherry"

    def test_space_separated(self):
        words, sorted_words, sorted_str = _assemble_sorting(
            {"words": "zebra mango apple"})
        assert sorted_words == ["apple", "mango", "zebra"]

    def test_single_word_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            _assemble_sorting({"words": "apple"})

    def test_whitespace_handling(self):
        words, sorted_words, _ = _assemble_sorting(
            {"words": "  cherry , banana , apple  "})
        assert sorted_words == ["apple", "banana", "cherry"]


class TestCountingAssembler:
    def test_vowels(self):
        target, ctype, count, expr = _assemble_counting(
            {"target": "mississippi", "count_type": "vowels",
             "specific_letter": "a"})
        assert count == 4
        assert ctype == "vowels"

    def test_consonants(self):
        _, _, count, _ = _assemble_counting(
            {"target": "python", "count_type": "consonants",
             "specific_letter": "a"})
        assert count == 5

    def test_words(self):
        _, _, count, _ = _assemble_counting(
            {"target": "the quick brown fox", "count_type": "words",
             "specific_letter": "a"})
        assert count == 4

    def test_letters(self):
        _, _, count, _ = _assemble_counting(
            {"target": "hello world", "count_type": "letters",
             "specific_letter": "a"})
        assert count == 10

    def test_characters(self):
        _, _, count, _ = _assemble_counting(
            {"target": "hello world", "count_type": "characters",
             "specific_letter": "a"})
        assert count == 11

    def test_specific_letter(self):
        _, ctype, count, _ = _assemble_counting(
            {"target": "strawberry", "count_type": "specific_letter",
             "specific_letter": "r"})
        assert count == 3
        assert ctype == "'r'"

    def test_strips_quotes(self):
        target, _, count, _ = _assemble_counting(
            {"target": "'strawberry'", "count_type": "specific_letter",
             "specific_letter": "r"})
        assert target == "strawberry"
        assert count == 3


class TestBooleanAssembler:
    def test_simple_and(self):
        expr, result, result_str = _assemble_boolean(
            {"expression": "True and False"})
        assert result is False
        assert result_str == "False"

    def test_not(self):
        _, result, _ = _assemble_boolean({"expression": "not False"})
        assert result is True

    def test_case_normalization(self):
        expr, result, _ = _assemble_boolean(
            {"expression": "true AND false"})
        assert result is False
        assert "True" in expr and "False" in expr

    def test_complex(self):
        _, result, _ = _assemble_boolean(
            {"expression": "True and not False or False"})
        assert result is True

    def test_invalid_tokens_raises(self):
        with pytest.raises(ValueError, match="Invalid tokens"):
            _assemble_boolean({"expression": "True and maybe False"})


class TestDyckAssembler:
    def test_simple_open(self):
        _, closing, _ = _assemble_dyck({"brackets": "("})
        assert closing == ")"

    def test_nested(self):
        _, closing, _ = _assemble_dyck({"brackets": "(("})
        assert closing == "))"

    def test_mixed(self):
        _, closing, _ = _assemble_dyck({"brackets": "([{"})
        assert closing == "}])"

    def test_partially_closed(self):
        _, closing, _ = _assemble_dyck({"brackets": "(()["})
        assert closing == "])"

    def test_balanced_raises(self):
        with pytest.raises(ValueError, match="already balanced"):
            _assemble_dyck({"brackets": "()"})

    def test_mismatched_raises(self):
        with pytest.raises(ValueError, match="Mismatched"):
            _assemble_dyck({"brackets": "(]"})

    def test_filter_brackets(self):
        assert _filter_brackets("abc ( [ def ) ] ghi") == "([)]"
        assert _filter_brackets("no brackets here") == ""
        two_open = "(("
        assert _filter_brackets(two_open) == two_open


# ════════════════════════════════════════════════════════════════════════
# Date assembler
# ════════════════════════════════════════════════════════════════════════

class TestDateAssembler:
    def test_days_between(self):
        expr, answer, unit = _assemble_date(
            {"date1": "2026-01-01", "date2": "2026-01-31", "unit": "days"})
        assert answer == 30
        assert unit == "day"
        assert "days(" in expr

    def test_weeks_between(self):
        expr, answer, unit = _assemble_date(
            {"date1": "2026-01-01", "date2": "2026-03-01", "unit": "weeks"})
        assert answer == 8  # 59 days // 7
        assert unit == "week"

    def test_written_dates(self):
        _, answer, _ = _assemble_date(
            {"date1": "March 1, 2026", "date2": "March 31, 2026", "unit": "days"})
        assert answer == 30

    def test_bad_date_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            _assemble_date(
                {"date1": "not a date", "date2": "2026-01-01", "unit": "days"})

    def test_strips_punctuation(self):
        _, answer, _ = _assemble_date(
            {"date1": "2026-06-01.", "date2": "2026-12-31?", "unit": "days"})
        assert answer == 213


# ════════════════════════════════════════════════════════════════════════
# Percentage assembler
# ════════════════════════════════════════════════════════════════════════

class TestPercentageAssembler:
    def test_of(self):
        a, b, op, result, expr = _assemble_percentage(
            {"value_a": "15", "value_b": "230", "operation": "of"})
        assert result == 34.5

    def test_is_pct(self):
        _, _, _, result, _ = _assemble_percentage(
            {"value_a": "45", "value_b": "180", "operation": "is_pct"})
        assert result == 25.0

    def test_off(self):
        _, _, _, result, _ = _assemble_percentage(
            {"value_a": "25", "value_b": "200", "operation": "off"})
        assert result == 150.0

    def test_tip(self):
        _, _, _, result, _ = _assemble_percentage(
            {"value_a": "20", "value_b": "85", "operation": "tip"})
        assert result == 17.0

    def test_strips_dollar(self):
        _, _, _, result, _ = _assemble_percentage(
            {"value_a": "10", "value_b": "$500", "operation": "of"})
        assert result == 50.0

    def test_zero_division_raises(self):
        with pytest.raises(ValueError, match="Division by zero"):
            _assemble_percentage(
                {"value_a": "45", "value_b": "0", "operation": "is_pct"})


# ════════════════════════════════════════════════════════════════════════
# Sandbox assembler
# ════════════════════════════════════════════════════════════════════════

class TestSandboxAssembler:
    def test_basic_code(self):
        result = _assemble_sandbox({"code": "sum(range(101))"})
        assert isinstance(result, SandboxParsed)
        assert result.code == "sum(range(101))"

    def test_strips_whitespace(self):
        result = _assemble_sandbox({"code": "  len([1,2,3])  "})
        assert result.code == "len([1,2,3])"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty code"):
            _assemble_sandbox({"code": "   "})


# ════════════════════════════════════════════════════════════════════════
# Regex fast path
# ════════════════════════════════════════════════════════════════════════

class TestRegexFastPath:
    """Verify extract() returns REGEX method for standard phrasings."""

    def _make_turnstyle(self, parse_result):
        ts = MagicMock()
        ts.parse.return_value = parse_result
        return ts

    def test_sorting_regex(self):
        parsed = (["banana", "apple"], ["apple", "banana"], "apple banana")
        ts = self._make_turnstyle(parsed)
        result = extract("Sort: banana, apple", ts, SORTING_EXTRACTION_SPEC)
        assert result.method == ExtractionMethod.REGEX
        assert result.confidence == 1.0
        assert result.parsed == parsed

    def test_boolean_regex(self):
        parsed = ("True and False", False, "False")
        ts = self._make_turnstyle(parsed)
        result = extract("True and False", ts, BOOLEAN_EXTRACTION_SPEC)
        assert result.method == ExtractionMethod.REGEX
        assert result.parsed == parsed

    def test_no_spec_returns_none(self):
        ts = self._make_turnstyle(None)
        result = extract("some input", ts, None)
        assert result is None


# ════════════════════════════════════════════════════════════════════════
# classify_token with mock model
# ════════════════════════════════════════════════════════════════════════

class TestClassifyToken:
    def test_picks_highest_prob_option(self):
        """Mock model returns known logits; verify correct option selected."""
        model = MagicMock()
        tokenizer = MagicMock()
        device = "cpu"

        # Set up tokenizer
        tokenizer.apply_chat_template.return_value = "prompt text"
        tokenizer.return_value = {
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        # Make tokenizer callable (for tokenizer(text, ...))
        tokenizer.side_effect = None
        tokenizer.__call__ = MagicMock(return_value=MagicMock(
            to=MagicMock(return_value={
                "input_ids": torch.zeros(1, 5, dtype=torch.long),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            })
        ))

        # Token IDs for options: "vowels"=100, " vowels"=101, "consonants"=200
        def mock_encode(text, add_special_tokens=False):
            mapping = {
                "vowels": [100], " vowels": [101],
                "consonants": [200], " consonants": [201],
            }
            return mapping.get(text, [0])
        tokenizer.encode = mock_encode

        # Logits: token 100 and 101 have high prob (vowels), 200 low
        logits = torch.full((1, 1, 300), -10.0)
        logits[0, 0, 100] = 5.0   # "vowels"
        logits[0, 0, 101] = 3.0   # " vowels"
        logits[0, 0, 200] = -1.0  # "consonants"
        logits[0, 0, 201] = -2.0  # " consonants"

        model_output = MagicMock()
        model_output.logits = logits
        model.return_value = model_output

        # Mock tokenizer as callable that returns object with .to()
        inputs_obj = MagicMock()
        inputs_obj.to.return_value = {
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
        }
        tokenizer.side_effect = lambda *a, **kw: inputs_obj

        idx, prob = classify_token(
            model, tokenizer, device, "test prompt", ["vowels", "consonants"])

        assert idx == 0  # "vowels" has higher combined probability
        assert prob > 0.0


# ════════════════════════════════════════════════════════════════════════
# Confidence gating
# ════════════════════════════════════════════════════════════════════════

class TestConfidenceGating:
    def test_low_confidence_returns_failed(self):
        """When LLM extraction has low confidence, result should be FAILED."""
        ts = MagicMock()
        ts.parse.return_value = None
        ts.model = MagicMock()
        ts.tokenizer = MagicMock()
        ts.device = "cpu"

        spec = ExtractionSpec(
            fields=[
                FieldSpec(
                    name="test_field",
                    prompt_template="Extract: {input}",
                    options=["a", "b"],
                ),
            ],
            assemble=lambda f: f["test_field"],
            min_confidence=0.9,  # very high threshold
        )

        # Mock classify_token to return low confidence
        with patch.object(_extract_mod, "classify_token", return_value=(0, 0.1)):
            result = extract("test input", ts, spec)

        assert result.method == ExtractionMethod.FAILED
        assert result.parsed is None
        assert result.confidence < 0.9

    def test_high_confidence_returns_llm(self):
        """When LLM extraction has high confidence, result should be LLM."""
        ts = MagicMock()
        ts.parse.return_value = None
        ts.model = MagicMock()
        ts.tokenizer = MagicMock()
        ts.device = "cpu"

        spec = ExtractionSpec(
            fields=[
                FieldSpec(
                    name="test_field",
                    prompt_template="Extract: {input}",
                    options=["a", "b"],
                ),
            ],
            assemble=lambda f: f["test_field"],
            min_confidence=0.3,
        )

        with patch.object(_extract_mod, "classify_token", return_value=(0, 0.8)):
            result = extract("test input", ts, spec)

        assert result.method == ExtractionMethod.LLM
        assert result.parsed == "a"
        assert result.confidence >= 0.3


# ════════════════════════════════════════════════════════════════════════
# Assembler error handling
# ════════════════════════════════════════════════════════════════════════

class TestAssemblerErrors:
    def test_assemble_failure_returns_failed(self):
        """If assemble raises, result should be FAILED."""
        ts = MagicMock()
        ts.parse.return_value = None

        def bad_assemble(fields):
            raise ValueError("boom")

        spec = ExtractionSpec(
            fields=[
                FieldSpec(
                    name="test_field",
                    prompt_template="Extract: {input}",
                    options=["a", "b"],
                ),
            ],
            assemble=bad_assemble,
            min_confidence=0.0,
        )

        with patch.object(_extract_mod, "classify_token", return_value=(0, 0.8)):
            result = extract("test input", ts, spec)

        assert result.method == ExtractionMethod.FAILED
        assert result.parsed is None


# ════════════════════════════════════════════════════════════════════════
# Extraction spec presence on turnstyle classes
# ════════════════════════════════════════════════════════════════════════

class TestExtractionSpecsExist:
    """Verify each turnstyle has its extraction_spec wired up."""

    def test_sorting_has_spec(self):
        from turnstyle.sorting import SortingTurnstyle
        assert SortingTurnstyle.extraction_spec is not None
        assert len(SortingTurnstyle.extraction_spec.fields) == 1

    def test_counting_has_spec(self):
        from turnstyle.counting import CountingTurnstyle
        assert CountingTurnstyle.extraction_spec is not None
        assert len(CountingTurnstyle.extraction_spec.fields) == 3

    def test_boolean_has_spec(self):
        from turnstyle.boolean import BooleanTurnstyle
        assert BooleanTurnstyle.extraction_spec is not None
        assert len(BooleanTurnstyle.extraction_spec.fields) == 1

    def test_dyck_has_spec(self):
        from turnstyle.dyck import DyckTurnstyle
        assert DyckTurnstyle.extraction_spec is not None
        assert len(DyckTurnstyle.extraction_spec.fields) == 1

    def test_date_has_spec(self):
        from turnstyle.dates import DateTurnstyle
        assert DateTurnstyle.extraction_spec is not None
        assert len(DateTurnstyle.extraction_spec.fields) == 3

    def test_percentage_has_spec(self):
        from turnstyle.percentage import PercentageTurnstyle
        assert PercentageTurnstyle.extraction_spec is not None
        assert len(PercentageTurnstyle.extraction_spec.fields) == 3

    def test_sandbox_has_spec(self):
        from turnstyle.sandbox import SandboxTurnstyle
        assert SandboxTurnstyle.extraction_spec is not None
        assert len(SandboxTurnstyle.extraction_spec.fields) == 1

    def test_base_turnstyle_has_none(self):
        from turnstyle.core import Turnstyle
        assert Turnstyle.extraction_spec is None

    def test_parse_returns_none_for_non_matching_prompts(self):
        """Each deterministic parser must return None on out-of-domain input
        (the 'a parse() that cannot return None is a bug' rule). These
        turnstyles now parse their own domain deterministically, so this checks
        graceful failure rather than the old extraction-only premise."""
        from turnstyle.sorting import SortingTurnstyle
        from turnstyle.counting import CountingTurnstyle
        from turnstyle.boolean import BooleanTurnstyle
        from turnstyle.dyck import DyckTurnstyle
        from turnstyle.dates import DateTurnstyle
        from turnstyle.percentage import PercentageTurnstyle
        from turnstyle.sandbox import SandboxTurnstyle

        model = MagicMock()
        tokenizer = MagicMock()
        device = "cpu"
        unrelated = "What is the capital of France?"

        for cls in [SortingTurnstyle, CountingTurnstyle, BooleanTurnstyle,
                    DyckTurnstyle, DateTurnstyle, PercentageTurnstyle]:
            t = cls(model, tokenizer, device)
            assert t.parse(unrelated) is None

        t = SandboxTurnstyle(model, tokenizer, device, backend=MagicMock())
        assert t.parse(unrelated) is None
