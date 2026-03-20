"""Tests for sandbox code extraction — no Deno needed."""

from turnstyle.sandbox import parse_sandbox_code


class TestFencedCodeBlocks:
    def test_python_fenced_block(self):
        text = 'What is the result?\n```python\nsum(range(101))\n```'
        result = parse_sandbox_code(text)
        assert result is not None
        assert result.code == "sum(range(101))"
        assert result.description == "fenced code block"

    def test_unfenced_block(self):
        text = 'Run this:\n```\nlen("hello")\n```'
        result = parse_sandbox_code(text)
        assert result is not None
        assert result.code == 'len("hello")'

    def test_multiline_fenced_block(self):
        text = '```python\nx = 10\ny = 20\nx + y\n```'
        result = parse_sandbox_code(text)
        assert result is not None
        assert "x = 10" in result.code
        assert "x + y" in result.code

    def test_empty_fenced_block_returns_none(self):
        text = '```python\n\n```'
        result = parse_sandbox_code(text)
        assert result is None


class TestWhatDoesReturn:
    def test_what_does_return(self):
        result = parse_sandbox_code("What does `sum(range(101))` return?")
        assert result is not None
        assert result.code == "sum(range(101))"

    def test_what_is_output_of(self):
        result = parse_sandbox_code("What is the output of `len([1,2,3])`?")
        assert result is not None
        assert result.code == "len([1,2,3])"

    def test_what_is_result_of(self):
        result = parse_sandbox_code("What is the result of `2**10`?")
        assert result is not None
        assert result.code == "2**10"


class TestInlineBacktick:
    def test_function_call(self):
        result = parse_sandbox_code("Calculate `len('hello world')`")
        assert result is not None
        assert result.code == "len('hello world')"

    def test_list_comprehension(self):
        result = parse_sandbox_code("Evaluate `sum([x**2 for x in range(10)])`")
        assert result is not None
        assert result.code == "sum([x**2 for x in range(10)])"

    def test_bare_arithmetic_rejected(self):
        """Bare arithmetic is ArithmeticTurnstyle's domain."""
        result = parse_sandbox_code("What is `445 + 152`?")
        assert result is None

    def test_bare_number_rejected(self):
        result = parse_sandbox_code("What is `42`?")
        assert result is None

    def test_attribute_access(self):
        result = parse_sandbox_code("What is `'hello'.upper()`?")
        assert result is not None
        assert result.code == "'hello'.upper()"


class TestDirectives:
    def test_execute_directive(self):
        result = parse_sandbox_code("Execute: print(2 + 2)")
        assert result is not None
        assert result.code == "print(2 + 2)"

    def test_evaluate_directive(self):
        result = parse_sandbox_code("Evaluate: sum(range(10))")
        assert result is not None
        assert result.code == "sum(range(10))"

    def test_run_directive(self):
        result = parse_sandbox_code("Run: len([1,2,3])")
        assert result is not None
        assert result.code == "len([1,2,3])"


class TestNoMatch:
    def test_plain_text_returns_none(self):
        assert parse_sandbox_code("What color is the sky?") is None

    def test_bare_arithmetic_returns_none(self):
        assert parse_sandbox_code("What is 445 + 152?") is None

    def test_empty_string_returns_none(self):
        assert parse_sandbox_code("") is None


class TestPriorityOrder:
    def test_fenced_block_wins_over_inline(self):
        text = 'Try `len("hi")` or:\n```python\nsum(range(5))\n```'
        result = parse_sandbox_code(text)
        assert result is not None
        assert result.code == "sum(range(5))"
        assert result.description == "fenced code block"
