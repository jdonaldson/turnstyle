"""Integration tests for Deno + Pyodide sandbox. Skipped if Deno not installed."""

import shutil

import pytest

from turnstyle.sandbox_backend import DenoPyodideBackend

pytestmark = pytest.mark.skipif(
    shutil.which("deno") is None,
    reason="Deno not installed",
)


@pytest.fixture(scope="module")
def backend():
    return DenoPyodideBackend()


class TestDenoPyodideBackend:
    def test_available(self, backend):
        assert backend.available()

    def test_simple_expression(self, backend):
        result = backend.execute("2 + 2")
        assert result.error is None
        assert result.numeric_value == 4

    def test_sum_range(self, backend):
        result = backend.execute("sum(range(101))")
        assert result.error is None
        assert result.numeric_value == 5050

    def test_multiline_code(self, backend):
        code = "x = 10\ny = 20\nx + y"
        result = backend.execute(code)
        assert result.error is None
        assert result.numeric_value == 30

    def test_stdout_capture(self, backend):
        code = "print('hello')\n42"
        result = backend.execute(code)
        assert result.error is None
        assert "hello" in result.stdout
        assert result.numeric_value == 42

    def test_assignment_falls_back_to_stdout(self, backend):
        code = "x = 100\nprint(x)"
        result = backend.execute(code)
        assert result.error is None
        assert result.numeric_value == 100

    def test_syntax_error(self, backend):
        result = backend.execute("def (")
        assert result.error is not None

    def test_runtime_error(self, backend):
        result = backend.execute("1 / 0")
        assert result.error is not None

    def test_list_comprehension(self, backend):
        code = "sum([x**2 for x in range(10)])"
        result = backend.execute(code)
        assert result.error is None
        assert result.numeric_value == 285

    def test_float_result(self, backend):
        result = backend.execute("22 / 7")
        assert result.error is None
        assert result.numeric_value is not None
        assert abs(result.numeric_value - 3.142857) < 0.001

    def test_non_numeric_result(self, backend):
        result = backend.execute("'hello world'")
        assert result.error is None
        assert result.numeric_value is None
        assert "hello" in result.return_value

    def test_stdlib_available(self, backend):
        code = "import math\nmath.factorial(10)"
        result = backend.execute(code)
        assert result.error is None
        assert result.numeric_value == 3628800
