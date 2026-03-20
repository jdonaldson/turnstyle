"""Tests for sandbox backends — no Deno needed."""

from turnstyle.sandbox_backend import MockBackend, SandboxResult, _parse_numeric


class TestParseNumeric:
    def test_integer(self):
        assert _parse_numeric("42") == 42

    def test_negative_integer(self):
        assert _parse_numeric("-7") == -7

    def test_float(self):
        assert _parse_numeric("3.14") == 3.14

    def test_empty_string(self):
        assert _parse_numeric("") is None

    def test_non_numeric(self):
        assert _parse_numeric("hello") is None

    def test_zero(self):
        assert _parse_numeric("0") == 0

    def test_large_number(self):
        assert _parse_numeric("1000000") == 1000000


class TestSandboxResult:
    def test_result_fields(self):
        r = SandboxResult(
            stdout="hello\n", stderr="", return_value="42",
            numeric_value=42, execution_time_ms=10.5, error=None,
        )
        assert r.numeric_value == 42
        assert r.error is None
        assert r.stdout == "hello\n"

    def test_error_result(self):
        r = SandboxResult(
            stdout="", stderr="traceback", return_value="",
            numeric_value=None, execution_time_ms=5.0,
            error="NameError: name 'x' is not defined",
        )
        assert r.error is not None
        assert r.numeric_value is None


class TestMockBackend:
    def test_available(self):
        mock = MockBackend()
        assert mock.available()

    def test_registered_response(self):
        mock = MockBackend()
        expected = SandboxResult(
            stdout="", stderr="", return_value="5050",
            numeric_value=5050, execution_time_ms=1.0, error=None,
        )
        mock.add("sum(range(101))", expected)
        result = mock.execute("sum(range(101))")
        assert result.numeric_value == 5050
        assert result.error is None

    def test_unregistered_code_returns_error(self):
        mock = MockBackend()
        result = mock.execute("unknown_code()")
        assert result.error is not None
        assert "no response registered" in result.error

    def test_constructor_with_responses(self):
        responses = {
            "2+2": SandboxResult(
                stdout="", stderr="", return_value="4",
                numeric_value=4, execution_time_ms=0.1, error=None,
            )
        }
        mock = MockBackend(responses)
        result = mock.execute("2+2")
        assert result.numeric_value == 4

    def test_protocol_compliance(self):
        from turnstyle.sandbox_backend import SandboxBackend
        mock = MockBackend()
        assert isinstance(mock, SandboxBackend)
