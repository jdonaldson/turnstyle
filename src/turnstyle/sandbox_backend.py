"""Sandbox execution backends — run arbitrary Python in WASM isolation.

SandboxResult captures stdout, stderr, return value, and timing.
DenoPyodideBackend spawns Deno + Pyodide for actual execution.
MockBackend maps code strings to canned results for tests.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class SandboxResult:
    """Result of executing Python code in a sandbox."""
    stdout: str
    stderr: str
    return_value: str          # repr() of last expression
    numeric_value: float | int | None  # parsed if numeric
    execution_time_ms: float
    error: str | None          # exception message or None


def _parse_numeric(value: str) -> float | int | None:
    """Try to parse a string as int or float."""
    if not value:
        return None
    try:
        n = int(value)
        return n
    except ValueError:
        pass
    try:
        n = float(value)
        return n
    except ValueError:
        return None


@runtime_checkable
class SandboxBackend(Protocol):
    def execute(self, code: str, timeout: float = 5.0) -> SandboxResult: ...
    def available(self) -> bool: ...


class DenoPyodideBackend:
    """Executes Python code via Deno + Pyodide WASM sandbox."""

    def __init__(self):
        self._runner_path = Path(__file__).parent / "_runner.js"

    def available(self) -> bool:
        return shutil.which("deno") is not None and self._runner_path.exists()

    def execute(self, code: str, timeout: float = 5.0) -> SandboxResult:
        if not self.available():
            return SandboxResult(
                stdout="", stderr="", return_value="",
                numeric_value=None, execution_time_ms=0.0,
                error="Deno not found or _runner.js missing",
            )

        payload = json.dumps({"code": code, "timeout": timeout})
        proc_timeout = timeout + 10.0  # grace for Pyodide cold start

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                [
                    "deno", "run",
                    "--allow-read",
                    "--allow-net=cdn.jsdelivr.net",
                    str(self._runner_path),
                ],
                input=payload,
                capture_output=True,
                text=True,
                timeout=proc_timeout,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - t0) * 1000
            return SandboxResult(
                stdout="", stderr="", return_value="",
                numeric_value=None, execution_time_ms=elapsed,
                error=f"Execution timed out after {timeout}s",
            )
        except FileNotFoundError:
            return SandboxResult(
                stdout="", stderr="", return_value="",
                numeric_value=None, execution_time_ms=0.0,
                error="Deno executable not found",
            )

        elapsed = (time.monotonic() - t0) * 1000

        if result.returncode != 0:
            return SandboxResult(
                stdout="", stderr=result.stderr, return_value="",
                numeric_value=None, execution_time_ms=elapsed,
                error=result.stderr.strip() or f"Deno exited with code {result.returncode}",
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return SandboxResult(
                stdout="", stderr=result.stderr, return_value=result.stdout,
                numeric_value=None, execution_time_ms=elapsed,
                error=f"Invalid JSON from runner: {result.stdout[:200]}",
            )

        return_value = str(data.get("return_value", ""))
        stdout = data.get("stdout", "")

        # If return_value is "None" (assignment), use stdout
        effective_value = return_value if return_value != "None" else stdout.strip()

        return SandboxResult(
            stdout=stdout,
            stderr=data.get("stderr", ""),
            return_value=return_value,
            numeric_value=_parse_numeric(effective_value),
            execution_time_ms=elapsed,
            error=data.get("error"),
        )


class MockBackend:
    """Maps code strings to canned SandboxResults. For tests."""

    def __init__(self, responses: dict[str, SandboxResult] | None = None):
        self._responses: dict[str, SandboxResult] = responses or {}

    def add(self, code: str, result: SandboxResult) -> None:
        self._responses[code] = result

    def available(self) -> bool:
        return True

    def execute(self, code: str, timeout: float = 5.0) -> SandboxResult:
        if code in self._responses:
            return self._responses[code]
        return SandboxResult(
            stdout="", stderr="", return_value="",
            numeric_value=None, execution_time_ms=0.0,
            error=f"MockBackend: no response registered for code: {code[:100]}",
        )
