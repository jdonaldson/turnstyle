"""Sandbox execution backends — run arbitrary Python in WASM isolation.

SandboxResult captures stdout, stderr, return value, and timing.
WasmtimeBackend runs CPython WASM via wasmtime (preferred — pip-installable).
DenoPyodideBackend spawns Deno + Pyodide for actual execution (fallback).
MockBackend maps code strings to canned results for tests.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# WasmtimeBackend — CPython WASM via wasmtime
# ---------------------------------------------------------------------------

_CPYTHON_WASM_URL = (
    "https://github.com/vmware-labs/webassembly-language-runtimes/"
    "releases/download/python/3.12.0%2B20231211-040d5a6/"
    "python-3.12.0-wasi-sdk-20.0.tar.gz"
)
_DEFAULT_CACHE_DIR = Path("~/.cache/turnstyle/cpython-wasm").expanduser()


def _download_cpython_wasm(cache_dir: Path, url: str = _CPYTHON_WASM_URL) -> None:
    """Download and extract CPython WASM tarball into cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading CPython WASM from %s ...", url)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(cache_dir)
    logger.info("Extracted CPython WASM to %s", cache_dir)


def _ensure_cpython_wasm(cache_dir: Path) -> Path:
    """Ensure python.wasm + stdlib exist in cache_dir. Returns wasm path."""
    wasm_path = _find_wasm_binary(cache_dir)
    if wasm_path is not None:
        return wasm_path
    _download_cpython_wasm(cache_dir)
    wasm_path = _find_wasm_binary(cache_dir)
    if wasm_path is None:
        raise FileNotFoundError(
            f"python*.wasm not found in {cache_dir} after download"
        )
    return wasm_path


def _find_wasm_binary(cache_dir: Path) -> Path | None:
    """Find the python WASM binary (may be named python.wasm or python-X.Y.Z.wasm)."""
    for p in cache_dir.rglob("python*.wasm"):
        if p.is_file():
            return p
    return None


def _find_preopen_root(cache_dir: Path) -> tuple[Path, str]:
    """Find the host path and guest mount point for stdlib access.

    The VMware CPython WASM build expects stdlib at /usr/local/lib/python312.zip
    and /usr/local/lib/python3.12/. We find the `usr` directory in the cache
    and mount it as /usr in the guest.

    Returns (host_path, guest_path) for preopen_dir().
    """
    # Look for the python312.zip or os.py to locate the lib tree
    for p in cache_dir.rglob("python312.zip"):
        # p = .../usr/local/lib/python312.zip
        # We want to mount the "usr" directory as "/usr"
        # Walk up to find "usr"
        candidate = p.parent  # lib/
        while candidate != cache_dir:
            if candidate.name == "usr":
                return candidate, "/usr"
            candidate = candidate.parent

    # Fallback: look for os.py
    for p in cache_dir.rglob("os.py"):
        candidate = p.parent.parent  # lib/
        while candidate != cache_dir:
            if candidate.name == "usr":
                return candidate, "/usr"
            candidate = candidate.parent

    raise FileNotFoundError(f"Python stdlib not found in {cache_dir}")


class WasmtimeBackend:
    """Executes Python code via CPython compiled to WASM, using wasmtime.

    Auto-downloads CPython WASM on first use (~11MB). Module compilation
    is cached for instant subsequent loads.

    Sandbox guarantees:
    - No network access
    - No host filesystem access (only preopened stdlib, read-only)
    - Epoch-based timeout interruption
    """

    def __init__(self, wasm_dir: Path | None = None):
        self._wasm_dir = wasm_dir or _DEFAULT_CACHE_DIR
        self._engine = None
        self._module = None
        self._wasm_path: Path | None = None
        self._preopen_host: Path | None = None
        self._preopen_guest: str | None = None

    def available(self) -> bool:
        try:
            import wasmtime  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_ready(self) -> None:
        """Download python.wasm if needed, compile/cache the module."""
        import wasmtime

        if self._module is not None:
            return

        self._wasm_path = _ensure_cpython_wasm(self._wasm_dir)
        self._preopen_host, self._preopen_guest = _find_preopen_root(self._wasm_dir)

        config = wasmtime.Config()
        config.epoch_interruption = True
        self._engine = wasmtime.Engine(config)

        # Try deserializing cached compiled module
        cwasm_path = self._wasm_dir / "python.cwasm"
        if cwasm_path.exists():
            try:
                self._module = wasmtime.Module.deserialize(
                    self._engine, cwasm_path.read_bytes()
                )
                return
            except Exception:
                logger.debug("Cached cwasm invalid, recompiling")

        # Compile from .wasm and cache
        self._module = wasmtime.Module.from_file(
            self._engine, str(self._wasm_path)
        )
        try:
            cwasm_path.write_bytes(self._module.serialize())
        except Exception:
            logger.debug("Failed to cache compiled module", exc_info=True)

    def execute(self, code: str, timeout: float = 5.0) -> SandboxResult:
        if not self.available():
            return SandboxResult(
                stdout="", stderr="", return_value="",
                numeric_value=None, execution_time_ms=0.0,
                error="wasmtime not installed",
            )

        t0 = time.monotonic()
        try:
            self._ensure_ready()
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return SandboxResult(
                stdout="", stderr="", return_value="",
                numeric_value=None, execution_time_ms=elapsed,
                error=f"Failed to initialize CPython WASM: {e}",
            )

        import wasmtime

        # Wrap code to capture the last expression's value
        wrapper = _wrap_code(code)

        # Create a fresh store per execution (isolation)
        store = wasmtime.Store(self._engine)
        store.set_epoch_deadline(1)

        # Set up timeout via epoch increment on a background thread
        timer = threading.Timer(timeout, self._engine.increment_epoch)
        timer.daemon = True
        timer.start()

        try:
            return self._run_in_store(store, wrapper, t0)
        finally:
            timer.cancel()

    def _run_in_store(self, store, wrapper: str, t0: float) -> SandboxResult:
        import wasmtime

        with tempfile.TemporaryDirectory() as tmpdir:
            stdout_path = Path(tmpdir) / "stdout"
            stderr_path = Path(tmpdir) / "stderr"
            stdout_path.touch()
            stderr_path.touch()

            wasi = wasmtime.WasiConfig()
            wasi.argv = ["python", "-c", wrapper]
            wasi.stdout_file = str(stdout_path)
            wasi.stderr_file = str(stderr_path)

            # Preopen stdlib as read-only
            wasi.preopen_dir(
                str(self._preopen_host),
                self._preopen_guest,
                wasmtime.DirPerms.READ_ONLY,
                wasmtime.FilePerms.READ_ONLY,
            )

            store.set_wasi(wasi)
            linker = wasmtime.Linker(store.engine)
            linker.define_wasi()

            try:
                instance = linker.instantiate(store, self._module)
            except Exception as e:
                elapsed = (time.monotonic() - t0) * 1000
                return SandboxResult(
                    stdout="", stderr="", return_value="",
                    numeric_value=None, execution_time_ms=elapsed,
                    error=f"WASM instantiation failed: {e}",
                )

            try:
                start = instance.exports(store)["_start"]
                start(store)
            except wasmtime.ExitTrap as e:
                elapsed = (time.monotonic() - t0) * 1000
                stdout = stdout_path.read_text()
                stderr = stderr_path.read_text()
                if e.code != 0:
                    return SandboxResult(
                        stdout=stdout, stderr=stderr, return_value="",
                        numeric_value=None, execution_time_ms=elapsed,
                        error=stderr.strip() or f"Python exited with code {e.code}",
                    )
                # exit(0) is normal for CPython WASM
                return self._parse_output(stdout, stderr, elapsed)
            except wasmtime.WasmtimeError as e:
                elapsed = (time.monotonic() - t0) * 1000
                err_str = str(e)
                if "epoch" in err_str.lower() or "interrupt" in err_str.lower():
                    return SandboxResult(
                        stdout="", stderr="", return_value="",
                        numeric_value=None, execution_time_ms=elapsed,
                        error=f"Execution timed out after {elapsed/1000:.1f}s",
                    )
                return SandboxResult(
                    stdout="", stderr="", return_value="",
                    numeric_value=None, execution_time_ms=elapsed,
                    error=f"WASM error: {e}",
                )

            elapsed = (time.monotonic() - t0) * 1000
            stdout = stdout_path.read_text()
            stderr = stderr_path.read_text()
            return self._parse_output(stdout, stderr, elapsed)

    @staticmethod
    def _parse_output(stdout: str, stderr: str, elapsed: float) -> SandboxResult:
        """Parse stdout for return value, mirroring DenoPyodideBackend."""
        stdout = stdout.rstrip("\n")
        # Last line is the return value (from our wrapper)
        lines = stdout.split("\n")
        return_value = lines[-1] if lines else ""
        # Everything before last line is user stdout
        user_stdout = "\n".join(lines[:-1]) if len(lines) > 1 else ""

        effective_value = return_value if return_value != "None" else user_stdout.strip()

        return SandboxResult(
            stdout=user_stdout,
            stderr=stderr,
            return_value=return_value,
            numeric_value=_parse_numeric(effective_value),
            execution_time_ms=elapsed,
            error=None,
        )


def _wrap_code(code: str) -> str:
    """Wrap user code to print the result of the last expression.

    If the code compiles as a single eval expression, wraps in print(repr(...)).
    Otherwise executes as statements and prints the last expression if possible.
    """
    code = code.strip()

    # Single expression — wrap in print(repr())
    try:
        compile(code, "<sandbox>", "eval")
        return f"print(repr(eval({code!r})))"
    except SyntaxError:
        pass

    # Multi-line: try to make the last line an expression we can capture
    lines = code.split("\n")
    last_line = lines[-1].strip()
    if last_line:
        try:
            compile(last_line, "<sandbox>", "eval")
            # Last line is an expression — replace it with print(repr(...))
            prefix = "\n".join(lines[:-1])
            return f"{prefix}\nprint(repr({last_line}))"
        except SyntaxError:
            pass

    # Pure statements — just run as-is, output goes to stdout
    return code
