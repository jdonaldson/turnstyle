# Changelog

## 0.3.0

### Added
- **WasmtimeBackend** — pip-installable sandbox backend using CPython WASM via wasmtime (~50ms warm execution vs 3-5s Deno cold start)
- Auto-downloads CPython 3.12 WASM on first use, caches compiled module for instant subsequent loads
- `SandboxTurnstyle` auto-selects best available backend (wasmtime preferred, Deno fallback)

### Changed
- `sandbox` extra now installs `wasmtime>=42.0` (previously empty, required external Deno)
- Default backend priority: WasmtimeBackend > DenoPyodideBackend

## 0.2.0

### Added
- **SandboxTurnstyle** — execute arbitrary Python in a WASM sandbox and ground digit generation in the computed result
- `DenoPyodideBackend` — production backend spawning Deno subprocess with Pyodide
- `MockBackend` — test backend mapping code strings to canned results
- `SandboxResult` dataclass capturing stdout, stderr, return value, numeric parsing, timing, and errors
- `parse_sandbox_code()` — extract Python from fenced blocks, inline backticks, "what does X return" patterns, and directives
- `_runner.js` — Deno Pyodide host script
- `sandbox` optional extra in pyproject.toml
- `docs/sandbox.md` — full reference documentation

### Changed
- README updated with all 8 turnstyles in a table format

## 0.1.0

### Added
- `ArithmeticTurnstyle` — `+`, `-`, `*`, `/` (integer division)
- `DateTurnstyle` — days/weeks between dates
- `UnitTurnstyle` — physical unit conversion (miles/km, F/C, kg/lb, etc.)
- `CurrencyTurnstyle` — currency conversion with configurable rates
- `PercentageTurnstyle` — percentages, tips, discounts
- `CountingTurnstyle` — letters, vowels, consonants, words, characters
- `BaseConversionTurnstyle` — binary, hex, octal with hex-aware digit biasing
- `CoprocessorDiagnostic` with inline/summary/detail formatting and annotation marks
- `Turnstyle` base class with `parse()` / `make_processor()` / `generate()` pattern
