# SandboxTurnstyle

Execute arbitrary Python in a WASM sandbox and ground LLM digit generation in the computed result.

## Architecture

```
Prompt → parse_sandbox_code() → extract Python
                                      ↓
                              DenoPyodideBackend.execute()
                                      ↓
                              Deno spawns → loads Pyodide WASM
                                      ↓
                              Python runs in sandbox (no network, no FS)
                                      ↓
                              SandboxResult (numeric_value, stdout, stderr)
                                      ↓
                              ArithmeticLogitsProcessor biases digits
```

Same architecture as every other turnstyle: code runs at parse time, the numeric result feeds into digit biasing. The sandbox guarantees isolation — no network, no filesystem, no syscalls.

## Quick Start

```python
from turnstyle import SandboxTurnstyle, DenoPyodideBackend

backend = DenoPyodideBackend()
t = SandboxTurnstyle(model, tokenizer, device, backend=backend)

# Inline expression
text, proof = t.generate("What does `sum(range(101))` return?")

# Fenced code block
text, proof = t.generate("""What does this return?
```python
primes = [n for n in range(2, 100) if all(n % i != 0 for i in range(2, int(n**0.5)+1))]
len(primes)
```""")

# Directive
text, proof = t.generate("Evaluate: sum(int(d) for d in str(2**100))")
```

## Code Extraction

`parse_sandbox_code(text)` extracts Python from a prompt using these patterns (first match wins):

### 1. Fenced code blocks

````
```python
x = [i**2 for i in range(10)]
sum(x)
```
````

Also matches untagged fences (` ``` ` without `python`).

### 2. "What does/is" patterns

```
What does `sum(range(101))` return?
What is the output of `len([1,2,3])`?
What is the result of `2**10`?
```

These explicit phrasings accept any expression inside backticks, including bare operators like `2**10` that would otherwise be rejected.

### 3. Inline backtick

```
Calculate `len('hello world')`
The value of `"mississippi".count("ss")`
```

Requires the expression to look like code (function calls, attribute access, keywords, etc.). Bare arithmetic like `` `445 + 152` `` is rejected — that's `ArithmeticTurnstyle`'s domain.

### 4. Directives

```
Execute: print(2 + 2)
Evaluate: sum(range(10))
Run: len([1,2,3])
```

### What doesn't match

- Bare arithmetic: "What is 445 + 152?" → `None` (ArithmeticTurnstyle handles this)
- Plain text: "What color is the sky?" → `None`
- Empty strings → `None`

## Backends

### DenoPyodideBackend

The production backend. Spawns a Deno subprocess that loads Pyodide (Python compiled to WASM).

```python
from turnstyle import DenoPyodideBackend

backend = DenoPyodideBackend()
assert backend.available()  # True if `deno` is on PATH

result = backend.execute("sum(range(101))")
print(result.numeric_value)  # 5050
```

**Sandbox guarantees:**
- No network access (Deno permissions restrict to Pyodide CDN only)
- No filesystem access
- No syscalls beyond what WASM provides
- Timeout enforcement (default 5s)

**Performance:** ~3-5s cold start (Pyodide WASM download + init, cached by Deno after first run). Each call spawns a fresh subprocess — no persistent state between calls.

**Requires:** [Deno](https://deno.land) installed and on PATH.

### MockBackend

For testing. Maps code strings to canned `SandboxResult`s.

```python
from turnstyle import MockBackend, SandboxResult

mock = MockBackend()
mock.add("sum(range(101))", SandboxResult(
    stdout="", stderr="", return_value="5050",
    numeric_value=5050, execution_time_ms=1.0, error=None,
))

result = mock.execute("sum(range(101))")
assert result.numeric_value == 5050
```

Unregistered code returns an error result (not an exception).

## SandboxResult

```python
@dataclass
class SandboxResult:
    stdout: str                        # captured print() output
    stderr: str                        # captured stderr
    return_value: str                  # repr() of last expression
    numeric_value: float | int | None  # parsed if numeric
    execution_time_ms: float
    error: str | None                  # exception message or None
```

**Numeric parsing:** If the return value is `"None"` (e.g., last statement is an assignment), falls back to `stdout.strip()`. Attempts `int()` then `float()` parsing.

**Non-numeric results:** `numeric_value` is `None`. The turnstyle returns `None` from `parse()`, and the model generates freely without biasing.

## Behavior on Errors

| Scenario | `parse()` returns | Model behavior |
|----------|-------------------|----------------|
| Numeric result | `(SandboxParsed, SandboxResult)` | Digit biasing active |
| Non-numeric result (e.g., string) | `None` | Free generation |
| Runtime error (e.g., ZeroDivisionError) | `None` | Free generation |
| Syntax error | `None` | Free generation |
| Timeout | `None` | Free generation |
| Deno not installed | `None` | Free generation |

The turnstyle never raises — it degrades to ungrounded generation.

## Number Formatting

Numeric results are formatted with `f"{answer:.6g}"` for floats and `str(answer)` for ints:

| Result | `answer_str` | Digits biased |
|--------|-------------|---------------|
| `5050` | `"5050"` | 5, 0, 5, 0 |
| `3.14159265` | `"3.14159"` | 3, 1, 4, 1, 5, 9 |
| `-42` | `"-42"` | 4, 2 |

## Testing

```bash
# Parser tests (no external deps)
python -m pytest tests/test_sandbox_parser.py -v

# Backend tests (MockBackend, no external deps)
python -m pytest tests/test_sandbox_backend.py -v

# Integration tests (requires Deno — auto-skipped without it)
python -m pytest tests/test_sandbox_integration.py -v

# Standalone runner test
echo '{"code":"sum(range(101))","timeout":5}' | deno run --allow-read --allow-net=cdn.jsdelivr.net src/turnstyle/_runner.js
```

## Limitations (V1)

- **Numeric results only** — digit biasing works on numbers. Non-numeric results fall through.
- **Fresh subprocess per call** — ~3-5s cold start. No persistent Pyodide process.
- **Stdlib only** — no `micropip.install()`. The sandbox runs pure Python.
- **Deno only** — Node.js backend is a natural follow-up.
